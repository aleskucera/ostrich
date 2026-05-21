"""Top-K-by-depth contact reducer.

For each (body0, body1) pair active in a given simulation step, keeps at
most K contacts (the K with the largest penetration depth). Implemented
as a two-kernel pipeline plus eight memcpys back to the canonical
``AxionContacts`` arrays:

    1. ``rank_within_pair_kernel``  — for each contact, count how many
       contacts in the same world with the same (b0, b1) pair are
       deeper. The result ``rank`` is the contact's 0-based rank within
       its pair (0 = deepest).

    2. ``scatter_top_k_kernel``     — every contact with rank < K
       atomically reserves a slot in a per-world counter and writes
       itself to the matching index in seven scratch arrays.

    3. ``wp.copy`` back the seven scratch arrays plus the new
       ``contact_count`` into the ``AxionContacts`` instance.

Depth is computed on the fly inside both kernels — we are launch- and
cache-bound for this scene, not compute-bound, so duplicating the
transform-and-dot work in ranking is cheaper than allocating a separate
depth scratch buffer + extra launch.

The compaction is necessary because downstream constraint kernels gate
on ``c_idx >= contact_count[w]``, so survivors must live at the front
of the array for the gate to drop the rejected ones.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .base import ContactReducer
from .config import ContactReductionConfig

if TYPE_CHECKING:
    from axion.core.contacts import AxionContacts
    from axion.core.engine_data import EngineData
    from axion.core.engine_dims import EngineDimensions
    from axion.core.model import AxionModel


@wp.func
def _resolve_body(world_idx: int, shape_idx: int,
                  shape_body: wp.array(dtype=wp.int32, ndim=2)) -> int:
    if shape_idx < 0:
        return -1
    return shape_body[world_idx, shape_idx]


@wp.func
def _contact_depth(
    world_idx: int,
    c_idx: int,
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    body_a: int,
    body_b: int,
    body_pose: wp.array(dtype=wp.transform, ndim=2),
) -> float:
    """World-space penetration depth (positive = penetrating).

    Mirrors the geometry used by ``contact_constraint_kernel`` but computed
    here from current body poses. Returns ``dot(n, p_b_world − p_a_world)``
    after the thickness offsets are applied along the normal.
    """
    n = contact_normal[world_idx, c_idx]
    p_a_local = contact_point0[world_idx, c_idx]
    p_b_local = contact_point1[world_idx, c_idx]
    t_a = contact_thickness0[world_idx, c_idx]
    t_b = contact_thickness1[world_idx, c_idx]

    pose_a = wp.transform_identity()
    if body_a >= 0:
        pose_a = body_pose[world_idx, body_a]
    pose_b = wp.transform_identity()
    if body_b >= 0:
        pose_b = body_pose[world_idx, body_b]

    p_a_world = wp.transform_point(pose_a, p_a_local) - t_a * n
    p_b_world = wp.transform_point(pose_b, p_b_local) + t_b * n
    return wp.dot(n, p_b_world - p_a_world)


@wp.kernel
def rank_within_pair_kernel(
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    rank: wp.array(dtype=wp.int32, ndim=2),
):
    """For each active contact, compute how many contacts in the same
    (b0, b1) pair are deeper.

    Tie-break by original index so ranks are unique and the policy is
    deterministic across runs (and identical under graph capture replay).
    Out-of-range threads (c_idx >= contact_count[w]) write a sentinel
    rank that is always ≥ K, so they're never selected.
    """
    world_idx, c_idx = wp.tid()
    n_count = contact_count[world_idx]
    if c_idx >= n_count:
        rank[world_idx, c_idx] = wp.int32(2147483647)
        return

    s0 = contact_shape0[world_idx, c_idx]
    s1 = contact_shape1[world_idx, c_idx]
    if s0 == s1:
        # Self-pair sentinel; should already be filtered upstream.
        rank[world_idx, c_idx] = wp.int32(2147483647)
        return

    b0 = _resolve_body(world_idx, s0, shape_body)
    b1 = _resolve_body(world_idx, s1, shape_body)

    my_depth = _contact_depth(
        world_idx, c_idx,
        contact_point0, contact_point1, contact_normal,
        contact_thickness0, contact_thickness1,
        b0, b1, body_pose,
    )

    r = wp.int32(0)
    for j in range(n_count):
        if j == c_idx:
            continue
        sj0 = contact_shape0[world_idx, j]
        sj1 = contact_shape1[world_idx, j]
        bj0 = _resolve_body(world_idx, sj0, shape_body)
        bj1 = _resolve_body(world_idx, sj1, shape_body)
        if bj0 != b0 or bj1 != b1:
            continue
        their_depth = _contact_depth(
            world_idx, j,
            contact_point0, contact_point1, contact_normal,
            contact_thickness0, contact_thickness1,
            bj0, bj1, body_pose,
        )
        if their_depth > my_depth:
            r += 1
        elif their_depth == my_depth and j < c_idx:
            r += 1

    rank[world_idx, c_idx] = r


@wp.kernel
def scatter_top_k_kernel(
    rank: wp.array(dtype=wp.int32, ndim=2),
    K: wp.int32,
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    # Source contact arrays (read).
    src_point0: wp.array(dtype=wp.vec3, ndim=2),
    src_point1: wp.array(dtype=wp.vec3, ndim=2),
    src_normal: wp.array(dtype=wp.vec3, ndim=2),
    src_shape0: wp.array(dtype=wp.int32, ndim=2),
    src_shape1: wp.array(dtype=wp.int32, ndim=2),
    src_thickness0: wp.array(dtype=wp.float32, ndim=2),
    src_thickness1: wp.array(dtype=wp.float32, ndim=2),
    # Per-world atomic counter + scratch destinations (write).
    new_count: wp.array(dtype=wp.int32, ndim=1),
    dst_point0: wp.array(dtype=wp.vec3, ndim=2),
    dst_point1: wp.array(dtype=wp.vec3, ndim=2),
    dst_normal: wp.array(dtype=wp.vec3, ndim=2),
    dst_shape0: wp.array(dtype=wp.int32, ndim=2),
    dst_shape1: wp.array(dtype=wp.int32, ndim=2),
    dst_thickness0: wp.array(dtype=wp.float32, ndim=2),
    dst_thickness1: wp.array(dtype=wp.float32, ndim=2),
):
    """Survivors (rank < K) atomically reserve a slot in ``new_count[w]``
    and copy themselves to that slot in the scratch arrays."""
    world_idx, c_idx = wp.tid()
    if c_idx >= contact_count[world_idx]:
        return
    if rank[world_idx, c_idx] >= K:
        return

    slot = wp.atomic_add(new_count, world_idx, 1)
    dst_point0[world_idx, slot] = src_point0[world_idx, c_idx]
    dst_point1[world_idx, slot] = src_point1[world_idx, c_idx]
    dst_normal[world_idx, slot] = src_normal[world_idx, c_idx]
    dst_shape0[world_idx, slot] = src_shape0[world_idx, c_idx]
    dst_shape1[world_idx, slot] = src_shape1[world_idx, c_idx]
    dst_thickness0[world_idx, slot] = src_thickness0[world_idx, c_idx]
    dst_thickness1[world_idx, slot] = src_thickness1[world_idx, c_idx]


class TopKReducer(ContactReducer):
    """Per-pair top-K-by-depth contact reduction."""

    def __init__(
        self,
        cfg: ContactReductionConfig,
        axion_model: "AxionModel",
        data: "EngineData",
        dims: "EngineDimensions",
        device: wp.Device,
    ):
        self._K = int(cfg.max_per_pair)
        self._device = device
        # Persistent references — body_pose_prev holds the current-step
        # pose during load_data (when reduction runs), because the engine
        # copies body_pose itself only AFTER load_data returns.
        self._shape_body = axion_model.shape_body
        self._body_pose = data.body_pose_prev

        N_w = dims.num_worlds
        N_c = dims.contact_count

        with wp.ScopedDevice(device):
            self._rank = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._new_count = wp.zeros((N_w,), dtype=wp.int32)
            self._scratch_point0 = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._scratch_point1 = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._scratch_normal = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._scratch_shape0 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._scratch_shape1 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._scratch_thickness0 = wp.zeros((N_w, N_c), dtype=wp.float32)
            self._scratch_thickness1 = wp.zeros((N_w, N_c), dtype=wp.float32)

    def apply(self, contacts: "AxionContacts") -> None:
        N_w = contacts.num_worlds
        N_c = contacts.max_contacts

        wp.launch(
            kernel=rank_within_pair_kernel,
            dim=(N_w, N_c),
            inputs=[
                contacts.contact_count,
                contacts.contact_point0,
                contacts.contact_point1,
                contacts.contact_normal,
                contacts.contact_shape0,
                contacts.contact_shape1,
                contacts.contact_thickness0,
                contacts.contact_thickness1,
                self._shape_body,
                self._body_pose,
            ],
            outputs=[self._rank],
            device=self._device,
        )

        # Reset survivor counter, then scatter.
        self._new_count.zero_()
        wp.launch(
            kernel=scatter_top_k_kernel,
            dim=(N_w, N_c),
            inputs=[
                self._rank,
                self._K,
                contacts.contact_count,
                contacts.contact_point0,
                contacts.contact_point1,
                contacts.contact_normal,
                contacts.contact_shape0,
                contacts.contact_shape1,
                contacts.contact_thickness0,
                contacts.contact_thickness1,
            ],
            outputs=[
                self._new_count,
                self._scratch_point0,
                self._scratch_point1,
                self._scratch_normal,
                self._scratch_shape0,
                self._scratch_shape1,
                self._scratch_thickness0,
                self._scratch_thickness1,
            ],
            device=self._device,
        )

        wp.copy(contacts.contact_count, self._new_count)
        wp.copy(contacts.contact_point0, self._scratch_point0)
        wp.copy(contacts.contact_point1, self._scratch_point1)
        wp.copy(contacts.contact_normal, self._scratch_normal)
        wp.copy(contacts.contact_shape0, self._scratch_shape0)
        wp.copy(contacts.contact_shape1, self._scratch_shape1)
        wp.copy(contacts.contact_thickness0, self._scratch_thickness0)
        wp.copy(contacts.contact_thickness1, self._scratch_thickness1)
