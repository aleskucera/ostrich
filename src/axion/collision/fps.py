"""Greedy farthest-point-sampling contact reducer.

For each (body0, body1) pair, picks K contacts:
  1. Seed: the deepest contact in the pair.
  2. Iteratively pick the contact whose minimum 3D distance to any
     already-picked contact in the pair is maximal.

This is the classic FPS pattern. Compared to ``top_k`` it spreads
survivors across the contact patch instead of keeping only the
deepest cluster, which usually produces a more "tetrahedral" support
polygon — the structure stable contact relies on.

Implementation detail: per-pair work is serialized inside one thread
(the "pair leader" — the smallest c_idx in the pair). Threads from
different pairs run in parallel and write to disjoint cells of a
keep-mask, so there is no race even though the leader emits multiple
writes for its pair. K is a runtime int — the inner loops use
runtime bounds, which Warp accepts for plain ``for ... in range(...)``.

The compaction step (move kept contacts to the front, decrement
``contact_count``) follows the same pattern as ``top_k.py``: an
atomic-counter scatter into per-reducer scratch buffers, then 8
``wp.copy``s back into ``AxionContacts``.
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
def _depth_and_midpoint(
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
):
    """Return ``(depth, midpoint_world)`` for one contact.

    Depth: ``dot(n, p_b_world − p_a_world)`` after thickness offsets;
    positive when penetrating. Midpoint: 0.5 * (p_a_world + p_b_world)
    after the same offsets — used as the FPS distance metric.
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

    depth = wp.dot(n, p_b_world - p_a_world)
    midpoint = 0.5 * (p_a_world + p_b_world)
    return depth, midpoint


@wp.kernel
def fps_per_pair_kernel(
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
    K: wp.int32,
    keep_mask: wp.array(dtype=wp.int32, ndim=2),
):
    """Per-thread (one per (world, c_idx)). The thread whose c_idx is
    the smallest in its (b0, b1) pair becomes the "pair leader" and
    runs the FPS picking serially inside its thread; other threads in
    the same pair return immediately (their cells of ``keep_mask`` are
    written by the leader).

    Threads from different pairs touch disjoint ``keep_mask`` cells, so
    leader-vs-leader races cannot happen. Other-thread cells start at
    0 (must be zeroed by the caller before launch) and are flipped to
    1 only by the leader of their pair.
    """
    world_idx, c_idx = wp.tid()
    n_count = contact_count[world_idx]
    if c_idx >= n_count:
        return

    s0 = contact_shape0[world_idx, c_idx]
    s1 = contact_shape1[world_idx, c_idx]
    if s0 == s1:
        return

    b0 = _resolve_body(world_idx, s0, shape_body)
    b1 = _resolve_body(world_idx, s1, shape_body)

    # Am I this pair's leader? Smallest c_idx with this (b0, b1).
    is_leader = wp.bool(True)
    for j in range(c_idx):
        sj0 = contact_shape0[world_idx, j]
        sj1 = contact_shape1[world_idx, j]
        if sj0 == sj1:
            continue
        bj0 = _resolve_body(world_idx, sj0, shape_body)
        bj1 = _resolve_body(world_idx, sj1, shape_body)
        if bj0 == b0 and bj1 == b1:
            is_leader = wp.bool(False)
    if not is_leader:
        return

    # Round 0 — seed: pick the deepest contact in this pair.
    seed_idx = wp.int32(-1)
    seed_depth = wp.float32(-1.0e30)
    for j in range(n_count):
        sj0 = contact_shape0[world_idx, j]
        sj1 = contact_shape1[world_idx, j]
        if sj0 == sj1:
            continue
        bj0 = _resolve_body(world_idx, sj0, shape_body)
        bj1 = _resolve_body(world_idx, sj1, shape_body)
        if bj0 != b0 or bj1 != b1:
            continue
        dj, _mp_unused = _depth_and_midpoint(
            world_idx, j,
            contact_point0, contact_point1, contact_normal,
            contact_thickness0, contact_thickness1,
            bj0, bj1, body_pose,
        )
        # Tie-break by smallest c_idx for determinism.
        if dj > seed_depth or (dj == seed_depth and j < seed_idx):
            seed_depth = dj
            seed_idx = j
    if seed_idx < 0:
        return  # no candidates (shouldn't happen since c_idx itself is one)

    keep_mask[world_idx, seed_idx] = 1
    n_picked = wp.int32(1)

    # Rounds 1..K-1 — farthest-point sampling.
    for r in range(K - 1):
        if n_picked >= K:
            break
        best_idx = wp.int32(-1)
        best_min_dist_sq = wp.float32(-1.0)
        for j in range(n_count):
            sj0 = contact_shape0[world_idx, j]
            sj1 = contact_shape1[world_idx, j]
            if sj0 == sj1:
                continue
            bj0 = _resolve_body(world_idx, sj0, shape_body)
            bj1 = _resolve_body(world_idx, sj1, shape_body)
            if bj0 != b0 or bj1 != b1:
                continue
            if keep_mask[world_idx, j] == 1:
                continue
            _depth_unused_j, mp_j = _depth_and_midpoint(
                world_idx, j,
                contact_point0, contact_point1, contact_normal,
                contact_thickness0, contact_thickness1,
                bj0, bj1, body_pose,
            )
            # min-distance over already-picked contacts in this pair.
            min_dist_sq = wp.float32(1.0e30)
            for k in range(n_count):
                if keep_mask[world_idx, k] != 1:
                    continue
                sk0 = contact_shape0[world_idx, k]
                sk1 = contact_shape1[world_idx, k]
                if sk0 == sk1:
                    continue
                bk0 = _resolve_body(world_idx, sk0, shape_body)
                bk1 = _resolve_body(world_idx, sk1, shape_body)
                if bk0 != b0 or bk1 != b1:
                    continue
                _depth_unused_k, mp_k = _depth_and_midpoint(
                    world_idx, k,
                    contact_point0, contact_point1, contact_normal,
                    contact_thickness0, contact_thickness1,
                    bk0, bk1, body_pose,
                )
                d_sq = wp.length_sq(mp_j - mp_k)
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
            # Tie-break by smallest c_idx.
            if min_dist_sq > best_min_dist_sq or (
                min_dist_sq == best_min_dist_sq and j < best_idx
            ):
                best_min_dist_sq = min_dist_sq
                best_idx = j
        if best_idx < 0:
            break  # ran out of candidates in this pair
        keep_mask[world_idx, best_idx] = 1
        n_picked += 1


@wp.kernel
def scatter_by_keep_kernel(
    keep_mask: wp.array(dtype=wp.int32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    src_point0: wp.array(dtype=wp.vec3, ndim=2),
    src_point1: wp.array(dtype=wp.vec3, ndim=2),
    src_normal: wp.array(dtype=wp.vec3, ndim=2),
    src_shape0: wp.array(dtype=wp.int32, ndim=2),
    src_shape1: wp.array(dtype=wp.int32, ndim=2),
    src_thickness0: wp.array(dtype=wp.float32, ndim=2),
    src_thickness1: wp.array(dtype=wp.float32, ndim=2),
    new_count: wp.array(dtype=wp.int32, ndim=1),
    dst_point0: wp.array(dtype=wp.vec3, ndim=2),
    dst_point1: wp.array(dtype=wp.vec3, ndim=2),
    dst_normal: wp.array(dtype=wp.vec3, ndim=2),
    dst_shape0: wp.array(dtype=wp.int32, ndim=2),
    dst_shape1: wp.array(dtype=wp.int32, ndim=2),
    dst_thickness0: wp.array(dtype=wp.float32, ndim=2),
    dst_thickness1: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, c_idx = wp.tid()
    if c_idx >= contact_count[world_idx]:
        return
    if keep_mask[world_idx, c_idx] != 1:
        return

    slot = wp.atomic_add(new_count, world_idx, 1)
    dst_point0[world_idx, slot] = src_point0[world_idx, c_idx]
    dst_point1[world_idx, slot] = src_point1[world_idx, c_idx]
    dst_normal[world_idx, slot] = src_normal[world_idx, c_idx]
    dst_shape0[world_idx, slot] = src_shape0[world_idx, c_idx]
    dst_shape1[world_idx, slot] = src_shape1[world_idx, c_idx]
    dst_thickness0[world_idx, slot] = src_thickness0[world_idx, c_idx]
    dst_thickness1[world_idx, slot] = src_thickness1[world_idx, c_idx]


class FPSReducer(ContactReducer):
    """Greedy farthest-point-sampling per-pair contact reduction."""

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
        self._shape_body = axion_model.shape_body
        # body_pose_prev is the current-step pose during load_data.
        self._body_pose = data.body_pose_prev

        N_w = dims.num_worlds
        N_c = dims.contact_count

        with wp.ScopedDevice(device):
            self._keep_mask = wp.zeros((N_w, N_c), dtype=wp.int32)
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

        # FPS kernel reads-AND-writes keep_mask, so it must start zeroed.
        self._keep_mask.zero_()

        wp.launch(
            kernel=fps_per_pair_kernel,
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
                self._K,
            ],
            outputs=[self._keep_mask],
            device=self._device,
        )

        self._new_count.zero_()
        wp.launch(
            kernel=scatter_by_keep_kernel,
            dim=(N_w, N_c),
            inputs=[
                self._keep_mask,
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
