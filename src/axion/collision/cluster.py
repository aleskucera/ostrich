"""Cluster-based contact reducer.

For each (body0, body1) pair, contacts are clustered by a "near-duplicate"
criterion: two contacts agree if their normals are within an angular
threshold AND their midpoints are within a distance threshold. The
representative of each cluster is its deepest contact; among
representatives, the K with the largest depth survive.

Compared to FPS, cluster reduction has different goals:
  - FPS *spreads* survivors maximally across the contact patch.
  - Cluster reduction *removes only true duplicates*; if the original
    contacts are already well-spread, clustering is a near-no-op.

The two policies trade off coverage vs. fidelity. For our obstacle
scene, a wheel sitting flat on a box has many near-coplanar contact
points whose normals all point straight up and whose midpoints are
within a few mm; cluster reduction collapses those to one.

Implementation:
  Pass 1 (per pair, single-threaded by the leader):
      For each contact j in the pair, j is a "representative" iff no
      deeper contact in the same pair matches it (normal-dot > thresh
      AND midpoint-distance < thresh). The match relation is non-
      transitive in general, but using "no deeper match" yields the
      max element of each connected component under the dominance
      relation, which is what we want.

  Pass 2 (same leader):
      Among representatives, compute rank by depth (descending).
      Keep contacts with rank < K.

Both passes are O(N_pair²) inside a single leader thread, so total
cost per pair is bounded by ~225 contact comparisons for our scenes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .base import ContactReducer
from .config import ContactReductionConfig
from .fps import scatter_by_keep_kernel

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
def cluster_per_pair_kernel(
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
    normal_dot_thresh: wp.float32,
    pos_thresh: wp.float32,
    keep_mask: wp.array(dtype=wp.int32, ndim=2),
):
    """One thread per (world, c_idx). The pair-leader thread (smallest
    c_idx with this (b0, b1)) runs both passes serially and writes to
    keep-mask cells of contacts in its pair only — disjoint across
    leaders."""
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

    pos_thresh_sq = pos_thresh * pos_thresh

    # Pass 1: mark representatives. j is a rep iff no deeper contact in
    # the same pair matches it. We use keep_mask as scratch first
    # (1 = rep, 0 = not), then overwrite in pass 2.
    for j in range(n_count):
        sj0 = contact_shape0[world_idx, j]
        sj1 = contact_shape1[world_idx, j]
        if sj0 == sj1:
            continue
        bj0 = _resolve_body(world_idx, sj0, shape_body)
        bj1 = _resolve_body(world_idx, sj1, shape_body)
        if bj0 != b0 or bj1 != b1:
            continue

        nj = contact_normal[world_idx, j]
        depth_j, mp_j = _depth_and_midpoint(
            world_idx, j,
            contact_point0, contact_point1, contact_normal,
            contact_thickness0, contact_thickness1,
            bj0, bj1, body_pose,
        )

        is_rep = wp.bool(True)
        for k in range(n_count):
            if k == j:
                continue
            sk0 = contact_shape0[world_idx, k]
            sk1 = contact_shape1[world_idx, k]
            if sk0 == sk1:
                continue
            bk0 = _resolve_body(world_idx, sk0, shape_body)
            bk1 = _resolve_body(world_idx, sk1, shape_body)
            if bk0 != b0 or bk1 != b1:
                continue
            nk = contact_normal[world_idx, k]
            depth_k, mp_k = _depth_and_midpoint(
                world_idx, k,
                contact_point0, contact_point1, contact_normal,
                contact_thickness0, contact_thickness1,
                bk0, bk1, body_pose,
            )
            n_dot = wp.dot(nj, nk)
            mp_dist_sq = wp.length_sq(mp_j - mp_k)
            if n_dot > normal_dot_thresh and mp_dist_sq < pos_thresh_sq:
                # k is in same cluster as j. Is k deeper?
                if depth_k > depth_j or (depth_k == depth_j and k < j):
                    is_rep = wp.bool(False)

        if is_rep:
            keep_mask[world_idx, j] = 1
        else:
            keep_mask[world_idx, j] = 0

    # Pass 2: among representatives, rank by depth and keep top K.
    # We do this in-place on keep_mask: for each rep, count deeper reps
    # in the same pair; if rank >= K, drop the keep flag.
    for j in range(n_count):
        if keep_mask[world_idx, j] != 1:
            continue
        sj0 = contact_shape0[world_idx, j]
        sj1 = contact_shape1[world_idx, j]
        bj0 = _resolve_body(world_idx, sj0, shape_body)
        bj1 = _resolve_body(world_idx, sj1, shape_body)

        depth_j, _mp_unused_j = _depth_and_midpoint(
            world_idx, j,
            contact_point0, contact_point1, contact_normal,
            contact_thickness0, contact_thickness1,
            bj0, bj1, body_pose,
        )

        rank = wp.int32(0)
        for k in range(n_count):
            if k == j:
                continue
            if keep_mask[world_idx, k] != 1:
                continue
            sk0 = contact_shape0[world_idx, k]
            sk1 = contact_shape1[world_idx, k]
            bk0 = _resolve_body(world_idx, sk0, shape_body)
            bk1 = _resolve_body(world_idx, sk1, shape_body)
            if bk0 != b0 or bk1 != b1:
                continue
            depth_k, _mp_unused_k = _depth_and_midpoint(
                world_idx, k,
                contact_point0, contact_point1, contact_normal,
                contact_thickness0, contact_thickness1,
                bk0, bk1, body_pose,
            )
            if depth_k > depth_j or (depth_k == depth_j and k < j):
                rank += 1

        if rank >= K:
            keep_mask[world_idx, j] = 0


class ClusterReducer(ContactReducer):
    """Cluster-then-keep-top-K-deepest-cluster-rep contact reduction."""

    def __init__(
        self,
        cfg: ContactReductionConfig,
        axion_model: "AxionModel",
        data: "EngineData",
        dims: "EngineDimensions",
        device: wp.Device,
    ):
        self._K = int(cfg.max_per_pair)
        self._normal_dot_thresh = float(cfg.cluster_normal_dot_thresh)
        self._pos_thresh = float(cfg.cluster_pos_thresh)
        self._device = device
        self._shape_body = axion_model.shape_body
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

        self._keep_mask.zero_()

        wp.launch(
            kernel=cluster_per_pair_kernel,
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
                self._normal_dot_thresh,
                self._pos_thresh,
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
