"""Cross-step contact warm-start.

Predicted-position matching:
  At each new simulation step, for every contact in the (post-reducer)
  active set, find the closest contact from the previous step's
  converged state where "closest" is measured against a position
  predicted from body velocities. Copy the matched contact's converged
  λ_n and λ_t (with friction-basis projection) into
  ``_constr_force_prev_iter`` so the friction kernel sees real values
  at NR iter 0 instead of starting from zero.

Phase 0 (this commit) ships only the scaffolding: persistent buffers,
lifecycle hooks, and disabled-by-default no-op apply/snapshot kernels.
The matching and snapshot logic lands in subsequent phases.

The component lives parallel to ``ContactReducer`` because:
  - both run between Newton's narrow-phase output and Axion's
    constraint kernels,
  - the warm-starter relies on the reducer's stable per-pair structure
    (matching is done within (b0, b1) groups, with K survivors per
    pair after reduction),
  - keeping them as siblings under ``axion/collision/`` makes the
    contact-pipeline pre-processing obvious from the package layout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from axion.mechanics import orthogonal_basis

if TYPE_CHECKING:
    from axion.core.contacts import AxionContacts
    from axion.core.engine_data import EngineData
    from axion.core.engine_dims import EngineDimensions
    from axion.core.model import AxionModel


@wp.func
def _resolve_body(
    world_idx: int,
    shape_idx: int,
    shape_body: wp.array(dtype=wp.int32, ndim=2),
) -> int:
    if shape_idx < 0:
        return -1
    return shape_body[world_idx, shape_idx]


@wp.func
def _world_midpoint(
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
) -> wp.vec3:
    """Contact midpoint in world frame, after thickness offsets along
    the normal. Mirrors the geometry used by the contact constraint
    kernel."""
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
    return 0.5 * (p_a_world + p_b_world)


@wp.kernel
def _match_kernel(
    # Current step inputs (post-reduction).
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
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    g_accel: wp.array(dtype=wp.vec3, ndim=1),
    # Previous step state.
    prev_count: wp.array(dtype=wp.int32, ndim=1),
    prev_b0: wp.array(dtype=wp.int32, ndim=2),
    prev_b1: wp.array(dtype=wp.int32, ndim=2),
    prev_p_world: wp.array(dtype=wp.vec3, ndim=2),
    prev_normal: wp.array(dtype=wp.vec3, ndim=2),
    prev_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    prev_lambda_t: wp.array(dtype=wp.vec2, ndim=2),
    # Parameters.
    dt: wp.float32,
    alpha: wp.float32,
    min_threshold: wp.float32,
    n_dot_up_min: wp.float32,
    offset_n: wp.int32,
    offset_f: wp.int32,
    # Output: warm-started friction-lag values.
    constr_force_prev_iter: wp.array(dtype=wp.float32, ndim=2),
    # Phase 2.5: per-contact match flag and per-body support reductions.
    is_matched: wp.array(dtype=wp.int32, ndim=2),
    body_matched_support: wp.array(dtype=wp.float32, ndim=2),
    body_unmatched_count: wp.array(dtype=wp.int32, ndim=2),
    # Diagnostics.
    diag_attempts: wp.array(dtype=wp.int32, ndim=1),
    diag_matched: wp.array(dtype=wp.int32, ndim=1),
    diag_max_dist: wp.array(dtype=wp.float32, ndim=1),
    # Failure-mode breakdown for the matcher:
    #   no_pair     = current contact's (b0,b1) appears nowhere in prev
    #   over_thresh = pair existed but closest distance exceeded threshold
    # closest_d_sum / closest_d_count compute the mean of "closest prev
    # contact in matching pair" distance (in meters) across all
    # attempted contacts, so we can see how far off the predictor is.
    # closest_d_raw_sum is the same metric WITHOUT the v_translate*dt
    # prediction step, so we can compare raw vs predicted distances and
    # tell whether the predictor is helping.
    diag_no_pair: wp.array(dtype=wp.int32, ndim=1),
    diag_over_thresh: wp.array(dtype=wp.int32, ndim=1),
    diag_closest_d_sum: wp.array(dtype=wp.float32, ndim=1),
    diag_closest_d_raw_sum: wp.array(dtype=wp.float32, ndim=1),
    diag_closest_d_count: wp.array(dtype=wp.int32, ndim=1),
):
    """One thread per (world, current contact slot).

    For each active contact, scan previous-step contacts in the same
    (b0, b1) pair, predict each prev contact's expected position via
    ``v_translate * dt``, and pick the closest one within the adaptive
    threshold ``max(min_threshold, alpha * |v_translate| * dt)``.

    On match: project the prev λ_t from prev's (t1, t2) basis through
    world frame into the current (t1, t2) basis, then write λ_n and
    the projected λ_t into ``constr_force_prev_iter`` so the friction
    kernel sees real values at NR iter 0.

    On no match: leave ``constr_force_prev_iter`` zero (the caller has
    already cleared it before launching us). NR cold-starts that
    contact, which is the correct behavior for new-contact-on-impact.

    Greedy matching only — two current contacts may pick the same prev
    contact, in which case its λ is effectively duplicated. With
    cluster reduction enabled the contact set is already spatially
    diverse, so this is rare in practice. A per-pair leader pattern
    that enforces 1:1 assignment is left as a follow-up.
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

    # Current-frame midpoint and normal.
    p_curr = _world_midpoint(
        world_idx, c_idx,
        contact_point0, contact_point1, contact_normal,
        contact_thickness0, contact_thickness1,
        b0, b1, body_pose,
    )
    n_curr = contact_normal[world_idx, c_idx]

    # Translation between snapshot frame (start of last step) and
    # current frame (start of this step). For ground contacts (b1<0)
    # only b0 contributes; symmetric for body-body contacts.
    v_translate_full = wp.vec3(0.0, 0.0, 0.0)
    if b0 >= 0:
        v_translate_full += wp.spatial_top(body_vel[world_idx, b0])
    if b1 >= 0:
        v_translate_full -= wp.spatial_top(body_vel[world_idx, b1])
    # Project onto the contact plane: a contact midpoint can only slide
    # along the surface, not penetrate it. The normal-velocity component
    # represents penetration/separation, which the constraint absorbs —
    # in equilibrium the contact patch in world stays at the surface
    # regardless of v_n. Including v_n in the prediction was off by up
    # to v_n·dt per step (the entire error budget for vertical motion
    # against a static ground).
    v_translate = v_translate_full - wp.dot(v_translate_full, n_curr) * n_curr
    v_mag = wp.length(v_translate)
    threshold = wp.max(min_threshold, alpha * v_mag * dt)
    threshold_sq = threshold * threshold

    wp.atomic_add(diag_attempts, world_idx, 1)

    # Scan previous-step contacts in the same body pair.
    n_prev = prev_count[world_idx]
    best_j = wp.int32(-1)
    best_d_sq = threshold_sq
    # Closest-in-pair distance (with prediction) and raw (no prediction),
    # both regardless of threshold, for diagnostics.
    closest_in_pair_sq = wp.float32(1.0e30)
    closest_raw_sq = wp.float32(1.0e30)
    pair_seen = wp.bool(False)
    for j in range(n_prev):
        if prev_b0[world_idx, j] != b0 or prev_b1[world_idx, j] != b1:
            continue
        pair_seen = wp.bool(True)
        prev_p = prev_p_world[world_idx, j]
        # Predicted: prev contact translated forward by body lin·dt.
        p_pred = prev_p + v_translate * dt
        d_sq = wp.length_sq(p_curr - p_pred)
        # Raw: no prediction, just compare current to prev location.
        d_raw_sq = wp.length_sq(p_curr - prev_p)
        if d_sq < closest_in_pair_sq:
            closest_in_pair_sq = d_sq
        if d_raw_sq < closest_raw_sq:
            closest_raw_sq = d_raw_sq
        if d_sq < best_d_sq:
            best_d_sq = d_sq
            best_j = j

    # Failure-mode accounting: only run after we've scanned all prev j's.
    if pair_seen:
        wp.atomic_add(diag_closest_d_sum, world_idx, wp.sqrt(closest_in_pair_sq))
        wp.atomic_add(diag_closest_d_raw_sum, world_idx, wp.sqrt(closest_raw_sq))
        wp.atomic_add(diag_closest_d_count, world_idx, 1)
        if best_j < 0:
            wp.atomic_add(diag_over_thresh, world_idx, 1)
    else:
        wp.atomic_add(diag_no_pair, world_idx, 1)

    # Anti-gravity direction (used by both branches to credit a body's
    # vertical-support budget). g_accel is a length-1 array of vec3.
    g = g_accel[0]
    g_mag = wp.length(g)
    up = wp.vec3(0.0, 0.0, 1.0)
    if g_mag > 1.0e-6:
        up = -g / g_mag
    n_dot_up = wp.dot(n_curr, up)
    # contact_force on b0 = -n * λ_n  →  up-component coefficient = -n_dot_up
    # contact_force on b1 = +n * λ_n  →  up-component coefficient = +n_dot_up
    supports_b0 = (-n_dot_up) > n_dot_up_min
    supports_b1 = ( n_dot_up) > n_dot_up_min

    if best_j < 0:
        # Cold start — leave constr_force_prev_iter at the caller's zero,
        # but credit the unmatched-count buffer so the cold-start kernel
        # can distribute residual gravity load.
        is_matched[world_idx, c_idx] = 0
        if b0 >= 0 and supports_b0:
            wp.atomic_add(body_unmatched_count, world_idx, b0, 1)
        if b1 >= 0 and supports_b1:
            wp.atomic_add(body_unmatched_count, world_idx, b1, 1)
        return

    # Project λ_t from prev's tangent basis to current's via world frame.
    prev_n = prev_normal[world_idx, best_j]
    prev_t1, prev_t2 = orthogonal_basis(prev_n)
    curr_t1, curr_t2 = orthogonal_basis(n_curr)
    prev_lt = prev_lambda_t[world_idx, best_j]
    lt_world = prev_lt[0] * prev_t1 + prev_lt[1] * prev_t2
    new_lt_x = wp.dot(lt_world, curr_t1)
    new_lt_y = wp.dot(lt_world, curr_t2)

    matched_lambda_n = prev_lambda_n[world_idx, best_j]
    constr_force_prev_iter[world_idx, offset_n + c_idx] = matched_lambda_n
    constr_force_prev_iter[world_idx, offset_f + 2 * c_idx] = new_lt_x
    constr_force_prev_iter[world_idx, offset_f + 2 * c_idx + 1] = new_lt_y

    is_matched[world_idx, c_idx] = 1

    # Credit each supported body with this contact's vertical-force
    # contribution so the cold-start kernel knows how much gravity load
    # is already accounted for by warm-started contacts.
    if b0 >= 0 and supports_b0:
        wp.atomic_add(
            body_matched_support, world_idx, b0,
            (-n_dot_up) * matched_lambda_n,
        )
    if b1 >= 0 and supports_b1:
        wp.atomic_add(
            body_matched_support, world_idx, b1,
            n_dot_up * matched_lambda_n,
        )

    wp.atomic_add(diag_matched, world_idx, 1)
    wp.atomic_max(diag_max_dist, world_idx, wp.sqrt(best_d_sq))


@wp.kernel
def _cold_start_kernel(
    # Current step inputs.
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    g_accel: wp.array(dtype=wp.vec3, ndim=1),
    # Phase 2.5 reductions from match kernel.
    is_matched: wp.array(dtype=wp.int32, ndim=2),
    body_matched_support: wp.array(dtype=wp.float32, ndim=2),
    body_unmatched_count: wp.array(dtype=wp.int32, ndim=2),
    # Parameters.
    dt: wp.float32,
    n_dot_up_min: wp.float32,
    v_t_threshold_sq: wp.float32,
    enable_gravity: wp.int32,
    enable_impact: wp.int32,
    enable_friction: wp.int32,
    offset_n: wp.int32,
    offset_f: wp.int32,
    # Output.
    constr_force_prev_iter: wp.array(dtype=wp.float32, ndim=2),
    # Diagnostics.
    diag_cold_normal: wp.array(dtype=wp.int32, ndim=1),
    diag_cold_friction: wp.array(dtype=wp.int32, ndim=1),
):
    """Fill `constr_force_prev_iter` for contact slots that the match
    kernel left at zero. Per-thread (world × contact slot).

    Three terms, all gated by config flags:
      * α (gravity): residual gravity load on each body, evenly split
        across its unmatched contacts that point sufficiently upward.
      * β (impact): impulse needed to kill the inbound normal velocity
        at the contact point, scaled to a force via /dt.
      * γ (sliding friction): if tangential speed exceeds the static
        threshold, seed kinetic friction along -v̂_t with magnitude
        μ·λ_n_cold projected into the contact's tangent basis.

    α and β are combined via max() — at impact, β includes inertia and
    α would double-count.
    """
    world_idx, c_idx = wp.tid()
    n_count = contact_count[world_idx]
    if c_idx >= n_count:
        return
    if is_matched[world_idx, c_idx] == 1:
        return

    s0 = contact_shape0[world_idx, c_idx]
    s1 = contact_shape1[world_idx, c_idx]
    if s0 == s1:
        return

    b0 = _resolve_body(world_idx, s0, shape_body)
    b1 = _resolve_body(world_idx, s1, shape_body)

    n = contact_normal[world_idx, c_idx]
    p = _world_midpoint(
        world_idx, c_idx,
        contact_point0, contact_point1, contact_normal,
        contact_thickness0, contact_thickness1,
        b0, b1, body_pose,
    )

    # Up direction (anti-gravity).
    g = g_accel[0]
    g_mag = wp.length(g)
    up = wp.vec3(0.0, 0.0, 1.0)
    if g_mag > 1.0e-6:
        up = -g / g_mag

    # Velocity at contact point (linear + angular cross arm).
    v0_at_p = wp.vec3(0.0, 0.0, 0.0)
    v1_at_p = wp.vec3(0.0, 0.0, 0.0)
    if b0 >= 0:
        twist0 = body_vel[world_idx, b0]
        com0 = wp.transform_get_translation(body_pose[world_idx, b0])
        v0_at_p = wp.spatial_top(twist0) + wp.cross(
            wp.spatial_bottom(twist0), p - com0
        )
    if b1 >= 0:
        twist1 = body_vel[world_idx, b1]
        com1 = wp.transform_get_translation(body_pose[world_idx, b1])
        v1_at_p = wp.spatial_top(twist1) + wp.cross(
            wp.spatial_bottom(twist1), p - com1
        )

    v_rel = v1_at_p - v0_at_p
    v_n = wp.dot(v_rel, n)
    v_t = v_rel - v_n * n

    # ---- β: impact ----
    lambda_n_impact = wp.float32(0.0)
    if enable_impact == 1 and v_n < 0.0:
        inv_m_sum = wp.float32(0.0)
        if b0 >= 0:
            inv_m_sum += body_inv_mass[world_idx, b0]
        if b1 >= 0:
            inv_m_sum += body_inv_mass[world_idx, b1]
        if inv_m_sum > 1.0e-9:
            m_eff = 1.0 / inv_m_sum
            lambda_n_impact = m_eff * (-v_n) / dt

    # ---- α: gravity ----
    n_dot_up = wp.dot(n, up)
    lambda_n_gravity = wp.float32(0.0)
    if enable_gravity == 1:
        # Force on b0 along up = (-n_dot_up) * λ_n. Only credit b0 when
        # this contact pushes it upward (-n_dot_up > floor).
        if b0 >= 0 and (-n_dot_up) > n_dot_up_min:
            cnt = body_unmatched_count[world_idx, b0]
            if cnt > 0:
                m_b = body_mass[world_idx, b0]
                supplied = body_matched_support[world_idx, b0]
                residual = m_b * g_mag - supplied
                if residual > 0.0:
                    candidate = residual / (float(cnt) * (-n_dot_up))
                    lambda_n_gravity = wp.max(lambda_n_gravity, candidate)
        if b1 >= 0 and n_dot_up > n_dot_up_min:
            cnt = body_unmatched_count[world_idx, b1]
            if cnt > 0:
                m_b = body_mass[world_idx, b1]
                supplied = body_matched_support[world_idx, b1]
                residual = m_b * g_mag - supplied
                if residual > 0.0:
                    candidate = residual / (float(cnt) * n_dot_up)
                    lambda_n_gravity = wp.max(lambda_n_gravity, candidate)

    lambda_n_cold = wp.max(lambda_n_impact, lambda_n_gravity)

    if lambda_n_cold > 0.0:
        constr_force_prev_iter[world_idx, offset_n + c_idx] = lambda_n_cold
        wp.atomic_add(diag_cold_normal, world_idx, 1)

    # ---- γ: sliding friction ----
    if enable_friction == 1 and lambda_n_cold > 0.0:
        v_t_sq = wp.length_sq(v_t)
        if v_t_sq > v_t_threshold_sq:
            v_t_mag = wp.sqrt(v_t_sq)
            t_hat = -v_t / v_t_mag
            mu_a = shape_material_mu[world_idx, s0]
            mu_b = shape_material_mu[world_idx, s1]
            mu = 0.5 * (mu_a + mu_b)
            f_t_world = mu * lambda_n_cold * t_hat
            t1, t2 = orthogonal_basis(n)
            constr_force_prev_iter[world_idx, offset_f + 2 * c_idx] = (
                wp.dot(f_t_world, t1)
            )
            constr_force_prev_iter[world_idx, offset_f + 2 * c_idx + 1] = (
                wp.dot(f_t_world, t2)
            )
            wp.atomic_add(diag_cold_friction, world_idx, 1)


@wp.kernel
def _snapshot_kernel(
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
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    offset_n: wp.int32,
    offset_f: wp.int32,
    # Outputs
    prev_b0: wp.array(dtype=wp.int32, ndim=2),
    prev_b1: wp.array(dtype=wp.int32, ndim=2),
    prev_p_world: wp.array(dtype=wp.vec3, ndim=2),
    prev_normal: wp.array(dtype=wp.vec3, ndim=2),
    prev_lambda_n: wp.array(dtype=wp.float32, ndim=2),
    prev_lambda_t: wp.array(dtype=wp.vec2, ndim=2),
):
    """One thread per (world, contact slot). Slots beyond
    contact_count[w] are untouched (stale; the next step's apply will
    iterate only [0, prev_count) so it doesn't matter what's there)."""
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

    midpoint = _world_midpoint(
        world_idx, c_idx,
        contact_point0, contact_point1, contact_normal,
        contact_thickness0, contact_thickness1,
        b0, b1, body_pose,
    )

    prev_b0[world_idx, c_idx] = b0
    prev_b1[world_idx, c_idx] = b1
    prev_p_world[world_idx, c_idx] = midpoint
    prev_normal[world_idx, c_idx] = contact_normal[world_idx, c_idx]
    prev_lambda_n[world_idx, c_idx] = constr_force[world_idx, offset_n + c_idx]
    prev_lambda_t[world_idx, c_idx] = wp.vec2(
        constr_force[world_idx, offset_f + 2 * c_idx],
        constr_force[world_idx, offset_f + 2 * c_idx + 1],
    )


# Adaptive match-distance threshold parameters. Used by the phase-2 match
# kernel; collected here so phase 0 already exposes them on the config.
DEFAULT_ALPHA = 0.1     # fraction of v·dt allowed as prediction error
DEFAULT_MIN_THRESHOLD = 5.0e-3   # 5 mm absolute floor (contact-discovery noise)

# Cold-start gates (phase 2.5).
# n_dot_up_min: only credit a body's gravity-support budget when the
# contact normal projects onto the up axis with at least this magnitude.
# Below 0.3 the contact is mostly horizontal and divides poorly.
DEFAULT_N_DOT_UP_MIN = 0.3
# v_t_threshold: friction heuristic only fires above this tangential
# speed. Below it, the direction is undetermined (stiction band) and a
# wrong-sign seed would cost more iters than starting from zero.
DEFAULT_V_T_THRESHOLD = 0.1   # m/s


class ContactWarmStarter:
    """Cross-step warm-start of contact normal/friction forces.

    Lifecycle, per simulation step:
      1. ``apply()`` runs in ``load_data``, after contact reduction has
         settled the active contact set. It populates
         ``data._constr_force_prev_iter`` so the friction model has
         non-degenerate ``f_n_prev`` at NR iter 0.
      2. Newton-Raphson runs.
      3. ``snapshot()`` runs at the end of ``_solve``, after
         ``perform_backtracking`` has gathered the picked iterate into
         ``data._constr_force``. It saves the converged contact set
         and forces into per-instance ``_prev_*`` buffers so the next
         step can match against them.

    Phase 0: both methods are no-ops. The buffers are allocated and
    the lifecycle hooks are wired so disabling/enabling the feature is
    just a config flip in later phases.
    """

    def __init__(
        self,
        enabled: bool,
        axion_model: "AxionModel",
        data: "EngineData",
        dims: "EngineDimensions",
        device: wp.Device,
        alpha: float = DEFAULT_ALPHA,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
        cold_start_gravity: bool = True,
        cold_start_impact: bool = True,
        cold_start_friction_v_threshold: float = DEFAULT_V_T_THRESHOLD,
        n_dot_up_min: float = DEFAULT_N_DOT_UP_MIN,
        method: str = "position_match",
    ):
        self._enabled = bool(enabled)
        self._device = device
        self._alpha = float(alpha)
        self._min_threshold = float(min_threshold)
        self._cold_gravity = bool(cold_start_gravity)
        self._cold_impact = bool(cold_start_impact)
        self._cold_friction = float(cold_start_friction_v_threshold) > 0.0
        self._v_t_threshold_sq = float(cold_start_friction_v_threshold) ** 2
        self._n_dot_up_min = float(n_dot_up_min)
        if method not in ("position_match", "pair_aggregate"):
            raise ValueError(
                f"WarmStartConfig.method must be 'position_match' or "
                f"'pair_aggregate', got {method!r}"
            )
        self._method = method

        # References needed by apply/snapshot.
        self._shape_body = axion_model.shape_body
        self._shape_material_mu = axion_model.shape_material_mu
        self._body_mass = axion_model.body_mass
        self._body_inv_mass = axion_model.body_inv_mass
        self._g_accel = axion_model.g_accel
        # body_pose_prev is the right reference for BOTH lifecycle hooks:
        #   - At apply time (in step N+1's load_data, after body_pose_prev
        #     was just set to state_in.body_q): equals start-of-step N+1
        #     pose, which is what collide() ran at, what contact_point0/1
        #     are anchored to, and what we want for current midpoints.
        #   - At snapshot time (end of step N's _solve, after
        #     perform_backtracking): body_pose_prev still holds step N's
        #     start-of-step pose because nothing in _solve writes to it.
        #     This is where step N's contacts were anchored, so the
        #     stored midpoint is the actual contact location at step N.
        # Using the same reference keeps both code paths simple AND
        # makes the (apply, snapshot) midpoints comparable across steps
        # via the v·dt prediction we compute in apply.
        self._body_pose = data.body_pose_prev
        self._body_vel = data.body_vel_prev

        # Constraint-vector offsets so the kernels can index
        # ``data._constr_force`` correctly.
        self._offset_n = int(dims.offset_n)
        self._offset_f = int(dims.offset_f)

        N_w = dims.num_worlds
        N_c = dims.contact_count
        N_b = dims.body_count

        # Persistent state: previous step's converged contact set + forces.
        # Indexed by (world, contact_slot) — same shape as AxionContacts
        # arrays so we can reuse the indexing math. Slots beyond
        # ``_prev_count[w]`` are stale and must be ignored on read.
        with wp.ScopedDevice(device):
            self._prev_count = wp.zeros((N_w,), dtype=wp.int32)
            self._prev_b0 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._prev_b1 = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._prev_p_world = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._prev_normal = wp.zeros((N_w, N_c), dtype=wp.vec3)
            self._prev_lambda_n = wp.zeros((N_w, N_c), dtype=wp.float32)
            self._prev_lambda_t = wp.zeros((N_w, N_c), dtype=wp.vec2)

            # Phase 2.5 scratch buffers (reset each apply()).
            self._is_matched = wp.zeros((N_w, N_c), dtype=wp.int32)
            self._body_matched_support = wp.zeros((N_w, N_b), dtype=wp.float32)
            self._body_unmatched_count = wp.zeros((N_w, N_b), dtype=wp.int32)

            # Per-step diagnostics (reset at start of each apply()).
            self._diag_attempts = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_matched = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_max_dist = wp.zeros((N_w,), dtype=wp.float32)
            self._diag_cold_normal = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_cold_friction = wp.zeros((N_w,), dtype=wp.int32)
            # Match failure-mode counters.
            self._diag_no_pair = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_over_thresh = wp.zeros((N_w,), dtype=wp.int32)
            self._diag_closest_d_sum = wp.zeros((N_w,), dtype=wp.float32)
            self._diag_closest_d_raw_sum = wp.zeros((N_w,), dtype=wp.float32)
            self._diag_closest_d_count = wp.zeros((N_w,), dtype=wp.int32)

            # Pair-aggregate state. Body indices are stored as ``b + 1``
            # so b = -1 (ground / world) maps to 0; the +1 padding gives
            # us [N_b + 1, N_b + 1] addressing.
            B1 = N_b + 1
            self._pair_lambda_n_sum = wp.zeros((N_w, B1, B1), dtype=wp.float32)
            self._pair_count_prev   = wp.zeros((N_w, B1, B1), dtype=wp.int32)
            self._pair_count_curr   = wp.zeros((N_w, B1, B1), dtype=wp.int32)
            self._diag_pair_seeded  = wp.zeros((N_w,), dtype=wp.int32)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def apply(self, contacts: "AxionContacts", data: "EngineData", dt: float) -> None:
        """Match every active contact against the snapshot stored from
        the previous step and populate ``data._constr_force_prev_iter``
        with the matched (and basis-projected) λ_n / λ_t values.

        Caller must have zeroed ``data._constr_force_prev_iter`` before
        invoking this — unmatched contacts rely on it staying zero.
        """
        if not self._enabled:
            return

        # Reset per-step diagnostic counters and phase-2.5 scratch buffers.
        self._diag_attempts.zero_()
        self._diag_matched.zero_()
        self._diag_max_dist.zero_()
        self._diag_cold_normal.zero_()
        self._diag_cold_friction.zero_()
        self._diag_no_pair.zero_()
        self._diag_over_thresh.zero_()
        self._diag_closest_d_sum.zero_()
        self._diag_closest_d_raw_sum.zero_()
        self._diag_closest_d_count.zero_()
        self._diag_pair_seeded.zero_()
        self._is_matched.zero_()
        self._body_matched_support.zero_()
        self._body_unmatched_count.zero_()

        if self._method == "pair_aggregate":
            self._launch_pair_aggregate(contacts, data)
            if self._cold_gravity or self._cold_impact:
                self._launch_cold_start(contacts, data, dt)
            return

        wp.launch(
            kernel=_match_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
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
                self._body_vel,
                self._g_accel,
                self._prev_count,
                self._prev_b0,
                self._prev_b1,
                self._prev_p_world,
                self._prev_normal,
                self._prev_lambda_n,
                self._prev_lambda_t,
                float(dt),
                self._alpha,
                self._min_threshold,
                self._n_dot_up_min,
                self._offset_n,
                self._offset_f,
            ],
            outputs=[
                data._constr_force_prev_iter,
                self._is_matched,
                self._body_matched_support,
                self._body_unmatched_count,
                self._diag_attempts,
                self._diag_matched,
                self._diag_max_dist,
                self._diag_no_pair,
                self._diag_over_thresh,
                self._diag_closest_d_sum,
                self._diag_closest_d_raw_sum,
                self._diag_closest_d_count,
            ],
            device=self._device,
        )

        # Cold-start kernel only does work when at least one term is on.
        if self._cold_gravity or self._cold_impact:
            self._launch_cold_start(contacts, data, dt)
        return

    def _launch_pair_aggregate(self, contacts, data) -> None:
        """Distribute prev-step pair-total λ_n uniformly over current-step
        contacts in the same body pair. Two passes: count current contacts
        per pair, then write seeds.
        """
        self._pair_count_curr.zero_()
        wp.launch(
            kernel=_pair_count_current_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
            inputs=[
                contacts.contact_count,
                contacts.contact_shape0,
                contacts.contact_shape1,
                self._shape_body,
            ],
            outputs=[self._pair_count_curr],
            device=self._device,
        )
        wp.launch(
            kernel=_pair_apply_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
            inputs=[
                contacts.contact_count,
                contacts.contact_shape0,
                contacts.contact_shape1,
                self._shape_body,
                self._pair_lambda_n_sum,
                self._pair_count_prev,
                self._pair_count_curr,
                self._offset_n,
            ],
            outputs=[
                data._constr_force_prev_iter,
                self._is_matched,
                self._diag_pair_seeded,
            ],
            device=self._device,
        )

    def _launch_cold_start(self, contacts, data, dt) -> None:
        wp.launch(
            kernel=_cold_start_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
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
                self._shape_material_mu,
                self._body_pose,
                self._body_vel,
                self._body_mass,
                self._body_inv_mass,
                self._g_accel,
                self._is_matched,
                self._body_matched_support,
                self._body_unmatched_count,
                float(dt),
                self._n_dot_up_min,
                self._v_t_threshold_sq,
                wp.int32(1 if self._cold_gravity else 0),
                wp.int32(1 if self._cold_impact else 0),
                wp.int32(1 if self._cold_friction else 0),
                self._offset_n,
                self._offset_f,
            ],
            outputs=[
                data._constr_force_prev_iter,
                self._diag_cold_normal,
                self._diag_cold_friction,
            ],
            device=self._device,
        )

    def snapshot(self, contacts: "AxionContacts", data: "EngineData") -> None:
        """Save the post-backtrack converged contact set + forces into
        the persistent ``_prev_*`` buffers so the next step's ``apply``
        can match against them.

        Reads ``data._constr_force`` (post-backtrack picked iter's
        forces), ``data.body_pose_prev`` (pre-step pose, the frame in
        which contact_point0/1 are anchored), and the current contact
        set from ``contacts``.
        """
        if not self._enabled:
            return

        if self._method == "pair_aggregate":
            self._pair_lambda_n_sum.zero_()
            self._pair_count_prev.zero_()
            wp.launch(
                kernel=_pair_snapshot_kernel,
                dim=(contacts.num_worlds, contacts.max_contacts),
                inputs=[
                    contacts.contact_count,
                    contacts.contact_shape0,
                    contacts.contact_shape1,
                    self._shape_body,
                    data._constr_force,
                    self._offset_n,
                ],
                outputs=[
                    self._pair_lambda_n_sum,
                    self._pair_count_prev,
                ],
                device=self._device,
            )
            return

        # Mirror the active count so apply knows how many slots to scan.
        wp.copy(self._prev_count, contacts.contact_count)

        wp.launch(
            kernel=_snapshot_kernel,
            dim=(contacts.num_worlds, contacts.max_contacts),
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
                data._constr_force,
                self._offset_n,
                self._offset_f,
            ],
            outputs=[
                self._prev_b0,
                self._prev_b1,
                self._prev_p_world,
                self._prev_normal,
                self._prev_lambda_n,
                self._prev_lambda_t,
            ],
            device=self._device,
        )


# ============================================================================
#                       PAIR-AGGREGATE WARM-START METHOD
# ============================================================================
# Aggregate prev-step λ_n by body pair, then distribute the pair total
# uniformly across current-step contacts in the same pair. Body indices
# are stored as ``b + 1`` so b = -1 (ground / static world) maps to 0.
# Robust to per-contact identity drift caused by mesh-vertex jitter,
# terrain-triangle changes, and rolling-wheel kinematics — the things
# that broke ``_match_kernel`` for moving contacts. See
# docs/warm_start_iterate_seeding_issue.md for the diagnostic that led
# here.


@wp.kernel
def _pair_snapshot_kernel(
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    offset_n: wp.int32,
    # Outputs (zeroed by caller before launch).
    pair_lambda_n_sum: wp.array(dtype=wp.float32, ndim=3),  # [W, B+1, B+1]
    pair_count_prev: wp.array(dtype=wp.int32, ndim=3),       # [W, B+1, B+1]
):
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

    lam_n = constr_force[world_idx, offset_n + c_idx]
    i0 = b0 + 1
    i1 = b1 + 1

    wp.atomic_add(pair_lambda_n_sum, world_idx, i0, i1, lam_n)
    wp.atomic_add(pair_count_prev, world_idx, i0, i1, 1)


@wp.kernel
def _pair_count_current_kernel(
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    pair_count_curr: wp.array(dtype=wp.int32, ndim=3),  # [W, B+1, B+1] (zeroed by caller)
):
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
    i0 = b0 + 1
    i1 = b1 + 1
    wp.atomic_add(pair_count_curr, world_idx, i0, i1, 1)


@wp.kernel
def _pair_apply_kernel(
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    pair_lambda_n_sum: wp.array(dtype=wp.float32, ndim=3),
    pair_count_prev: wp.array(dtype=wp.int32, ndim=3),
    pair_count_curr: wp.array(dtype=wp.int32, ndim=3),
    offset_n: wp.int32,
    # Outputs.
    constr_force_prev_iter: wp.array(dtype=wp.float32, ndim=2),
    is_matched: wp.array(dtype=wp.int32, ndim=2),
    diag_pair_seeded: wp.array(dtype=wp.int32, ndim=1),
):
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
    i0 = b0 + 1
    i1 = b1 + 1

    n_prev = pair_count_prev[world_idx, i0, i1]
    if n_prev <= 0:
        return  # genuinely new body pair — leave for cold-start kernel

    n_curr = pair_count_curr[world_idx, i0, i1]
    if n_curr <= 0:
        return  # shouldn't happen since this thread IS in the pair, but guard

    total = pair_lambda_n_sum[world_idx, i0, i1]
    avg = total / wp.float32(n_curr)

    constr_force_prev_iter[world_idx, offset_n + c_idx] = avg
    is_matched[world_idx, c_idx] = 1
    wp.atomic_add(diag_pair_seeded, world_idx, 1)
