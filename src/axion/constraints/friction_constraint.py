import warp as wp
from axion.mechanics import orthogonal_basis
from axion.mechanics import scaled_fisher_burmeister

from .utils import compute_effective_mass

# -----------------------------------------------------------------------------
# 1. Low-Level Helpers
# -----------------------------------------------------------------------------


@wp.func
def resolve_body_indices(
    world_idx: int,
    shape0: int,
    shape1: int,
    shape_body: wp.array(dtype=wp.int32, ndim=2),
):
    """Resolves shape indices to body indices. Returns -1 for static shapes."""
    body0 = -1
    if shape0 >= 0:
        body0 = shape_body[world_idx, shape0]

    body1 = -1
    if shape1 >= 0:
        body1 = shape_body[world_idx, shape1]

    return body0, body1


@wp.func
def resolve_friction_frame(
    n: wp.vec3,
    axis_local_0: wp.vec3,
    pose0_prev: wp.transform,
    body0: int,
    axis_local_1: wp.vec3,
    pose1_prev: wp.transform,
    body1: int,
    mu0: wp.float32,
    mu_perp_0: wp.float32,
    mu1: wp.float32,
    mu_perp_1: wp.float32,
):
    """Resolves the friction tangent frame and per-axis coefficients at a contact.

    If neither shape has a non-zero `axis_local`, returns the isotropic basis
    `orthogonal_basis(n)` and the scalar-average mu for both axes (existing
    behavior). Otherwise rotates the non-zero axis into world space, projects
    onto the tangent plane, and uses it as t1; mu_x applies along t1, mu_y
    along t2 = n x t1.

    Combine rule: when exactly one shape is anisotropic, average its per-axis
    coefficients with the other shape's isotropic mu (component-wise). If both
    are anisotropic, the first shape's frame wins; mu_y for it averages with
    the other shape's scalar mu.

    A negative `mu_perp` is a sentinel meaning "fall back to mu" (i.e. that
    shape is effectively isotropic even if an axis was set).
    """
    eps_len = 1e-6

    aniso0 = wp.length_sq(axis_local_0) > eps_len and body0 >= 0
    aniso1 = wp.length_sq(axis_local_1) > eps_len and body1 >= 0

    if not aniso0 and not aniso1:
        t1, t2 = orthogonal_basis(n)
        mu = (mu0 + mu1) * 0.5
        return t1, t2, mu, mu

    axis_world = wp.vec3(0.0, 0.0, 0.0)
    mu_along_aniso = 0.0
    mu_perp_aniso = 0.0
    mu_other = 0.0

    if aniso0:
        q0 = wp.transform_get_rotation(pose0_prev)
        axis_world = wp.quat_rotate(q0, axis_local_0)
        mu_along_aniso = mu0
        if mu_perp_0 >= 0.0:
            mu_perp_aniso = mu_perp_0
        else:
            mu_perp_aniso = mu0
        mu_other = mu1
    else:
        q1 = wp.transform_get_rotation(pose1_prev)
        axis_world = wp.quat_rotate(q1, axis_local_1)
        mu_along_aniso = mu1
        if mu_perp_1 >= 0.0:
            mu_perp_aniso = mu_perp_1
        else:
            mu_perp_aniso = mu1
        mu_other = mu0

    axis_tan = axis_world - wp.dot(axis_world, n) * n
    tan_len2 = wp.length_sq(axis_tan)
    if tan_len2 < 1e-8:
        # Axis is parallel to the contact normal — no preferred tangent direction.
        t1, t2 = orthogonal_basis(n)
        mu_iso = (mu_along_aniso + mu_other) * 0.5
        return t1, t2, mu_iso, mu_iso

    t1 = axis_tan / wp.sqrt(tan_len2)
    t2 = wp.cross(n, t1)

    mu_x = (mu_along_aniso + mu_other) * 0.5
    mu_y = (mu_perp_aniso + mu_other) * 0.5
    return t1, t2, mu_x, mu_y


@wp.func
def compute_friction_model(
    mu_x: wp.float32,
    mu_y: wp.float32,
    J_t1_0: wp.spatial_vector,
    J_t2_0: wp.spatial_vector,
    J_t1_1: wp.spatial_vector,
    J_t2_1: wp.spatial_vector,
    vel0: wp.spatial_vector,
    vel1: wp.spatial_vector,
    force_f_prev: wp.vec2,
    force_n_prev: wp.float32,
    dt: wp.float32,
    precond: wp.float32,
):
    """Impulse-level Fisher-Burmeister for the (elliptical) Coulomb cone.

    Two regimes:

    * **Isotropic** (``mu_x == mu_y``): runs the original scalar formulation
      verbatim and returns ``(v_t, w, w)``. This is bit-identical to the
      pre-anisotropic code — `resolve_friction_frame` returns equal mu for
      every isotropic / fallback path, so all existing scenes hit this branch
      and are unaffected.

    * **Anisotropic** (``mu_x != mu_y``): elliptical cone
      ``sqrt((f.x/mu_x)^2 + (f.y/mu_y)^2) <= f_n``. Reformulated in tilde-space
      (``v~ = D v``, ``f~ = D^{-1} f``, ``D = diag(mu_x, mu_y)``) so the same FB
      machinery applies; the returned per-direction weights map the tilde-space
      scalar weight back to the residual ``v_t.i + w_i * lambda_i = 0``.

    NOTE: the two regimes are *not* a continuous limit of each other inside the
    FB smoothing (sticking/sliding transition) region — they only coincide in
    pure sliding and pure sticking. The hard branch on ``mu_x == mu_y`` is
    therefore deliberate: it guarantees the isotropic path exactly reproduces
    the original solver behavior rather than an FB-smoothing approximation of
    it.
    """
    v_t_0 = wp.dot(J_t1_0, vel0) + wp.dot(J_t1_1, vel1)
    v_t_1 = wp.dot(J_t2_0, vel0) + wp.dot(J_t2_1, vel1)
    v_t = wp.vec2(v_t_0, v_t_1)

    eps = 1e-8
    r = precond
    denom_eps = 1e-6 * dt

    if mu_x == mu_y:
        # ---- Isotropic: original scalar formulation, verbatim ----
        mu = mu_x

        d_t = v_t * dt
        d_t_norm = wp.sqrt(wp.dot(d_t, d_t) + eps)

        impulse_f_prev = force_f_prev * dt
        raw_imp_norm = wp.length(impulse_f_prev)

        impulse_n_prev = force_n_prev * dt
        limit = mu * impulse_n_prev
        clamped_imp_norm = wp.min(raw_imp_norm, limit)

        gap = limit - clamped_imp_norm

        phi_f = scaled_fisher_burmeister(d_t_norm, gap, 1.0, r)

        denominator = r * raw_imp_norm + phi_f + denom_eps
        numerator = d_t_norm - phi_f

        w = r * (numerator / denominator)
        w = wp.max(w, 0.0)
        w = wp.min(w, 1e5)

        return v_t, w, w

    # ---- Anisotropic: elliptical cone via tilde-space ----
    # Floor mu to avoid 1/mu blow-up when a direction is set to zero friction.
    # A direction with mu=0 is a hard "no friction force allowed there"; a tiny
    # floor preserves that intent (the corresponding f component is driven to
    # ~0 by the elliptical cone) while keeping the tilde-space math finite.
    mu_floor = wp.float32(1e-6)
    mu_x_safe = wp.max(mu_x, mu_floor)
    mu_y_safe = wp.max(mu_y, mu_floor)

    # Impulse-level displacement, then map to tilde-space: d_tilde = D d_t
    d_t = v_t * dt
    d_x_tilde = mu_x_safe * d_t.x
    d_y_tilde = mu_y_safe * d_t.y
    d_t_norm = wp.sqrt(d_x_tilde * d_x_tilde + d_y_tilde * d_y_tilde + eps)

    # Previous friction impulse, mapped to tilde-space: f_tilde = D^{-1} f
    impulse_f_prev = force_f_prev * dt
    inv_mux = 1.0 / mu_x_safe
    inv_muy = 1.0 / mu_y_safe
    f_x_tilde = impulse_f_prev.x * inv_mux
    f_y_tilde = impulse_f_prev.y * inv_muy
    raw_imp_norm = wp.sqrt(f_x_tilde * f_x_tilde + f_y_tilde * f_y_tilde)

    impulse_n_prev = force_n_prev * dt
    # In tilde-space the cone is the unit cone ||f_tilde|| <= f_n
    limit = impulse_n_prev
    clamped_imp_norm = wp.min(raw_imp_norm, limit)

    gap = limit - clamped_imp_norm

    phi_f = scaled_fisher_burmeister(d_t_norm, gap, 1.0, r)

    denominator = r * raw_imp_norm + phi_f + denom_eps
    numerator = d_t_norm - phi_f

    w_tilde = r * (numerator / denominator)
    w_tilde = wp.max(w_tilde, 0.0)
    w_tilde = wp.min(w_tilde, 1e5)

    # Map tilde-space scalar weight back to per-direction weights in the
    # original (un-scaled) frame. At sliding this gives the correct
    # max-dissipation force f_t.i = -f_n * mu_i^2 * v_t.i / ||D v_t||.
    w_x = w_tilde * inv_mux * inv_mux
    w_y = w_tilde * inv_muy * inv_muy

    return v_t, w_x, w_y


# -----------------------------------------------------------------------------
# 2. The SINGLE Core Logic Function
# -----------------------------------------------------------------------------


@wp.func
def compute_friction_core(
    body0: int,
    body1: int,
    n: wp.vec3,
    t1: wp.vec3,
    t2: wp.vec3,
    mu_x: float,
    mu_y: float,
    p0_local: wp.vec3,
    p1_local: wp.vec3,
    thickness0: float,
    thickness1: float,
    vel0: wp.spatial_vector,
    pose0_prev: wp.transform,
    m_inv0: float,
    I_inv0: wp.mat33,
    com0: wp.vec3,
    vel1: wp.spatial_vector,
    pose1_prev: wp.transform,
    m_inv1: float,
    I_inv1: wp.mat33,
    com1: wp.vec3,
    lambda_t1: float,
    lambda_t2: float,
    lambda_t1_prev: float,
    lambda_t2_prev: float,
    force_n_prev: float,
    dt: float,
    compliance: float,
):
    """
    Computes all Jacobians and friction residuals dynamically. Per-direction
    friction coefficients (mu_x along t1, mu_y along t2) define the elliptical
    Coulomb cone; the residual and compliance are emitted per-direction.
    """
    J_t1_0 = wp.spatial_vector()
    J_t2_0 = wp.spatial_vector()

    if body0 >= 0:
        p0_world_prev = wp.transform_point(pose0_prev, p0_local)
        p0_adj_prev = p0_world_prev - (thickness0 * n)
        com0_world_prev = wp.transform_point(pose0_prev, com0)
        r0 = p0_adj_prev - com0_world_prev
        J_t1_0 = wp.spatial_vector(t1, wp.cross(r0, t1))
        J_t2_0 = wp.spatial_vector(t2, wp.cross(r0, t2))

    J_t1_1 = wp.spatial_vector()
    J_t2_1 = wp.spatial_vector()

    if body1 >= 0:
        p1_world_prev = wp.transform_point(pose1_prev, p1_local)
        p1_adj_prev = p1_world_prev + (thickness1 * n)
        com1_world_prev = wp.transform_point(pose1_prev, com1)
        r1 = p1_adj_prev - com1_world_prev
        J_t1_1 = wp.spatial_vector(-t1, -wp.cross(r1, t1))
        J_t2_1 = wp.spatial_vector(-t2, -wp.cross(r1, t2))

    w_t1 = compute_effective_mass(
        pose0_prev, pose1_prev, J_t1_0, J_t1_1, m_inv0, I_inv0, m_inv1, I_inv1, body0, body1
    )
    w_t2 = compute_effective_mass(
        pose0_prev, pose1_prev, J_t2_0, J_t2_1, m_inv0, I_inv0, m_inv1, I_inv1, body0, body1
    )

    effective_mass = (w_t1 + w_t2) * 0.5
    precond = effective_mass  # dt-independent for impulse-level FB
    force_f_prev = wp.vec2(lambda_t1_prev, lambda_t2_prev)

    v_t, w_x, w_y = compute_friction_model(
        mu_x, mu_y, J_t1_0, J_t2_0, J_t1_1, J_t2_1, vel0, vel1, force_f_prev, force_n_prev, dt, precond
    )

    d_res_d0 = -dt * (J_t1_0 * lambda_t1 + J_t2_0 * lambda_t2)
    d_res_d1 = -dt * (J_t1_1 * lambda_t1 + J_t2_1 * lambda_t2)
    res_f0 = v_t.x + w_x * lambda_t1
    res_f1 = v_t.y + w_y * lambda_t2
    c_f0 = w_x / dt + compliance
    c_f1 = w_y / dt + compliance

    return d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f0, c_f1


# -----------------------------------------------------------------------------
# 3. Standard Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def friction_residual_kernel(
    # State variables
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_friction_axis_local: wp.array(dtype=wp.vec3, ndim=2),
    shape_mu_perp: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    compliance: wp.float32,
    # Outputs
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_f: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()
    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    mu0 = shape_material_mu[world_idx, shape0]
    mu1 = shape_material_mu[world_idx, shape1]
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    # EARLY EXIT: skip when the cone budget is essentially zero in both shapes.
    mu_max = wp.max(mu0, mu1)
    if mu_max * force_n_prev <= 1e-6:
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    axis0 = shape_friction_axis_local[world_idx, shape0]
    axis1 = shape_friction_axis_local[world_idx, shape1]
    mu_perp_0 = shape_mu_perp[world_idx, shape0]
    mu_perp_1 = shape_mu_perp[world_idx, shape1]
    t1, t2, mu_x, mu_y = resolve_friction_frame(
        n, axis0, pose0_prev, body0, axis1, pose1_prev, body1,
        mu0, mu_perp_0, mu1, mu_perp_1,
    )

    lam_t1 = constr_force[world_idx, constr_idx0]
    lam_t2 = constr_force[world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f0, c_f1 = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu_x,
        mu_y,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
        compliance,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_f[world_idx, constr_idx0] = res_f0
    res_f[world_idx, constr_idx1] = res_f1


@wp.kernel
def friction_constraint_kernel(
    # State variables
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_friction_axis_local: wp.array(dtype=wp.vec3, ndim=2),
    shape_mu_perp: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    compliance: wp.float32,
    # Outputs
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_f: wp.array(dtype=wp.float32, ndim=2),
    J_hat_f_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_f_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    mu0 = shape_material_mu[world_idx, shape0]
    mu1 = shape_material_mu[world_idx, shape1]
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    mu_max = wp.max(mu0, mu1)
    if mu_max * force_n_prev <= 1e-6:
        constr_active_mask[world_idx, constr_idx0] = 0.0
        constr_active_mask[world_idx, constr_idx1] = 0.0
        constr_force[world_idx, constr_idx0] = 0.0
        constr_force[world_idx, constr_idx1] = 0.0
        res_f[world_idx, constr_idx0] = 0.0
        res_f[world_idx, constr_idx1] = 0.0

        J_hat_f_values[world_idx, constr_idx0, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 0] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx0, 1] = wp.spatial_vector()
        J_hat_f_values[world_idx, constr_idx1, 1] = wp.spatial_vector()

        C_f_values[world_idx, constr_idx0] = 0.0
        C_f_values[world_idx, constr_idx1] = 0.0
        return

    constr_active_mask[world_idx, constr_idx0] = 1.0
    constr_active_mask[world_idx, constr_idx1] = 1.0

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)

    constr_body_idx[world_idx, constr_idx0, 0] = body0
    constr_body_idx[world_idx, constr_idx0, 1] = body1
    constr_body_idx[world_idx, constr_idx1, 0] = body0
    constr_body_idx[world_idx, constr_idx1, 1] = body1

    n = contact_normal[world_idx, contact_idx]

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    axis0 = shape_friction_axis_local[world_idx, shape0]
    axis1 = shape_friction_axis_local[world_idx, shape1]
    mu_perp_0 = shape_mu_perp[world_idx, shape0]
    mu_perp_1 = shape_mu_perp[world_idx, shape1]
    t1, t2, mu_x, mu_y = resolve_friction_frame(
        n, axis0, pose0_prev, body0, axis1, pose1_prev, body1,
        mu0, mu_perp_0, mu1, mu_perp_1,
    )

    lam_t1 = constr_force[world_idx, constr_idx0]
    lam_t2 = constr_force[world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f0, c_f1 = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu_x,
        mu_y,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
        compliance,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_f[world_idx, constr_idx0] = res_f0
    res_f[world_idx, constr_idx1] = res_f1

    J_hat_f_values[world_idx, constr_idx0, 0] = J_t1_0
    J_hat_f_values[world_idx, constr_idx1, 0] = J_t2_0
    J_hat_f_values[world_idx, constr_idx0, 1] = J_t1_1
    J_hat_f_values[world_idx, constr_idx1, 1] = J_t2_1

    C_f_values[world_idx, constr_idx0] = c_f0
    C_f_values[world_idx, constr_idx1] = c_f1


# -----------------------------------------------------------------------------
# 4. Batched Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def batch_friction_residual_kernel(
    # State variables (3D)
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),  # Prev step remains 2D
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_friction_axis_local: wp.array(dtype=wp.vec3, ndim=2),
    shape_mu_perp: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    compliance: wp.float32,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_f: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    mu0 = shape_material_mu[world_idx, shape0]
    mu1 = shape_material_mu[world_idx, shape1]
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    mu_max = wp.max(mu0, mu1)
    if mu_max * force_n_prev <= 1e-6:
        res_f[batch_idx, world_idx, constr_idx0] = 0.0
        res_f[batch_idx, world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]

    vel0, pose0_prev, m_inv0, I_inv0, com0 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body0 >= 0:
        vel0 = body_vel[batch_idx, world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]

    vel1, pose1_prev, m_inv1, I_inv1, com1 = (
        wp.spatial_vector(),
        wp.transform_identity(),
        0.0,
        wp.mat33(0.0),
        wp.vec3(),
    )
    if body1 >= 0:
        vel1 = body_vel[batch_idx, world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]

    axis0 = shape_friction_axis_local[world_idx, shape0]
    axis1 = shape_friction_axis_local[world_idx, shape1]
    mu_perp_0 = shape_mu_perp[world_idx, shape0]
    mu_perp_1 = shape_mu_perp[world_idx, shape1]
    t1, t2, mu_x, mu_y = resolve_friction_frame(
        n, axis0, pose0_prev, body0, axis1, pose1_prev, body1,
        mu0, mu_perp_0, mu1, mu_perp_1,
    )

    lam_t1 = constr_force[batch_idx, world_idx, constr_idx0]
    lam_t2 = constr_force[batch_idx, world_idx, constr_idx1]
    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f0, c_f1 = compute_friction_core(
        body0,
        body1,
        n,
        t1,
        t2,
        mu_x,
        mu_y,
        p0,
        p1,
        thickness0,
        thickness1,
        vel0,
        pose0_prev,
        m_inv0,
        I_inv0,
        com0,
        vel1,
        pose1_prev,
        m_inv1,
        I_inv1,
        com1,
        lam_t1,
        lam_t2,
        lam_t1_p,
        lam_t2_p,
        force_n_prev,
        dt,
        compliance,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body1, d_res_d1)

    res_f[batch_idx, world_idx, constr_idx0] = res_f0
    res_f[batch_idx, world_idx, constr_idx1] = res_f1


@wp.kernel
def fused_batch_friction_residual_kernel(
    # State variables (3D)
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    constr_force_prev: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n_prev: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_friction_axis_local: wp.array(dtype=wp.vec3, ndim=2),
    shape_mu_perp: wp.array(dtype=wp.float32, ndim=2),
    # Contact properties
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    # Simulation parameters
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_f: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    constr_idx0 = 2 * contact_idx
    constr_idx1 = 2 * contact_idx + 1

    if contact_idx >= contact_count[world_idx]:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    # Load shared contact and mass parameters exactly ONCE
    mu0 = shape_material_mu[world_idx, shape0]
    mu1 = shape_material_mu[world_idx, shape1]
    force_n_prev = constr_force_n_prev[world_idx, contact_idx]

    mu_max = wp.max(mu0, mu1)
    if mu_max * force_n_prev <= 1e-6:
        for b in range(num_batches):
            res_f[b, world_idx, constr_idx0] = 0.0
            res_f[b, world_idx, constr_idx1] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    m_inv0, I_inv0, com0 = 0.0, wp.mat33(0.0), wp.vec3()
    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]

    m_inv1, I_inv1, com1 = 0.0, wp.mat33(0.0), wp.vec3()
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]

    # Friction frame is shape/pose data — same for all batches, hoist outside loop.
    axis0 = shape_friction_axis_local[world_idx, shape0]
    axis1 = shape_friction_axis_local[world_idx, shape1]
    mu_perp_0 = shape_mu_perp[world_idx, shape0]
    mu_perp_1 = shape_mu_perp[world_idx, shape1]
    t1, t2, mu_x, mu_y = resolve_friction_frame(
        n, axis0, pose0_prev, body0, axis1, pose1_prev, body1,
        mu0, mu_perp_0, mu1, mu_perp_1,
    )

    lam_t1_p = constr_force_prev[world_idx, constr_idx0]
    lam_t2_p = constr_force_prev[world_idx, constr_idx1]

    # --- Iterate through batches utilizing the preloaded memory ---
    for b in range(num_batches):
        vel0 = wp.spatial_vector()
        if body0 >= 0:
            vel0 = body_vel[b, world_idx, body0]

        vel1 = wp.spatial_vector()
        if body1 >= 0:
            vel1 = body_vel[b, world_idx, body1]

        lam_t1 = constr_force[b, world_idx, constr_idx0]
        lam_t2 = constr_force[b, world_idx, constr_idx1]

        d_res_d0, d_res_d1, res_f0, res_f1, J_t1_0, J_t2_0, J_t1_1, J_t2_1, c_f0, c_f1 = (
            compute_friction_core(
                body0,
                body1,
                n,
                t1,
                t2,
                mu_x,
                mu_y,
                p0,
                p1,
                thickness0,
                thickness1,
                vel0,
                pose0_prev,
                m_inv0,
                I_inv0,
                com0,
                vel1,
                pose1_prev,
                m_inv1,
                I_inv1,
                com1,
                lam_t1,
                lam_t2,
                lam_t1_p,
                lam_t2_p,
                force_n_prev,
                dt,
                compliance,
            )
        )

        if body0 >= 0:
            wp.atomic_add(res_d, b, world_idx, body0, d_res_d0)
        if body1 >= 0:
            wp.atomic_add(res_d, b, world_idx, body1, d_res_d1)

        res_f[b, world_idx, constr_idx0] = res_f0
        res_f[b, world_idx, constr_idx1] = res_f1
