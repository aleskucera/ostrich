import warp as wp
from axion.mechanics import scaled_fisher_burmeister_diff

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


# -----------------------------------------------------------------------------
# 2. The SINGLE Core Logic Function
# -----------------------------------------------------------------------------


@wp.func
def compute_contact_core(
    body0: int,
    body1: int,
    n: wp.vec3,
    p0_local: wp.vec3,
    p1_local: wp.vec3,
    thickness0: float,
    thickness1: float,
    pose0: wp.transform,
    m_inv0: float,
    I_inv0: wp.mat33,
    com0: wp.vec3,
    pose1: wp.transform,
    m_inv1: float,
    I_inv1: wp.mat33,
    com1: wp.vec3,
    pose0_prev: wp.transform,
    pose1_prev: wp.transform,
    f_n: float,
    dt: float,
    compliance: float,
    fb_eps_sq: float,
):
    """
    Computes all Jacobians and contact residuals dynamically.
    Array-agnostic, supporting both standard and batched execution.
    """
    # Jacobian calculation is based on PREVIOUS poses
    p0_world_prev = wp.transform_point(pose0_prev, p0_local)
    p0_adj_prev = p0_world_prev - (thickness0 * n)
    p1_world_prev = wp.transform_point(pose1_prev, p1_local)
    p1_adj_prev = p1_world_prev + (thickness1 * n)

    # Penetration depth is based on CURRENT poses
    p0_world = wp.transform_point(pose0, p0_local)
    p0_adj = p0_world - (thickness0 * n)
    p1_world = wp.transform_point(pose1, p1_local)
    p1_adj = p1_world + (thickness1 * n)

    # Compute Jacobians ONLY for dynamic bodies
    J0 = wp.spatial_vector()
    if body0 >= 0:
        com0_world_prev = wp.transform_point(pose0_prev, com0)
        lever0 = p0_adj_prev - com0_world_prev
        J0 = wp.spatial_vector(n, wp.cross(lever0, n))

    J1 = wp.spatial_vector()
    if body1 >= 0:
        com1_world_prev = wp.transform_point(pose1_prev, com1)
        lever1 = p1_adj_prev - com1_world_prev
        J1 = wp.spatial_vector(-n, wp.cross(lever1, -n))

    # --- 1. Compute Signed Distance ---
    signed_dist = wp.dot(n, p0_adj - p1_adj)

    # --- 2. Compute Effective Mass ---
    effective_mass = compute_effective_mass(
        pose0_prev, pose1_prev, J0, J1, m_inv0, I_inv0, m_inv1, I_inv1, body0, body1
    )
    precond = wp.pow(dt, 2.0) * effective_mass

    # --- 3. Fisher-Burmeister Complementarity ---
    phi_n, dphi_dc_n, dphi_dlambda_n = scaled_fisher_burmeister_diff(signed_dist, f_n, 1.0, precond, fb_eps_sq)

    # --- 4. Final Solver Terms ---
    J_hat_0 = dphi_dc_n * J0
    J_hat_1 = dphi_dc_n * J1

    d_res_d0 = -J_hat_0 * f_n * dt
    d_res_d1 = -J_hat_1 * f_n * dt
    res_n_val = phi_n / dt
    c_val = (dphi_dlambda_n + compliance) / wp.pow(dt, 2.0)

    return d_res_d0, d_res_d1, res_n_val, J_hat_0, J_hat_1, c_val


# -----------------------------------------------------------------------------
# 3. Standard Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def contact_residual_kernel(
    # State variables
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
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
    fb_eps_sq: wp.float32,
    # Outputs
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_n: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        res_n[world_idx, contact_idx] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_n[world_idx, contact_idx] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    f_n = constr_force[world_idx, contact_idx]

    pose0, m_inv0, I_inv0, com0 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        pose0 = body_pose[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]

    pose1, m_inv1, I_inv1, com1 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        pose1 = body_pose[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_n_val, J0, J1, c_val = compute_contact_core(
        body0,
        body1,
        n,
        p0,
        p1,
        thickness0,
        thickness1,
        pose0,
        m_inv0,
        I_inv0,
        com0,
        pose1,
        m_inv1,
        I_inv1,
        com1,
        pose0_prev,
        pose1_prev,
        f_n,
        dt,
        compliance,
        fb_eps_sq,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_n[world_idx, contact_idx] = res_n_val


@wp.kernel
def contact_constraint_kernel(
    # State variables
    body_pose: wp.array(dtype=wp.transform, ndim=2),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=2),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
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
    fb_eps_sq: wp.float32,
    # Outputs
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),
    constr_body_idx: wp.array(dtype=wp.int32, ndim=3),
    res_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    res_n: wp.array(dtype=wp.float32, ndim=2),
    J_hat_n_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_n_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        constr_active_mask[world_idx, contact_idx] = 0.0
        constr_force[world_idx, contact_idx] = 0.0
        res_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        constr_active_mask[world_idx, contact_idx] = 0.0
        constr_force[world_idx, contact_idx] = 0.0
        res_n[world_idx, contact_idx] = 0.0
        J_hat_n_values[world_idx, contact_idx, 0] = wp.spatial_vector()
        J_hat_n_values[world_idx, contact_idx, 1] = wp.spatial_vector()
        C_n_values[world_idx, contact_idx] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    f_n = constr_force[world_idx, contact_idx]

    pose0, m_inv0, I_inv0, com0 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        pose0 = body_pose[world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]

    pose1, m_inv1, I_inv1, com1 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        pose1 = body_pose[world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_n_val, J0, J1, c_val = compute_contact_core(
        body0,
        body1,
        n,
        p0,
        p1,
        thickness0,
        thickness1,
        pose0,
        m_inv0,
        I_inv0,
        com0,
        pose1,
        m_inv1,
        I_inv1,
        com1,
        pose0_prev,
        pose1_prev,
        f_n,
        dt,
        compliance,
        fb_eps_sq,
    )

    constr_active_mask[world_idx, contact_idx] = 1.0
    constr_body_idx[world_idx, contact_idx, 0] = body0
    constr_body_idx[world_idx, contact_idx, 1] = body1

    if body0 >= 0:
        wp.atomic_add(res_d, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, world_idx, body1, d_res_d1)

    res_n[world_idx, contact_idx] = res_n_val
    C_n_values[world_idx, contact_idx] = c_val
    J_hat_n_values[world_idx, contact_idx, 0] = J0
    J_hat_n_values[world_idx, contact_idx, 1] = J1


# -----------------------------------------------------------------------------
# 4. Batched Kernels
# -----------------------------------------------------------------------------


@wp.kernel
def batch_contact_residual_kernel(
    # State variables (3D)
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
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
    fb_eps_sq: wp.float32,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_n: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        res_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        res_n[batch_idx, world_idx, contact_idx] = 0.0
        return

    body0, body1 = resolve_body_indices(world_idx, shape0, shape1, shape_body)
    n = contact_normal[world_idx, contact_idx]
    f_n = constr_force[batch_idx, world_idx, contact_idx]

    pose0, m_inv0, I_inv0, com0 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose0_prev = wp.transform_identity()
    if body0 >= 0:
        pose0 = body_pose[batch_idx, world_idx, body0]
        m_inv0 = body_m_inv[world_idx, body0]
        I_inv0 = body_I_inv[world_idx, body0]
        com0 = body_com[world_idx, body0]
        pose0_prev = body_pose_prev[world_idx, body0]

    pose1, m_inv1, I_inv1, com1 = wp.transform_identity(), 0.0, wp.mat33(0.0), wp.vec3()
    pose1_prev = wp.transform_identity()
    if body1 >= 0:
        pose1 = body_pose[batch_idx, world_idx, body1]
        m_inv1 = body_m_inv[world_idx, body1]
        I_inv1 = body_I_inv[world_idx, body1]
        com1 = body_com[world_idx, body1]
        pose1_prev = body_pose_prev[world_idx, body1]

    p0 = contact_point0[world_idx, contact_idx]
    p1 = contact_point1[world_idx, contact_idx]
    thickness0 = contact_thickness0[world_idx, contact_idx]
    thickness1 = contact_thickness1[world_idx, contact_idx]

    d_res_d0, d_res_d1, res_n_val, J0, J1, c_val = compute_contact_core(
        body0,
        body1,
        n,
        p0,
        p1,
        thickness0,
        thickness1,
        pose0,
        m_inv0,
        I_inv0,
        com0,
        pose1,
        m_inv1,
        I_inv1,
        com1,
        pose0_prev,
        pose1_prev,
        f_n,
        dt,
        compliance,
        fb_eps_sq,
    )

    if body0 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body0, d_res_d0)
    if body1 >= 0:
        wp.atomic_add(res_d, batch_idx, world_idx, body1, d_res_d1)

    res_n[batch_idx, world_idx, contact_idx] = res_n_val


@wp.kernel
def fused_batch_contact_residual_kernel(
    # State variables (3D)
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    constr_force: wp.array(dtype=wp.float32, ndim=3),
    # Body properties
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Shape properties
    shape_body: wp.array(dtype=wp.int32, ndim=2),
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
    fb_eps_sq: wp.float32,
    num_batches: int,
    # Outputs (3D)
    res_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    res_n: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        for b in range(num_batches):
            res_n[b, world_idx, contact_idx] = 0.0
        return

    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]

    if shape0 == shape1:
        for b in range(num_batches):
            res_n[b, world_idx, contact_idx] = 0.0
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

    # --- Iterate through batches utilizing the preloaded memory ---
    for b in range(num_batches):
        pose0 = wp.transform_identity()
        if body0 >= 0:
            pose0 = body_pose[b, world_idx, body0]

        pose1 = wp.transform_identity()
        if body1 >= 0:
            pose1 = body_pose[b, world_idx, body1]

        f_n = constr_force[b, world_idx, contact_idx]

        d_res_d0, d_res_d1, res_n_val, J0, J1, c_val = compute_contact_core(
            body0,
            body1,
            n,
            p0,
            p1,
            thickness0,
            thickness1,
            pose0,
            m_inv0,
            I_inv0,
            com0,
            pose1,
            m_inv1,
            I_inv1,
            com1,
            pose0_prev,
            pose1_prev,
            f_n,
            dt,
            compliance,
            fb_eps_sq,
        )

        if body0 >= 0:
            wp.atomic_add(res_d, b, world_idx, body0, d_res_d0)
        if body1 >= 0:
            wp.atomic_add(res_d, b, world_idx, body1, d_res_d1)

        res_n[b, world_idx, contact_idx] = res_n_val
