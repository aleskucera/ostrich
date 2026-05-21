import warp as wp
from axion.mechanics import compute_spatial_momentum
from axion.mechanics import compute_world_inertia
from axion.mechanics.kinematic_mapping import Gt_matvec  # Assuming your import


# -----------------------------------------------------------------------------
# 1. Body Initialization Kernel
#    Computes: w_u = M^-1 * (grad_u + h * G^T * grad_q)
# -----------------------------------------------------------------------------
@wp.kernel
def compute_body_adjoint_init_kernel(
    grad_q: wp.array(dtype=wp.transform, ndim=2),
    grad_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),  # Local Scalar
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),  # Local Matrix
    dt: float,
    # Output
    w_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()

    # 1. Fetch Gradients
    g_u = grad_u[world_idx, body_idx]
    g_q_geom = grad_q[world_idx, body_idx]
    q_geom = body_q[world_idx, body_idx]
    com = body_com[world_idx, body_idx]

    # --- Transform Gradient from Geometric Origin to CoM ---
    # The solver works at the CoM, but the tape/loss might be defined at the geometric origin.
    # p_geom = p_com + R * L (where L = -com)
    # grad_p_com = grad_p_geom
    # grad_r_com = grad_r_geom + (R * L) x grad_p_geom

    g_p_geom = wp.transform_get_translation(g_q_geom)
    g_r_geom = wp.transform_get_rotation(g_q_geom)

    r_geom = wp.transform_get_rotation(q_geom)
    lever_world = wp.quat_rotate(r_geom, -com)

    # Torque from force g_p acting at lever_world
    torque_com = wp.cross(lever_world, g_p_geom)

    # Convert torque to quaternion-space gradient (compatible with G^T)
    # Since G^T will multiply by 0.5 * Q(r)^T, we need to provide the rotation gradient g_r.
    # Actually, G^T * g_q_com = [g_p_com, 0.5 * Q^T * g_r_com]
    # We want this to be [g_p_geom, torque_com + 0.5 * Q^T * g_r_geom]
    # Wait, the angular part of G^T * g_q is already 0.5 * Q^T * g_r.
    # So we can just add a 'virtual' g_r that produces torque_com.
    # Or more simply, we can just add torque_com to the angular part of f_drive directly.

    # 2. Compute Driving Force: f_drive = - grad_u - dt * G^T * grad_q
    # We first compute G^T * g_q_geom
    f_drive = (-1.0) * Gt_matvec(dt, g_q_geom, g_u, q_geom)

    # Then add the torque correction from the CoM offset
    # Torque term: - dt * torque_com
    f_drive_v = wp.spatial_top(f_drive)
    f_drive_w = wp.spatial_bottom(f_drive) - torque_com * dt
    f_drive = wp.spatial_vector(f_drive_v, f_drive_w)

    # 3. Prepare Inverse Inertia in World Frame
    m_inv = body_m_inv[world_idx, body_idx]
    I_loc_inv = body_I_inv[world_idx, body_idx]

    # Note: I_loc_inv should be at the CoM.
    # If the pose q_geom is used, we must be careful.
    # But R_geom == R_com, so it's fine.
    I_w_inv = compute_world_inertia(q_geom, I_loc_inv)

    # 4. Apply Inverse Mass to Force (M^-1 * f) using your momentum function
    w_u[world_idx, body_idx] = compute_spatial_momentum(m_inv, I_w_inv, f_drive)


# -----------------------------------------------------------------------------
# 2. RHS Projection Kernel (Gather Bodies -> Constraints)
#    Computes: b = J * w_u
# -----------------------------------------------------------------------------
@wp.kernel
def compute_adjoint_rhs_kernel(
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    w_u_body: wp.array(dtype=wp.spatial_vector, ndim=2),
    # Output
    adjoint_rhs: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    if constraint_active_mask[world_idx, constraint_idx] == 0.0:
        adjoint_rhs[world_idx, constraint_idx] = 0.0
        return

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]

    val = float(0.0)

    # Project: J * w_u
    if body_1 >= 0:
        J_1 = J_values[world_idx, constraint_idx, 0]
        w_1 = w_u_body[world_idx, body_1]
        val += wp.dot(J_1, w_1)

    if body_2 >= 0:
        J_2 = J_values[world_idx, constraint_idx, 1]
        w_2 = w_u_body[world_idx, body_2]
        val += wp.dot(J_2, w_2)

    adjoint_rhs[world_idx, constraint_idx] = val


# -----------------------------------------------------------------------------
# 3. Constraint Feedback Kernel (Scatter Constraints -> Bodies)
#    Computes: w_u -= M^-1 * (J^T * w_lambda)
# -----------------------------------------------------------------------------
@wp.kernel
def subtract_constraint_feedback_kernel(
    w_lambda: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_m_inv: wp.array(dtype=wp.float32, ndim=2),
    body_I_inv: wp.array(dtype=wp.mat33, ndim=2),
    # Output (In-Place Update)
    w_u: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, constraint_idx = wp.tid()

    if constraint_active_mask[world_idx, constraint_idx] == 0.0:
        return

    lam = w_lambda[world_idx, constraint_idx]

    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]

    if body_1 >= 0:
        # 1. Compute Constraint Force: f_c = J^T * lambda
        f_c_1 = lam * J_1

        # 2. Transform Inertia
        q_1 = body_q[world_idx, body_1]
        m_inv_1 = body_m_inv[world_idx, body_1]
        I_loc_inv_1 = body_I_inv[world_idx, body_1]
        I_w_inv_1 = compute_world_inertia(q_1, I_loc_inv_1)

        # 3. Apply Inverse Inertia: accel = compute_spatial_momentum(m_inv, I_w_inv, force)
        accel_1 = compute_spatial_momentum(m_inv_1, I_w_inv_1, f_c_1)

        # 4. Atomic Subtract
        wp.atomic_add(w_u, world_idx, body_1, -accel_1)

    if body_2 >= 0:
        f_c_2 = lam * J_2

        q_2 = body_q[world_idx, body_2]
        m_inv_2 = body_m_inv[world_idx, body_2]
        I_loc_inv_2 = body_I_inv[world_idx, body_2]
        I_w_inv_2 = compute_world_inertia(q_2, I_loc_inv_2)

        accel_2 = compute_spatial_momentum(m_inv_2, I_w_inv_2, f_c_2)

        wp.atomic_add(w_u, world_idx, body_2, -accel_2)
