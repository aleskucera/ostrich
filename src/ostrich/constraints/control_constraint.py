import warp as wp
from ostrich.constraints.joint_kinematics import compute_joint_transforms
from ostrich.core.types import JointMode


# --- 1. KINEMATICS ---


@wp.func
def compute_revolute_q_qd(
    X_w_p: wp.transform,
    X_w_c: wp.transform,
    axis_local: wp.vec3,
    body_u_p: wp.spatial_vector,
    body_u_c: wp.spatial_vector,
):
    # 1. Compute q (angle)
    q_p = wp.transform_get_rotation(X_w_p)
    q_c = wp.transform_get_rotation(X_w_c)

    q_p_inv = wp.quat_inverse(q_p)
    q_rel = wp.mul(q_p_inv, q_c)

    vec_part = wp.vec3(q_rel[0], q_rel[1], q_rel[2])
    sin_half_theta = wp.dot(vec_part, axis_local)
    cos_half_theta = q_rel[3]

    q = 2.0 * wp.atan2(sin_half_theta, cos_half_theta)

    # 2. Compute qd (velocity)
    w_p = wp.spatial_bottom(body_u_p)
    w_c = wp.spatial_bottom(body_u_c)

    w_rel_world = w_c - w_p
    axis_world = wp.quat_rotate(q_p, axis_local)
    qd = wp.dot(w_rel_world, axis_world)

    return q, qd


@wp.func
def compute_prismatic_q_qd(
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    X_w_p: wp.transform,
    axis_local: wp.vec3,
    body_u_p: wp.spatial_vector,
    body_u_c: wp.spatial_vector,
    r_p: wp.vec3,
    r_c: wp.vec3,
):
    # 1. q (Position along axis)
    q_p = wp.transform_get_rotation(X_w_p)
    axis_w = wp.quat_rotate(q_p, axis_local)

    delta = pos_c - pos_p
    q = wp.dot(delta, axis_w)

    # 2. qd (Velocity along axis)
    w_p = wp.spatial_bottom(body_u_p)
    v_p = wp.spatial_top(body_u_p)
    w_c = wp.spatial_bottom(body_u_c)
    v_c = wp.spatial_top(body_u_c)

    vel_c = v_c + wp.cross(w_c, r_c)
    vel_p = v_p + wp.cross(w_p, r_p)

    delta_dot = vel_c - vel_p
    axis_dot = wp.cross(w_p, axis_w)

    qd = wp.dot(delta_dot, axis_w) + wp.dot(delta, axis_dot)

    return q, qd


# --- 2. GEOMETRY (JACOBIANS) ---


@wp.func
def compute_revolute_jacobians(
    X_w_p: wp.transform,
    axis_local: wp.vec3,
):
    q_p = wp.transform_get_rotation(X_w_p)
    axis_w = wp.quat_rotate(q_p, axis_local)

    J_c = wp.spatial_vector(wp.vec3(0.0), axis_w)
    J_p = wp.spatial_vector(wp.vec3(0.0), -axis_w)

    return J_p, J_c


@wp.func
def compute_prismatic_jacobians(
    X_w_p: wp.transform,
    axis_local: wp.vec3,
    r_c: wp.vec3,
    pos_c: wp.vec3,
    com_p: wp.vec3,
):
    q_p = wp.transform_get_rotation(X_w_p)
    axis_w = wp.quat_rotate(q_p, axis_local)

    ang_c = wp.cross(r_c, axis_w)
    J_c = wp.spatial_vector(axis_w, ang_c)

    r_p_plus_delta = pos_c - com_p
    ang_p = wp.cross(r_p_plus_delta, axis_w)
    J_p = wp.spatial_vector(-axis_w, -ang_p)

    return J_p, J_c


# --- 3. CONTROL LAW (PHYSICS) ---


@wp.func
def compute_control_properties(
    q: float,
    qd: float,
    target: float,
    mode: int,
    ke: float,
    kd: float,
    dt: float,
    is_angular: bool,
):
    error = 0.0
    alpha = 0.0

    if mode == JointMode.TARGET_POSITION:
        raw_error = q - target

        # If it's a rotating joint, find the shortest path (-PI to +PI)
        # to prevent explosion when the angle wraps around.
        if is_angular:
            PI = 3.141592653589793
            TWO_PI = 6.283185307179586

            # Shift domain to [0, 2PI)
            raw_error_shifted = raw_error + PI
            # Modulo arithmetic
            mod_error = raw_error_shifted - TWO_PI * wp.floor(raw_error_shifted / TWO_PI)
            # Shift back to [-PI, PI)
            error = mod_error - PI
        else:
            error = raw_error

        # Convert position error to velocity level for the solver
        error = error / dt

        denom = dt * dt * ke + dt * kd
        if denom > 1e-6:
            alpha = 1.0 / denom
        else:
            alpha = 1.0e8

    elif mode == JointMode.TARGET_VELOCITY:
        error = qd - target
        denom = dt * ke
        if denom > 1e-6:
            alpha = 1.0 / denom
        else:
            alpha = 1.0e8

    return error, alpha


# --- 4. ORCHESTRATOR ---


@wp.func
def compute_control_local(
    j_type: int,
    X_w_p: wp.transform,
    X_w_c: wp.transform,
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    com_p: wp.vec3,
    axis_local: wp.vec3,
    body_u_p: wp.spatial_vector,
    body_u_c: wp.spatial_vector,
    target: float,
    mode: int,
    ke: float,
    kd: float,
    current_lambda: float,
    dt: float,
):
    J_p = wp.spatial_vector()
    J_c = wp.spatial_vector()

    current_q = 0.0
    current_qd = 0.0
    is_angular = False

    if j_type == 1:
        # REVOLUTE
        current_q, current_qd = compute_revolute_q_qd(X_w_p, X_w_c, axis_local, body_u_p, body_u_c)
        J_p, J_c = compute_revolute_jacobians(X_w_p, axis_local)
        is_angular = True
    elif j_type == 0:
        # PRISMATIC
        current_q, current_qd = compute_prismatic_q_qd(
            pos_p, pos_c, X_w_p, axis_local, body_u_p, body_u_c, r_p, r_c
        )
        J_p, J_c = compute_prismatic_jacobians(X_w_p, axis_local, r_c, pos_c, com_p)
        is_angular = False

    # Compliance & Error (Unified)
    error_vel, alpha = compute_control_properties(
        current_q, current_qd, target, mode, ke, kd, dt, is_angular
    )

    return (
        -J_p * current_lambda * dt,  # delta_h_d_p
        -J_c * current_lambda * dt,  # delta_h_d_c
        error_vel + alpha * current_lambda * dt,  # h_ctrl_val
        J_p,  # J_p
        J_c,  # J_c
        alpha,  # alpha
    )


# --- 5. KERNELS ---


@wp.kernel
def control_constraint_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_target_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_ctrl: wp.array(dtype=wp.float32, ndim=2),
    J_hat_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]
    constraint_active_mask[world_idx, ctrl_offset] = 1.0

    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx], body_com[world_idx, c_idx], joint_X_c[world_idx, joint_idx]
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    body_u_p = wp.spatial_vector()
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        body_u_p = body_u[world_idx, p_idx]

    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    axis_local = joint_axis[world_idx, qd_start_idx]
    body_u_c = body_u[world_idx, c_idx]

    # --- Select Target based on Mode ---
    target = 0.0
    if mode == JointMode.TARGET_POSITION:
        target = joint_target_pos[world_idx, qd_start_idx]
    elif mode == JointMode.TARGET_VELOCITY:
        target = joint_target_vel[world_idx, qd_start_idx]

    ke = joint_target_ke[world_idx, qd_start_idx]
    # wp.printf("Joint target ke: %f", ke)
    kd = joint_target_kd[world_idx, qd_start_idx]
    # wp.printf("Joint target kd: %f", kd)
    current_lambda = body_lambda_ctrl[world_idx, ctrl_offset]

    (res_hdp, res_hdc, res_hctrl, res_jp, res_jc, res_alpha) = compute_control_local(
        j_type,
        X_w_p,
        X_w_c,
        r_p,
        r_c,
        pos_p,
        pos_c,
        com_p,
        axis_local,
        body_u_p,
        body_u_c,
        target,
        mode,
        ke,
        kd,
        current_lambda,
        dt,
    )

    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, res_hdp)
    wp.atomic_add(h_d, world_idx, c_idx, res_hdc)

    h_ctrl[world_idx, ctrl_offset] = res_hctrl
    J_hat_values[world_idx, ctrl_offset, 0] = res_jp
    J_hat_values[world_idx, ctrl_offset, 1] = res_jc
    C_values[world_idx, ctrl_offset] = res_alpha


@wp.kernel
def control_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_target_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_ctrl: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]

    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx], body_com[world_idx, c_idx], joint_X_c[world_idx, joint_idx]
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    body_u_p = wp.spatial_vector()
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        body_u_p = body_u[world_idx, p_idx]

    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    axis_local = joint_axis[world_idx, qd_start_idx]
    body_u_c = body_u[world_idx, c_idx]

    # --- Select Target based on Mode ---
    target = 0.0
    if mode == JointMode.TARGET_POSITION:
        target = joint_target_pos[world_idx, qd_start_idx]
    elif mode == JointMode.TARGET_VELOCITY:
        target = joint_target_vel[world_idx, qd_start_idx]

    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]
    current_lambda = body_lambda_ctrl[world_idx, ctrl_offset]

    (res_hdp, res_hdc, res_hctrl, skip_jp, skip_jc, skip_alpha) = compute_control_local(
        j_type,
        X_w_p,
        X_w_c,
        r_p,
        r_c,
        pos_p,
        pos_c,
        com_p,
        axis_local,
        body_u_p,
        body_u_c,
        target,
        mode,
        ke,
        kd,
        current_lambda,
        dt,
    )

    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, res_hdp)
    wp.atomic_add(h_d, world_idx, c_idx, res_hdc)

    h_ctrl[world_idx, ctrl_offset] = res_hctrl


@wp.kernel
def batch_control_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=3),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_target_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_ctrl: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]

    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[batch_idx, world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx],
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    body_u_p = wp.spatial_vector()
    if p_idx >= 0:
        X_body_p = body_q[batch_idx, world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
        body_u_p = body_u[batch_idx, world_idx, p_idx]

    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    axis_local = joint_axis[world_idx, qd_start_idx]
    body_u_c = body_u[batch_idx, world_idx, c_idx]

    # --- Select Target based on Mode ---
    target = 0.0
    if mode == JointMode.TARGET_POSITION:
        target = joint_target_pos[world_idx, qd_start_idx]
    elif mode == JointMode.TARGET_VELOCITY:
        target = joint_target_vel[world_idx, qd_start_idx]

    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]
    current_lambda = body_lambda_ctrl[batch_idx, world_idx, ctrl_offset]

    (b_hdp, b_hdc, b_hctrl, skip_jp, skip_jc, skip_alpha) = compute_control_local(
        j_type,
        X_w_p,
        X_w_c,
        r_p,
        r_c,
        pos_p,
        pos_c,
        com_p,
        axis_local,
        body_u_p,
        body_u_c,
        target,
        mode,
        ke,
        kd,
        current_lambda,
        dt,
    )

    if p_idx >= 0:
        wp.atomic_add(h_d, batch_idx, world_idx, p_idx, b_hdp)
    wp.atomic_add(h_d, batch_idx, world_idx, c_idx, b_hdc)

    h_ctrl[batch_idx, world_idx, ctrl_offset] = b_hctrl


@wp.kernel
def fused_batch_control_residual_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_lambda_ctrl: wp.array(dtype=wp.float32, ndim=3),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    control_constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_target_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32, ndim=2),
    joint_target_ke: wp.array(dtype=wp.float32, ndim=2),
    joint_target_kd: wp.array(dtype=wp.float32, ndim=2),
    dt: wp.float32,
    num_batches: int,
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_ctrl: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, joint_idx = wp.tid()

    if joint_idx >= joint_type.shape[1]:
        return
    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    if j_type != 1 and j_type != 0:
        return

    qd_start_idx = joint_qd_start[world_idx, joint_idx]
    mode = joint_dof_mode[world_idx, qd_start_idx]
    if mode == 0:
        return

    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    if c_idx < 0:
        return

    ctrl_offset = control_constraint_offsets[world_idx, joint_idx]

    joint_X_c_val = joint_X_c[world_idx, joint_idx]
    joint_X_p_val = joint_X_p[world_idx, joint_idx]
    com_c = body_com[world_idx, c_idx]
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        com_p = body_com[world_idx, p_idx]

    axis_local = joint_axis[world_idx, qd_start_idx]

    # --- Select Target based on Mode ---
    target = 0.0
    if mode == JointMode.TARGET_POSITION:
        target = joint_target_pos[world_idx, qd_start_idx]
    elif mode == JointMode.TARGET_VELOCITY:
        target = joint_target_vel[world_idx, qd_start_idx]

    ke = joint_target_ke[world_idx, qd_start_idx]
    kd = joint_target_kd[world_idx, qd_start_idx]

    for b in range(num_batches):
        X_w_c, r_c, pos_c = compute_joint_transforms(
            body_q[b, world_idx, c_idx], com_c, joint_X_c_val
        )
        X_body_p = wp.transform_identity()
        if p_idx >= 0:
            X_body_p = body_q[b, world_idx, p_idx]
        X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p_val)

        body_u_c = body_u[b, world_idx, c_idx]
        body_u_p = wp.spatial_vector()
        if p_idx >= 0:
            body_u_p = body_u[b, world_idx, p_idx]

        current_lambda = body_lambda_ctrl[b, world_idx, ctrl_offset]

        (f_hdp, f_hdc, f_hctrl, skip_jp, skip_jc, skip_alpha) = compute_control_local(
            j_type,
            X_w_p,
            X_w_c,
            r_p,
            r_c,
            pos_p,
            pos_c,
            com_p,
            axis_local,
            body_u_p,
            body_u_c,
            target,
            mode,
            ke,
            kd,
            current_lambda,
            dt,
        )

        if p_idx >= 0:
            wp.atomic_add(h_d, b, world_idx, p_idx, f_hdp)
        wp.atomic_add(h_d, b, world_idx, c_idx, f_hdc)

        h_ctrl[b, world_idx, ctrl_offset] = f_hctrl


@wp.kernel
def fill_control_constraint_body_idx_kernel(
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    control_offsets: wp.array(dtype=wp.int32, ndim=2),
    constraint_body_idx_ctrl: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, joint_idx = wp.tid()
    j_type = joint_type[world_idx, joint_idx]
    count = 0
    if j_type == 1 or j_type == 0:
        qd_start = joint_qd_start[world_idx, joint_idx]
        mode = joint_dof_mode[world_idx, qd_start]
        if mode != 0:
            count = 1
    if count == 0:
        return

    offset = control_offsets[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    constraint_body_idx_ctrl[world_idx, offset, 0] = p_idx
    constraint_body_idx_ctrl[world_idx, offset, 1] = c_idx
