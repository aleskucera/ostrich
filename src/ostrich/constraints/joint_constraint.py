import warp as wp

from .joint_kinematics import compute_joint_row
from .joint_kinematics import compute_joint_transforms

# ---------------------------------------------------------------------------- #
#                           Shared Logic                                       #
# ---------------------------------------------------------------------------- #


@wp.func
def submit_row(
    world_idx: int,
    constraint_idx: int,
    p_idx: int,
    c_idx: int,
    J_p: wp.spatial_vector,
    J_c: wp.spatial_vector,
    error: float,
    active: float,
    dt: float,
    compliance: float,
    # Outputs
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),  # Optional
    C_j_values: wp.array(dtype=wp.float32, ndim=2),  # Optional
    is_solver: bool,
):
    if active == 0.0:
        # Zero out if needed (important for sparsity)
        h_j[world_idx, constraint_idx] = 0.0
        if is_solver:
            J_hat_j_values[world_idx, constraint_idx, 0] = wp.spatial_vector()
            J_hat_j_values[world_idx, constraint_idx, 1] = wp.spatial_vector()
            C_j_values[world_idx, constraint_idx] = 0.0
        return

    current_lambda = body_lambda_j[world_idx, constraint_idx]

    # 1. Atomic Force Update
    if p_idx >= 0:
        wp.atomic_add(h_d, world_idx, p_idx, -J_p * current_lambda * dt)
    wp.atomic_add(h_d, world_idx, c_idx, -J_c * current_lambda * dt)

    # 2. Residual
    h_j[world_idx, constraint_idx] = (error + compliance * current_lambda) / dt

    # 3. Solver Data
    if is_solver:
        J_hat_j_values[world_idx, constraint_idx, 0] = J_p
        J_hat_j_values[world_idx, constraint_idx, 1] = J_c
        C_j_values[world_idx, constraint_idx] = compliance / (dt * dt)


# ---------------------------------------------------------------------------- #
#                           Kernels                                            #
# ---------------------------------------------------------------------------- #


@wp.kernel
def joint_constraint_kernel(
    # State
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # Def
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_compliance_override: wp.array(dtype=wp.float32, ndim=2),
    # Params
    dt: wp.float32,
    global_compliance: wp.float32,
    # Outputs
    constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
    J_hat_j_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    C_j_values: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    # Data Loading
    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    comp = joint_compliance_override[world_idx, joint_idx]
    if comp < 0.0:
        comp = global_compliance

    # Kinematics
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx], body_com[world_idx, c_idx], joint_X_c[world_idx, joint_idx]
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    # Axis (for Revolute/Prismatic)
    axis_local = wp.vec3(0.0)
    if j_type == 0 or j_type == 1:
        # Revert to using qd_start to find the correct packed axis
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

    row_count = 0
    if j_type == 0:  # Prismatic
        row_count = 5
    elif j_type == 1:  # Revolute
        row_count = 5
    elif j_type == 2:  # Ball
        row_count = 3
    elif j_type == 3:  # Fixed
        row_count = 6

    # Process all possible 6 rows
    for i in range(wp.static(6)):
        if i >= row_count:
            continue  # [FIX] Stop processing to prevent overwriting next joint's memory
        J_p, J_c, err, active = compute_joint_row(
            j_type, i, X_w_p, X_w_c, r_p, r_c, pos_p, pos_c, com_p, axis_local
        )

        row_global_idx = start_offset + i
        constraint_active_mask[world_idx, row_global_idx] = active

        if active > 0.0:
            submit_row(
                world_idx,
                row_global_idx,
                p_idx,
                c_idx,
                J_p,
                J_c,
                err,
                active,
                dt,
                comp,
                h_d,
                h_j,
                body_lambda_j,
                J_hat_j_values,
                C_j_values,
                True,
            )


@wp.kernel
def joint_residual_kernel(
    # State
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    # Def
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    joint_compliance_override: wp.array(dtype=wp.float32, ndim=2),
    # Params
    dt: wp.float32,
    global_compliance: wp.float32,
    # Outputs
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
    h_j: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    # Data Loading
    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    comp = joint_compliance_override[world_idx, joint_idx]
    if comp < 0.0:
        comp = global_compliance

    # Kinematics
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[world_idx, c_idx], body_com[world_idx, c_idx], joint_X_c[world_idx, joint_idx]
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    # Axis (for Revolute/Prismatic)
    axis_local = wp.vec3(0.0)
    if j_type == 0 or j_type == 1:
        # Revert to using qd_start to find the correct packed axis
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

    row_count = 0
    if j_type == 0:  # Prismatic
        row_count = 5
    elif j_type == 1:  # Revolute
        row_count = 5
    elif j_type == 2:  # Ball
        row_count = 3
    elif j_type == 3:  # Fixed
        row_count = 6

    # Process all possible 6 rows
    for i in range(wp.static(6)):
        if i >= row_count:
            continue  # [FIX] Stop processing to prevent overwriting next joint's memory

        J_p, J_c, err, active = compute_joint_row(
            j_type, i, X_w_p, X_w_c, r_p, r_c, pos_p, pos_c, com_p, axis_local
        )

        if active > 0.0:
            row_global_idx = start_offset + i
            # Inline submit for residual (no J/C output)
            lam = body_lambda_j[world_idx, row_global_idx]
            if p_idx >= 0:
                wp.atomic_add(h_d, world_idx, p_idx, -J_p * lam * dt)
            wp.atomic_add(h_d, world_idx, c_idx, -J_c * lam * dt)
            h_j[world_idx, row_global_idx] = (err + comp * lam) / dt


# ---------------------------------------------------------------------------- #
#                           Batched Kernels                                    #
# ---------------------------------------------------------------------------- #


@wp.kernel
def batch_joint_residual_kernel(
    # State (Batched)
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    # Model Data (Shared)
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # Params
    dt: wp.float32,
    compliance: wp.float32,
    # Outputs (Batched)
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
):
    batch_idx, world_idx, joint_idx = wp.tid()

    if joint_enabled[world_idx, joint_idx] == 0:
        return

    # Data Loading
    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Kinematics
    X_w_c, r_c, pos_c = compute_joint_transforms(
        body_q[batch_idx, world_idx, c_idx],
        body_com[world_idx, c_idx],
        joint_X_c[world_idx, joint_idx],
    )

    X_body_p = wp.transform_identity()
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        X_body_p = body_q[batch_idx, world_idx, p_idx]
        com_p = body_com[world_idx, p_idx]
    X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p[world_idx, joint_idx])

    # Axis (for Revolute/Prismatic)
    axis_local = wp.vec3(0.0)
    if j_type == 0 or j_type == 1:
        # Revert to using qd_start to find the correct packed axis
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

    row_count = 0
    if j_type == 0:  # Prismatic
        row_count = 5
    elif j_type == 1:  # Revolute
        row_count = 5
    elif j_type == 2:  # Ball
        row_count = 3
    elif j_type == 3:  # Fixed
        row_count = 6

    for i in range(wp.static(6)):
        if i >= row_count:
            continue  # [FIX] Stop processing to prevent overwriting next joint's memory

        J_p, J_c, err, active = compute_joint_row(
            j_type, i, X_w_p, X_w_c, r_p, r_c, pos_p, pos_c, com_p, axis_local
        )

        if active > 0.0:
            row_idx = start_offset + i
            lam = body_lambda_j[batch_idx, world_idx, row_idx]
            if p_idx >= 0:
                wp.atomic_add(h_d, batch_idx, world_idx, p_idx, -J_p * lam * dt)
            wp.atomic_add(h_d, batch_idx, world_idx, c_idx, -J_c * lam * dt)
            h_j[batch_idx, world_idx, row_idx] = (err + compliance * lam) / dt


@wp.kernel
def fused_batch_joint_residual_kernel(
    # State (Batched)
    body_q: wp.array(dtype=wp.transform, ndim=3),
    body_lambda_j: wp.array(dtype=wp.float32, ndim=3),
    # Model Data (Shared)
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    joint_X_p: wp.array(dtype=wp.transform, ndim=2),
    joint_X_c: wp.array(dtype=wp.transform, ndim=2),
    joint_axis: wp.array(dtype=wp.vec3, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.bool, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # Params
    dt: wp.float32,
    compliance: wp.float32,
    num_batches: int,
    # Outputs (Batched)
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
    h_j: wp.array(dtype=wp.float32, ndim=3),
):
    world_idx, joint_idx = wp.tid()

    if joint_idx >= joint_type.shape[1]:
        return
    if joint_enabled[world_idx, joint_idx] == 0:
        return

    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Pre-load shared data
    joint_X_c_val = joint_X_c[world_idx, joint_idx]
    com_c = body_com[world_idx, c_idx]
    joint_X_p_val = joint_X_p[world_idx, joint_idx]
    com_p = wp.vec3(0.0)
    if p_idx >= 0:
        com_p = body_com[world_idx, p_idx]

    # Axis (for Revolute/Prismatic)
    axis_local = wp.vec3(0.0)
    if j_type == 0 or j_type == 1:
        # Revert to using qd_start to find the correct packed axis
        axis_idx = joint_qd_start[world_idx, joint_idx]
        axis_local = joint_axis[world_idx, axis_idx]

    # Loop batches
    for b in range(num_batches):
        X_w_c, r_c, pos_c = compute_joint_transforms(
            body_q[b, world_idx, c_idx], com_c, joint_X_c_val
        )

        X_body_p = wp.transform_identity()
        if p_idx >= 0:
            X_body_p = body_q[b, world_idx, p_idx]
        X_w_p, r_p, pos_p = compute_joint_transforms(X_body_p, com_p, joint_X_p_val)

        row_count = 0
        if j_type == 0:  # Prismatic
            row_count = 5
        elif j_type == 1:  # Revolute
            row_count = 5
        elif j_type == 2:  # Ball
            row_count = 3
        elif j_type == 3:  # Fixed
            row_count = 6

        for i in range(wp.static(6)):
            if i >= row_count:
                continue  # [FIX] Stop processing to prevent overwriting next joint's memory

            J_p, J_c, err, active = compute_joint_row(
                j_type, i, X_w_p, X_w_c, r_p, r_c, pos_p, pos_c, com_p, axis_local
            )

            if active > 0.0:
                row_idx = start_offset + i
                lam = body_lambda_j[b, world_idx, row_idx]
                if p_idx >= 0:
                    wp.atomic_add(h_d, b, world_idx, p_idx, -J_p * lam * dt)
                wp.atomic_add(h_d, b, world_idx, c_idx, -J_c * lam * dt)
                h_j[b, world_idx, row_idx] = (err + compliance * lam) / dt
