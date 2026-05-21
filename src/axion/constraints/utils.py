import warp as wp
from axion.mechanics import compute_spatial_momentum
from axion.mechanics import compute_world_inertia


@wp.func
def compute_effective_mass(
    body_q_1: wp.transform,
    body_q_2: wp.transform,
    J_1: wp.spatial_vector,
    J_2: wp.spatial_vector,
    m_inv_1: wp.float32,
    I_inv_b_1: wp.mat33,
    m_inv_2: wp.float32,
    I_inv_b_2: wp.mat33,
    body_1_idx: int,
    body_2_idx: int,
) -> float:
    """
    Computes the diagonal term (effective mass) J M^-1 J^T.
    Expects M_inv to be in WORLD frame.
    """
    val = 0.0
    if body_1_idx >= 0:
        # compute J M^-1 J^T
        I_inv_w_1 = compute_world_inertia(body_q_1, I_inv_b_1)
        val += wp.dot(J_1, compute_spatial_momentum(m_inv_1, I_inv_w_1, J_1))

    if body_2_idx >= 0:
        I_inv_w_2 = compute_world_inertia(body_q_2, I_inv_b_2)
        val += wp.dot(J_2, compute_spatial_momentum(m_inv_2, I_inv_w_2, J_2))

    return val


@wp.kernel
def fill_joint_constraint_body_idx_kernel(
    # --- Joint Definition ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_parent: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Output ---
    joint_constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
):
    world_idx, joint_idx = wp.tid()

    # Check bounds (though dims should match)
    if world_idx >= joint_type.shape[0] or joint_idx >= joint_type.shape[1]:
        return

    j_type = joint_type[world_idx, joint_idx]
    p_idx = joint_parent[world_idx, joint_idx]
    c_idx = joint_child[world_idx, joint_idx]
    start_offset = constraint_offsets[world_idx, joint_idx]

    # Determine constraint count
    count = 0
    if j_type == 0:  # PRISMATIC
        count = 5
    elif j_type == 1:  # REVOLUTE
        count = 5
    elif j_type == 2:  # BALL
        count = 3
    elif j_type == 3:  # FIXED
        count = 6
    elif j_type == 7:  # CABLE
        count = 0

    for k in range(count):
        offset = start_offset + k
        # Safety check for output bounds
        if offset < joint_constraint_body_idx.shape[1]:
            joint_constraint_body_idx[world_idx, offset, 0] = p_idx
            joint_constraint_body_idx[world_idx, offset, 1] = c_idx


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


@wp.kernel
def fill_joint_constraint_active_mask_kernel(
    # --- Joint Definition ---
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_enabled: wp.array(dtype=wp.int32, ndim=2),
    joint_child: wp.array(dtype=wp.int32, ndim=2),
    constraint_offsets: wp.array(dtype=wp.int32, ndim=2),
    # --- Output ---
    joint_constraint_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, joint_idx = wp.tid()

    if world_idx >= joint_type.shape[0] or joint_idx >= joint_type.shape[1]:
        return

    j_type = joint_type[world_idx, joint_idx]
    is_enabled = joint_enabled[world_idx, joint_idx] != 0

    # Check if valid child
    if joint_child[world_idx, joint_idx] < 0:
        is_enabled = False

    start_offset = constraint_offsets[world_idx, joint_idx]

    count = 0
    if j_type == 0:  # PRISMATIC
        count = 5
    elif j_type == 1:  # REVOLUTE
        count = 5
    elif j_type == 2:  # BALL
        count = 3
    elif j_type == 3:  # FIXED
        count = 6
    elif j_type == 7:  # CABLE
        count = 0

    val = 1.0 if is_enabled else 0.0

    for k in range(count):
        offset = start_offset + k
        if offset < joint_constraint_active_mask.shape[1]:
            joint_constraint_active_mask[world_idx, offset] = val
