import warp as wp
from axion.mechanics import orthogonal_basis


@wp.func
def compute_joint_transforms(
    body_q: wp.transform,
    body_com: wp.vec3,
    joint_X_local: wp.transform,
):
    """
    Computes the World Space joint frame and the lever arm (vector from COM to Joint).
    """
    # Joint Frame in World Space: X_w = X_body * X_local
    X_w = body_q * joint_X_local

    # Center of Mass in World Space
    com_w = wp.transform_point(body_q, body_com)

    # Joint Position in World Space
    pos_w = wp.transform_get_translation(X_w)

    # Lever Arm: r = pos_joint - pos_com
    r = pos_w - com_w

    return X_w, r, pos_w


# ---------------------------------------------------------------------------- #
#                               Constraint Helpers                             #
# ---------------------------------------------------------------------------- #


@wp.func
def vector_dot_axis(v: wp.vec3, axis_idx: wp.int32):
    if axis_idx == 0:
        return v[0]
    elif axis_idx == 1:
        return v[1]
    return v[2]


@wp.func
def get_linear_component(
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    axis_idx: wp.int32,  # 0=X, 1=Y, 2=Z
):
    """
    Generates the Jacobian data and Error for a linear constraint along a global axis.
    Used by: Spherical, Revolute, Fixed.
    """
    # 1. Define the Global Axis (World Space)
    axis_vec = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        axis_vec = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        axis_vec = wp.vec3(0.0, 1.0, 0.0)
    else:
        axis_vec = wp.vec3(0.0, 0.0, 1.0)

    ang_p = wp.cross(r_p, axis_vec)
    ang_c = wp.cross(r_c, axis_vec)

    J_c = wp.spatial_vector(axis_vec, ang_c)
    J_p = wp.spatial_vector(-axis_vec, -ang_p)

    # 3. Compute Error (Distance)
    delta = pos_c - pos_p
    error = delta[axis_idx]

    return J_p, J_c, error


@wp.func
def get_angular_component(
    X_wp: wp.transform,
    X_wc: wp.transform,
    axis_idx: wp.int32,  # 0, 1, or 2 relative to the Joint Frame
):
    """
    Generates the Jacobian data and Error for an angular constraint.
    This locks the rotation around a specific local axis of the joint.
    """
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    local_axis = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        local_axis = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        local_axis = wp.vec3(0.0, 1.0, 0.0)
    else:
        local_axis = wp.vec3(0.0, 0.0, 1.0)

    axis_w = wp.quat_rotate(q_p, local_axis)

    J_c = wp.spatial_vector(wp.vec3(0.0), axis_w)
    J_p = wp.spatial_vector(wp.vec3(0.0), -axis_w)

    q_p_inv = wp.quat_inverse(q_p)
    q_rel = wp.mul(q_p_inv, q_c)

    vec_part = wp.vec3(q_rel[0], q_rel[1], q_rel[2])

    error = 2.0 * vector_dot_axis(vec_part, axis_idx)

    if q_rel[3] < 0.0:
        error = -error

    return J_p, J_c, error


@wp.func
def get_revolute_angular_component(
    X_wp: wp.transform,
    X_wc: wp.transform,
    hinge_axis_local: wp.vec3,
    ortho_idx: wp.int32,  # 0 or 1
):
    """
    Specialized helper for Revolute joints.
    Locks the two axes ORTHOGONAL to the hinge axis.
    """
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    b1_local, b2_local = orthogonal_basis(hinge_axis_local)

    axis_c_world = wp.quat_rotate(q_c, hinge_axis_local)

    target_basis_p_world = wp.vec3(0.0)
    if ortho_idx == 0:
        target_basis_p_world = wp.quat_rotate(q_p, b1_local)
    else:
        target_basis_p_world = wp.quat_rotate(q_p, b2_local)

    error = wp.dot(axis_c_world, target_basis_p_world)

    rot_axis = wp.cross(axis_c_world, target_basis_p_world)

    J_c = wp.spatial_vector(wp.vec3(0.0), rot_axis)
    J_p = wp.spatial_vector(wp.vec3(0.0), -rot_axis)

    return J_p, J_c, error


@wp.func
def get_prismatic_linear_component(
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    com_p: wp.vec3,
    r_c: wp.vec3,
    X_wp: wp.transform,
    axis_local: wp.vec3,
    ortho_idx: wp.int32,  # 0 or 1
):
    """
    Constrains linear motion to be along the specified axis (allows sliding).
    """
    q_p = wp.transform_get_rotation(X_wp)

    b1_local, b2_local = orthogonal_basis(axis_local)

    normal_vec = wp.vec3(0.0)
    if ortho_idx == 0:
        normal_vec = wp.quat_rotate(q_p, b1_local)
    else:
        normal_vec = wp.quat_rotate(q_p, b2_local)

    delta = pos_c - pos_p
    error = wp.dot(delta, normal_vec)

    ang_c = wp.cross(r_c, normal_vec)
    J_c = wp.spatial_vector(normal_vec, ang_c)

    r_p_plus_delta = pos_c - com_p
    ang_p = wp.cross(normal_vec, r_p_plus_delta)
    J_p = wp.spatial_vector(-normal_vec, ang_p)

    return J_p, J_c, error


# ---------------------------------------------------------------------------- #
#                           Unified Dispatcher                                 #
# ---------------------------------------------------------------------------- #


@wp.func
def compute_joint_row(
    j_type: int,
    row_idx: int,
    # Kinematics
    X_wp: wp.transform,
    X_wc: wp.transform,
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    com_p: wp.vec3,
    # Definition
    axis_local: wp.vec3,
):
    """
    Unified dispatcher that computes the J and Error for a specific row
    of a specific joint type.

    Returns:
        J_p, J_c, error, active_flag (1.0 if this row exists, 0.0 if not)
    """
    J_p = wp.spatial_vector()
    J_c = wp.spatial_vector()
    err = 0.0
    active = 0.0

    # === PRISMATIC (0) ===
    # 5 constraints: 2 Linear Ortho, 3 Angular Locked
    if j_type == 0:
        if row_idx < 2:
            # Linear Ortho Constraints (0, 1)
            J_p, J_c, err = get_prismatic_linear_component(
                pos_p, pos_c, com_p, r_c, X_wp, axis_local, row_idx
            )
            active = 1.0
        elif row_idx < 5:
            # Angular Constraints (2, 3, 4) maps to axis 0, 1, 2
            J_p, J_c, err = get_angular_component(X_wp, X_wc, row_idx - 2)
            active = 1.0

    # === REVOLUTE (1) ===
    # 5 constraints: 3 Linear Locked, 2 Angular Ortho
    elif j_type == 1:
        if row_idx < 3:
            # Linear Locked (0, 1, 2)
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, row_idx)
            active = 1.0
        elif row_idx < 5:
            # Angular Ortho (3, 4) maps to ortho 0, 1
            J_p, J_c, err = get_revolute_angular_component(X_wp, X_wc, axis_local, row_idx - 3)
            active = 1.0

    # === BALL (2) ===
    # 3 constraints: 3 Linear Locked
    elif j_type == 2:
        if row_idx < 3:
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, row_idx)
            active = 1.0

    # === FIXED (3) ===
    # 6 constraints: 3 Linear Locked, 3 Angular Locked
    elif j_type == 3:
        if row_idx < 3:
            J_p, J_c, err = get_linear_component(r_p, r_c, pos_p, pos_c, row_idx)
            active = 1.0
        elif row_idx < 6:
            J_p, J_c, err = get_angular_component(X_wp, X_wc, row_idx - 3)
            active = 1.0

    return J_p, J_c, err, active
