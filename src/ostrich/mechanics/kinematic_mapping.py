import warp as wp


@wp.func
def mul_G(
    q: wp.transform,
    u: wp.spatial_vector,
):
    """
    Computes the state derivative: q_dot = G(q) * u

    Mapping to Paper Notation:
      - q:      Generalized coordinates (Position x, Quaternion theta)
      - u:      Generalized velocities (Linear v, Angular w)
      - theta:  Quaternion components [theta_1, theta_2, theta_3, theta_4]
                where theta_1 is scalar, theta_2..4 are vector x,y,z.
    """

    # --- 1. Decomposition (Paper Variables) ---

    # Extract Linear (v) and Angular (w) velocities from u
    v = wp.spatial_top(u)
    w = wp.spatial_bottom(u)

    # Extract components of vector w
    w_1 = w[0]
    w_2 = w[1]
    w_3 = w[2]

    # Extract Quaternion theta from q
    # Warp stores quats as (x, y, z, w) -> (theta_2, theta_3, theta_4, theta_1)
    rot = wp.transform_get_rotation(q)
    theta_2 = rot[0]
    theta_3 = rot[1]
    theta_4 = rot[2]
    theta_1 = rot[3]  # Scalar part

    # --- 2. Matrix Multiplication G * u ---

    # Top Block: x_dot = I * v
    x_dot = v

    # Bottom Block: theta_dot = 0.5 * Q(theta) * w
    # We explicitly write the rows of matrix Q(theta) from the paper image

    # Row 1 (Scalar derivative dot_theta_1): [-t2, -t3, -t4] . w
    dot_theta_1 = -theta_2 * w_1 - theta_3 * w_2 - theta_4 * w_3

    # Row 2 (X derivative dot_theta_2): [ t1,  t4, -t3] . w
    dot_theta_2 = theta_1 * w_1 + theta_4 * w_2 - theta_3 * w_3

    # Row 3 (Y derivative dot_theta_3): [-t4,  t1,  t2] . w
    dot_theta_3 = -theta_4 * w_1 + theta_1 * w_2 + theta_2 * w_3

    # Row 4 (Z derivative dot_theta_4): [ t3, -t2,  t1] . w
    dot_theta_4 = theta_3 * w_1 - theta_2 * w_2 + theta_1 * w_3

    # --- 3. Reassembly ---

    # Construct the derivative quaternion (theta_dot)
    # Note: We must map theta indices back to Warp's (x, y, z, w) storage order
    theta_dot = wp.quat(dot_theta_2 * 0.5, dot_theta_3 * 0.5, dot_theta_4 * 0.5, dot_theta_1 * 0.5)

    # Pack result into a transform container (representing q_dot)
    q_dot = wp.transform(x_dot, theta_dot)

    return q_dot


@wp.func
def G_matvec(alpha: float, u: wp.spatial_vector, y: wp.transform, q_params: wp.transform):
    """
    Computes: result = y + alpha * (G(q_params) * u)

    This performs the kinematic update in one fused step.

    Args:
       alpha:    Step size (e.g., dt)
       u:        Input vector (spatial velocity)
       y:        Accumulator (e.g., previous state q_prev)
       q_params: State defining G (usually q_prev, same as y)
    """

    # --- 1. Decomposition ---
    v = wp.spatial_top(u)
    w = wp.spatial_bottom(u)

    # We unpack 'y' (the state we are adding to)
    y_pos = wp.transform_get_translation(y)
    y_rot = wp.transform_get_rotation(y)

    # We unpack 'q_params' (the state defining the G matrix)
    # usually q_params == y, but in some gradients they might differ.
    rot = wp.transform_get_rotation(q_params)
    theta_2 = rot[0]
    theta_3 = rot[1]
    theta_4 = rot[2]
    theta_1 = rot[3]  # scalar

    w_1 = w[0]
    w_2 = w[1]
    w_3 = w[2]

    # --- 2. Fused Calculation (alpha * G * u + y) ---

    # LINEAR PART:
    # x_new = y_pos + alpha * (I * v)
    out_pos = y_pos + v * alpha

    # ANGULAR PART:
    # theta_new = y_rot + alpha * (0.5 * Q * w)

    # Pre-multiply alpha * 0.5 to save ops
    s = alpha * 0.5

    # Compute Q*w rows directly and add to y_rot
    # Note: We match the paper's matrix Q rows to the output indices

    # Row 2 of output (x / theta_2)
    d_theta_2 = theta_1 * w_1 + theta_4 * w_2 - theta_3 * w_3
    out_rot_x = y_rot[0] + d_theta_2 * s

    # Row 3 of output (y / theta_3)
    d_theta_3 = -theta_4 * w_1 + theta_1 * w_2 + theta_2 * w_3
    out_rot_y = y_rot[1] + d_theta_3 * s

    # Row 4 of output (z / theta_4)
    d_theta_4 = theta_3 * w_1 - theta_2 * w_2 + theta_1 * w_3
    out_rot_z = y_rot[2] + d_theta_4 * s

    # Row 1 of output (w / theta_1 - Scalar)
    d_theta_1 = -theta_2 * w_1 - theta_3 * w_2 - theta_4 * w_3
    out_rot_w = y_rot[3] + d_theta_1 * s

    # --- 3. Pack Output ---
    # Note: We do NOT normalize here. A true AXPY operation shouldn't normalize.
    # The kernel should handle normalization after the update if needed.

    out_rot = wp.quat(out_rot_x, out_rot_y, out_rot_z, out_rot_w)

    return wp.transform(out_pos, out_rot)


@wp.func
def mul_Gt(q: wp.transform, input_vec: wp.transform):
    """
    Implements the multiplication result = G^T * z

    Mapping to Paper Notation:
      - q:         Current State (Position x, Quaternion theta)
      - input_vec: 7D Vector in maximal coords (e.g., Gradient dL/dq or Force)
                   Packed as wp.transform(vec3, quat)

    Output:
      - output_spatial: 6D Spatial Vector (Linear v, Angular w)
    """

    # --- 1. Decomposition ---

    # Unpack the "State" q to get theta (the rotation matrix terms)
    # theta = [theta_2, theta_3, theta_4, theta_1] (x, y, z, w)
    rot = wp.transform_get_rotation(q)
    theta_2 = rot[0]
    theta_3 = rot[1]
    theta_4 = rot[2]
    theta_1 = rot[3]  # Real part

    # Unpack the "Input Vector" (z)
    # Corresponds to d_x (3D) and d_theta (4D)
    d_x = wp.transform_get_translation(input_vec)
    d_quat = wp.transform_get_rotation(input_vec)

    # Map input quaternion parts to paper indices
    d_theta_2 = d_quat[0]
    d_theta_3 = d_quat[1]
    d_theta_4 = d_quat[2]
    d_theta_1 = d_quat[3]  # Scalar derivative part

    # --- 2. Matrix Multiplication G^T * z ---

    # Top Block: v = I * d_x
    out_v = d_x

    # Bottom Block: w = 0.5 * Q(theta)^T * d_theta
    # We transpose the matrix Q from the image (Swap rows/cols)

    # Col 1 of Q becomes Row 1 (Result w_1):
    # [-theta_2, theta_1, -theta_4, theta_3] dot [d_theta_1, d_theta_2, d_theta_3, d_theta_4]
    w_1 = -theta_2 * d_theta_1 + theta_1 * d_theta_2 - theta_4 * d_theta_3 + theta_3 * d_theta_4

    # Col 2 of Q becomes Row 2 (Result w_2):
    # [-theta_3, theta_4, theta_1, -theta_2] dot ...
    w_2 = -theta_3 * d_theta_1 + theta_4 * d_theta_2 + theta_1 * d_theta_3 - theta_2 * d_theta_4

    # Col 3 of Q becomes Row 3 (Result w_3):
    # [-theta_4, -theta_3, theta_2, theta_1] dot ...
    w_3 = -theta_4 * d_theta_1 - theta_3 * d_theta_2 + theta_2 * d_theta_3 + theta_1 * d_theta_4

    # --- 3. Reassembly ---

    # Apply the 0.5 factor from G definition
    out_w = wp.vec3(w_1 * 0.5, w_2 * 0.5, w_3 * 0.5)

    # Return as spatial vector (Linear, Angular)
    return wp.spatial_vector(out_v, out_w)


@wp.func
def Gt_matvec(alpha: float, z: wp.transform, y: wp.spatial_vector, q_params: wp.transform):
    """
    Computes: result = y + alpha * (G(q_params)^T * z)

    This is the FUSED AXPY for the transpose.
    Useful for Backpropagation / Gradient Accumulation.

    Args:
       alpha:    Scalar multiplier (e.g., 1.0 or dt)
       z:        Input 7D vector (e.g., gradient dL/dq) packed as transform
       y:        Accumulator 6D vector (e.g., gradient dL/du) packed as spatial_vector
       q_params: State defining G (the configuration at which Jacobian is evaluated)

    Output:
       result:   6D Spatial Vector (accumulated gradient)
    """

    # --- 1. Decomposition ---

    # Unpack q_params (defines the matrix values)
    # theta = [theta_2, theta_3, theta_4, theta_1] -> (x, y, z, w)
    rot = wp.transform_get_rotation(q_params)
    theta_2 = rot[0]
    theta_3 = rot[1]
    theta_4 = rot[2]
    theta_1 = rot[3]  # Real part

    # Unpack Input z (the 7D vector we are multiplying)
    d_x = wp.transform_get_translation(z)
    d_quat = wp.transform_get_rotation(z)

    # Map input quaternion to paper indices
    d_theta_2 = d_quat[0]
    d_theta_3 = d_quat[1]
    d_theta_4 = d_quat[2]
    d_theta_1 = d_quat[3]  # Scalar derivative part

    # Unpack Accumulator y (the 6D vector we are adding to)
    y_v = wp.spatial_top(y)
    y_w = wp.spatial_bottom(y)

    # --- 2. Fused Calculation (alpha * G^T * z + y) ---

    # LINEAR PART:
    # v_new = y_v + alpha * (I * d_x)
    out_v = y_v + d_x * alpha

    # ANGULAR PART:
    # w_new = y_w + alpha * (0.5 * Q^T * d_theta)

    # Pre-multiply alpha * 0.5
    s = alpha * 0.5

    # Compute Q^T * d_theta rows directly and add to y_w

    # Row 1 (Result w_1): [-t2, t1, -t4, t3] dot [dt1, dt2, dt3, dt4]
    val_w1 = -theta_2 * d_theta_1 + theta_1 * d_theta_2 - theta_4 * d_theta_3 + theta_3 * d_theta_4
    out_w1 = y_w[0] + val_w1 * s

    # Row 2 (Result w_2): [-t3, t4, t1, -t2] dot ...
    val_w2 = -theta_3 * d_theta_1 + theta_4 * d_theta_2 + theta_1 * d_theta_3 - theta_2 * d_theta_4
    out_w2 = y_w[1] + val_w2 * s

    # Row 3 (Result w_3): [-t4, -t3, t2, t1] dot ...
    val_w3 = -theta_4 * d_theta_1 - theta_3 * d_theta_2 + theta_2 * d_theta_3 + theta_1 * d_theta_4
    out_w3 = y_w[2] + val_w3 * s

    # --- 3. Pack Output ---
    out_w = wp.vec3(out_w1, out_w2, out_w3)

    return wp.spatial_vector(out_v, out_w)
