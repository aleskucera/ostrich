import warp as wp

from .kinematic_mapping import G_matvec


@wp.func
def integrate_body_pose(
    vel: wp.spatial_vector,
    pose_prev: wp.transform,
    com: wp.vec3,
    dt: float,
):
    # 1. Shift State to Center of Mass
    p_geom = wp.transform_get_translation(pose_prev)
    r_prev = wp.transform_get_rotation(pose_prev)

    # Calculate position of CoM in World Frame
    p_com = p_geom + wp.quat_rotate(r_prev, com)

    # Construct temporary CoM transform
    # Note: Rotation is the same for CoM and Geometric Origin
    pose_prev_com = wp.transform(p_com, r_prev)

    # 2. Fused Integration Step (AXPY)
    # q_new = q_prev + dt * (G(q) * u)
    pose_new_com = G_matvec(dt, vel, pose_prev_com, pose_prev_com)

    # 3. Post-Process
    p_com_new = wp.transform_get_translation(pose_new_com)
    r_new_raw = wp.transform_get_rotation(pose_new_com)

    # Normalize quaternion (Explicit integration drifts off manifold)
    r_new = wp.normalize(r_new_raw)

    # Shift Position back to Geometric Origin
    p_geom_new = p_com_new - wp.quat_rotate(r_new, com)

    return wp.transform(p_geom_new, r_new)


@wp.kernel
def integrate_body_pose_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    dt: wp.float32,
    body_pose: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()

    # 1. Fetch Inputs
    vel = body_vel[world_idx, body_idx]
    pose_prev = body_pose_prev[world_idx, body_idx]
    com = body_com[world_idx, body_idx]

    # 2. Run Shared Logic
    pose_new = integrate_body_pose(vel, pose_prev, com, dt)

    # 3. Store Result
    body_pose[world_idx, body_idx] = pose_new


@wp.kernel
def integrate_batched_body_pose_kernel(
    batch_body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_pose_prev: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    dt: wp.float32,
    batch_body_pose: wp.array(dtype=wp.transform, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()

    # 1. Fetch Inputs (Note the 3D indexing for velocity)
    vel = batch_body_vel[batch_idx, world_idx, body_idx]

    # Pose/CoM are shared across the batch, so we use 2D indexing
    pose_prev = body_pose_prev[world_idx, body_idx]
    com = body_com[world_idx, body_idx]

    # 2. Run Shared Logic
    pose_new = integrate_body_pose(vel, pose_prev, com, dt)

    # 3. Store Result (Note the 3D indexing for output)
    batch_body_pose[batch_idx, world_idx, body_idx] = pose_new
