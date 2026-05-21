import warp as wp


@wp.struct
class SpatialInertia:
    m: wp.float32
    inertia: wp.mat33


@wp.func
def compute_spatial_momentum(
    mass: wp.float32,
    inertia: wp.mat33,
    velocity: wp.spatial_vector,
) -> wp.spatial_vector:
    top = mass * wp.spatial_top(velocity)
    bot = inertia @ wp.spatial_bottom(velocity)
    return wp.spatial_vector(top, bot)


@wp.func
def compute_world_inertia(
    body_q: wp.transform,
    body_inertia: wp.mat33,
) -> wp.mat33:
    # Get orientation quaternion from transform
    orientation = wp.transform_get_rotation(body_q)
    R = wp.quat_to_matrix(orientation)

    # Transform inertia tensor: I_w = R * I_body * R^T
    I_w = R @ body_inertia @ wp.transpose(R)

    return I_w


@wp.kernel
def spatial_inertia_kernel(
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat33),
    # Outputs
    spatial_inertia: wp.array(dtype=SpatialInertia),
):
    tid = wp.tid()
    spatial_inertia[tid] = SpatialInertia(mass[tid], inertia[tid])


@wp.kernel
def world_spatial_inertia_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # Outputs
    world_spatial_inertia: wp.array(dtype=SpatialInertia, ndim=2),
):
    world_idx, body_idx = wp.tid()

    transform = body_q[world_idx, body_idx]
    m = body_mass[world_idx, body_idx]
    I = body_inertia[world_idx, body_idx]

    orientation = wp.transform_get_rotation(transform)
    R = wp.quat_to_matrix(orientation)
    I_w = R @ I @ wp.transpose(R)

    world_spatial_inertia[world_idx, body_idx] = SpatialInertia(m, I_w)
