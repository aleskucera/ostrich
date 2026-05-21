import warp as wp
from axion.mechanics import compute_spatial_momentum
from axion.mechanics import compute_world_inertia


@wp.kernel
def unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Property Inputs ---
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    if body_idx >= body_u.shape[1]:
        return

    q = body_q[world_idx, body_idx]
    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]

    # Compute World Inertia
    I_world = compute_world_inertia(q, I_body)

    u = body_u[world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    # Gravity force: f_g = [m * g, 0]
    # to_spatial_momentum was used before, which for [g, 0] gives [m*g, I_w*0] = [m*g, 0]
    f_g = wp.spatial_vector(m * g_accel[0], wp.vec3(0.0))

    # Gyroscopic term: w x (I @ w)
    w = wp.spatial_bottom(u)
    f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, I_world @ w))

    # momentum_diff = M @ (u - u_prev)
    momentum_diff = compute_spatial_momentum(m, I_world, u - u_prev)

    h_d[world_idx, body_idx] = momentum_diff - (f + f_g + f_gyro) * dt


@wp.kernel
def batch_unconstrained_dynamics_kernel(
    # --- Body State Inputs ---
    batch_body_q: wp.array(dtype=wp.transform, ndim=3),
    batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    # --- Body Property Inputs ---
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Simulation Parameters ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    batch_idx, world_idx, body_idx = wp.tid()
    if body_idx >= batch_body_u.shape[2]:
        return
    q = batch_body_q[batch_idx, world_idx, body_idx]
    u = batch_body_u[batch_idx, world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]

    # Compute World Inertia
    I_world = compute_world_inertia(q, I_body)

    f_g = wp.spatial_vector(m * g_accel[0], wp.vec3(0.0))

    # Gyroscopic term: w x (I @ w)
    w = wp.spatial_bottom(u)
    f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, I_world @ w))

    # momentum_diff = M @ (u - u_prev)
    momentum_diff = compute_spatial_momentum(m, I_world, u - u_prev)

    h_d[batch_idx, world_idx, body_idx] = momentum_diff - (f + f_g + f_gyro) * dt


@wp.kernel
def fused_batch_unconstrained_dynamics_kernel(
    # --- Batched Inputs ---
    batch_body_q: wp.array(dtype=wp.transform, ndim=3),  # [Batch, World, Body]
    batch_body_u: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
    # --- Static Inputs (Loaded ONCE) ---
    body_u_prev: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_f: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # --- Params ---
    dt: wp.float32,
    g_accel: wp.array(dtype=wp.vec3),
    num_batches: int,
    # --- Output ---
    h_d: wp.array(dtype=wp.spatial_vector, ndim=3),  # [Batch, World, Body]
):
    # Launch dim: (World, Body) - NO Batch dim here!
    world_idx, body_idx = wp.tid()

    if body_idx >= batch_body_u.shape[2]:
        return

    # 1. Load Static Data into Registers
    m = body_mass[world_idx, body_idx]
    I_body = body_inertia[world_idx, body_idx]
    u_prev = body_u_prev[world_idx, body_idx]
    f = body_f[world_idx, body_idx]

    # 2. Inner Loop over Batch Steps
    for b in range(num_batches):
        q = batch_body_q[b, world_idx, body_idx]
        u = batch_body_u[b, world_idx, body_idx]

        # Recompute World Inertia (q dependent)
        I_world = compute_world_inertia(q, I_body)

        f_g = wp.spatial_vector(m * g_accel[0], wp.vec3(0.0))

        # Gyroscopic term
        w = wp.spatial_bottom(u)
        f_gyro = wp.spatial_vector(wp.vec3(), wp.cross(w, I_world @ w))

        # momentum_diff = M @ (u - u_prev)
        momentum_diff = compute_spatial_momentum(m, I_world, u - u_prev)

        # Write Output
        h_d[b, world_idx, body_idx] = momentum_diff - (f + f_g + f_gyro) * dt

