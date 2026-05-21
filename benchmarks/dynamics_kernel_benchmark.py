import time

import numpy as np
import warp as wp
from axion.constraints import unconstrained_dynamics_kernel
from axion.mechanics import spatial_inertia_kernel
from axion.mechanics import SpatialInertia


def setup_data(num_bodies, device):
    """
    Generates random but physically plausible input arrays for the kernel benchmark.

    Args:
        num_bodies (int): The number of bodies to create data for.
        device (str): The Warp device ('cpu' or 'cuda') to create arrays on.
    """
    B = num_bodies

    # Generate random inertia tensors that are symmetric positive-definite
    rand_mat = np.random.rand(B, 3, 3)
    inertia_tensors = (rand_mat + rand_mat.transpose((0, 2, 1))) / 2.0  # Make symmetric
    inertia_tensors += np.expand_dims(np.identity(3) * 0.1, axis=0)  # Ensure positive-definite

    mass = np.random.rand(B).astype(np.float32) + 1.0

    # Create temporary Warp arrays for raw mass and inertia
    body_mass_wp = wp.from_numpy(mass, dtype=wp.float32, device=device)
    body_inertia_wp = wp.from_numpy(
        inertia_tensors.astype(np.float32), dtype=wp.mat33, device=device
    )

    # Create the output array for the structured generalized mass
    gen_mass_wp = wp.zeros(B, dtype=SpatialInertia, device=device)

    # Launch kernel to populate the generalized mass array from the raw components
    wp.launch(
        kernel=spatial_inertia_kernel,
        dim=B,
        inputs=[body_mass_wp, body_inertia_wp, gen_mass_wp],
        device=device,
    )

    data = {
        "body_qd": wp.from_numpy(
            (np.random.rand(B, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_qd_prev": wp.from_numpy(
            (np.random.rand(B, 6) - 0.5).astype(np.float32),
            dtype=wp.spatial_vector,
            device=device,
        ),
        "body_f": wp.from_numpy(
            np.zeros((B, 6), dtype=np.float32), dtype=wp.spatial_vector, device=device
        ),
        "gen_mass": gen_mass_wp,  # Use the new structured array
        "dt": 1.0 / 60.0,
        "g_accel": wp.vec3(0.0, -9.8, 0.0),
        "g": wp.zeros(B, dtype=wp.spatial_vector, device=device),
    }
    return data


def run_benchmark(num_bodies, num_iterations=200):
    """Measures execution time of the kernel using standard launch vs. CUDA graph."""
    print(
        f"\n--- Benchmarking Unconstrained Dynamics Kernel: N_b={num_bodies}, Iterations={num_iterations} ---"
    )
    device = wp.get_device()
    data = setup_data(num_bodies, device)

    # Assemble the list of arguments in the exact order the kernel expects
    kernel_args = [
        data["body_qd"],
        data["body_qd_prev"],
        data["body_f"],
        data["gen_mass"],  # Pass the single generalized mass array
        data["dt"],
        data["g_accel"],
        data["g"],
    ]

    # --- Standard Launch Benchmark ---
    print("1. Benching Standard Kernel Launch...")
    # Warm-up launch to compile the kernel
    wp.launch(
        kernel=unconstrained_dynamics_kernel,
        dim=num_bodies,
        inputs=kernel_args,
        device=device,
    )
    wp.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        data["g"].zero_()
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=kernel_args,
            device=device,
        )
    wp.synchronize()
    standard_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000
    print(f"   Avg. Time: {standard_launch_ms:.3f} ms")

    # --- CUDA Graph Benchmark (only on GPU) ---
    # CUDA graphs significantly reduce CPU launch overhead for repeated calls.
    if device.is_cuda:
        print("2. Benching CUDA Graph Launch...")
        wp.capture_begin()
        data["g"].zero_()
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=num_bodies,
            inputs=kernel_args,
            device=device,
        )
        graph = wp.capture_end()

        # Warm-up launch
        wp.capture_launch(graph)
        wp.synchronize()

        start_time = time.perf_counter()
        for _ in range(num_iterations):
            wp.capture_launch(graph)
        wp.synchronize()
        graph_launch_ms = (time.perf_counter() - start_time) / num_iterations * 1000

        print(f"   Avg. Time: {graph_launch_ms:.3f} ms")
        if graph_launch_ms > 0:
            speedup = standard_launch_ms / graph_launch_ms
            print(f"   Speedup from Graph: {speedup:.2f}x")
    else:
        print("2. Skipping CUDA Graph benchmark (not on a CUDA-enabled device).")

    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    wp.init()

    device_name = wp.get_device().name
    print(f"Initialized Warp on device: {device_name}")
    if not wp.get_device().is_cuda:
        print(
            "\nWarning: No CUDA device found. Performance will be limited and CUDA Graph test will be skipped."
        )

    # Benchmark with a range of body counts to observe scaling
    run_benchmark(num_bodies=100)
    run_benchmark(num_bodies=500)
    run_benchmark(num_bodies=1000)
    run_benchmark(num_bodies=4000)
