import gc
import time

import ostrich.optim.A_matvec as kernels
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

wp.init()


def measure_graph_throughput(launch_lambda, graph_batch_size=10, measure_launches=20):
    """
    Benchmarks using CUDA Graphs to hide Python/Driver overhead.
    Records 'graph_batch_size' calls into one graph, then launches that graph 'measure_launches' times.
    """
    # 1. Warmup (compile kernels)
    launch_lambda()
    wp.synchronize()

    # 2. Capture Graph (Batch of N calls)
    with wp.ScopedCapture() as cap:
        for _ in range(graph_batch_size):
            launch_lambda()
    graph = cap.graph

    # 3. Benchmark
    wp.synchronize()
    t0 = time.time()
    for _ in range(measure_launches):
        wp.capture_launch(graph)
    wp.synchronize()
    t1 = time.time()

    total_calls = graph_batch_size * measure_launches
    return ((t1 - t0) / total_calls) * 1000.0


def run_benchmark_case(num_bodies, j_per_body, c_per_body, block_size):
    num_joints = int(num_bodies * (j_per_body / 2.0 * 0.9))
    num_contacts = num_bodies * c_per_body

    # ---------------------------
    # 1. DATA GENERATION
    # ---------------------------
    M_inv_np = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)

    # Joints
    map_j_np = kernels.generate_safe_joint_map(num_joints, num_bodies, j_per_body)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)
    J_j_flat_np, body_to_flat_np = kernels.build_joint_graph(
        num_bodies, j_per_body, J_j_np, map_j_np
    )

    J_j_flat = wp.array(J_j_flat_np, dtype=wp.spatial_vector)
    map_j = wp.array(map_j_np, dtype=wp.int32)

    # Contacts
    J_c = wp.array(np.random.randn(num_contacts, 6).astype(np.float32), dtype=wp.spatial_vector)

    # Vectors
    x_j = wp.array(np.random.randn(num_joints).astype(np.float32), dtype=wp.float32)
    x_c = wp.array(np.random.randn(num_contacts).astype(np.float32), dtype=wp.float32)

    # Compliance (dummy)
    C_j = wp.zeros_like(x_j)
    C_c = wp.zeros_like(x_c)

    # ---------------------------
    # 2. PRECOMPUTE
    # ---------------------------
    W_j_flat = wp.zeros(num_bodies * j_per_body, dtype=wp.spatial_vector)
    x_j_idx = wp.zeros(num_bodies * j_per_body, dtype=wp.int32)
    map_struct = wp.array(body_to_flat_np, dtype=wp.int32, ndim=2)

    wp.launch(
        kernels.kernel_bake_joints_flat,
        dim=num_bodies,
        inputs=[J_j_flat, map_struct, M_inv],
        outputs=[W_j_flat, x_j_idx, j_per_body],
    )

    W_c_flat = wp.zeros(num_contacts, dtype=wp.spatial_vector)
    wp.launch(
        kernels.kernel_bake_contacts_flat,
        dim=num_bodies,
        inputs=[J_c, M_inv],
        outputs=[W_c_flat, c_per_body],
    )

    # ---------------------------
    # 3. KERNEL DEFINITION
    # ---------------------------
    v_body = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    z_j = wp.zeros_like(x_j)
    z_c = wp.zeros_like(x_c)

    k_gather = kernels.create_body_gather_kernel(block_size, j_per_body, c_per_body)
    k_apply_c = kernels.create_apply_contacts_kernel(c_per_body)

    # We pass variables explicitly to closure (optional, but good practice)
    def run_optimized():
        # Phase 1: Body Gather (Fused)
        wp.launch_tiled(
            k_gather,
            dim=[num_bodies // block_size],
            inputs=[x_j, W_j_flat, x_j_idx, x_c, W_c_flat],
            outputs=[v_body],
            block_dim=block_size * max(j_per_body, c_per_body),
        )
        # Phase 2: Apply
        wp.launch(
            kernels.kernel_apply_joints,
            dim=num_joints,
            inputs=[v_body, J_j_flat, map_j, C_j, x_j, z_j],
        )
        wp.launch(
            k_apply_c,
            dim=num_contacts,
            inputs=[v_body, J_c, C_c, x_c, z_c],
        )

    def run_baseline():
        # Reset Accumulator
        v_body.zero_()
        # Phase 1: Scatter (Atomic)
        wp.launch(
            kernels.kernel_baseline_scatter_joints,
            dim=num_joints,
            inputs=[x_j, J_j_flat, map_j, M_inv, v_body],
        )
        wp.launch(
            kernels.kernel_baseline_scatter_contacts,
            dim=num_contacts,
            inputs=[x_c, J_c, M_inv, v_body, c_per_body],
        )
        # Phase 2: Apply
        wp.launch(
            kernels.kernel_apply_joints,
            dim=num_joints,
            inputs=[v_body, J_j_flat, map_j, C_j, x_j, z_j],
        )
        wp.launch(k_apply_c, dim=num_contacts, inputs=[v_body, J_c, C_c, x_c, z_c])

    # ---------------------------
    # 4. MEASURE (With CUDA Graphs)
    # ---------------------------

    # Using 50 internal iterations per graph launch to saturate the GPU
    t_base = measure_graph_throughput(run_baseline, graph_batch_size=20, measure_launches=20)
    t_opt = measure_graph_throughput(run_optimized, graph_batch_size=20, measure_launches=20)

    # Cleanup VRAM
    del M_inv, J_j_flat, J_c, x_j, x_c, W_j_flat, W_c_flat

    return t_base, t_opt


def main():
    # Settings
    j_per_body = 32
    c_per_body = 39  # Higher contact count to stress atomics
    block_size = 4

    # Sweep Configuration
    problem_sizes = [4, 16, 32, 64, 256, 512, 1024, 2048, 4096, 8196, 16384]

    print("===============================================================")
    print(f"BENCHMARKING A*x (Joints={j_per_body}/body, Contacts={c_per_body}/body)")
    print("===============================================================")
    print(f"{'Bodies':<10} | {'Baseline (ms)':<15} | {'Optimized (ms)':<15} | {'Speedup':<10}")
    print("-" * 63)

    results_base = []
    results_opt = []
    results_speedup = []

    for nb in problem_sizes:
        gc.collect()
        try:
            t_b, t_o = run_benchmark_case(nb, j_per_body, c_per_body, block_size)
            speedup = t_b / t_o

            results_base.append(t_b)
            results_opt.append(t_o)
            results_speedup.append(speedup)

            print(f"{nb:<10} | {t_b:<15.4f} | {t_o:<15.4f} | {speedup:<9.2f}x")

        except Exception as e:
            print(f"{nb:<10} | FAILED: {e}")
            results_base.append(None)
            results_opt.append(None)
            results_speedup.append(None)

    # ---------------------------
    # PLOTTING
    # ---------------------------
    valid_indices = [i for i, v in enumerate(results_base) if v is not None]
    x_vals = [problem_sizes[i] for i in valid_indices]
    y_base = [results_base[i] for i in valid_indices]
    y_opt = [results_opt[i] for i in valid_indices]
    y_speed = [results_speedup[i] for i in valid_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graph 1: Performance (Time)
    ax1.plot(x_vals, y_base, "o--", label="Baseline (Scatter)", color="red")
    ax1.plot(x_vals, y_opt, "o-", label="Optimized (Gather/Tiled)", color="green")
    ax1.set_xlabel("Number of Bodies")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("Operator Performance (Lower is Better)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend()

    # Graph 2: Speedup
    ax2.plot(x_vals, y_speed, "s-", color="blue", linewidth=2)
    ax2.set_xlabel("Number of Bodies")
    ax2.set_ylabel("Speedup Factor (x times faster)")
    ax2.set_title("Optimization Speedup")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)

    # Annotate points
    for x, y in zip(x_vals, y_speed):
        ax2.annotate(f"{y:.2f}x", xy=(x, y), xytext=(0, 5), textcoords="offset points", ha="center")

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\n[Info] Plots saved to 'benchmark_results.png'")


if __name__ == "__main__":
    main()
