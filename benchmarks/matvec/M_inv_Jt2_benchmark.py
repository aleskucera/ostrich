import gc
import time

import ostrich.optim.M_inv_Jt2 as kernels
import numpy as np
import warp as wp

# Import the kernels and helpers from your file

wp.init()

# =================================================================================
# 1. GRAPH MEASUREMENT UTILITY
# =================================================================================


def measure_graph_throughput(launch_lambda, graph_batch_size=10, measure_launches=20):
    """
    Benchmarks a specific kernel launch function using CUDA Graphs.

    Args:
        launch_lambda: A function () -> None that enqueues the kernels.
        graph_batch_size: How many times to repeat the kernel INSIDE the graph.
                          High numbers amortize launch cost (like a solver loop).
        measure_launches: How many times to execute the full graph.
    """

    # 1. Warmup (Compile kernels)
    launch_lambda()
    wp.synchronize()

    # 2. Capture Graph
    # We unroll the loop inside the graph to simulate a solver iteration loop
    # and reduce the graph launch overhead per kernel execution.
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

    # Calculate average time per SINGLE 'launch_lambda' execution
    total_calls = graph_batch_size * measure_launches
    avg_ms = ((t1 - t0) / total_calls) * 1000.0

    return avg_ms


# =================================================================================
# 2. BENCHMARK SUITE
# =================================================================================


def run_benchmark_case(num_bodies, j_per_body, c_per_body, bodies_in_tile):

    num_joints = int(num_bodies * (j_per_body / 2.0 * 0.9))  # e.g. 4 joints/body avg
    num_contacts = num_bodies * c_per_body  # Max contacts

    # ---------------------------
    # A. DATA PREP
    # ---------------------------

    # Generate Topology
    # Use the safe generator from the imported module to ensure tiling compatibility
    try:
        map_j_np = kernels.generate_safe_joint_map(num_joints, num_bodies, j_per_body)
    except Exception as e:
        print(f"Graph Gen failed: {e}")
        return None

    # Physics Data
    M_inv_np = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T

    lam_j_np = np.random.randn(num_joints).astype(np.float32)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)

    lam_c_np = np.random.randn(num_contacts).astype(np.float32)
    J_c_np = np.random.randn(num_contacts, 6).astype(np.float32)
    map_c_np = np.repeat(np.arange(num_bodies, dtype=np.int32), c_per_body)

    # Unified Data Structures
    lam_all = np.concatenate([lam_j_np, lam_c_np])

    J_c_padded = np.zeros((num_contacts, 2, 6), dtype=np.float32)
    J_c_padded[:, 0, :] = J_c_np
    J_all = np.concatenate([J_j_np, J_c_padded])

    map_c_padded = np.full((num_contacts, 2), -1, dtype=np.int32)
    map_c_padded[:, 0] = map_c_np
    map_all = np.concatenate([map_j_np, map_c_padded])

    # ---------------------------
    # B. GPU ALLOCATION
    # ---------------------------
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    du_out = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_out_j = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_out_c = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    # 1. Unified Inputs
    lam_u = wp.array(lam_all, dtype=wp.float32)
    J_u = wp.array(J_all, dtype=wp.spatial_vector)
    map_u = wp.array(map_all, dtype=wp.int32)

    # 2. Separated Inputs
    lam_j = wp.array(lam_j_np, dtype=wp.float32)
    J_j = wp.array(J_j_np, dtype=wp.spatial_vector)
    map_j = wp.array(map_j_np, dtype=wp.int32)

    lam_c = wp.array(lam_c_np, dtype=wp.float32)
    J_c = wp.array(J_c_np, dtype=wp.spatial_vector)
    map_c = wp.array(map_c_np, dtype=wp.int32)

    # 3. Tiled Inputs (Precompute)
    # Joints
    J_j_flat_np, map_j_flat_np = kernels.build_joint_graph(num_bodies, j_per_body, J_j_np, map_j_np)
    lam_j_pad_np = np.zeros(num_joints + 1, dtype=np.float32)
    lam_j_pad_np[1:] = lam_j_np

    lam_j_tiled = wp.array(lam_j_pad_np, dtype=wp.float32)
    J_j_flat = wp.array(J_j_flat_np, dtype=wp.spatial_vector)
    map_j_flat = wp.array(map_j_flat_np, dtype=wp.int32)
    W_j_flat = wp.zeros_like(J_j_flat)

    # Contacts
    W_c = wp.zeros_like(J_c)

    # Run Precompute Once
    wp.launch(
        kernels.kernel_precompute_joint_W,
        dim=len(J_j_flat),
        inputs=[J_j_flat, map_j, M_inv],
        outputs=[W_j_flat],
    )
    wp.launch(
        kernels.kernel_precompute_contact_W,
        dim=num_contacts,
        inputs=[J_c, map_c, M_inv],
        outputs=[W_c],
    )
    wp.synchronize()

    # Create Tiled Kernel Instances
    k_joint_tiled = kernels.create_tiled_joint_solver(bodies_in_tile, j_per_body)
    k_contact_tiled = kernels.create_tiled_contact_solver(bodies_in_tile, c_per_body)

    stream_j = wp.Stream()
    stream_c = wp.Stream()

    event_j = wp.Event()

    # ---------------------------
    # C. RUN TESTS
    # ---------------------------

    # 1. UNIFIED
    def launch_unified():
        # Note: We optionally zero output to simulate fresh step,
        # but for pure throughput check we can skip it or include it.
        # du_out.zero_()
        wp.launch(
            kernels.kernel_unified_scatter,
            dim=len(lam_all),
            inputs=[lam_u, J_u, map_u, M_inv],
            outputs=[du_out],
        )

    t_unified = measure_graph_throughput(launch_unified, graph_batch_size=10)

    # 2. SEPARATED
    def launch_separated():
        du_out_j.zero_()
        du_out_c.zero_()
        wp.launch(
            kernels.kernel_scatter_joints,
            dim=num_joints,
            inputs=[lam_j, J_j, map_j, M_inv],
            outputs=[du_out_j],
        )
        wp.launch(
            kernels.kernel_scatter_contacts,
            dim=num_contacts,
            inputs=[lam_c, J_c, map_c, M_inv],
            outputs=[du_out_c],
        )
        wp.launch(
            kernels.kernel_sum_vectors,
            dim=num_bodies,
            inputs=[du_out_j, du_out_c],
            outputs=[du_out],
        )

    t_separated = measure_graph_throughput(launch_separated, graph_batch_size=10)

    # 3. TILED
    def launch_tiled():
        with wp.ScopedStream(stream_j):
            wp.launch_tiled(
                k_joint_tiled,
                dim=[num_bodies // bodies_in_tile],
                inputs=[lam_j_tiled, W_j_flat, map_j_flat],
                outputs=[du_out_j],
                block_dim=bodies_in_tile * j_per_body,
            )
            wp.record_event(event_j)
        with wp.ScopedStream(stream_c):
            wp.launch_tiled(
                k_contact_tiled,
                dim=[num_bodies // bodies_in_tile],
                inputs=[lam_c, W_c],
                outputs=[du_out_c],
                block_dim=bodies_in_tile * c_per_body,
            )
            wp.wait_event(event_j)
            wp.launch(
                kernels.kernel_sum_vectors,
                dim=num_bodies,
                inputs=[du_out_j, du_out_c],
                outputs=[du_out],
            )

    t_tiled = measure_graph_throughput(launch_tiled, graph_batch_size=10)

    return t_unified, t_separated, t_tiled


# =================================================================================
# 3. MAIN
# =================================================================================


def main():
    # Configuration
    j_per_body = 16
    c_per_body = 16
    bodies_in_tile = 8

    # Scale up to test bandwidth saturation
    problem_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    print(f"\nBenchmark: 10 solver iterations inside Graph")
    print(
        f"{'BODIES':<10} | {'UNIFIED (ms)':<15} | {'SEPARATED (ms)':<15} | {'TILED (ms)':<15} | {'SPEEDUP':<10}"
    )
    print("-" * 85)

    for nb in problem_sizes:
        gc.collect()
        try:
            res = run_benchmark_case(nb, j_per_body, c_per_body, bodies_in_tile)
            if res:
                t_u, t_s, t_t = res
                speedup = t_u / t_t
                print(f"{nb:<10} | {t_u:<15.4f} | {t_s:<15.4f} | {t_t:<15.4f} | {speedup:<10.2f}x")
            else:
                print(f"{nb:<10} | Skipped")

        except Exception as e:
            print(f"{nb:<10} | FAILED: {e}")


if __name__ == "__main__":
    main()
