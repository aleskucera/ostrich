import time

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from ostrich.optim.M_inv_Jt import create_M_inv_Jct_matvec_tiled
from ostrich.optim.M_inv_Jt import kernel_M_inv_Jct_matvec_scatter
from ostrich.optim.M_inv_Jt import kernel_M_inv_Jjt_matvec_scatter
from ostrich.optim.M_inv_Jt import kernel_M_inv_Jt_matvec_scatter

# Imports from your specific module structure

wp.init()

# =================================================================================
# 1. DATA PREPARATION
# =================================================================================


def build_benchmark_data(num_bodies, j_per_body, c_per_body, bodies_in_tile):
    num_joints = num_bodies * j_per_body
    num_contacts = num_bodies * c_per_body

    # --- Generate CPU Data ---
    M_inv_np = np.random.randn(num_bodies, 6, 6).astype(np.float32)

    # Joints
    dlambda_j_np = np.random.randn(num_joints).astype(np.float32)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)
    joint_map_np = np.random.randint(0, num_bodies, size=(num_joints, 2)).astype(np.int32)

    # Contacts (Sorted for Tiling: [0,0,0, 1,1,1...])
    dlambda_c_np = np.random.randn(num_contacts).astype(np.float32)
    J_c_np = np.random.randn(num_contacts, 6).astype(np.float32)
    contact_map_np = np.repeat(np.arange(num_bodies, dtype=np.int32), c_per_body)

    # --- Prepare "Unified" Data (Padding) ---
    dlambda_all_np = np.concatenate([dlambda_j_np, dlambda_c_np])

    # Pad Contact Map: [BodyID, -1]
    contact_map_padded = np.full((num_contacts, 2), -1, dtype=np.int32)
    contact_map_padded[:, 0] = contact_map_np
    map_all_np = np.concatenate([joint_map_np, contact_map_padded], axis=0)

    # Pad Contact Jacobian: [J, Zero]
    J_c_padded = np.zeros((num_contacts, 2, 6), dtype=np.float32)
    J_c_padded[:, 0, :] = J_c_np
    J_all_np = np.concatenate([J_j_np, J_c_padded], axis=0)

    # --- Upload to Warp ---
    M_inv = wp.array(M_inv_np, dtype=wp.spatial_matrix)

    # Split Data
    dlambda_j = wp.array(dlambda_j_np, dtype=wp.float32)
    J_j = wp.array(J_j_np, dtype=wp.spatial_vector)
    map_j = wp.array(joint_map_np, dtype=wp.int32)

    dlambda_c = wp.array(dlambda_c_np, dtype=wp.float32)
    J_c = wp.array(J_c_np, dtype=wp.spatial_vector)
    map_c = wp.array(contact_map_np, dtype=wp.int32)

    # Unified Data
    dlambda_all = wp.array(dlambda_all_np, dtype=wp.float32)
    J_all = wp.array(J_all_np, dtype=wp.spatial_vector)
    map_all = wp.array(map_all_np, dtype=wp.int32)

    # Output Buffer
    du = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_j = wp.zeros(num_bodies, dtype=wp.spatial_vector)
    du_c = wp.zeros(num_bodies, dtype=wp.spatial_vector)

    return {
        "M_inv": M_inv,
        "split": {
            "j": {"du": du_j, "dlambda": dlambda_j, "J": J_j, "map": map_j, "dim": num_joints},
            "c": {"du": du_c, "dlambda": dlambda_c, "J": J_c, "map": map_c, "dim": num_contacts},
            "tiled_dim": num_contacts // (bodies_in_tile * c_per_body),
        },
        "unified": {
            "du": du,
            "dlambda": dlambda_all,
            "J": J_all,
            "map": map_all,
            "dim": num_joints + num_contacts,
        },
    }


# =================================================================================
# 2. BENCHMARK RUNNER (With Inner Loop Capture)
# =================================================================================


def measure_performance(setup_closure, num_graph_iters=100, ops_per_graph=10):
    """
    Args:
        setup_closure: A function that takes no args and builds the graph body
                       (including the loop for ops_per_graph).
        num_graph_iters: How many times to replay the captured graph.
        ops_per_graph: Handled inside setup_closure, but passed here for calculation.
    """

    # 1. Warmup (Run once immediately)
    setup_closure()
    wp.synchronize()

    # 2. Capture
    with wp.ScopedCapture() as cap:
        setup_closure()

    graph = cap.graph

    # 3. Timing
    t0 = time.perf_counter()
    for _ in range(num_graph_iters):
        wp.capture_launch(graph)
    wp.synchronize()
    t1 = time.perf_counter()

    total_ops = num_graph_iters * ops_per_graph
    return (t1 - t0) / total_ops


# =================================================================================
# 3. KERNEL SPECIFIC BENCHMARKS
# =================================================================================


def run_benchmark_unified(data, ops_per_graph=10):
    d = data["unified"]
    M_inv = data["M_inv"]

    def graph_body():
        for _ in range(ops_per_graph):
            d["du"].zero_()  # CRITICAL: Reset for atomic add
            wp.launch(
                kernel_M_inv_Jt_matvec_scatter,
                dim=d["dim"],
                inputs=[d["dlambda"], d["J"], d["map"], M_inv],
                outputs=[d["du"]],
            )

    return measure_performance(graph_body, ops_per_graph=ops_per_graph)


def run_benchmark_scatter(data, stream_j, stream_c, ops_per_graph=10):
    d = data["split"]
    M_inv = data["M_inv"]

    def graph_body():
        for _ in range(ops_per_graph):
            # Using streams inside capture records the concurrency
            with wp.ScopedStream(stream_j):
                d["j"]["du"].zero_()
                wp.launch(
                    kernel_M_inv_Jjt_matvec_scatter,
                    dim=d["j"]["dim"],
                    inputs=[d["j"]["dlambda"], d["j"]["J"], d["j"]["map"], M_inv],
                    outputs=[d["j"]["du"]],
                )

            with wp.ScopedStream(stream_c):
                d["c"]["du"].zero_()
                wp.launch(
                    kernel_M_inv_Jct_matvec_scatter,
                    dim=d["c"]["dim"],
                    inputs=[d["c"]["dlambda"], d["c"]["J"], d["c"]["map"], M_inv],
                    outputs=[d["c"]["du"]],
                )

    return measure_performance(graph_body, ops_per_graph=ops_per_graph)


def run_benchmark_tiled(data, kernel_tiled, tile_block_dim, stream_j, stream_c, ops_per_graph=10):
    d = data["split"]
    M_inv = data["M_inv"]

    def graph_body():
        for _ in range(ops_per_graph):
            with wp.ScopedStream(stream_j):
                d["j"]["du"].zero_()
                wp.launch(
                    kernel_M_inv_Jjt_matvec_scatter,
                    dim=d["j"]["dim"],
                    inputs=[d["j"]["dlambda"], d["j"]["J"], d["j"]["map"], M_inv],
                    outputs=[d["j"]["du"]],
                )

            with wp.ScopedStream(stream_c):
                wp.launch_tiled(
                    kernel_tiled,
                    dim=d["tiled_dim"],
                    inputs=[d["c"]["dlambda"], d["c"]["J"], M_inv],
                    outputs=[d["c"]["du"]],
                    block_dim=tile_block_dim,
                )

            # d["j"]["du"].zero_()
            # wp.launch(
            #     kernel_M_inv_Jjt_matvec_scatter,
            #     dim=d["j"]["dim"],
            #     inputs=[d["j"]["dlambda"], d["j"]["J"], d["j"]["map"], M_inv],
            #     outputs=[d["j"]["du"]],
            # )
            #
            # wp.launch_tiled(
            #     kernel_tiled,
            #     dim=d["tiled_dim"],
            #     inputs=[d["c"]["dlambda"], d["c"]["J"], M_inv],
            #     outputs=[d["c"]["du"]],
            #     block_dim=tile_block_dim,
            # )

    return measure_performance(graph_body, ops_per_graph=ops_per_graph)


# =================================================================================
# 4. MAIN SUITE
# =================================================================================


def run_suite():
    # Benchmark Config
    bodies_in_tile = 8
    j_per_body = 10
    c_per_body = 32 * 3
    ops_per_graph = 10  # Number of kernel launches per graph submission

    # Sweep Config
    body_counts = [128, 512, 1024, 2048, 4096, 8192, 16384]  # 32768]

    # Kernel Creation (Only need to create the tiled kernel once)
    # Note: create_M_inv_Jct_matvec_tiled logic depends on how you implemented the factory.
    # Assuming factory signature: (num_bodies, constraints_per_body, bodies_in_tile)
    # Since num_bodies changes, we might need to recreate it or just create a generic one
    # if the kernel code doesn't hardcode num_bodies.
    # Usually Warp kernels are JIT compiled based on types, so creating it once is fine
    # IF the factory doesn't bake in `num_bodies`.
    # Based on your previous snippet: `num_bodies` was an arg to the factory but mostly for assertion.
    # We will regenerate it per loop to be safe or use a large enough dummy if strict checking isn't used.
    # Let's regenerate inside the loop to match your factory signature strictly.

    stream_j = wp.Stream()
    stream_c = wp.Stream()

    results = {"unified": [], "scatter": [], "tiled": []}

    print(f"{'Bodies':<10} | {'Unified (ms)':<15} | {'Scatter (ms)':<15} | {'Tiled (ms)':<15}")
    print("-" * 65)

    for n_bodies in body_counts:
        # 1. Build Data
        data = build_benchmark_data(n_bodies, j_per_body, c_per_body, bodies_in_tile)

        # 2. Get Tiled Kernel Instance for this size
        kernel_tiled_instance = create_M_inv_Jct_matvec_tiled(
            num_bodies=n_bodies, constraints_per_body=c_per_body, bodies_in_tile=bodies_in_tile
        )
        tile_block_dim = bodies_in_tile * c_per_body

        # 3. Run Benchmarks
        t_uni = run_benchmark_unified(data, ops_per_graph) * 1000.0
        time.sleep(0.1)
        t_sca = run_benchmark_scatter(data, stream_j, stream_c, ops_per_graph) * 1000.0
        time.sleep(0.1)
        t_til = (
            run_benchmark_tiled(
                data, kernel_tiled_instance, tile_block_dim, stream_j, stream_c, ops_per_graph
            )
            * 1000.0
        )

        results["unified"].append(t_uni)
        results["scatter"].append(t_sca)
        results["tiled"].append(t_til)

        print(f"{n_bodies:<10} | {t_uni:<15.4f} | {t_sca:<15.4f} | {t_til:<15.4f}")

    return body_counts, results


if __name__ == "__main__":
    counts, res = run_suite()

    plt.figure(figsize=(12, 6))
    plt.plot(counts, res["unified"], "r-o", label="Unified (Atomic)")
    plt.plot(counts, res["scatter"], "b--s", label="Split Scatter (Atomic)")
    plt.plot(counts, res["tiled"], "g-.^", label="Split Tiled (Store)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Bodies")
    plt.ylabel("Time (ms)")
    plt.title("M_inv * J^T performance: Unified vs Split vs Tiled")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    output_file = "benchmark_results_graph.png"
    plt.savefig(output_file)
    print(f"\nBenchmark complete. Results saved to {output_file}")
    plt.show()
