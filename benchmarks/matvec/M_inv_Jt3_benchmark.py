import gc
import time

import ostrich.optim.M_inv_Jt3 as kernels  # Imports the new file
import numpy as np
import warp as wp

wp.init()


def measure_graph_throughput(launch_lambda, graph_batch_size=10, measure_launches=20):
    launch_lambda()
    wp.synchronize()

    with wp.ScopedCapture() as cap:
        for _ in range(graph_batch_size):
            launch_lambda()
    graph = cap.graph

    wp.synchronize()
    t0 = time.time()
    for _ in range(measure_launches):
        wp.capture_launch(graph)
    wp.synchronize()
    t1 = time.time()

    total_calls = graph_batch_size * measure_launches
    return ((t1 - t0) / total_calls) * 1000.0


def run_benchmark_case(num_bodies, j_per_body, c_per_body, bodies_in_tile):
    num_joints = int(num_bodies * (j_per_body / 2.0 * 0.9))
    num_contacts = num_bodies * c_per_body

    # ---------------------------
    # A. DATA PREP
    # ---------------------------
    try:
        map_j_np = kernels.generate_safe_joint_map(num_joints, num_bodies, j_per_body)
    except Exception as e:
        print(f"Gen failed: {e}")
        return None

    M_inv_np = np.random.rand(num_bodies, 6, 6).astype(np.float32)
    for i in range(num_bodies):
        M_inv_np[i] = M_inv_np[i] @ M_inv_np[i].T

    lam_j_np = np.random.randn(num_joints).astype(np.float32)
    J_j_np = np.random.randn(num_joints, 2, 6).astype(np.float32)

    lam_c_np = np.random.randn(num_contacts).astype(np.float32)
    J_c_np = np.random.randn(num_contacts, 6).astype(np.float32)
    map_c_np = np.repeat(np.arange(num_bodies, dtype=np.int32), c_per_body)

    # Unified
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

    # Baseline Inputs
    lam_u = wp.array(lam_all, dtype=wp.float32)
    J_u = wp.array(J_all, dtype=wp.spatial_vector)
    map_u = wp.array(map_all, dtype=wp.int32)

    lam_j = wp.array(lam_j_np, dtype=wp.float32)
    J_j = wp.array(J_j_np, dtype=wp.spatial_vector)
    map_j = wp.array(map_j_np, dtype=wp.int32)
    map_j_flat_dev = wp.array(map_j_np, dtype=wp.int32)  # For Naive Precompute

    lam_c = wp.array(lam_c_np, dtype=wp.float32)
    J_c = wp.array(J_c_np, dtype=wp.spatial_vector)
    map_c = wp.array(map_c_np, dtype=wp.int32)

    # Tiled Graph Construction
    J_j_flat_np, body_to_flat_np = kernels.build_joint_graph(
        num_bodies, j_per_body, J_j_np, map_j_np
    )

    lam_j_pad_np = np.zeros(num_joints + 1, dtype=np.float32)
    lam_j_pad_np[1:] = lam_j_np
    lam_j_tiled = wp.array(lam_j_pad_np, dtype=wp.float32)

    J_j_flat = wp.array(J_j_flat_np, dtype=wp.spatial_vector)

    # Map Variants
    map_naive = wp.array(body_to_flat_np.flatten(), dtype=wp.int32)
    map_opt_struct = wp.array(body_to_flat_np, dtype=wp.int32, ndim=2)

    # ---------------------------
    # C. PRECOMPUTE
    # ---------------------------

    # 1. Naive Precompute (W_flat)
    W_j_naive = wp.zeros_like(J_j_flat)
    wp.launch(
        kernels.kernel_precompute_joint_W_naive,
        dim=len(J_j_flat),
        inputs=[J_j_flat, map_j_flat_dev, M_inv],
        outputs=[W_j_naive],
    )

    # 2. Optimized Precompute (W_binned_flat - 1D)
    flat_size = num_bodies * j_per_body
    W_j_opt = wp.zeros(flat_size, dtype=wp.spatial_vector)
    lam_idx_opt = wp.zeros(flat_size, dtype=wp.int32)

    wp.launch(
        kernels.kernel_bake_joint_data_flat,
        dim=num_bodies,
        inputs=[J_j_flat, map_opt_struct, M_inv],
        outputs=[W_j_opt, lam_idx_opt, j_per_body],
    )

    # 3. Contacts Precompute
    W_c = wp.zeros_like(J_c)
    wp.launch(
        kernels.kernel_precompute_contact_W,
        dim=num_contacts,
        inputs=[J_c, map_c, M_inv],
        outputs=[W_c],
    )
    wp.synchronize()

    # ---------------------------
    # D. KERNEL CREATION
    # ---------------------------
    k_naive_j = kernels.create_naive_tiled_joint_solver(bodies_in_tile, j_per_body)
    k_opt_j = kernels.create_optimized_tiled_joint_solver(bodies_in_tile, j_per_body)
    k_cont = kernels.create_tiled_contact_solver(bodies_in_tile, c_per_body)

    stream_j = wp.Stream()
    stream_c = wp.Stream()
    event_j = wp.Event()

    # ---------------------------
    # E. RUN BENCHMARKS
    # ---------------------------

    # 1. UNIFIED
    def launch_unified():
        wp.launch(
            kernels.kernel_unified_scatter,
            dim=len(lam_all),
            inputs=[lam_u, J_u, map_u, M_inv],
            outputs=[du_out],
        )

    t_unified = measure_graph_throughput(launch_unified)

    # 2. SEPARATED
    def launch_separated():
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

    t_separated = measure_graph_throughput(launch_separated)

    # 3. NAIVE TILED
    def launch_naive_tiled():
        with wp.ScopedStream(stream_j):
            wp.launch_tiled(
                k_naive_j,
                dim=[num_bodies // bodies_in_tile],
                inputs=[lam_j_tiled, W_j_naive, map_naive],
                outputs=[du_out_j],
                block_dim=bodies_in_tile * j_per_body,
            )
            wp.record_event(event_j)
        with wp.ScopedStream(stream_c):
            wp.launch_tiled(
                k_cont,
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

    t_naive = measure_graph_throughput(launch_naive_tiled)

    # 4. OPTIMIZED TILED
    def launch_opt_tiled():
        with wp.ScopedStream(stream_j):
            # Key difference: Inputs are W_j_opt (Flat 1D) and lam_idx_opt (Flat 1D)
            wp.launch_tiled(
                k_opt_j,
                dim=[num_bodies // bodies_in_tile],
                inputs=[lam_j_tiled, W_j_opt, lam_idx_opt],
                outputs=[du_out_j],
                block_dim=bodies_in_tile * j_per_body,
            )
            wp.record_event(event_j)
        with wp.ScopedStream(stream_c):
            wp.launch_tiled(
                k_cont,
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

    t_opt = measure_graph_throughput(launch_opt_tiled)

    return t_unified, t_separated, t_naive, t_opt


def main():
    j_per_body = 16
    c_per_body = 16
    bodies_in_tile = 8

    # 8k and 16k are the interesting ones for throughput
    problem_sizes = [512, 1024, 2048, 4096, 8192, 16384]

    print(f"\nBenchmark Results (ms per iteration):")
    print(
        f"{'BODIES':<10} | {'UNIFIED':<12} | {'SEPARATED':<12} | {'NAIVE TILE':<12} | {'OPT TILE':<12} | {'SPEEDUP':<10}"
    )
    print("-" * 80)

    for nb in problem_sizes:
        gc.collect()
        try:
            res = run_benchmark_case(nb, j_per_body, c_per_body, bodies_in_tile)
            if res:
                t_u, t_s, t_n, t_o = res
                speedup = t_u / t_o
                print(
                    f"{nb:<10} | {t_u:<12.4f} | {t_s:<12.4f} | {t_n:<12.4f} | {t_o:<12.4f} | {speedup:<10.2f}x"
                )
        except Exception as e:
            print(f"{nb:<10} | FAILED: {e}")


if __name__ == "__main__":
    main()
