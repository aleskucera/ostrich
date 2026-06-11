import time

import numpy as np
import warp as wp
from ostrich.sparse.J_tiled_v import CONSTRAINTS_PER_BODY
from ostrich.sparse.J_tiled_v import kernel_J_matvec_scatter
from ostrich.sparse.J_tiled_v import kernel_J_matvec_tiled
from ostrich.sparse.J_tiled_v import NUM_BODIES
from ostrich.sparse.J_tiled_v import NUM_CONSTRAINTS
from ostrich.sparse.J_tiled_v import TILE_SIZE

wp.init()


# ----------------------------------------------------------
# Build test data (shared for both kernels)
# ----------------------------------------------------------
def build_data():
    # Host data
    x_host = np.random.randn(NUM_BODIES * 6).astype(np.float32)
    J_values_host = np.random.randn(NUM_CONSTRAINTS * 6).astype(np.float32)

    # For tiled kernel: flat float arrays
    x_flat = wp.array(x_host, dtype=wp.float32)  # (NUM_BODIES * 6)
    J_flat = wp.array(J_values_host, dtype=wp.float32)  # (NUM_CONSTRAINTS * 6)

    # For scatter kernel: spatial vectors
    x_v = wp.array(x_host.reshape(NUM_BODIES, 6), dtype=wp.spatial_vector)  # (NUM_BODIES)
    J_v = wp.array(
        J_values_host.reshape(NUM_CONSTRAINTS, 6), dtype=wp.spatial_vector
    )  # (NUM_CONSTRAINTS)

    # Body index per constraint: [0,0,...,1,1,...]
    constraint_body_idx = wp.array(
        np.repeat(np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY),
        dtype=wp.int32,
    )

    out_scatter = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)
    out_tiled = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)

    return x_v, J_v, constraint_body_idx, out_scatter, x_flat, J_flat, out_tiled


# ----------------------------------------------------------
# Standard timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time(num_repeats=50):
    x_v, J_v, constraint_body_idx, out_scatter, _, _, _ = build_data()

    # warmup
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x_v, J_v, constraint_body_idx],
        outputs=[out_scatter],
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        wp.launch(
            kernel_J_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[x_v, J_v, constraint_body_idx],
            outputs=[out_scatter],
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# Standard timing: tiled kernel
# ----------------------------------------------------------
def measure_tiled_time(num_repeats=50):
    _, _, _, _, x_flat, J_flat, out_tiled = build_data()

    # warmup
    wp.launch_tiled(
        kernel=kernel_J_matvec_tiled,
        dim=(NUM_BODIES,),
        inputs=[x_flat, J_flat],
        outputs=[out_tiled],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        wp.launch_tiled(
            kernel=kernel_J_matvec_tiled,
            dim=(NUM_BODIES,),
            inputs=[x_flat, J_flat],
            outputs=[out_tiled],
            block_dim=TILE_SIZE,
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA graph timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    x_v, J_v, constraint_body_idx, out_scatter, _, _, _ = build_data()

    # warmup
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x_v, J_v, constraint_body_idx],
        outputs=[out_scatter],
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch(
                kernel_J_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[x_v, J_v, constraint_body_idx],
                outputs=[out_scatter],
            )
    wp.synchronize()

    # repeated graph launches
    t0 = time.time()
    for _ in range(num_graph_iters):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / (num_graph_iters * num_ops_in_graph)


# ----------------------------------------------------------
# CUDA graph timing: tiled kernel
# ----------------------------------------------------------
def measure_tiled_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    _, _, _, _, x_flat, J_flat, out_tiled = build_data()

    # warmup
    wp.launch_tiled(
        kernel=kernel_J_matvec_tiled,
        dim=(NUM_BODIES,),
        inputs=[x_flat, J_flat],
        outputs=[out_tiled],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch_tiled(
                kernel=kernel_J_matvec_tiled,
                dim=(NUM_BODIES,),
                inputs=[x_flat, J_flat],
                outputs=[out_tiled],
                block_dim=TILE_SIZE,
            )
    wp.synchronize()

    # repeated graph launches
    t0 = time.time()
    for _ in range(num_graph_iters):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / (num_graph_iters * num_ops_in_graph)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("\n=== Benchmark: Scatter vs Tiled kernels (J_tiled_v) ===\n")
    print(f"NUM_BODIES={NUM_BODIES}")
    print(f"CONSTRAINTS_PER_BODY={CONSTRAINTS_PER_BODY}")
    print(f"NUM_CONSTRAINTS={NUM_CONSTRAINTS}")
    print(f"TILE_SIZE={TILE_SIZE}\n")

    # ---------------- Standard ----------------
    print("Measuring standard scatter...")
    t_s_std = measure_scatter_time()
    print(f"Scatter (standard): {t_s_std * 1e3:.4f} ms")

    print("Measuring standard tiled...")
    t_t_std = measure_tiled_time()
    print(f"Tiled (standard):   {t_t_std * 1e3:.4f} ms\n")

    # --------------- CUDA graphs --------------
    print("Measuring CUDA graph scatter...")
    t_s_graph = measure_scatter_time_cuda_graph()
    print(f"Scatter (CUDA graph): {t_s_graph * 1e3:.4f} ms")

    print("Measuring CUDA graph tiled...")
    t_t_graph = measure_tiled_time_cuda_graph()
    print(f"Tiled (CUDA graph):   {t_t_graph * 1e3:.4f} ms\n")


if __name__ == "__main__":
    main()
