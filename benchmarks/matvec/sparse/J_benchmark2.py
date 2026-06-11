import time

import numpy as np
import warp as wp
from ostrich.sparse.J_tiled import CONSTRAINTS_PER_BODY
from ostrich.sparse.J_tiled import kernel_J_matvec_scatter
from ostrich.sparse.J_tiled import kernel_J_matvec_tiled
from ostrich.sparse.J_tiled import NUM_BODIES
from ostrich.sparse.J_tiled import NUM_CONSTRAINTS
from ostrich.sparse.J_tiled import TILE_SIZE  # <-- NEW: import TILE_SIZE

wp.init()

# Number of tiles for the new tiled kernel
TILED_LAUNCH_DIM = NUM_CONSTRAINTS // TILE_SIZE


# ----------------------------------------------------------
# Build test data
# ----------------------------------------------------------
def build_data():
    x = wp.array(np.random.randn(NUM_BODIES).astype(np.float32), dtype=wp.float32)

    J_values = wp.array(np.random.randn(NUM_CONSTRAINTS).astype(np.float32), dtype=wp.float32)

    constraint_body_idx = wp.array(
        np.repeat(np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY),
        dtype=wp.int32,
    )

    out = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)
    return x, J_values, constraint_body_idx, out


# ----------------------------------------------------------
# Standard timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time(num_repeats=50):
    x, J_values, constraint_body_idx, out = build_data()

    # warmup
    wp.launch(
        kernel_J_matvec_scatter, dim=NUM_CONSTRAINTS, inputs=[x, J_values, constraint_body_idx, out]
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        wp.launch(
            kernel_J_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[x, J_values, constraint_body_idx, out],
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# Standard timing: tiled kernel
# ----------------------------------------------------------
def measure_tiled_time(num_repeats=50):
    x, J_values, constraint_body_idx, out = build_data()

    # warmup
    wp.launch_tiled(
        kernel_J_matvec_tiled,
        dim=TILED_LAUNCH_DIM,  # <-- CHANGED
        inputs=[x, J_values, constraint_body_idx, out],
        block_dim=TILE_SIZE,  # <-- CHANGED
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        wp.launch_tiled(
            kernel_J_matvec_tiled,
            dim=TILED_LAUNCH_DIM,  # <-- CHANGED
            inputs=[x, J_values, constraint_body_idx, out],
            block_dim=TILE_SIZE,  # <-- CHANGED
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA graph timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    x, J_values, constraint_body_idx, out = build_data()

    # warmup
    wp.launch(
        kernel_J_matvec_scatter, dim=NUM_CONSTRAINTS, inputs=[x, J_values, constraint_body_idx, out]
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch(
                kernel_J_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[x, J_values, constraint_body_idx, out],
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
    x, J_values, constraint_body_idx, out = build_data()

    # warmup
    wp.launch_tiled(
        kernel_J_matvec_tiled,
        dim=TILED_LAUNCH_DIM,  # <-- CHANGED
        inputs=[x, J_values, constraint_body_idx, out],
        block_dim=TILE_SIZE,  # <-- CHANGED
    )
    wp.synchronize()

    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch_tiled(
                kernel_J_matvec_tiled,
                dim=TILED_LAUNCH_DIM,  # <-- CHANGED
                inputs=[x, J_values, constraint_body_idx, out],
                block_dim=TILE_SIZE,  # <-- CHANGED
            )
    wp.synchronize()

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
    print("\n=== Benchmark: Scatter vs Tiled kernels ===\n")
    print(f"NUM_BODIES={NUM_BODIES}")
    print(f"CONSTRAINTS_PER_BODY={CONSTRAINTS_PER_BODY}")
    print(f"NUM_CONSTRAINTS={NUM_CONSTRAINTS}")
    print(f"TILE_SIZE={TILE_SIZE}")
    print(f"TILED_LAUNCH_DIM={TILED_LAUNCH_DIM}\n")

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
