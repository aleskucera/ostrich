import time

import numpy as np
import warp as wp
from ostrich.sparse.M_inv_Jt_tiled import BODIES_IN_TILE
from ostrich.sparse.M_inv_Jt_tiled import CONSTRAINTS_PER_BODY
from ostrich.sparse.M_inv_Jt_tiled import create_M_inv_Jt_matvec_tiled
from ostrich.sparse.M_inv_Jt_tiled import kernel_M_inv_Jt_matvec_scatter
from ostrich.sparse.M_inv_Jt_tiled import NUM_BODIES
from ostrich.sparse.M_inv_Jt_tiled import NUM_CONSTRAINTS
from ostrich.sparse.M_inv_Jt_tiled import TILE_SIZE

wp.init()

# instantiate tiled kernel once
kernel_M_inv_Jt_matvec_tiled = create_M_inv_Jt_matvec_tiled()


# ----------------------------------------------------------
# Build test data
# ----------------------------------------------------------
def build_data():
    # x is on constraints (NUM_CONSTRAINTS)
    x_host = np.random.randn(NUM_CONSTRAINTS).astype(np.float32)
    x = wp.array(x_host, dtype=wp.float32)

    # J_values: per-constraint spatial vectors (NUM_CONSTRAINTS x 6)
    J_values_host = np.random.randn(NUM_CONSTRAINTS, 6).astype(np.float32)
    J_values = wp.array(J_values_host, dtype=wp.spatial_vector)

    # M_inv: per-body spatial matrices (NUM_BODIES x 6 x 6)
    M_inv_host = np.random.randn(NUM_BODIES, 6, 6).astype(np.float32)
    M_inv = wp.array(M_inv_host, dtype=wp.spatial_matrix)

    # [0,...,0, 1,...,1, 2,...,2, ...] length = NUM_CONSTRAINTS
    constraint_body_idx_host = np.repeat(
        np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY
    )
    constraint_body_idx = wp.array(constraint_body_idx_host, dtype=wp.int32)

    # out lives on bodies (NUM_BODIES), spatial vectors
    out = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)

    return x, J_values, constraint_body_idx, M_inv, out


# compute tiled launch dimension once (must be integer)
assert NUM_CONSTRAINTS % TILE_SIZE == 0, "NUM_CONSTRAINTS must be divisible by TILE_SIZE"
TILED_LAUNCH_DIM = NUM_CONSTRAINTS // TILE_SIZE


# ----------------------------------------------------------
# Standard timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time(num_repeats=50):
    x, J_values, constraint_body_idx, M_inv, out = build_data()

    # warmup
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x, J_values, constraint_body_idx, M_inv],
        outputs=[out],
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        out.zero_()
        wp.launch(
            kernel_M_inv_Jt_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[x, J_values, constraint_body_idx, M_inv],
            outputs=[out],
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# Standard timing: tiled kernel
# ----------------------------------------------------------
def measure_tiled_time(num_repeats=50):
    x, J_values, constraint_body_idx, M_inv, out = build_data()

    # warmup
    wp.launch_tiled(
        kernel=kernel_M_inv_Jt_matvec_tiled,
        dim=TILED_LAUNCH_DIM,
        inputs=[x, J_values, M_inv],
        outputs=[out],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        out.zero_()
        wp.launch_tiled(
            kernel=kernel_M_inv_Jt_matvec_tiled,
            dim=TILED_LAUNCH_DIM,
            inputs=[x, J_values, M_inv],
            outputs=[out],
            block_dim=TILE_SIZE,
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA graph timing: scatter kernel
# ----------------------------------------------------------
def measure_scatter_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    x, J_values, constraint_body_idx, M_inv, out = build_data()

    # warmup
    wp.launch(
        kernel_M_inv_Jt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x, J_values, constraint_body_idx, M_inv],
        outputs=[out],
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            out.zero_()
            wp.launch(
                kernel_M_inv_Jt_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[x, J_values, constraint_body_idx, M_inv],
                outputs=[out],
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
    x, J_values, constraint_body_idx, M_inv, out = build_data()

    # warmup
    wp.launch_tiled(
        kernel=kernel_M_inv_Jt_matvec_tiled,
        dim=TILED_LAUNCH_DIM,
        inputs=[x, J_values, M_inv],
        outputs=[out],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            out.zero_()
            wp.launch_tiled(
                kernel=kernel_M_inv_Jt_matvec_tiled,
                dim=TILED_LAUNCH_DIM,
                inputs=[x, J_values, M_inv],
                outputs=[out],
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
    print("\n=== Benchmark: M_inv * J^T * x (scatter vs tiled) ===\n")
    print(f"NUM_BODIES={NUM_BODIES}")
    print(f"CONSTRAINTS_PER_BODY={CONSTRAINTS_PER_BODY}")
    print(f"NUM_CONSTRAINTS={NUM_CONSTRAINTS}")
    print(f"TILE_SIZE={TILE_SIZE}")
    print(f"BODIES_IN_TILE={BODIES_IN_TILE}")
    print(f"TILED_LAUNCH_DIM={TILED_LAUNCH_DIM}\n")

    # ---------------- Standard ----------------
    print("Measuring standard scatter...")
    t_s_std = measure_scatter_time()
    print(f"Scatter (standard): {t_s_std * 1e3:.4f} ms")

    print("Measuring standard tiled...")
    t_t_std = measure_tiled_time()
    print(f"Tiled   (standard): {t_t_std * 1e3:.4f} ms\n")

    # --------------- CUDA graphs --------------
    print("Measuring CUDA graph scatter...")
    t_s_graph = measure_scatter_time_cuda_graph()
    print(f"Scatter (CUDA graph): {t_s_graph * 1e3:.4f} ms")

    print("Measuring CUDA graph tiled...")
    t_t_graph = measure_tiled_time_cuda_graph()
    print(f"Tiled   (CUDA graph): {t_t_graph * 1e3:.4f} ms\n")


if __name__ == "__main__":
    main()
