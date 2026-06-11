import time

import numpy as np
import warp as wp
from ostrich.sparse.J_Jt_tiled import CONSTRAINTS_PER_BODY
from ostrich.sparse.J_Jt_tiled import kernel_J_Jt_matvec_tiled
from ostrich.sparse.J_Jt_tiled import kernel_J_matvec_scatter
from ostrich.sparse.J_Jt_tiled import kernel_Jt_matvec_scatter
from ostrich.sparse.J_Jt_tiled import NUM_BODIES
from ostrich.sparse.J_Jt_tiled import NUM_CONSTRAINTS

wp.init()


# ----------------------------------------------------------
# Build test data
# ----------------------------------------------------------
def build_data():
    # x lives on constraints (JJ^T : num_constraints -> num_constraints)
    x = wp.array(
        np.random.randn(NUM_CONSTRAINTS).astype(np.float32),
        dtype=wp.float32,
    )

    # J_values: one value per constraint
    J_values = wp.array(
        np.random.randn(NUM_CONSTRAINTS).astype(np.float32),
        dtype=wp.float32,
    )

    # [0,...,0, 1,...,1, 2,...,2, ...] length = NUM_CONSTRAINTS
    constraint_body_idx = wp.array(
        np.repeat(np.arange(NUM_BODIES, dtype=np.int32), CONSTRAINTS_PER_BODY),
        dtype=wp.int32,
    )

    # scratch and outputs
    tmp_scatter = wp.zeros(NUM_BODIES, dtype=wp.float32)  # y = J^T * x
    out_scatter = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)  # JJ^T * x
    out_tiled = wp.zeros(NUM_CONSTRAINTS, dtype=wp.float32)  # JJ^T * x

    return x, J_values, constraint_body_idx, tmp_scatter, out_scatter, out_tiled


# ----------------------------------------------------------
# Standard timing: two scattered kernels (J^T then J)
# ----------------------------------------------------------
def measure_scatter_chain_time(num_repeats=50):
    (
        x,
        J_values,
        constraint_body_idx,
        tmp_scatter,
        out_scatter,
        _,
    ) = build_data()

    # warmup
    wp.launch(
        kernel_Jt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x, J_values, constraint_body_idx],
        outputs=[tmp_scatter],
    )
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[tmp_scatter, J_values, constraint_body_idx],
        outputs=[out_scatter],
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        # Note: we do not reset tmp_scatter here, since we only care about kernel timing.
        wp.launch(
            kernel_Jt_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[x, J_values, constraint_body_idx],
            outputs=[tmp_scatter],
        )
        wp.launch(
            kernel_J_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[tmp_scatter, J_values, constraint_body_idx],
            outputs=[out_scatter],
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# Standard timing: single fused tiled kernel (JJ^T)
# ----------------------------------------------------------
def measure_tiled_fused_time(num_repeats=50):
    (
        x,
        J_values,
        constraint_body_idx,
        _,
        _,
        out_tiled,
    ) = build_data()

    # warmup
    wp.launch_tiled(
        kernel_J_Jt_matvec_tiled,
        dim=NUM_BODIES,
        inputs=[x, J_values, constraint_body_idx],
        outputs=[out_tiled],
        block_dim=CONSTRAINTS_PER_BODY,
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        wp.launch_tiled(
            kernel_J_Jt_matvec_tiled,
            dim=NUM_BODIES,
            inputs=[x, J_values, constraint_body_idx],
            outputs=[out_tiled],
            block_dim=CONSTRAINTS_PER_BODY,
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA graph timing: two scattered kernels (J^T then J)
# ----------------------------------------------------------
def measure_scatter_chain_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    (
        x,
        J_values,
        constraint_body_idx,
        tmp_scatter,
        out_scatter,
        _,
    ) = build_data()

    # warmup
    wp.launch(
        kernel_Jt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[x, J_values, constraint_body_idx],
        outputs=[tmp_scatter],
    )
    wp.launch(
        kernel_J_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[tmp_scatter, J_values, constraint_body_idx],
        outputs=[out_scatter],
    )
    wp.synchronize()

    # capture: JJ^T via two kernels, repeated num_ops_in_graph times
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch(
                kernel_Jt_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[x, J_values, constraint_body_idx],
                outputs=[tmp_scatter],
            )
            wp.launch(
                kernel_J_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[tmp_scatter, J_values, constraint_body_idx],
                outputs=[out_scatter],
            )
    wp.synchronize()

    # repeated graph launches
    t0 = time.time()
    for _ in range(num_graph_iters):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    t1 = time.time()

    # average time per "JJ^T application" (i.e., per J^T+J pair)
    return (t1 - t0) / (num_graph_iters * num_ops_in_graph)


# ----------------------------------------------------------
# CUDA graph timing: single fused tiled kernel (JJ^T)
# ----------------------------------------------------------
def measure_tiled_fused_time_cuda_graph(num_graph_iters=100, num_ops_in_graph=10):
    (
        x,
        J_values,
        constraint_body_idx,
        _,
        _,
        out_tiled,
    ) = build_data()

    # warmup
    wp.launch_tiled(
        kernel_J_Jt_matvec_tiled,
        dim=NUM_BODIES,
        inputs=[x, J_values, constraint_body_idx],
        outputs=[out_tiled],
        block_dim=CONSTRAINTS_PER_BODY,
    )
    wp.synchronize()

    # capture: fused JJ^T, repeated num_ops_in_graph times
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            wp.launch_tiled(
                kernel_J_Jt_matvec_tiled,
                dim=NUM_BODIES,
                inputs=[x, J_values, constraint_body_idx],
                outputs=[out_tiled],
                block_dim=CONSTRAINTS_PER_BODY,
            )
    wp.synchronize()

    # repeated graph launches
    t0 = time.time()
    for _ in range(num_graph_iters):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    t1 = time.time()

    # average time per fused JJ^T application
    return (t1 - t0) / (num_graph_iters * num_ops_in_graph)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("\n=== Benchmark: JJ^T via 2x scatter vs fused tiled ===\n")
    print(f"NUM_BODIES={NUM_BODIES}")
    print(f"CONSTRAINTS_PER_BODY={CONSTRAINTS_PER_BODY}")
    print(f"NUM_CONSTRAINTS={NUM_CONSTRAINTS}\n")

    # ---------------- Standard ----------------
    print("Measuring standard JJ^T (2x scatter: J^T then J)...")
    t_s_std = measure_scatter_chain_time()
    print(f"JJ^T (2x scatter, standard): {t_s_std * 1e3:.4f} ms")

    print("Measuring standard JJ^T (fused tiled)...")
    t_t_std = measure_tiled_fused_time()
    print(f"JJ^T (fused tiled, standard): {t_t_std * 1e3:.4f} ms\n")

    # --------------- CUDA graphs --------------
    print("Measuring CUDA graph JJ^T (2x scatter: J^T then J)...")
    t_s_graph = measure_scatter_chain_time_cuda_graph()
    print(f"JJ^T (2x scatter, CUDA graph): {t_s_graph * 1e3:.4f} ms")

    print("Measuring CUDA graph JJ^T (fused tiled)...")
    t_t_graph = measure_tiled_fused_time_cuda_graph()
    print(f"JJ^T (fused tiled, CUDA graph): {t_t_graph * 1e3:.4f} ms\n")


if __name__ == "__main__":
    main()
