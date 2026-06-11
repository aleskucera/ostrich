import time

import numpy as np
import warp as wp
from ostrich.optim.M_inv_Jjt import build_tiled_graph_data
from ostrich.optim.M_inv_Jjt import create_M_inv_Jjt_matvec_tiled
from ostrich.optim.M_inv_Jjt import generate_constr_to_body
from ostrich.optim.M_inv_Jjt import generate_M_inv
from ostrich.optim.M_inv_Jjt import kernel_M_inv_Jjt_matvec_scatter

wp.init()

# =================================================================================
# 0. CONSTANTS (Based on user's desired setup)
# =================================================================================

NUM_BODIES = 1024 * 4
NUM_CONSTRAINTS = NUM_BODIES * 4
CONSTRAINTS_PER_BODY = 16
BODIES_IN_TILE = 8
TILE_SIZE = BODIES_IN_TILE * CONSTRAINTS_PER_BODY

assert NUM_BODIES % BODIES_IN_TILE == 0, "NUM_BODIES must be divisible by BODIES_IN_TILE"
TILED_LAUNCH_DIM = NUM_BODIES // BODIES_IN_TILE


# ----------------------------------------------------------
# Build test data (modified to return inputs for both kernels)
# ----------------------------------------------------------
def build_data():
    # A. Generate Random Data
    M_inv_np = generate_M_inv(NUM_BODIES)
    dlambda_np = np.random.randn(NUM_CONSTRAINTS).astype(np.float32)
    J_np = np.random.randn(NUM_CONSTRAINTS, 2, 6).astype(np.float32)
    # Use a fixed seed for deterministic constraint generation
    constr_map_np = generate_constr_to_body(
        NUM_CONSTRAINTS, NUM_BODIES, CONSTRAINTS_PER_BODY, seed=42
    )

    # --- Data for Scatter Kernel ---
    dlambda_wp_s = wp.array(dlambda_np, dtype=wp.float32)
    J_wp_s = wp.array(J_np, dtype=wp.spatial_vector)
    constr_map_wp_s = wp.array(constr_map_np, dtype=wp.int32)
    M_inv_wp = wp.array(M_inv_np, dtype=wp.spatial_matrix)
    du_scatter = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)

    scatter_inputs = [dlambda_wp_s, J_wp_s, constr_map_wp_s, M_inv_wp, du_scatter]

    # --- Data for Tiled Kernel ---
    J_flat_np, body_to_J_np = build_tiled_graph_data(
        NUM_BODIES, CONSTRAINTS_PER_BODY, J_np, constr_map_np
    )

    # Pad lambda for the tiled kernel's safe access (index 0 is padding)
    dlambda_padded_np = np.zeros(NUM_CONSTRAINTS + 1, dtype=np.float32)
    dlambda_padded_np[1:] = dlambda_np

    lambda_wp_t = wp.array(dlambda_padded_np, dtype=wp.float32)
    J_flat_wp = wp.array(J_flat_np, dtype=wp.spatial_vector)
    body_to_J_wp = wp.array(body_to_J_np, dtype=wp.int32)
    du_tiled = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)

    # Instantiate the kernel factory result once
    tiled_kernel = create_M_inv_Jjt_matvec_tiled(BODIES_IN_TILE, CONSTRAINTS_PER_BODY)

    # lambda_, J, J_to_lambda, body_to_constraints, M_inv, du
    tiled_inputs = [lambda_wp_t, J_flat_wp, body_to_J_wp, M_inv_wp, du_tiled]

    return scatter_inputs, tiled_inputs, tiled_kernel


# =================================================================================
# 4. TIMING FUNCTIONS
# =================================================================================


# ----------------------------------------------------------
# Standard timing: scatter kernel (M_inv J^T J)
# ----------------------------------------------------------
def measure_scatter_time(scatter_inputs, num_repeats=50):
    dlambda, J, constr_map, M_inv, du = scatter_inputs

    # warmup
    wp.launch(
        kernel_M_inv_Jjt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[dlambda, J, constr_map, M_inv],
        outputs=[du],
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        du.zero_()
        wp.launch(
            kernel_M_inv_Jjt_matvec_scatter,
            dim=NUM_CONSTRAINTS,
            inputs=[dlambda, J, constr_map, M_inv],
            outputs=[du],
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# Standard timing: tiled kernel (M_inv J^T J)
# ----------------------------------------------------------
def measure_tiled_time(tiled_inputs, tiled_kernel, num_repeats=50):
    lambda_, J_flat, body_to_J, M_inv, du = tiled_inputs

    # warmup: launch with tiles
    wp.launch_tiled(
        kernel=tiled_kernel,
        dim=TILED_LAUNCH_DIM,
        inputs=[lambda_, J_flat, body_to_J, M_inv],
        outputs=[du],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    t0 = time.time()
    for _ in range(num_repeats):
        du.zero_()
        wp.launch_tiled(
            kernel=tiled_kernel,
            dim=TILED_LAUNCH_DIM,
            inputs=[lambda_, J_flat, body_to_J, M_inv],
            outputs=[du],
            block_dim=TILE_SIZE,
        )
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA graph timing: scatter kernel (M_inv J^T J)
# ----------------------------------------------------------
def measure_scatter_time_cuda_graph(scatter_inputs, num_graph_iters=100, num_ops_in_graph=10):
    dlambda, J, constr_map, M_inv, du = scatter_inputs

    # warmup
    wp.launch(
        kernel_M_inv_Jjt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[dlambda, J, constr_map, M_inv],
        outputs=[du],
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            du.zero_()
            wp.launch(
                kernel_M_inv_Jjt_matvec_scatter,
                dim=NUM_CONSTRAINTS,
                inputs=[dlambda, J, constr_map, M_inv],
                outputs=[du],
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
# CUDA graph timing: tiled kernel (M_inv J^T J)
# ----------------------------------------------------------
def measure_tiled_time_cuda_graph(
    tiled_inputs, tiled_kernel, num_graph_iters=100, num_ops_in_graph=10
):
    lambda_, J_flat, body_to_J, M_inv, du = tiled_inputs

    # warmup
    wp.launch_tiled(
        kernel=tiled_kernel,
        dim=TILED_LAUNCH_DIM,
        inputs=[lambda_, J_flat, body_to_J, M_inv],
        outputs=[du],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    # capture
    with wp.ScopedCapture() as cap:
        for _ in range(num_ops_in_graph):
            du.zero_()
            wp.launch_tiled(
                kernel=tiled_kernel,
                dim=TILED_LAUNCH_DIM,
                inputs=[lambda_, J_flat, body_to_J, M_inv],
                outputs=[du],
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


# =================================================================================
# 5. MAIN EXECUTION
# =================================================================================


def main():
    print("\n=== Benchmark: M_inv J^T J Scatter vs Tiled kernels ===\n")
    print(f"NUM_BODIES={NUM_BODIES}")
    print(f"NUM_CONSTRAINTS={NUM_CONSTRAINTS}")
    print(f"CONSTRAINTS_PER_BODY={CONSTRAINTS_PER_BODY}")
    print(f"TILE_SIZE={TILE_SIZE}")
    print(f"TILED_LAUNCH_DIM={TILED_LAUNCH_DIM}\n")

    # Build data once
    scatter_inputs, tiled_inputs, tiled_kernel = build_data()

    # ---------------- Correctness Check ----------------
    # Run kernels once for comparison
    du_scatter_check = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)
    dlambda, J, constr_map, M_inv, _ = scatter_inputs
    wp.launch(
        kernel_M_inv_Jjt_matvec_scatter,
        dim=NUM_CONSTRAINTS,
        inputs=[dlambda, J, constr_map, M_inv],
        outputs=[du_scatter_check],
    )

    du_tiled_check = wp.zeros(NUM_BODIES, dtype=wp.spatial_vector)
    lambda_, J_flat, body_to_J, M_inv, _ = tiled_inputs
    wp.launch_tiled(
        kernel=tiled_kernel,
        dim=TILED_LAUNCH_DIM,
        inputs=[lambda_, J_flat, body_to_J, M_inv],
        outputs=[du_tiled_check],
        block_dim=TILE_SIZE,
    )
    wp.synchronize()

    res_scatter = du_scatter_check.numpy()
    res_tiled = du_tiled_check.numpy()

    diff = np.abs(res_scatter - res_tiled)
    max_err = np.max(diff)

    print(f"Correctness Check:")
    print(f"-------------------")
    print(f"Max Absolute Error: {max_err:.8f}")

    if max_err < 1e-4:
        print("SUCCESS: Tiled implementation matches Scatter implementation (within tolerance).")
    else:
        print("WARNING: Mismatch detected. Benchmark results may not be comparable.")
        for i in range(min(5, NUM_BODIES)):
            print(f"Body {i}: Scatter {res_scatter[i,0]:.4f} | Tiled {res_tiled[i,0]:.4f}")

    print("\n" + "=" * 30 + "\n")

    # ---------------- Standard Timing ----------------
    print("Measuring standard M_inv J^T J scatter...")
    t_s_std = measure_scatter_time(scatter_inputs)
    print(f"M_inv J^T J Scatter (standard): {t_s_std * 1e3:.4f} ms")

    print("Measuring standard M_inv J^T J tiled...")
    t_t_std = measure_tiled_time(tiled_inputs, tiled_kernel)
    print(f"M_inv J^T J Tiled (standard):   {t_t_std * 1e3:.4f} ms\n")

    # --------------- CUDA graph Timing --------------
    print("Measuring CUDA graph M_inv J^T J scatter...")
    t_s_graph = measure_scatter_time_cuda_graph(scatter_inputs)
    print(f"M_inv J^T J Scatter (CUDA graph): {t_s_graph * 1e3:.4f} ms")

    print("Measuring CUDA graph M_inv J^T J tiled...")
    t_t_graph = measure_tiled_time_cuda_graph(tiled_inputs, tiled_kernel)
    print(f"M_inv J^T J Tiled (CUDA graph):   {t_t_graph * 1e3:.4f} ms\n")


if __name__ == "__main__":
    main()
