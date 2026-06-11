import time

import numpy as np
import warp as wp
from ostrich.sparse.old_operator import MAX_CONTACTS_BETWEEN_BODIES_PER_WORLD
from ostrich.sparse.old_operator import MAX_GROUND_CONTACTS_PER_BODY
from ostrich.sparse.old_operator import NUM_BODIES
from ostrich.sparse.old_operator import NUM_JOINT_CONSTRAINTS
from ostrich.sparse.old_operator import NUM_WORLDS
from ostrich.sparse.old_operator import OldOperator

wp.init()


# ----------------------------------------------------------
# Build test operator
# ----------------------------------------------------------
def build_operator():
    device = wp.get_device()

    op = OldOperator(
        device=device,
        num_worlds=NUM_WORLDS,
        num_bodies=NUM_BODIES,
        num_joint_constraints=NUM_JOINT_CONSTRAINTS,
        max_ground_contacts_per_body=MAX_GROUND_CONTACTS_PER_BODY,
        max_contacts_between_bodies_per_world=MAX_CONTACTS_BETWEEN_BODIES_PER_WORLD,
    )

    return op


# ----------------------------------------------------------
# Measure standard matvec performance
# ----------------------------------------------------------
def measure_matvec_time(op, num_repeats=50):
    x = wp.ones((op.shape[0], op.shape[2]), dtype=wp.spatial_vector)
    y = wp.zeros((op.shape[0], op.shape[1]), dtype=wp.float32)
    z = wp.zeros((op.shape[0], op.shape[1]), dtype=wp.float32)

    # Warmup
    op.matvec(x, y, z, alpha=1.0, beta=0.0)
    wp.synchronize()

    # Timing
    t0 = time.time()
    for _ in range(num_repeats):
        op.matvec(x, y, z, alpha=1.0, beta=0.0)
    wp.synchronize()
    t1 = time.time()

    return (t1 - t0) / num_repeats


# ----------------------------------------------------------
# CUDA Graph timing — executes multiple matvecs inside graph
# ----------------------------------------------------------
def measure_matvec_time_cuda_graph(op, num_graph_iters=100, num_ops_in_graph=10):
    x = wp.ones((op.shape[0], op.shape[2]), dtype=wp.spatial_vector)
    y = wp.zeros((op.shape[0], op.shape[1]), dtype=wp.float32)
    z = wp.zeros((op.shape[0], op.shape[1]), dtype=wp.float32)

    # Warmup
    op.matvec(x, y, z, alpha=1.0, beta=0.0)
    wp.synchronize()

    # Capture CUDA graph
    with wp.ScopedCapture() as capture:
        for _ in range(num_ops_in_graph):
            op.matvec(x, y, z, alpha=1.0, beta=0.0)

    wp.synchronize()

    # Launch graph repeatedly
    t0 = time.time()
    for _ in range(num_graph_iters):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    t1 = time.time()

    total_ops = num_graph_iters * num_ops_in_graph
    return (t1 - t0) / total_ops


# ----------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------
def main():
    print("\n=== JacobianOperator matvec benchmark ===\n")

    op = build_operator()

    print(f"Operator shape: {op.shape}")
    print(
        f"Total batches: {op.shape[0]}, rows per batch: {op.shape[1]}, cols per batch: {op.shape[2]}\n"
        f"  worlds={NUM_WORLDS}, bodies={NUM_BODIES}, joints={NUM_JOINT_CONSTRAINTS}\n"
    )

    print("Measuring standard matvec time...")
    t_std = measure_matvec_time(op, num_repeats=50)
    print(f"Average standard matvec = {t_std * 1e3:.4f} ms\n")

    print("Measuring CUDA graph matvec time...")
    t_graph = measure_matvec_time_cuda_graph(op, num_graph_iters=100, num_ops_in_graph=10)
    print(f"Average CUDA graph matvec = {t_graph * 1e3:.4f} ms\n")


if __name__ == "__main__":
    main()
