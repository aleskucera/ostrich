"""Test 5: Single-step pose gradient via finite differences.

Perturbs initial body positions (translation only), runs one forward step,
and compares dL/d(body_q_in) from step_backward() against central FD.

Tests the body_pose_prev_grad_kernel (kinematic gradient propagation).
Quaternion components are not tested (tangent-space perturbation needed).
"""

import numpy as np
import warp as wp

wp.init()

from axion.simulation.trajectory_buffer import TrajectoryBuffer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_free_box, make_engine, forward_one_step, scalar_loss_vel


def test_pose_gradient_single_step():
    print("\n=== Test: Single-step pose gradient (FD) ===")

    model = build_free_box()
    engine = make_engine(model, sim_steps=3)
    dt = 0.01

    np.random.seed(42)

    state_in = model.state()
    control = model.control()

    qd_size = state_in.body_qd.numpy().flatten().shape[0]
    w = np.random.randn(qd_size).astype(np.float32)

    # --- Analytical gradient ---
    state_out = model.state()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, dt)

    buffer = TrajectoryBuffer(
        data=engine.data,
        contacts=engine.axion_contacts,
        dims=engine.dims,
        num_steps=1,
        device=model.device,
    )
    buffer.save_step(0, engine.data, engine.axion_contacts)
    buffer.zero_grad()

    w_reshaped = w.reshape(engine.data.body_vel_grad.numpy().shape)
    wp.copy(
        buffer.body_vel.grad[1],
        wp.array(w_reshaped, dtype=wp.spatial_vector, device=model.device),
    )

    buffer.load_step(0, engine.data, engine.axion_contacts)
    engine.data.zero_gradients()
    engine.step_backward()

    pose_grad_raw = engine.data.body_pose_prev.grad.numpy().flatten().copy()

    # Extract position (translation) part only — first 3 of 7 components per body
    base_q = state_in.body_q.numpy()
    q_flat = base_q.reshape(-1, 7)
    total_bodies = q_flat.shape[0]
    pose_grad_flat = pose_grad_raw.reshape(-1, 7)

    grad_analytical_pos = []
    for b in range(total_bodies):
        grad_analytical_pos.extend(pose_grad_flat[b, :3])
    grad_analytical_pos = np.array(grad_analytical_pos, dtype=np.float32)

    # --- Finite differences (position only) ---
    eps = 1e-4
    grad_fd_pos = np.zeros_like(grad_analytical_pos)

    idx = 0
    for b in range(total_bodies):
        for c in range(3):  # x, y, z position
            q_plus = q_flat.copy()
            q_plus[b, c] += eps
            wp.copy(
                state_in.body_q,
                wp.array(q_plus.reshape(base_q.shape), dtype=wp.transform, device=model.device),
            )
            s_plus = forward_one_step(engine, model, state_in, control, dt)
            loss_plus = scalar_loss_vel(s_plus, w)

            q_minus = q_flat.copy()
            q_minus[b, c] -= eps
            wp.copy(
                state_in.body_q,
                wp.array(q_minus.reshape(base_q.shape), dtype=wp.transform, device=model.device),
            )
            s_minus = forward_one_step(engine, model, state_in, control, dt)
            loss_minus = scalar_loss_vel(s_minus, w)

            grad_fd_pos[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            idx += 1

    # Restore
    wp.copy(state_in.body_q, wp.array(base_q, dtype=wp.transform, device=model.device))

    # Compare
    abs_err = np.abs(grad_analytical_pos - grad_fd_pos)
    denom = np.maximum(np.abs(grad_fd_pos), np.abs(grad_analytical_pos))
    denom = np.maximum(denom, 1e-8)
    rel_err = abs_err / denom
    max_rel_err = np.max(rel_err)

    print(f"  Analytical grad (pos): {grad_analytical_pos}")
    print(f"  FD grad (pos):         {grad_fd_pos}")
    print(f"  Max abs error:         {np.max(abs_err):.2e}")
    print(f"  Max rel error:         {max_rel_err:.4f}")

    for i in range(len(grad_fd_pos)):
        if rel_err[i] > 0.05:
            print(f"    idx={i}: analytical={grad_analytical_pos[i]:.6f}, fd={grad_fd_pos[i]:.6f}, "
                  f"rel_err={rel_err[i]:.4f}")

    assert max_rel_err < 0.1, f"Pose gradient FD check failed: max rel error {max_rel_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_pose_gradient_single_step()
