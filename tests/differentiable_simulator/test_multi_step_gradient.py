"""Test 4: Multi-step velocity gradient via finite differences.

Runs 3 forward steps with a terminal loss on body_qd, then compares
dL/d(body_qd_0) from the full backward pass (TrajectoryBuffer + step_backward)
against finite differences through the entire trajectory.

Tests that gradient accumulation across time steps via save_gradients /
save_pose_gradients / load_step is correct.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_free_box, make_engine, scalar_loss_vel


def test_multi_step_velocity_gradient():
    print("\n=== Test: Multi-step velocity gradient (FD) ===")

    model = build_free_box()
    num_steps = 3
    engine = make_engine(model, sim_steps=num_steps)
    dt = 0.01

    np.random.seed(42)

    state_init = model.state()
    control = model.control()

    qd_size = state_init.body_qd.numpy().flatten().shape[0]
    w = np.random.randn(qd_size).astype(np.float32)

    def run_forward_trajectory(state_in_override=None):
        """Run num_steps forward, return final state."""
        s_in = model.state()
        if state_in_override is not None:
            wp.copy(s_in.body_q, state_init.body_q)
            wp.copy(
                s_in.body_qd,
                wp.array(state_in_override.reshape(s_in.body_qd.numpy().shape),
                         dtype=wp.spatial_vector, device=model.device),
            )
        else:
            wp.copy(s_in.body_q, state_init.body_q)
            wp.copy(s_in.body_qd, state_init.body_qd)

        s_out = model.state()
        for _ in range(num_steps):
            contacts = model.collide(s_in)
            engine.step(s_in, s_out, control, contacts, dt)
            s_in, s_out = s_out, s_in
        return s_in  # after swap, s_in holds the final state

    # --- Analytical gradient via adjoint + trajectory buffer ---
    buffer = TrajectoryBuffer(
        data=engine.data,
        contacts=engine.axion_contacts,
        dims=engine.dims,
        num_steps=num_steps,
        device=model.device,
    )

    states = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states[0].body_q, state_init.body_q)
    wp.copy(states[0].body_qd, state_init.body_qd)

    for i in range(num_steps):
        contacts = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, contacts, dt)
        buffer.save_step(i, engine.data, engine.axion_contacts)

    buffer.zero_grad()

    w_reshaped = w.reshape(engine.data.body_vel_grad.numpy().shape)
    wp.copy(
        buffer.body_vel.grad[num_steps],
        wp.array(w_reshaped, dtype=wp.spatial_vector, device=model.device),
    )

    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)

    grad_analytical = buffer.body_vel.grad[0].numpy().flatten().copy()

    # --- Finite differences ---
    eps = 1e-4
    base_qd = state_init.body_qd.numpy().flatten().copy()
    grad_fd = np.zeros_like(base_qd)

    for i in range(len(base_qd)):
        qd_plus = base_qd.copy()
        qd_plus[i] += eps
        s_final_plus = run_forward_trajectory(qd_plus)
        loss_plus = scalar_loss_vel(s_final_plus, w)

        qd_minus = base_qd.copy()
        qd_minus[i] -= eps
        s_final_minus = run_forward_trajectory(qd_minus)
        loss_minus = scalar_loss_vel(s_final_minus, w)

        grad_fd[i] = (loss_plus - loss_minus) / (2.0 * eps)

    # Compare
    abs_err = np.abs(grad_analytical - grad_fd)
    denom = np.maximum(np.abs(grad_fd), np.abs(grad_analytical))
    denom = np.maximum(denom, 1e-8)
    rel_err = abs_err / denom
    max_rel_err = np.max(rel_err)

    print(f"  Analytical grad: {grad_analytical}")
    print(f"  FD grad:         {grad_fd}")
    print(f"  Max abs error:   {np.max(abs_err):.2e}")
    print(f"  Max rel error:   {max_rel_err:.4f}")

    for i in range(len(grad_fd)):
        if rel_err[i] > 0.05:
            print(f"    idx={i}: analytical={grad_analytical[i]:.6f}, fd={grad_fd[i]:.6f}, "
                  f"rel_err={rel_err[i]:.4f}")

    assert max_rel_err < 0.1, f"Multi-step velocity gradient FD check failed: max rel error {max_rel_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_multi_step_velocity_gradient()
