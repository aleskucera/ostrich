"""Test 2: Single-step velocity gradient via finite differences.

Perturbs initial body velocities, runs one forward step, and compares
dL/d(body_qd_in) from step_backward() against central finite differences.

This is the most fundamental gradient test — velocity enters the dynamics
residual directly (M(u - u_prev)/dt = ...), so any bug in the adjoint body
init or residual gradient will show up here.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_box_on_ground, make_engine, forward_one_step, scalar_loss_vel


def test_velocity_gradient_single_step():
    print("\n=== Test: Single-step velocity gradient (FD) ===")

    model = build_box_on_ground()
    engine = make_engine(model, sim_steps=3)
    dt = 0.01

    np.random.seed(42)

    state_in = model.state()
    control = model.control()

    qd_size = state_in.body_qd.numpy().flatten().shape[0]
    w = np.random.randn(qd_size).astype(np.float32)

    # --- Analytical gradient via adjoint ---
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

    grad_analytical = engine.data.body_vel_prev.grad.numpy().flatten().copy()

    # --- Finite differences ---
    eps = 1e-4
    grad_fd = np.zeros_like(grad_analytical)
    base_qd = state_in.body_qd.numpy().flatten().copy()

    for i in range(len(base_qd)):
        qd_plus = base_qd.copy()
        qd_plus[i] += eps
        wp.copy(
            state_in.body_qd,
            wp.array(qd_plus.reshape(state_in.body_qd.numpy().shape),
                     dtype=wp.spatial_vector, device=model.device),
        )
        state_out_plus = forward_one_step(engine, model, state_in, control, dt)
        loss_plus = scalar_loss_vel(state_out_plus, w)

        qd_minus = base_qd.copy()
        qd_minus[i] -= eps
        wp.copy(
            state_in.body_qd,
            wp.array(qd_minus.reshape(state_in.body_qd.numpy().shape),
                     dtype=wp.spatial_vector, device=model.device),
        )
        state_out_minus = forward_one_step(engine, model, state_in, control, dt)
        loss_minus = scalar_loss_vel(state_out_minus, w)

        grad_fd[i] = (loss_plus - loss_minus) / (2.0 * eps)

    # Restore
    wp.copy(
        state_in.body_qd,
        wp.array(base_qd.reshape(state_in.body_qd.numpy().shape),
                 dtype=wp.spatial_vector, device=model.device),
    )

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

    assert max_rel_err < 0.1, f"Velocity gradient FD check failed: max rel error {max_rel_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_velocity_gradient_single_step()
