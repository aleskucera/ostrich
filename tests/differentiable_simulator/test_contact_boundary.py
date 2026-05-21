"""Test 8: Contact boundary gradient test.

Tests velocity gradients for a box at two heights:
  - In contact (h=0.55): contact constraints active
  - Free flight (h=2.0): no contact constraints

Verifies gradients are correct (vs FD) on both sides, have no NaN/Inf,
and that the constr_active_mask properly gates the adjoint.
"""

import numpy as np
import warp as wp

wp.init()

from axion.simulation.trajectory_buffer import TrajectoryBuffer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_box_on_ground, make_engine, forward_one_step, scalar_loss_vel


def test_contact_boundary():
    print("\n=== Test: Contact boundary ===")

    dt = 0.01
    np.random.seed(42)

    for label, height in [("in contact", 0.55), ("free flight", 2.0)]:
        print(f"\n  --- {label} (height={height}) ---")

        model = build_box_on_ground(height=height)
        engine = make_engine(model, sim_steps=3)

        state_in = model.state()
        control = model.control()

        qd_size = state_in.body_qd.numpy().flatten().shape[0]
        w = np.random.randn(qd_size).astype(np.float32)

        # Analytical
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

        # FD
        eps = 1e-4
        base_qd = state_in.body_qd.numpy().flatten().copy()
        grad_fd = np.zeros_like(base_qd)

        for i in range(len(base_qd)):
            qd_plus = base_qd.copy()
            qd_plus[i] += eps
            wp.copy(
                state_in.body_qd,
                wp.array(qd_plus.reshape(state_in.body_qd.numpy().shape),
                         dtype=wp.spatial_vector, device=model.device),
            )
            s_plus = forward_one_step(engine, model, state_in, control, dt)
            loss_plus = scalar_loss_vel(s_plus, w)

            qd_minus = base_qd.copy()
            qd_minus[i] -= eps
            wp.copy(
                state_in.body_qd,
                wp.array(qd_minus.reshape(state_in.body_qd.numpy().shape),
                         dtype=wp.spatial_vector, device=model.device),
            )
            s_minus = forward_one_step(engine, model, state_in, control, dt)
            loss_minus = scalar_loss_vel(s_minus, w)

            grad_fd[i] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore
        wp.copy(
            state_in.body_qd,
            wp.array(base_qd.reshape(state_in.body_qd.numpy().shape),
                     dtype=wp.spatial_vector, device=model.device),
        )

        abs_err = np.abs(grad_analytical - grad_fd)
        denom = np.maximum(np.abs(grad_fd), np.abs(grad_analytical))
        denom = np.maximum(denom, 1e-8)
        rel_err = abs_err / denom
        max_rel_err = np.max(rel_err)

        assert not np.any(np.isnan(grad_analytical)), "NaN in analytical gradient!"
        assert not np.any(np.isinf(grad_analytical)), "Inf in analytical gradient!"

        print(f"  Analytical grad: {grad_analytical}")
        print(f"  FD grad:         {grad_fd}")
        print(f"  Max abs error:   {np.max(abs_err):.2e}")
        print(f"  Max rel error:   {max_rel_err:.4f}")

        for i in range(len(grad_fd)):
            if rel_err[i] > 0.05:
                print(f"    idx={i}: analytical={grad_analytical[i]:.6f}, fd={grad_fd[i]:.6f}, "
                      f"rel_err={rel_err[i]:.4f}")

        assert max_rel_err < 0.15, (
            f"Contact boundary gradient check failed ({label}): max rel error {max_rel_err:.4f}"
        )

    print("  PASSED")


if __name__ == "__main__":
    test_contact_boundary()
