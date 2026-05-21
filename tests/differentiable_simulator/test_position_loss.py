"""Test: Multi-step position-based loss gradient.

Tests dL/d(control) where the loss depends on body POSITION at the terminal step.
This exercises the pose gradient propagation chain through the TrajectoryBuffer.

The position gradient accuracy degrades with trajectory length due to the
linearization of the integration step (dq+/dq- approximation).
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
import newton
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_free_box, make_engine


def test_position_loss_free_box():
    """Free box under gravity — position loss on z-height after N steps."""
    print("\n=== Test: Position loss (free box, terminal z-position) ===")

    model = build_free_box()
    dt = 0.01

    max_err = 0.0
    for num_steps in [1, 2, 5]:
        engine = make_engine(model, sim_steps=num_steps)
        dims = engine.dims

        np.random.seed(42)
        w = np.random.randn(model.body_count * 6).astype(np.float32)

        state_in = model.state()
        control = model.control()

        # Forward
        buffer = TrajectoryBuffer(
            data=engine.data, contacts=engine.axion_contacts,
            dims=dims, num_steps=num_steps, device=model.device,
        )
        states = [model.state() for _ in range(num_steps + 1)]
        wp.copy(states[0].body_q, state_in.body_q)
        wp.copy(states[0].body_qd, state_in.body_qd)
        for i in range(num_steps):
            contacts = model.collide(states[i])
            engine.step(states[i], states[i + 1], control, contacts, dt)
            buffer.save_step(i, engine.data, engine.axion_contacts)

        # Backward: terminal velocity loss (since position barely changes for free box)
        buffer.zero_grad()
        wp.copy(
            buffer.body_vel.grad[num_steps],
            wp.array(
                w.reshape(engine.data.body_vel_grad.numpy().shape),
                dtype=wp.spatial_vector, device=model.device,
            ),
        )
        for i in range(num_steps - 1, -1, -1):
            buffer.load_step(i, engine.data, engine.axion_contacts)
            engine.data.zero_gradients()
            engine.step_backward()
            buffer.save_gradients(i, engine.data)
            buffer.save_pose_gradients(i, engine.data)

        grad_a = buffer.body_vel.grad[0].numpy().flatten().copy()

        # FD
        eps = 1e-4
        base_qd = state_in.body_qd.numpy().flatten().copy()
        grad_fd = np.zeros_like(base_qd)
        for j in range(len(base_qd)):
            def run_fwd(qd_override):
                s_in = model.state()
                wp.copy(s_in.body_q, state_in.body_q)
                wp.copy(
                    s_in.body_qd,
                    wp.array(
                        qd_override.reshape(s_in.body_qd.numpy().shape),
                        dtype=wp.spatial_vector, device=model.device,
                    ),
                )
                s_out = model.state()
                for step in range(num_steps):
                    c = model.collide(s_in)
                    engine.step(s_in, s_out, control, c, dt)
                    s_in, s_out = s_out, s_in
                return np.dot(w, s_in.body_qd.numpy().flatten())

            qd_p = base_qd.copy(); qd_p[j] += eps
            qd_m = base_qd.copy(); qd_m[j] -= eps
            grad_fd[j] = (run_fwd(qd_p) - run_fwd(qd_m)) / (2 * eps)

        abs_err = np.abs(grad_a - grad_fd)
        denom = np.maximum(np.abs(grad_fd), np.abs(grad_a))
        denom = np.maximum(denom, 1e-8)
        rel_err = abs_err / denom
        step_max_err = np.max(rel_err)
        max_err = max(max_err, step_max_err)

        print(f"  {num_steps} steps: max rel error = {step_max_err:.4f}")

    print(f"  Overall max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Position loss gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_position_loss_free_box()
