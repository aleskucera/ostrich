"""Test 6: Gradient-based optimization sanity check.

Optimizes initial z-velocity of a free-falling box to reach a target z-position
after N steps. Verifies that gradient descent using step_backward() gradients
actually reduces the loss — an end-to-end check that gradients are useful.

Catches sign errors and scaling issues that FD tests at small epsilon might miss.
"""

import numpy as np
import warp as wp

wp.init()

from axion.simulation.trajectory_buffer import TrajectoryBuffer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_free_box, make_engine


def test_optimization_sanity_check():
    print("\n=== Test: Optimization sanity check ===")

    model = build_free_box()
    num_steps = 5
    engine = make_engine(model, sim_steps=num_steps)
    dt = 0.01

    state_init = model.state()
    control = model.control()

    # Target: we want final vz = 5.0 (box starts at rest, gravity pulls it down)
    target_vz = 5.0

    def compute_trajectory_and_loss(v0_z):
        s_in = model.state()
        wp.copy(s_in.body_q, state_init.body_q)

        qd_np = state_init.body_qd.numpy().copy()
        qd_shape = qd_np.shape
        qd_flat = qd_np.flatten()
        # spatial_vector: (wx, wy, wz, vx, vy, vz)
        qd_flat[5] = v0_z  # vz
        wp.copy(
            s_in.body_qd,
            wp.array(qd_flat.reshape(qd_shape), dtype=wp.spatial_vector, device=model.device),
        )

        buffer = TrajectoryBuffer(
            data=engine.data,
            contacts=engine.axion_contacts,
            dims=engine.dims,
            num_steps=num_steps,
            device=model.device,
        )

        # Forward
        states = [model.state() for _ in range(num_steps + 1)]
        wp.copy(states[0].body_q, s_in.body_q)
        wp.copy(states[0].body_qd, s_in.body_qd)

        for i in range(num_steps):
            contacts = model.collide(states[i])
            engine.step(states[i], states[i + 1], control, contacts, dt)
            buffer.save_step(i, engine.data, engine.axion_contacts)

        # Loss = (vz_final - target_vz)^2
        qd_final = states[num_steps].body_qd.numpy().flatten()
        vz_final = qd_final[5]
        loss = (vz_final - target_vz) ** 2

        # dL/d(body_qd_final) — only vz component is nonzero
        buffer.zero_grad()
        vel_grad_terminal = np.zeros_like(buffer.body_vel.grad[num_steps].numpy())
        vel_grad_flat = vel_grad_terminal.reshape(-1, 6)
        vel_grad_flat[0, 5] = 2.0 * (vz_final - target_vz)
        wp.copy(
            buffer.body_vel.grad[num_steps],
            wp.array(vel_grad_terminal.reshape(buffer.body_vel.grad[num_steps].numpy().shape),
                     dtype=wp.spatial_vector, device=model.device),
        )

        # Backward
        for i in range(num_steps - 1, -1, -1):
            buffer.load_step(i, engine.data, engine.axion_contacts)
            engine.data.zero_gradients()
            engine.step_backward()
            buffer.save_gradients(i, engine.data)
            buffer.save_pose_gradients(i, engine.data)

        vel_grad = buffer.body_vel.grad[0].numpy().flatten()
        dL_dv0z = vel_grad[5]  # vz component

        return loss, dL_dv0z

    # Run gradient descent
    v0_z = 0.0
    lr = 0.5
    losses = []

    for iteration in range(30):
        loss, grad = compute_trajectory_and_loss(v0_z)
        losses.append(loss)
        v0_z -= lr * grad
        if iteration % 5 == 0 or iteration == 29:
            print(f"  iter {iteration:3d}: loss={loss:.6f}, v0_z={v0_z:.4f}, grad={grad:.6f}")

    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")

    assert losses[-1] < losses[0] * 0.01, (
        f"Optimization did not converge: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
    )
    print("  PASSED")


if __name__ == "__main__":
    test_optimization_sanity_check()
