"""Test 1: Zero-gradient baseline.

A loss independent of the simulation (zero loss gradient) should produce
exactly zero gradients from step_backward(). Catches gradient contamination
from uninitialized buffers or stale data in the TrajectoryBuffer.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent))
from helpers import build_box_on_ground, make_engine


def test_zero_gradient_baseline():
    print("\n=== Test: Zero-gradient baseline ===")

    model = build_box_on_ground()
    engine = make_engine(model, sim_steps=3)
    dt = 0.01

    state_in = model.state()
    state_out = model.state()
    control = model.control()
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

    # body_pose_grad and body_vel_grad are both zero (no loss depends on output)
    # => step_backward should produce zero gradients everywhere
    buffer.load_step(0, engine.data, engine.axion_contacts)
    engine.data.zero_gradients()
    engine.step_backward()

    vel_prev_grad = engine.data.body_vel_prev.grad.numpy()
    target_pos_grad = engine.data.joint_target_pos.grad.numpy()
    target_vel_grad = engine.data.joint_target_vel.grad.numpy()

    vel_err = np.max(np.abs(vel_prev_grad))
    tpos_err = np.max(np.abs(target_pos_grad))
    tvel_err = np.max(np.abs(target_vel_grad))

    print(f"  body_vel_prev.grad max: {vel_err:.2e}")
    print(f"  joint_target_pos.grad max: {tpos_err:.2e}")
    print(f"  joint_target_vel.grad max: {tvel_err:.2e}")

    assert vel_err < 1e-6, f"body_vel_prev.grad not zero: {vel_err:.2e}"
    assert tpos_err < 1e-6, f"joint_target_pos.grad not zero: {tpos_err:.2e}"
    assert tvel_err < 1e-6, f"joint_target_vel.grad not zero: {tvel_err:.2e}"
    print("  PASSED")


if __name__ == "__main__":
    test_zero_gradient_baseline()
