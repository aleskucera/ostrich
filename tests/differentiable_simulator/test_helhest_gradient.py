"""Test: Helhest robot gradient verification.

Tests dL/d(wheel_target_vel) for the 3-wheeled Helhest robot with friction contacts.
Compares adjoint gradients against centered finite differences.
"""

import sys
from pathlib import Path

import warp as wp

wp.init()

import numpy as np
import newton
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig, LinearSolverConfig, NewtonRaphsonConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.simulation.trajectory_buffer import TrajectoryBuffer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from helhest.common import HelhestConfig, create_helhest_model

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


def build_helhest(k_p=150.0, k_d=0.0, friction=0.7):
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane()
    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()),
        is_visible=False,
        control_mode="velocity",
        k_p=k_p,
        k_d=k_d,
        friction_left_right=friction,
        friction_rear=0.35,
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)
    return model


def make_engine(model, sim_steps=5):
    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=20),
        linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8),
    )
    return AxionEngine(
        model=model,
        sim_steps=sim_steps,
        config=config,
        logging_config=LoggingConfig(),
        differentiable_simulation=True,
    )


def compute_wheel_vel_gradient(model, engine, target_vel, w, dt=0.01):
    """Run forward + backward, return (analytical_grad, fd_grad) for wheel DOFs."""
    dims = engine.dims

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    wp.copy(
        control.joint_target_vel,
        wp.array(target_vel.reshape(1, -1), dtype=wp.float32, device=model.device),
    )

    # Forward
    state_out = model.state()
    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, dt)

    # Backward
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=1, device=model.device,
    )
    buffer.save_step(0, engine.data, engine.axion_contacts)
    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[1],
        wp.array(w.reshape(engine.data.body_vel_grad.numpy().shape),
                 dtype=wp.spatial_vector, device=model.device),
    )
    buffer.load_step(0, engine.data, engine.axion_contacts)
    engine.data.zero_gradients()
    engine.step_backward()

    grad_analytical = engine.data.joint_target_vel.grad.numpy().flatten().copy()

    # FD for wheel DOFs only (skip free joint DOFs 0-5)
    eps = 1e-4
    grad_fd = np.zeros(dims.joint_dof_count, dtype=np.float32)
    for dof in range(WHEEL_DOF_OFFSET, WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS):
        tv_p = target_vel.copy()
        tv_p[dof] += eps
        wp.copy(control.joint_target_vel,
                wp.array(tv_p.reshape(1, -1), dtype=wp.float32, device=model.device))
        c = model.collide(state_in)
        sp = model.state()
        engine.step(state_in, sp, control, c, dt)
        lp = np.dot(w, sp.body_qd.numpy().flatten())

        tv_m = target_vel.copy()
        tv_m[dof] -= eps
        wp.copy(control.joint_target_vel,
                wp.array(tv_m.reshape(1, -1), dtype=wp.float32, device=model.device))
        c = model.collide(state_in)
        sm = model.state()
        engine.step(state_in, sm, control, c, dt)
        lm = np.dot(w, sm.body_qd.numpy().flatten())

        grad_fd[dof] = (lp - lm) / (2 * eps)

    return grad_analytical, grad_fd


def test_straight_drive():
    """All wheels at same velocity — sticking friction."""
    print("\n=== Helhest: Straight drive (all wheels same velocity) ===")

    model = build_helhest()
    engine = make_engine(model)
    dims = engine.dims

    np.random.seed(42)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = 5.0

    grad_a, grad_fd = compute_wheel_vel_gradient(model, engine, target_vel, w)

    wheel_names = ["left", "right", "rear"]
    max_err = 0.0
    for i, dof in enumerate(range(WHEEL_DOF_OFFSET, WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS)):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  {wheel_names[i]:8s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_differential_turn():
    """Left and right wheels at different speeds — turning."""
    print("\n=== Helhest: Differential turn ===")

    model = build_helhest()
    engine = make_engine(model)
    dims = engine.dims

    np.random.seed(123)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[WHEEL_DOF_OFFSET + 0] = 2.0   # left slow
    target_vel[WHEEL_DOF_OFFSET + 1] = 10.0  # right fast
    target_vel[WHEEL_DOF_OFFSET + 2] = 0.0   # rear off

    grad_a, grad_fd = compute_wheel_vel_gradient(model, engine, target_vel, w)

    wheel_names = ["left", "right", "rear"]
    max_err = 0.0
    for i, dof in enumerate(range(WHEEL_DOF_OFFSET, WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS)):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  {wheel_names[i]:8s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.5, f"Failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_multi_step():
    """Multi-step trajectory (5 steps) — gradient through multiple contacts."""
    print("\n=== Helhest: Multi-step trajectory (5 steps) ===")

    model = build_helhest()
    engine = make_engine(model, sim_steps=5)
    dims = engine.dims

    np.random.seed(77)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = 3.0

    # For multi-step, run full trajectory
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    wp.copy(
        control.joint_target_vel,
        wp.array(target_vel.reshape(1, -1), dtype=wp.float32, device=model.device),
    )

    dt = 0.01
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=5, device=model.device,
    )

    # Forward
    state_out = model.state()
    for step in range(5):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        buffer.save_step(step, engine.data, engine.axion_contacts)
        state_in, state_out = state_out, state_in

    # Set loss gradient at final velocity
    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[5],
        wp.array(w.reshape(engine.data.body_vel_grad.numpy().shape),
                 dtype=wp.spatial_vector, device=model.device),
    )

    # Backward
    for step in range(4, -1, -1):
        buffer.load_step(step, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        if step > 0:
            # Propagate pose and vel gradients to previous step
            buffer.save_gradients(step, engine.data)

    grad_a = engine.data.joint_target_vel.grad.numpy().flatten().copy()

    # FD
    eps = 1e-4
    grad_fd = np.zeros(dims.joint_dof_count, dtype=np.float32)
    for dof in range(WHEEL_DOF_OFFSET, WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS):
        for sign, tv_arr in [(1, None), (-1, None)]:
            tv = target_vel.copy()
            tv[dof] += sign * eps
            wp.copy(control.joint_target_vel,
                    wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))

            s_in = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
            s_out = model.state()
            for step in range(5):
                c = model.collide(s_in)
                engine.step(s_in, s_out, control, c, dt)
                s_in, s_out = s_out, s_in

            loss_val = np.dot(w, s_in.body_qd.numpy().flatten())
            if sign == 1:
                lp = loss_val
            else:
                lm = loss_val

        grad_fd[dof] = (lp - lm) / (2 * eps)

    wheel_names = ["left", "right", "rear"]
    max_err = 0.0
    for i, dof in enumerate(range(WHEEL_DOF_OFFSET, WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS)):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  {wheel_names[i]:8s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Failed: max rel error {max_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_straight_drive()
    test_differential_turn()
    test_multi_step()
    print("\n=== All Helhest gradient tests done! ===")
