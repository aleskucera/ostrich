"""Test: Wheeled robot (Taros-4) gradient verification.

Tests dL/d(wheel_target_vel) for a 4-wheeled robot with sticking friction contacts.
This is the primary use case: torque → wheel → friction → chassis motion.

Also tests differential drive (left/right velocity difference) for turning,
which involves a mix of sticking and sliding contacts.
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
from taros_4.common import create_taros4_model


def build_taros4(control_mode="velocity", k_p=1000.0, k_d=0.0, friction=0.8):
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()
    chassis, wheels = create_taros4_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()),
        is_visible=False,
        control_mode=control_mode,
        k_p=k_p,
        k_d=k_d,
        friction=friction,
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


def get_contact_modes(engine):
    """Returns (n_sticking, n_sliding, n_inactive) for friction contacts."""
    dims = engine.dims
    active_n = engine.data.constr_active_mask.full.numpy()[0, dims.slice_n]
    lam_n = engine.data._constr_force.numpy()[0, dims.offset_n:dims.offset_f]
    lam_f = engine.data._constr_force.numpy()[0, dims.offset_f:dims.offset_f + 2 * dims.contact_count]
    shapes0 = engine.axion_contacts.contact_shape0.numpy()[0]
    shapes1 = engine.axion_contacts.contact_shape1.numpy()[0]
    mu_arr = engine.axion_model.shape_material_mu.numpy()[0]
    cc = engine.axion_contacts.contact_count.numpy()[0]

    n_stick, n_slide, n_inactive = 0, 0, 0
    for ci in range(cc):
        if active_n[ci] > 0:
            ln = lam_n[ci]
            lt1 = lam_f[2 * ci]
            lt2 = lam_f[2 * ci + 1]
            lf = np.sqrt(lt1 ** 2 + lt2 ** 2)
            mu = 0.5 * (mu_arr[shapes0[ci]] + mu_arr[shapes1[ci]])
            ratio = lf / (mu * ln + 1e-10)
            if ratio > 0.9:
                n_slide += 1
            else:
                n_stick += 1
        else:
            n_inactive += 1
    return n_stick, n_slide, n_inactive


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
    for dof in range(6, dims.joint_dof_count):
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

    return grad_analytical, grad_fd, get_contact_modes(engine)


def test_straight_drive():
    """All wheels at same velocity — pure sticking friction."""
    print("\n=== Test: Straight drive (all wheels same velocity) ===")

    model = build_taros4()
    engine = make_engine(model)
    dims = engine.dims

    np.random.seed(42)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[6:] = 5.0  # all 4 wheels at 5 rad/s

    grad_a, grad_fd, (n_stick, n_slide, n_inact) = compute_wheel_vel_gradient(
        model, engine, target_vel, w
    )

    print(f"  Contacts: {n_stick} sticking, {n_slide} sliding, {n_inact} inactive")

    max_err = 0.0
    for dof in range(6, dims.joint_dof_count):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  wheel DOF {dof-6}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Straight drive gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_differential_turn():
    """Left wheels slow, right wheels fast — differential turning with mix of sticking/sliding."""
    print("\n=== Test: Differential turn (left slow, right fast) ===")

    model = build_taros4(friction=0.5)
    engine = make_engine(model)
    dims = engine.dims

    np.random.seed(123)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    # DOFs: 6=front_left, 7=front_right, 8=rear_left, 9=rear_right
    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[6] = 2.0   # front left (slow)
    target_vel[7] = 10.0  # front right (fast)
    target_vel[8] = 2.0   # rear left (slow)
    target_vel[9] = 10.0  # rear right (fast)

    grad_a, grad_fd, (n_stick, n_slide, n_inact) = compute_wheel_vel_gradient(
        model, engine, target_vel, w
    )

    print(f"  Contacts: {n_stick} sticking, {n_slide} sliding, {n_inact} inactive")

    max_err = 0.0
    for dof in range(6, dims.joint_dof_count):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        wheel_names = ["front_left", "front_right", "rear_left", "rear_right"]
        print(f"  {wheel_names[dof-6]:12s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    # Looser tolerance for differential turn — sliding contacts are approximate
    assert max_err < 0.5, f"Differential turn gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_hard_turn():
    """Extreme differential — one side stopped, other at high speed. Should create sliding."""
    print("\n=== Test: Hard turn (left stopped, right fast) ===")

    model = build_taros4(friction=0.3)
    engine = make_engine(model)
    dims = engine.dims

    np.random.seed(99)
    w = np.random.randn(model.body_count * 6).astype(np.float32)

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[6] = 0.0    # front left stopped
    target_vel[7] = 20.0   # front right fast
    target_vel[8] = 0.0    # rear left stopped
    target_vel[9] = 20.0   # rear right fast

    grad_a, grad_fd, (n_stick, n_slide, n_inact) = compute_wheel_vel_gradient(
        model, engine, target_vel, w
    )

    print(f"  Contacts: {n_stick} sticking, {n_slide} sliding, {n_inact} inactive")

    max_err = 0.0
    wheel_names = ["front_left", "front_right", "rear_left", "rear_right"]
    for dof in range(6, dims.joint_dof_count):
        a, f = grad_a[dof], grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  {wheel_names[dof-6]:12s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.5, f"Hard turn gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_multi_step_trajectory():
    """Multi-step gradient accumulation through the trajectory buffer."""
    print("\n=== Test: Multi-step trajectory (5 steps) ===")

    model = build_taros4()
    num_steps = 5
    engine = make_engine(model, sim_steps=num_steps)
    dims = engine.dims
    dt = 0.01

    np.random.seed(42)
    w_terminal = np.random.randn(model.body_count * 6).astype(np.float32)
    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[6:] = 5.0

    # --- Analytical via trajectory buffer backward ---
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    wp.copy(
        control.joint_target_vel,
        wp.array(target_vel.reshape(1, -1), dtype=wp.float32, device=model.device),
    )

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

    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[num_steps],
        wp.array(w_terminal.reshape(engine.data.body_vel_grad.numpy().shape),
                 dtype=wp.spatial_vector, device=model.device),
    )
    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)

    ctrl_grad_a = np.zeros(dims.joint_dof_count, dtype=np.float32)
    for i in range(num_steps):
        ctrl_grad_a += buffer.joint_target_vel.grad[i].numpy().flatten()

    # --- FD for all wheel DOFs ---
    eps = 1e-3
    ctrl_grad_fd = np.zeros(dims.joint_dof_count, dtype=np.float32)
    for dof in range(6, dims.joint_dof_count):
        def run_traj(tv):
            s_in = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
            ctrl = model.control()
            wp.copy(ctrl.joint_target_vel,
                    wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))
            s_out = model.state()
            for step in range(num_steps):
                c = model.collide(s_in)
                engine.step(s_in, s_out, ctrl, c, dt)
                s_in, s_out = s_out, s_in
            return np.dot(w_terminal, s_in.body_qd.numpy().flatten())

        tv_p = target_vel.copy(); tv_p[dof] += eps
        tv_m = target_vel.copy(); tv_m[dof] -= eps
        ctrl_grad_fd[dof] = (run_traj(tv_p) - run_traj(tv_m)) / (2 * eps)

    wheel_names = ["front_left", "front_right", "rear_left", "rear_right"]
    max_err = 0.0
    for dof in range(6, dims.joint_dof_count):
        a, f = ctrl_grad_a[dof], ctrl_grad_fd[dof]
        err = abs(a - f) / max(abs(a), abs(f), 1e-10)
        max_err = max(max_err, err)
        print(f"  {wheel_names[dof - 6]:12s}: analytical={a:10.4f}  FD={f:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Multi-step trajectory gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_straight_drive()
    test_differential_turn()
    test_hard_turn()
    test_multi_step_trajectory()
    print("\n=== All wheeled robot tests done! ===")
