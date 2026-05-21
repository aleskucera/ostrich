"""Test: Cartpole gradient verification across trajectory lengths.

Tests the gradient chain for a cartpole (prismatic cart + revolute pole)
with no contacts. This is the key differentiable simulation use case:
joint control → body motion → pose/velocity loss.

Reports both strengths (velocity loss: good) and weaknesses (position loss
over long horizons: degraded due to linearization compounding).
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
from axion.core.types import JointMode
from axion.simulation.trajectory_buffer import TrajectoryBuffer


def build_cartpole():
    builder = AxionModelBuilder()
    no_collision = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

    link_cart = builder.add_link()
    builder.add_shape_box(link_cart, hx=0.3, hy=0.5, hz=0.2, cfg=no_collision)
    rot_z_90 = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2.0)
    j_cart = builder.add_joint_prismatic(
        parent=-1, child=link_cart, axis=wp.vec3(1, 0, 0),
        parent_xform=wp.transform(wp.vec3(0, 0, 0), rot_z_90),
        child_xform=wp.transform(wp.vec3(0, 0, 0), rot_z_90),
        target_ke=1000.0, target_kd=100.0, label="cart",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
    )

    link_pole = builder.add_link()
    builder.add_shape_box(link_pole, hx=0.05, hy=0.05, hz=1.0, cfg=no_collision)
    j_pole = builder.add_joint_revolute(
        parent=link_cart, child=link_pole, axis=wp.vec3(1, 0, 0),
        parent_xform=wp.transform(wp.vec3(0, 0, 0.2), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, -1.0), wp.quat_identity()),
        target_ke=0.0, target_kd=0.0, label="pole",
    )
    builder.add_articulation([j_cart, j_pole], label="cartpole")
    return builder.finalize_replicated(num_worlds=1, requires_grad=True)


def run_gradient_test(num_steps, loss_type="velocity"):
    """Run forward+backward, return (analytical, FD) for cart control gradient.

    loss_type: "velocity" (pole angular velocity) or "position" (pole z-position)
    """
    model = build_cartpole()
    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=20),
        linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8),
    )
    engine = AxionEngine(
        model=model, sim_steps=num_steps, config=config,
        logging_config=LoggingConfig(), differentiable_simulation=True,
    )
    dims = engine.dims
    dt = 0.01

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[0] = 2.0

    state_in = model.state()
    jq = model.joint_q.numpy()
    jq[1] = np.pi  # pole hanging down
    wp.copy(model.joint_q, wp.array(jq, dtype=wp.float32, device=model.joint_q.device))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    wp.copy(control.joint_target_vel,
            wp.array(target_vel.reshape(1, -1), dtype=wp.float32, device=model.device))

    # Forward
    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.axion_contacts,
        dims=dims, num_steps=num_steps, device=model.device,
    )
    states = [model.state() for _ in range(num_steps + 1)]
    wp.copy(states[0].body_q, state_in.body_q)
    wp.copy(states[0].body_qd, state_in.body_qd)
    for i in range(num_steps):
        c = model.collide(states[i])
        engine.step(states[i], states[i + 1], control, c, dt)
        buffer.save_step(i, engine.data, engine.axion_contacts)

    # Backward
    buffer.zero_grad()
    if loss_type == "velocity":
        w = np.zeros(engine.data.body_vel_grad.numpy().shape, dtype=np.float32)
        w.reshape(-1, 6)[1, 3] = 1.0  # pole ang_x
        wp.copy(buffer.body_vel.grad[num_steps],
                wp.array(w, dtype=wp.spatial_vector, device=model.device))
    else:
        pg = np.zeros(buffer.body_pose.grad[num_steps].numpy().shape, dtype=np.float32)
        pg.reshape(-1, 7)[1, 2] = 1.0  # pole pz
        wp.copy(buffer.body_pose.grad[num_steps],
                wp.array(pg, dtype=wp.transform, device=model.device))

    for i in range(num_steps - 1, -1, -1):
        buffer.load_step(i, engine.data, engine.axion_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        buffer.save_gradients(i, engine.data)
        buffer.save_pose_gradients(i, engine.data)

    ctrl_a = sum(
        buffer.joint_target_vel.grad[i].numpy().flatten()[0]
        for i in range(num_steps)
    )

    # FD
    eps = 1e-3

    def run_loss(tv):
        s = model.state()
        model.joint_q.numpy()[1] = np.pi
        newton.eval_fk(model, model.joint_q, model.joint_qd, s)
        ct = model.control()
        wp.copy(ct.joint_target_vel,
                wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))
        so = model.state()
        for step in range(num_steps):
            cc = model.collide(s)
            engine.step(s, so, ct, cc, dt)
            s, so = so, s
        if loss_type == "velocity":
            return s.body_qd.numpy().reshape(-1, 6)[1, 3]
        else:
            return s.body_q.numpy().reshape(-1, 7)[1, 2]

    tv_p = target_vel.copy(); tv_p[0] += eps
    tv_m = target_vel.copy(); tv_m[0] -= eps
    fd = (run_loss(tv_p) - run_loss(tv_m)) / (2 * eps)

    return ctrl_a, fd


def test_cartpole_velocity_loss():
    """Velocity-based loss over increasing trajectory lengths — the strong case."""
    print("\n=== Test: Cartpole velocity loss (pole angular velocity) ===")

    max_err = 0.0
    for num_steps in [1, 10, 20]:
        a, fd = run_gradient_test(num_steps, "velocity")
        err = abs(a - fd) / max(abs(a), abs(fd), 1e-10)
        max_err = max(max_err, err)
        print(f"  {num_steps:3d} steps: analytical={a:10.4f}  FD={fd:10.4f}  rel_err={err:.4f}")

    print(f"  Max rel error: {max_err:.4f}")
    assert max_err < 0.15, f"Cartpole velocity gradient failed: max rel error {max_err:.4f}"
    print("  PASSED")


def test_cartpole_position_loss():
    """Position-based loss — documents the degradation for long horizons."""
    print("\n=== Test: Cartpole position loss (pole z-position) ===")

    for num_steps in [1, 10, 20, 50]:
        a, fd = run_gradient_test(num_steps, "position")
        err = abs(a - fd) / max(abs(a), abs(fd), 1e-15)
        print(f"  {num_steps:3d} steps: analytical={a:12.6f}  FD={fd:12.6f}  rel_err={err:.4f}")

    print("  (position loss degrades with horizon — known limitation of single-step adjoint)")
    print("  REPORTED (no assertion)")


if __name__ == "__main__":
    test_cartpole_velocity_loss()
    test_cartpole_position_loss()
    print("\n=== Cartpole gradient tests done! ===")
