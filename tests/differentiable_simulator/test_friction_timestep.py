"""
Test: Does friction/driving behavior change with timestep?

Uses exact Taros-4 setup from working gradient tests.
Drives all wheels at constant speed, measures chassis motion after 1s.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from taros_4.common import create_taros4_model


def run_drive(dt, drive_time=1.0, settle_time=0.5, wheel_speed=5.0,
              k_p=1000.0, newton_iters=16, linear_iters=16):
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()
    create_taros4_model(
        builder,
        xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()),
        is_visible=False, control_mode="velocity",
        k_p=k_p, k_d=0.0, friction=0.8,
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    settle_steps = max(1, int(settle_time / dt))
    drive_steps = max(1, int(drive_time / dt))
    total_steps = settle_steps + drive_steps

    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=newton_iters),
        linear=LinearSolverConfig(max_iters=linear_iters),
    )
    engine = AxionEngine(
        model=model, sim_steps=total_steps, config=config,
        logging_config=LoggingConfig(),
    )

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    zero_ctrl = model.control()
    drive_ctrl = model.control()
    tv = np.zeros(model.joint_dof_count, dtype=np.float32)
    tv[6:] = wheel_speed
    wp.copy(drive_ctrl.joint_target_vel,
            wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))

    for step in range(total_steps):
        ctrl = zero_ctrl if step < settle_steps else drive_ctrl
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        state_in, state_out = state_out, state_in

    q = state_in.body_q.numpy().reshape(-1, 7)
    qd = state_in.body_qd.numpy().reshape(-1, 6)
    return q[0, 0], qd[0, 0]  # chassis x position and x velocity


print("=" * 70)
print("Taros-4: constant wheel_speed=5, settle 0.5s then drive 1.0s")
print("=" * 70)
print(f"{'dt':>8} {'drive_steps':>12} {'chassis_x':>12} {'chassis_vx':>12}")
print("-" * 50)

for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
    cx, cvx = run_drive(dt)
    drive_steps = max(1, int(1.0 / dt))
    print(f"{dt:8.4f} {drive_steps:12d} {cx:12.4f} {cvx:12.4f}")

print()
print("=" * 70)
print("Effect of Newton iterations at large dt")
print("=" * 70)
print(f"{'dt':>8} {'iters':>8} {'chassis_x':>12} {'chassis_vx':>12}")
print("-" * 45)

for dt in [0.05, 0.1]:
    for ni in [16, 32, 64, 128]:
        cx, cvx = run_drive(dt, newton_iters=ni, linear_iters=ni)
        print(f"{dt:8.4f} {ni:8d} {cx:12.4f} {cvx:12.4f}")

print()
print("=" * 70)
print("Effect of ke at different dt")
print("=" * 70)
print(f"{'dt':>8} {'k_p':>8} {'chassis_x':>12} {'chassis_vx':>12}")
print("-" * 45)

for dt in [0.01, 0.03, 0.05, 0.1]:
    for kp in [100, 500, 1000, 5000]:
        cx, cvx = run_drive(dt, k_p=float(kp))
        print(f"{dt:8.4f} {kp:8d} {cx:12.4f} {cvx:12.4f}")


# ─── Slope tests ────────────────────────────────────────────────────────────


def run_drive_on_slope(dt, slope_deg=20.0, drive_time=1.0, settle_time=0.5,
                       wheel_speed=5.0, k_p=1000.0, newton_iters=16, linear_iters=16,
                       use_heightfield=False):
    """Drive Taros-4 uphill on a tilted surface."""
    slope_rad = np.radians(slope_deg)

    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05

    if use_heightfield:
        # Heightfield ramp
        ncol, nrow = 30, 12
        length = 20.0
        z_per_m = np.tan(slope_rad)
        x = np.linspace(0.0, length, ncol)
        z = np.array([z_per_m * xi for xi in x], dtype=np.float32)
        data = np.tile(z, (nrow, 1))
        hf = newton.Heightfield(
            data=data, nrow=nrow, ncol=ncol,
            hx=length / 2.0, hy=5.0,
            min_z=float(z.min()), max_z=float(z.max()),
        )
        builder.add_shape_heightfield(
            heightfield=hf,
            xform=wp.transform(wp.vec3(length / 2.0, 0.0, 0.0), wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8),
            label="slope",
        )
        start_x = 3.0
        start_z = z_per_m * start_x + 0.8
    else:
        # Tilted flat box
        ground_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-slope_rad))
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(5.0, 0.0, -0.25), ground_rot),
            hx=15.0, hy=5.0, hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8),
        )
        start_x = 3.0
        start_z = np.tan(slope_rad) * start_x + 0.8

    create_taros4_model(
        builder,
        xform=wp.transform(wp.vec3(start_x, 0, start_z), wp.quat_identity()),
        is_visible=False, control_mode="velocity",
        k_p=k_p, k_d=0.0, friction=0.8,
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    settle_steps = max(1, int(settle_time / dt))
    drive_steps = max(1, int(drive_time / dt))
    total_steps = settle_steps + drive_steps

    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=newton_iters),
        linear=LinearSolverConfig(max_iters=linear_iters),
    )
    engine = AxionEngine(
        model=model, sim_steps=total_steps, config=config,
        logging_config=LoggingConfig(),
    )

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    zero_ctrl = model.control()
    drive_ctrl = model.control()
    tv = np.zeros(model.joint_dof_count, dtype=np.float32)
    tv[6:] = wheel_speed
    wp.copy(drive_ctrl.joint_target_vel,
            wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))

    for step in range(total_steps):
        ctrl = zero_ctrl if step < settle_steps else drive_ctrl
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        state_in, state_out = state_out, state_in

    q = state_in.body_q.numpy().reshape(-1, 7)
    qd = state_in.body_qd.numpy().reshape(-1, 6)
    chassis_z = q[0, 2]
    chassis_vx = qd[0, 0]
    return chassis_z, chassis_vx


print()
print("=" * 70)
print("SLOPE TEST: Taros-4 on 20° tilted BOX (no heightfield)")
print("=" * 70)
print(f"{'dt':>8} {'chassis_z':>12} {'chassis_vx':>12}")
print("-" * 35)

for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    cz, cvx = run_drive_on_slope(dt, use_heightfield=False)
    print(f"{dt:8.4f} {cz:12.4f} {cvx:12.4f}")

print()
print("=" * 70)
print("SLOPE TEST: Taros-4 on 20° HEIGHTFIELD")
print("=" * 70)
print(f"{'dt':>8} {'chassis_z':>12} {'chassis_vx':>12}")
print("-" * 35)

for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    cz, cvx = run_drive_on_slope(dt, use_heightfield=True)
    print(f"{dt:8.4f} {cz:12.4f} {cvx:12.4f}")
