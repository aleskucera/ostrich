"""
Rigorous friction accuracy tests.

Uses the 4-wheel robot from uphill_drive.py on a tilted flat surface.
The robot drops and settles for 1s, then we measure friction behavior for 1s.
Settling eliminates the initial contact transient.
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
from differentiable.uphill_drive import create_4wheel_robot, WHEEL_RADIUS


def run_robot_on_slope(
    dt, slope_deg=20.0, mu=0.5, wheel_speed=0.0, k_p=500.0,
    settle_time=1.0, drive_time=1.0, newton_iters=16, linear_iters=16,
):
    """4-wheel robot on tilted flat surface. Settle, then optionally drive."""
    slope_rad = np.radians(slope_deg)

    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05

    ground_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-slope_rad))
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), ground_rot),
        hx=30.0, hy=10.0, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
    )

    start_z = WHEEL_RADIUS + 0.5
    create_4wheel_robot(builder, xform=wp.transform(wp.vec3(0.0, 0.0, start_z), wp.quat_identity()),
                        is_visible=False, k_p=k_p, k_d=0.0, wheel_mu=mu)

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    settle_steps = max(1, int(settle_time / dt))
    drive_steps = max(1, int(drive_time / dt))
    total_steps = settle_steps + drive_steps

    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=newton_iters),
        linear=LinearSolverConfig(max_iters=linear_iters),
    )
    engine = AxionEngine(model=model, sim_steps=total_steps, config=config, logging_config=LoggingConfig())

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    zero_ctrl = model.control()
    drive_ctrl = model.control()
    if wheel_speed != 0.0:
        tv = np.zeros(model.joint_dof_count, dtype=np.float32)
        tv[8:10] = wheel_speed
        wp.copy(drive_ctrl.joint_target_vel, wp.array(tv.reshape(1, -1), dtype=wp.float32, device=model.device))

    settle_pos = None
    for step in range(total_steps):
        ctrl = zero_ctrl if step < settle_steps else drive_ctrl
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, ctrl, contacts, dt)
        if step == settle_steps - 1:
            settle_pos = state_out.body_q.numpy().reshape(-1, 7)[0, :3].copy()
        state_in, state_out = state_out, state_in

    final_pos = state_in.body_q.numpy().reshape(-1, 7)[0, :3]
    final_vel = state_in.body_qd.numpy().reshape(-1, 6)[0, :3]
    disp = np.linalg.norm(final_pos - settle_pos)
    speed = np.linalg.norm(final_vel)
    return disp, speed


def test_static_sticking():
    """Robot on 20° slope, no driving. Should not slide."""
    print("=" * 80)
    print("TEST 1: Static friction — robot on 20° slope, no driving")
    print("   mu=0.5, critical=26.57°. Settle 1s, measure 1s.")
    print("=" * 80)
    print(f"{'dt':>8} {'leakage (m/s)':>16} {'post-settle disp':>18}")
    print("-" * 45)
    for dt in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
        disp, speed = run_robot_on_slope(dt, slope_deg=20.0, mu=0.5)
        print(f"{dt:8.4f} {speed:16.8f} {disp:18.8f}")


def test_static_threshold():
    """Sweep slope angle across Coulomb boundary."""
    print()
    print("=" * 80)
    print("TEST 2: Static friction threshold (mu=0.5, critical=26.57°, dt=0.03)")
    print("=" * 80)
    critical = np.degrees(np.arctan(0.5))
    print(f"{'angle':>8} {'speed (m/s)':>14} {'expected':>10} {'actual':>10} {'correct':>8}")
    print("-" * 55)
    for angle in [10, 15, 20, 24, 25, 26, 27, 28, 30, 35]:
        disp, speed = run_robot_on_slope(0.03, slope_deg=angle, mu=0.5)
        expected = "STICK" if angle < critical else "SLIDE"
        actual = "STICK" if speed < 0.05 else "SLIDE"
        correct = "OK" if expected == actual else "WRONG"
        print(f"{angle:8.1f} {speed:14.6f} {expected:>10} {actual:>10} {correct:>8}")


def test_driven_flat():
    """Driven on flat ground — should be dt-independent."""
    print()
    print("=" * 80)
    print("TEST 3: Driven robot on flat ground (wheel_speed=5, mu=0.8)")
    print("=" * 80)
    print(f"{'dt':>8} {'disp':>14} {'speed':>14}")
    print("-" * 40)
    for dt in [0.01, 0.02, 0.03, 0.05, 0.1]:
        disp, speed = run_robot_on_slope(dt, slope_deg=0.0, mu=0.8, wheel_speed=5.0)
        print(f"{dt:8.4f} {disp:14.4f} {speed:14.4f}")


def test_driven_slope():
    """Driven on 20° slope — check dt consistency."""
    print()
    print("=" * 80)
    print("TEST 4: Driven robot on 20° slope (wheel_speed=10, mu=0.8)")
    print("=" * 80)
    print(f"{'dt':>8} {'disp':>14} {'speed':>14}")
    print("-" * 40)
    for dt in [0.01, 0.02, 0.03, 0.05, 0.1]:
        disp, speed = run_robot_on_slope(dt, slope_deg=20.0, mu=0.8, wheel_speed=10.0)
        print(f"{dt:8.4f} {disp:14.4f} {speed:14.4f}")


def test_convergence():
    """Does more iterations reduce leakage?"""
    print()
    print("=" * 80)
    print("TEST 5: Convergence (20° slope, no driving, dt=0.03)")
    print("=" * 80)
    print(f"  {'newton_iters':>14} {'linear_iters':>14} {'leakage (m/s)':>16}")
    print("  " + "-" * 48)
    for ni in [8, 16, 32, 64]:
        for li in [16, 64]:
            disp, speed = run_robot_on_slope(0.03, slope_deg=20.0, mu=0.5,
                                             newton_iters=ni, linear_iters=li)
            print(f"  {ni:14d} {li:14d} {speed:16.8f}")


if __name__ == "__main__":
    test_static_sticking()
    test_static_threshold()
    test_driven_flat()
    test_driven_slope()
    test_convergence()
