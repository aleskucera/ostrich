"""Interactive semi-implicit simulation of Helhest for parameter tuning.

The semi-implicit solver uses penalty-based contacts (ke, kd, kf) and explicit
PD control (target_ke, target_kd). Very different parameter regime from Axion.

Usage:
    python examples/comparison_gradient/helhest/parameter_sweep/tune_semi_implicit.py

    python examples/comparison_gradient/helhest/parameter_sweep/tune_semi_implicit.py \
        --ground-truth results/right_turn_b.json

    python examples/comparison_gradient/helhest/parameter_sweep/tune_semi_implicit.py \
        --ke 5000 --kd 200 --kf 2000 --k-p 100 --dt 0.001
"""
import argparse
import json
import os
from typing import override

import newton
import numpy as np
import warp as wp
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig

from examples.helhest.common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"


@wp.kernel
def set_control_from_sequence_kernel(
    step_idx: wp.array(dtype=wp.int32),
    ctrl_sequence: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32),
    num_steps: int,
):
    dof = wp.tid()
    s = step_idx[0]
    if s >= num_steps:
        s = num_steps - 1
    joint_target_vel[dof] = ctrl_sequence[s, dof]


@wp.kernel
def increment_step_kernel(
    step_idx: wp.array(dtype=wp.int32),
):
    step_idx[0] = step_idx[0] + 1


class HelhestSemiImplicitSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: SemiImplicitEngineConfig,
        logging_config: LoggingConfig,
        target_ctrl: list[float],
        k_p: float = 100.0,
        k_d: float = 10.0,
        mu: float = 0.5,
        ke: float = 2500.0,
        kd: float = 100.0,
        kf: float = 1000.0,
        gt_trajectory: list[dict] | None = None,
        wheel_timeseries: list[dict] | None = None,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._mu = mu
        self._ke = ke
        self._kd = kd
        self._kf = kf
        self._gt_trajectory = gt_trajectory
        self._gt_lines_initialized = False
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        num_dofs = 9
        total_steps = self.clock.total_sim_steps

        # Build per-step control sequence on GPU
        if wheel_timeseries:
            ts_t = np.array([p["t"] for p in wheel_timeseries], dtype=np.float32)
            ts_left = np.array([p["left"] for p in wheel_timeseries], dtype=np.float32)
            ts_right = np.array([p["right"] for p in wheel_timeseries], dtype=np.float32)
            ts_rear = np.array([p["rear"] for p in wheel_timeseries], dtype=np.float32)

            ctrl_np = np.zeros((total_steps, num_dofs), dtype=np.float32)
            for i in range(total_steps):
                t = (i + 1) * self.clock.dt
                ctrl_np[i, 6] = float(np.interp(t, ts_t, ts_left))
                ctrl_np[i, 7] = float(np.interp(t, ts_t, ts_right))
                ctrl_np[i, 8] = float(np.interp(t, ts_t, ts_rear))
        else:
            ctrl_np = np.zeros((total_steps, num_dofs), dtype=np.float32)
            ctrl_np[:, 6] = target_ctrl[0]
            ctrl_np[:, 7] = target_ctrl[1]
            ctrl_np[:, 8] = target_ctrl[2]

        self._ctrl_sequence = wp.array(ctrl_np, dtype=wp.float32)
        self._ctrl_step_idx = wp.zeros(1, dtype=wp.int32)
        self._ctrl_total_steps = total_steps

    @override
    def init_state_fn(self, current_state, next_state, contacts, dt):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state):
        wp.launch(
            kernel=set_control_from_sequence_kernel,
            dim=9,
            inputs=[
                self._ctrl_step_idx,
                self._ctrl_sequence,
                self.control.joint_target_vel,
                self._ctrl_total_steps,
            ],
        )
        wp.launch(
            kernel=increment_step_kernel,
            dim=1,
            inputs=[self._ctrl_step_idx],
        )

    def _render(self, segment_num: int):
        if self._gt_trajectory and not self._gt_lines_initialized:
            pts = self._gt_trajectory
            n = len(pts) - 1
            if n > 0:
                z = 0.05
                starts_np = np.array([[p["x"], p["y"], z] for p in pts[:-1]], dtype=np.float32)
                ends_np = np.array([[p["x"], p["y"], z] for p in pts[1:]], dtype=np.float32)
                starts = wp.array(starts_np, dtype=wp.vec3)
                ends = wp.array(ends_np, dtype=wp.vec3)
                self.viewer.log_lines("gt_trajectory", starts, ends, (1.0, 0.0, 0.0), width=0.02)
            self._gt_lines_initialized = True
        super()._render(segment_num)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self._mu,
            ke=self._ke,
            kd=self._kd,
            kf=self._kf,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=self._k_p,
            k_d=self._k_d,
            friction_left_right=self._mu,
            friction_rear=self._mu,
            ke=self._ke,
            kd=self._kd,
            kf=self._kf,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to extracted rosbag JSON",
    )
    parser.add_argument(
        "--ctrl",
        type=float,
        nargs=3,
        default=None,
        metavar=("LEFT", "RIGHT", "REAR"),
        help="Wheel velocities in rad/s",
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep (default: 0.001)")
    parser.add_argument("--k-p", type=float, default=0.0, help="Velocity servo gain (target_ke)")
    parser.add_argument(
        "--k-d", type=float, default=400.0, help="Velocity servo damping (target_kd)"
    )
    parser.add_argument("--mu", type=float, default=0.01, help="Friction coefficient")
    parser.add_argument("--ke", type=float, default=8000.0, help="Contact elastic stiffness")
    parser.add_argument("--kd", type=float, default=2000.0, help="Contact damping")
    parser.add_argument("--kf", type=float, default=800.0, help="Friction damping")
    parser.add_argument("--friction-smoothing", type=float, default=0.1)
    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # Determine target_ctrl, duration, and ground truth trajectory
    gt_trajectory = None
    wheel_timeseries = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt = json.load(f)
        target_ctrl = gt["target_ctrl_rad_s"]
        duration = args.duration or gt["trajectory"].get(
            "constant_speed_duration_s", gt["trajectory"]["duration_s"]
        )
        gt_trajectory = gt["trajectory"]["points"]
        if "wheel_velocities" in gt and "timeseries" in gt["wheel_velocities"]:
            wheel_timeseries = gt["wheel_velocities"]["timeseries"]
            print(f"  Using time-varying wheel velocities ({len(wheel_timeseries)} points)")
        print(f"Ground truth: {gt.get('bag_name', '?')}")
    elif args.ctrl:
        target_ctrl = args.ctrl
        duration = args.duration or 10.0
    else:
        target_ctrl = [1.829, 0.984, 1.395]
        duration = args.duration or 10.0

    print(f"  ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f} rad/s")
    print(f"  dt={args.dt}, k_p={args.k_p}, k_d={args.k_d}, mu={args.mu}")
    print(f"  ke={args.ke}, kd={args.kd}, kf={args.kf}")

    sim_config = SimulationConfig(
        duration_seconds=duration,
        target_timestep_seconds=args.dt,
        num_worlds=1,
    )
    render_config = RenderingConfig(
        vis_type="null" if args.headless else "gl",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=1,
    )
    engine_config = SemiImplicitEngineConfig(
        angular_damping=args.angular_damping,
        friction_smoothing=args.friction_smoothing,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = HelhestSemiImplicitSim(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        target_ctrl=target_ctrl,
        k_p=args.k_p,
        k_d=args.k_d,
        mu=args.mu,
        ke=args.ke,
        kd=args.kd,
        kf=args.kf,
        gt_trajectory=gt_trajectory,
        wheel_timeseries=wheel_timeseries,
    )

    # Hide the UI panel for a cleaner view
    if hasattr(sim.viewer, "show_ui"):
        sim.viewer.show_ui = False

    # Camera: auto-pick view based on trajectory extent
    if hasattr(sim.viewer, "set_camera"):
        if gt_trajectory:
            xs = [p["x"] for p in gt_trajectory]
            ys = [p["y"] for p in gt_trajectory]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            cx = (max(xs) + min(xs)) / 2.0
            cy = (max(ys) + min(ys)) / 2.0
            cam_h = max(x_range, y_range, 4.0) * 2.0
            sim.viewer.set_camera(pos=wp.vec3(cx, cy, cam_h), pitch=-90.0, yaw=90.0)
        else:
            sim.viewer.set_camera(pos=wp.vec3(0.0, -1.5, 8.0), pitch=-90.0, yaw=90.0)

    sim.run()


if __name__ == "__main__":
    main()
