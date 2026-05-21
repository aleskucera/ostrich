"""Interactive Axion simulation of Helhest with manually specified parameters.

No Hydra — configs are built directly from dataclasses.

Usage:
    # Run with real robot wheel velocities from a rosbag:
    python examples/comparison_accuracy/helhest/sweep_axion2.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json

    # Override parameters:
    python examples/comparison_accuracy/helhest/sweep_axion2.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json \
        --k-p 500 --friction-lr 0.5 --friction-rear 0.2

    # Manual wheel velocities (no ground truth file):
    python examples/comparison_accuracy/helhest/sweep_axion2.py \
        --ctrl 1.83 0.98 1.39

    # Headless (no viewer):
    python examples/comparison_accuracy/helhest/sweep_axion2.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json --headless
"""
import argparse
import json
import os
from typing import override

import newton
import numpy as np
import warp as wp
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig

from examples.helhest.common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"


@wp.kernel
def set_friction_coefficient_kernel(
    mu: float,
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, shape_idx = wp.tid()
    shape_material_mu[world_idx, shape_idx] = mu


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


class HelhestSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        target_ctrl: list[float],
        k_p: float = 150.0,
        k_d: float = 0.0,
        mu: float = 0.1,
        gt_trajectory: list[dict] | None = None,
        wheel_timeseries: list[dict] | None = None,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._gt_trajectory = gt_trajectory
        self._gt_lines_initialized = False
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.set_friction_coefficient(mu)

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
        # Draw ground truth trajectory on first frame
        if self._gt_trajectory and not self._gt_lines_initialized:
            pts = self._gt_trajectory
            n = len(pts) - 1
            if n > 0:
                z = 0.05  # slightly above ground
                starts_np = np.array([[p["x"], p["y"], z] for p in pts[:-1]], dtype=np.float32)
                ends_np = np.array([[p["x"], p["y"], z] for p in pts[1:]], dtype=np.float32)
                starts = wp.array(starts_np, dtype=wp.vec3)
                ends = wp.array(ends_np, dtype=wp.vec3)
                self.viewer.log_lines("gt_trajectory", starts, ends, (1.0, 0.0, 0.0), width=0.02)
            self._gt_lines_initialized = True
        super()._render(segment_num)

    def build_model(self) -> newton.Model:
        DUMMY_FRICTION = 0.0  # This is just dummy friction which should be overwritten by self.set_friction_coefficient method

        self.builder.rigid_gap = 0.1
        self.builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=DUMMY_FRICTION))
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=self._k_p,
            k_d=self._k_d,
            friction_left_right=DUMMY_FRICTION,
            friction_rear=DUMMY_FRICTION,
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )

    def set_friction_coefficient(self, mu: float):
        wp.launch(
            kernel=set_friction_coefficient_kernel,
            dim=(self.solver.dims.num_worlds, self.solver.axion_model.shape_count),
            inputs=[mu],
            outputs=[self.solver.axion_model.shape_material_mu],
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to extracted rosbag JSON (loads target_ctrl from it)",
    )
    parser.add_argument(
        "--ctrl",
        type=float,
        nargs=3,
        default=None,
        metavar=("LEFT", "RIGHT", "REAR"),
        help="Wheel velocities in rad/s (overrides ground truth)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration (default: from ground truth or 10s)",
    )
    parser.add_argument(
        "--bag",
        type=str,
        default=None,
        help="Rosbag dir; use /cmd_vel (diff-drive, fitted kinematics) as wheel target",
    )
    parser.add_argument(
        "--wheels-json",
        type=str,
        default=None,
        help="JSON with wheel_velocities.timeseries (e.g. synth output from "
        "fit_joy_to_wheels.py); overrides --bag and ground-truth joint_states",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep (default: 0.05)")
    parser.add_argument(
        "--k-p", type=float, default=2000.0, help="Velocity servo gain (default: 150)"
    )
    parser.add_argument("--mu", type=float, default=0.1, help="Coefficient of friction")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--headless", action="store_true", help="Run without viewer")
    output_group.add_argument(
        "--usd",
        type=str,
        default=None,
        metavar="PATH",
        help="Render to a USD file at PATH instead of opening the GL viewer",
    )
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
        if args.wheels_json:
            with open(args.wheels_json) as f:
                wheel_timeseries = json.load(f)["wheel_velocities"]["timeseries"]
            print(
                f"  Using synth wheel targets from {args.wheels_json} "
                f"({len(wheel_timeseries)} samples)"
            )
        elif args.bag:
            from diagnose_cmd_vs_joint import load_cmd_vel_timeseries, cmd_vel_to_wheel_ts
            import pathlib as _pl

            cmd_msgs = load_cmd_vel_timeseries(_pl.Path(args.bag))
            ref_ts = (
                gt["wheel_velocities"]["timeseries"]
                if "wheel_velocities" in gt and "timeseries" in gt["wheel_velocities"]
                else [{"t": t} for t in np.linspace(0, duration, 400)]
            )
            wheel_timeseries = cmd_vel_to_wheel_ts(cmd_msgs, ref_ts)
            print(
                f"  Using /cmd_vel-derived wheel targets "
                f"({len(cmd_msgs)} cmd_vel msgs → {len(wheel_timeseries)} samples)"
            )
        elif "wheel_velocities" in gt and "timeseries" in gt["wheel_velocities"]:
            wheel_timeseries = gt["wheel_velocities"]["timeseries"]
            print(f"  Using /joint_states wheel velocities ({len(wheel_timeseries)} points)")
        print(f"Ground truth: {gt.get('bag_name', '?')}")
    elif args.ctrl:
        target_ctrl = args.ctrl
        duration = args.duration or 10.0
    else:
        # Default: real robot wheel velocities from bag 14_46_18 (right turn)
        target_ctrl = [1.829, 0.984, 1.395]
        duration = args.duration or 10.0

    print(f"  ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f} rad/s")
    print(f"  duration={duration:.1f}s, dt={args.dt}, k_p={args.k_p}")
    print(f"  mu: {args.mu}")

    sim_config = SimulationConfig(
        duration_seconds=duration,
        target_timestep_seconds=args.dt,
        num_worlds=1,
    )
    if args.headless:
        vis_type = "null"
    elif args.usd:
        vis_type = "usd"
    else:
        vis_type = "gl"
    render_config = RenderingConfig(
        vis_type=vis_type,
        target_fps=30,
        usd_file=args.usd,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=1,
    )
    engine_config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
        backtrack_min_iter=12,
        newton_atol=1e-5,
        linear_atol=1e-5,
        linear_tol=1e-5,
        enable_linesearch=False,
        joint_compliance=6e-8,
        contact_compliance=1e-3,
        friction_compliance=2e-2,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = HelhestSim(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        target_ctrl=target_ctrl,
        k_p=args.k_p,
        mu=args.mu,
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
            # Height proportional to the larger dimension (FOV ~30°, so h ≈ extent / tan(15°))
            cam_h = max(x_range, y_range, 4.0) * 2.0
            sim.viewer.set_camera(pos=wp.vec3(cx, cy, cam_h), pitch=-90.0, yaw=90.0)
        else:
            sim.viewer.set_camera(pos=wp.vec3(0.0, -1.5, 8.0), pitch=-90.0, yaw=90.0)

    sim.run()


if __name__ == "__main__":
    main()
