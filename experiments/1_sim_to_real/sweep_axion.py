"""Fast Axion sweep — single process, no subprocess, no model rebuild.

Sweeps mu and friction_compliance by modifying solver state in-place between runs.
State is reset to initial pose after each simulation.

Usage:
    python examples/comparison_accuracy/helhest/sweep_axion_fast.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json

    python examples/comparison_accuracy/helhest/sweep_axion_fast.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json \
        --save results/sweep_axion_14_46_18.json
"""
import argparse
import dataclasses
import json
import os
import pathlib
import time
from copy import copy
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

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Fixed params (from interactive tuning)
DT = 0.15
K_P = 4000.0


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


class HelhestSweepSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        target_ctrl: list[float],
        k_p: float = K_P,
        k_d: float = 0.0,
        mu: float = 0.1,
        wheel_timeseries: list[dict] | None = None,
    ):
        self._k_p = k_p
        self._k_d = k_d
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.set_friction_coefficient(mu)

        # Build per-step control sequence on GPU
        num_dofs = 9
        total_steps = self.clock.total_sim_steps

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

        # Save initial state for reset
        self._initial_body_q = wp.clone(self.current_state.body_q)
        self._initial_body_qd = wp.clone(self.current_state.body_qd)
        self._initial_joint_q = wp.clone(self.current_state.joint_q)
        self._initial_joint_qd = wp.clone(self.current_state.joint_qd)

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

    def build_model(self) -> newton.Model:
        DUMMY_FRICTION = 0.0
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

    def set_friction_compliance(self, friction_compliance: float):
        self.solver.config = dataclasses.replace(
            self.solver.config, friction_compliance=friction_compliance
        )
        self.cuda_graph = None

    def set_contact_compliance(self, contact_compliance: float):
        self.solver.config = dataclasses.replace(
            self.solver.config, contact_compliance=contact_compliance
        )
        self.cuda_graph = None

    def reset_state(self):
        """Reset to initial pose and zero velocities."""
        wp.copy(self.current_state.body_q, self._initial_body_q)
        wp.copy(self.current_state.body_qd, self._initial_body_qd)
        wp.copy(self.current_state.joint_q, self._initial_joint_q)
        wp.copy(self.current_state.joint_qd, self._initial_joint_qd)
        wp.copy(self.next_state.body_q, self._initial_body_q)
        wp.copy(self.next_state.body_qd, self._initial_body_qd)
        wp.copy(self.next_state.joint_q, self._initial_joint_q)
        wp.copy(self.next_state.joint_qd, self._initial_joint_qd)

        self.clock._current_step = 0
        self.clock._current_time = 0.0
        self._ctrl_step_idx.zero_()
        self.solver.reset_timestep_counter()
        # Invalidate CUDA graph since config may have changed
        self.cuda_graph = None

    def simulate_trajectory(self) -> list[list[float]]:
        """Run simulation and record full trajectory."""
        traj = []
        body_q = self.current_state.body_q.numpy()
        traj.append([float(body_q[0, 0]), float(body_q[0, 1])])

        total_steps = self.clock.total_sim_steps
        for _ in range(total_steps):
            self._single_physics_step(0)
            wp.synchronize()
            body_q = self.current_state.body_q.numpy()
            traj.append([float(body_q[0, 0]), float(body_q[0, 1])])

        return traj


def trajectory_error(traj_sim, traj_gt, sim_duration=None, gt_duration=None):
    """Mean L2 distance in xy between sim and ground truth trajectories.

    If durations are given, resampled on common physical time over the shorter window.
    Otherwise falls back to normalized time [0, 1].
    """
    sim_np = np.array(traj_sim)
    gt_np = np.array(traj_gt)
    n = min(len(sim_np), len(gt_np), 500)

    if sim_duration is not None and gt_duration is not None:
        # Compare over common physical time window (min of the two)
        common_end = min(sim_duration, gt_duration)
        t_sim = np.linspace(0, sim_duration, len(sim_np))
        t_gt = np.linspace(0, gt_duration, len(gt_np))
        t_common = np.linspace(0, common_end, n)
    else:
        t_sim = np.linspace(0, 1, len(sim_np))
        t_gt = np.linspace(0, 1, len(gt_np))
        t_common = np.linspace(0, 1, n)

    sim_x = np.interp(t_common, t_sim, sim_np[:, 0])
    sim_y = np.interp(t_common, t_sim, sim_np[:, 1])
    gt_x = np.interp(t_common, t_gt, gt_np[:, 0])
    gt_y = np.interp(t_common, t_gt, gt_np[:, 1])
    return float(np.mean(np.sqrt((sim_x - gt_x) ** 2 + (sim_y - gt_y) ** 2)))


def load_ground_truth(path):
    with open(path) as f:
        gt = json.load(f)
    duration = gt["trajectory"].get("constant_speed_duration_s", gt["trajectory"]["duration_s"])
    traj_xy = [[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration]
    target_ctrl = gt["target_ctrl_rad_s"]
    wheel_ts = None
    if "wheel_velocities" in gt and "timeseries" in gt["wheel_velocities"]:
        wheel_ts = gt["wheel_velocities"]["timeseries"]
    return traj_xy, target_ctrl, duration, gt, wheel_ts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ground-truth", required=True, nargs="+", help="Path(s) to extracted rosbag JSON(s)"
    )
    parser.add_argument("--save", metavar="PATH", help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10, help="Print top N results")
    parser.add_argument(
        "--dt", type=float, nargs="+", default=[0.08, 0.1, 0.15], help="Timestep values to sweep"
    )
    parser.add_argument(
        "--mu", type=float, nargs="+", default=[0.1], help="Friction coefficient values to sweep"
    )
    parser.add_argument(
        "--fc",
        type=float,
        nargs="+",
        default=[1e-3, 5e-3, 1.2e-2, 2e-2, 5e-2, 1e-1],
        help="Friction compliance values to sweep",
    )
    parser.add_argument(
        "--cc", type=float, nargs="+", default=[1e-1], help="Contact compliance values to sweep"
    )
    parser.add_argument(
        "--cmd-vel-bag", type=str, nargs="+", default=None,
        help="Rosbag dir(s); use /cmd_vel (diff-drive kinematic) as wheel target "
             "instead of /joint_states. Must align 1:1 with --ground-truth paths.",
    )
    args = parser.parse_args()

    cmd_vel_bags = args.cmd_vel_bag
    if cmd_vel_bags and len(cmd_vel_bags) != len(args.ground_truth):
        parser.error("--cmd-vel-bag must have same count as --ground-truth")

    # Load all ground truth trajectories
    gt_data_list = []
    for i, gt_path in enumerate(args.ground_truth):
        traj_gt, target_ctrl, duration, gt_data, wheel_ts = load_ground_truth(gt_path)
        if cmd_vel_bags:
            import pathlib as _pl
            from diagnose_cmd_vs_joint import load_cmd_vel_timeseries, cmd_vel_to_wheel_ts
            cmd_msgs = load_cmd_vel_timeseries(_pl.Path(cmd_vel_bags[i]))
            ref_ts = wheel_ts if wheel_ts else [
                {"t": t} for t in np.linspace(0, duration, 400)
            ]
            wheel_ts = cmd_vel_to_wheel_ts(cmd_msgs, ref_ts)
            print(f"  [cmd_vel] {len(cmd_msgs)} msgs -> {len(wheel_ts)} samples")
        gt_data_list.append(
            {
                "path": gt_path,
                "traj_gt": traj_gt,
                "target_ctrl": target_ctrl,
                "duration": duration,
                "gt_data": gt_data,
                "wheel_ts": wheel_ts,
            }
        )
        gt_np = np.array(traj_gt)
        print(f"Ground truth: {gt_data.get('bag_name', '?')}")
        print(
            f"  duration={duration:.1f}s, {len(traj_gt)} points, "
            f"final=({gt_np[-1, 0]:.3f}, {gt_np[-1, 1]:.3f})"
        )
        if wheel_ts:
            print(f"  Using time-varying wheel velocities ({len(wheel_ts)} points)")

    print(f"\nFixed: k_p={K_P}")

    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    exec_config = ExecutionConfig(use_cuda_graph=True, headless_steps_per_segment=1)
    engine_config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
        backtrack_min_iter=12,
        newton_atol=1e-5,
        linear_atol=1e-5,
        linear_tol=1e-5,
        enable_linesearch=False,
        joint_compliance=6e-8,
        contact_compliance=1e-4,
        friction_compliance=1.2e-2,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    # --- Sweep grid (cartesian product of CLI args) ---
    configs = []
    for dt in args.dt:
        for mu in args.mu:
            for fc in args.fc:
                for cc in args.cc:
                    configs.append(
                        {"dt": dt, "mu": mu, "friction_compliance": fc, "contact_compliance": cc}
                    )

    total = len(configs)
    print(
        f"\n=== Sweep: {total} configs x {len(gt_data_list)} trajectories "
        f"(dt={args.dt}, mu={args.mu}, fc={args.fc}, cc={args.cc}) ==="
    )

    results = []
    current_dt = None
    sims = {}  # bag_name -> HelhestSweepSim
    global_i = 0

    for cfg in configs:
        # Rebuild sims when dt changes
        if cfg["dt"] != current_dt:
            current_dt = cfg["dt"]
            sims = {}
            print(f"\n  --- Building sims for dt={current_dt} ---")
            for gt_entry in gt_data_list:
                bag_name = gt_entry["gt_data"].get("bag_name", "?")
                sim_config_dt = SimulationConfig(
                    duration_seconds=gt_entry["duration"],
                    target_timestep_seconds=current_dt,
                    num_worlds=1,
                )
                sims[bag_name] = HelhestSweepSim(
                    sim_config_dt,
                    render_config,
                    exec_config,
                    engine_config,
                    logging_config,
                    target_ctrl=gt_entry["target_ctrl"],
                    k_p=K_P,
                    mu=cfg["mu"],
                    wheel_timeseries=gt_entry["wheel_ts"],
                )

        t0 = time.perf_counter()
        errors = []
        trajectories = {}

        for gt_entry in gt_data_list:
            bag_name = gt_entry["gt_data"].get("bag_name", "?")
            sim = sims[bag_name]
            try:
                sim.reset_state()
                sim.set_friction_coefficient(cfg["mu"])
                sim.set_friction_compliance(cfg["friction_compliance"])
                sim.set_contact_compliance(cfg["contact_compliance"])

                traj = sim.simulate_trajectory()
                sim_duration = sim.clock.total_sim_steps * sim.clock.dt
                err = trajectory_error(traj, gt_entry["traj_gt"],
                                       sim_duration=sim_duration,
                                       gt_duration=gt_entry["duration"])
                errors.append(err)
                trajectories[bag_name] = {"trajectory": traj, "error": err}
            except Exception as e:
                errors.append(float("inf"))
                trajectories[bag_name] = {"error": float("inf"), "exception": str(e)[:200]}

        elapsed = time.perf_counter() - t0
        combined_err = float(np.mean(errors))

        results.append(
            {
                "params": cfg,
                "error": combined_err,
                "per_trajectory": trajectories,
            }
        )
        global_i += 1

        err_strs = " + ".join(f"{e:.4f}" for e in errors)
        print(
            f"  [{global_i}/{total}] err={combined_err:.4f} ({err_strs}) "
            f"dt={cfg['dt']} mu={cfg['mu']} fc={cfg['friction_compliance']:.4f} "
            f"cc={cfg['contact_compliance']:.1e} ({elapsed:.1f}s)"
        )

    results.sort(key=lambda r: r["error"])

    print(f"\nTop {args.top} results:")
    for i, r in enumerate(results[: args.top]):
        p = r["params"]
        per_traj = r["per_trajectory"]
        errs = " + ".join(f"{v['error']:.4f}" for v in per_traj.values())
        print(
            f"  {i+1}. err={r['error']:.4f} ({errs}) | "
            f"dt={p['dt']} mu={p['mu']} fc={p['friction_compliance']:.4f} "
            f"cc={p['contact_compliance']:.1e}"
        )

    if args.save:
        best = results[0]
        output = {
            "simulator": "Axion",
            "ground_truth": args.ground_truth,
            "ground_truth_source": "real_robot",
            "fixed_params": {"k_p": K_P},
            "swept_param_values": {
                "dt": args.dt, "mu": args.mu, "fc": args.fc, "cc": args.cc,
            },
            "best_params": best["params"],
            "best_error": best["error"],
            "best_per_trajectory": best["per_trajectory"],
            "sweep_size": len(configs),
            "top_10": [{"error": r["error"], "params": r["params"]} for r in results[:10]],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
