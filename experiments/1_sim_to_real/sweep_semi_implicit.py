"""Parameter sweep for semi-implicit solver — minimize trajectory error vs real robot.

Sweeps dt, k_p, mu, ke, kd, kf. Semi-implicit uses explicit integration so
dt must be small and contact stiffness (ke) is critical for stability.

Usage:
    python examples/comparison_gradient/helhest/parameter_sweep/sweep_semi_implicit.py \
        --ground-truth results/right_turn_b.json

    python examples/comparison_gradient/helhest/parameter_sweep/sweep_semi_implicit.py \
        --ground-truth results/right_turn_b.json \
        --save results/sweep_semi_implicit_right_turn_b.json
"""
import argparse
import json
import math
import os
import pathlib
import time
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

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


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
        wheel_timeseries: list[dict] | None = None,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._mu = mu
        self._ke = ke
        self._kd = kd
        self._kf = kf
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

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
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self._mu, ke=self._ke, kd=self._kd, kf=self._kf,
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

    def simulate_trajectory(self) -> list[list[float]] | None:
        """Run simulation and return xy trajectory, or None if unstable."""
        traj = []
        body_q = self.current_state.body_q.numpy()
        traj.append([float(body_q[0, 0]), float(body_q[0, 1])])

        total_steps = self.clock.total_sim_steps
        for _ in range(total_steps):
            self._single_physics_step(0)
            wp.synchronize()
            body_q = self.current_state.body_q.numpy()
            x = float(body_q[0, 0])
            y = float(body_q[0, 1])
            z = float(body_q[0, 2])

            if math.isnan(x) or math.isnan(y) or math.isnan(z) or abs(z) > 10:
                return None

            traj.append([x, y])

        return traj


def trajectory_error(traj_sim, traj_gt, sim_duration=None, gt_duration=None):
    sim_np = np.array(traj_sim)
    gt_np = np.array(traj_gt)
    n = min(len(sim_np), len(gt_np), 500)
    if sim_duration is not None and gt_duration is not None:
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
    return float(np.mean(np.sqrt((sim_x - gt_x)**2 + (sim_y - gt_y)**2)))


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


KE = 8000.0
KD_CONTACT = 2000.0


def build_configs(dt_values, k_d_values, mu_values, kf_values):
    configs = []
    for dt in dt_values:
        for k_d_servo in k_d_values:
            for mu in mu_values:
                for kf in kf_values:
                    configs.append(dict(
                        dt=dt, k_p=0.0, k_d=k_d_servo,
                        mu=mu, ke=KE, kd=KD_CONTACT, kf=kf,
                    ))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", required=True, nargs="+",
                        help="Path(s) to extracted rosbag JSON(s)")
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dt", type=float, nargs="+", default=[0.0005],
                        help="Timestep values to sweep")
    parser.add_argument("--k-d", type=float, nargs="+", default=[200.0, 400.0, 800.0],
                        help="Velocity servo gain values to sweep")
    parser.add_argument("--mu", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05],
                        help="Friction coefficient values to sweep")
    parser.add_argument("--kf", type=float, nargs="+", default=[400.0, 800.0, 1500.0],
                        help="Friction damping values to sweep")
    args = parser.parse_args()

    gt_data_list = []
    for gt_path in args.ground_truth:
        traj_gt, target_ctrl, duration, gt_data, wheel_ts = load_ground_truth(gt_path)
        gt_data_list.append({
            "path": gt_path,
            "traj_gt": traj_gt,
            "target_ctrl": target_ctrl,
            "duration": duration,
            "gt_data": gt_data,
            "wheel_ts": wheel_ts,
        })
        gt_np = np.array(traj_gt)
        print(f"Ground truth: {gt_data.get('bag_name', '?')}")
        print(f"  duration={duration:.1f}s, {len(traj_gt)} points, "
              f"final=({gt_np[-1, 0]:.3f}, {gt_np[-1, 1]:.3f})")
        if wheel_ts:
            print(f"  Using time-varying wheel velocities ({len(wheel_ts)} points)")

    configs = build_configs(args.dt, args.k_d, args.mu, args.kf)
    print(f"\n=== Sweep: {len(configs)} configs x {len(gt_data_list)} trajectories "
          f"(dt={args.dt}, k_d={args.k_d}, mu={args.mu}, kf={args.kf}) ===")

    render_config = RenderingConfig(
        vis_type="null", target_fps=30, usd_file=None, start_paused=False,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    results = []

    for i, cfg in enumerate(configs):
        t0 = time.perf_counter()
        errors = []
        trajectories = {}

        for gt_entry in gt_data_list:
            bag_name = gt_entry["gt_data"].get("bag_name", "?")
            try:
                sim_config = SimulationConfig(
                    duration_seconds=gt_entry["duration"],
                    target_timestep_seconds=cfg["dt"],
                    num_worlds=1,
                )
                exec_config = ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1)
                engine_config = SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1)

                sim = HelhestSemiImplicitSim(
                    sim_config, render_config, exec_config, engine_config, logging_config,
                    target_ctrl=gt_entry["target_ctrl"],
                    k_p=cfg["k_p"], k_d=cfg["k_d"],
                    mu=cfg["mu"], ke=cfg["ke"], kd=cfg["kd"], kf=cfg["kf"],
                    wheel_timeseries=gt_entry["wheel_ts"],
                )
                traj = sim.simulate_trajectory()

                if traj is None:
                    errors.append(float("inf"))
                    trajectories[bag_name] = {"error": float("inf"), "exception": "unstable"}
                else:
                    sim_dur = len(traj) * cfg["dt"]
                    err = trajectory_error(traj, gt_entry["traj_gt"],
                                           sim_duration=sim_dur,
                                           gt_duration=gt_entry["duration"])
                    errors.append(err)
                    trajectories[bag_name] = {"trajectory": traj, "error": err}
            except Exception as e:
                errors.append(float("inf"))
                trajectories[bag_name] = {"error": float("inf"), "exception": str(e)[:200]}

        elapsed = time.perf_counter() - t0
        combined_err = float(np.mean(errors))
        results.append({"params": cfg, "error": combined_err, "per_trajectory": trajectories})

        err_strs = " + ".join(f"{e:.4f}" for e in errors)
        print(f"  [{i+1}/{len(configs)}] err={combined_err:.4f} ({err_strs}) "
              f"k_d={cfg['k_d']} mu={cfg['mu']} kf={cfg['kf']:.0f} ({elapsed:.1f}s)")

    results.sort(key=lambda r: r["error"])

    print(f"\nTop {args.top} results:")
    for i, r in enumerate(results[:args.top]):
        p = r["params"]
        per_traj = r["per_trajectory"]
        errs = " + ".join(f"{v['error']:.4f}" for v in per_traj.values())
        print(f"  {i+1}. err={r['error']:.4f} ({errs}) | "
              f"k_d={p['k_d']} mu={p['mu']} kf={p['kf']:.0f}")

    if args.save:
        best = results[0]
        output = {
            "simulator": "Semi-Implicit",
            "ground_truth": args.ground_truth,
            "ground_truth_source": "real_robot",
            "best_params": best["params"],
            "best_error": best["error"],
            "best_per_trajectory": best["per_trajectory"],
            "sweep_size": len(configs),
            "top_10": [{"error": r["error"], "params": r["params"]}
                       for r in results[:10]
                       if r["error"] != float("inf")],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
