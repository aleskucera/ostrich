"""Semi-implicit dt stability sweep — obstacle traversal with calibrated params.

Usage:
    python experiments/2_dt_stability/sweep_semi_implicit.py
    python experiments/2_dt_stability/sweep_semi_implicit.py \
        --save results/sweep_semi_implicit.json
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

# Calibrated params from 1_sim_to_real sweep
K_P = 0.0
K_D = 400.0
MU = 0.02
KE = 8000.0
KD_CONTACT = 2000.0
KF = 1500.0

# Obstacle config (nominal — trials perturb these)
OBSTACLE_X = 2.0
OBSTACLE_HEIGHT = 0.08  # half-height (full step 0.16m ≈ 44% of wheel radius)
WHEEL_VEL = 4.0
RAMP_TIME = 1.0
DURATION = 8.0

# Perturbation ranges
OBSTACLE_HEIGHT_RANGE = (0.07, 0.09)
OBSTACLE_X_RANGE = (1.5, 2.5)
WHEEL_VEL_RANGE = (3.5, 4.5)
INITIAL_YAW_RANGE = (-0.1, 0.1)

DT_PROBES = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]


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


class HelhestObstacleSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: SemiImplicitEngineConfig,
        logging_config: LoggingConfig,
        k_p: float = K_P,
        k_d: float = K_D,
        mu: float = MU,
        ke: float = KE,
        kd_contact: float = KD_CONTACT,
        kf: float = KF,
        wheel_vel: float = WHEEL_VEL,
        obstacle_x: float = OBSTACLE_X,
        obstacle_height: float = OBSTACLE_HEIGHT,
        initial_yaw: float = 0.0,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._mu = mu
        self._ke = ke
        self._kd_contact = kd_contact
        self._kf = kf
        self._obstacle_x = obstacle_x
        self._obstacle_height = obstacle_height
        self._initial_yaw = initial_yaw
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        # Build ramped control sequence
        num_dofs = 9
        total_steps = self.clock.total_sim_steps
        ctrl_np = np.zeros((total_steps, num_dofs), dtype=np.float32)
        for i in range(total_steps):
            t = (i + 1) * self.clock.dt
            ramp = min(t / RAMP_TIME, 1.0)
            wv = wheel_vel * ramp
            ctrl_np[i, 6] = wv
            ctrl_np[i, 7] = wv
            ctrl_np[i, 8] = wv

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
            inputs=[self._ctrl_step_idx, self._ctrl_sequence,
                    self.control.joint_target_vel, self._ctrl_total_steps],
        )
        wp.launch(
            kernel=increment_step_kernel,
            dim=1,
            inputs=[self._ctrl_step_idx],
        )

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self._mu, ke=self._ke, kd=self._kd_contact, kf=self._kf,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        yaw_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self._initial_yaw)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), yaw_quat),
            control_mode="velocity",
            k_p=self._k_p,
            k_d=self._k_d,
            friction_left_right=self._mu,
            friction_rear=self._mu,
            ke=self._ke,
            kd=self._kd_contact,
            kf=self._kf,
        )

        # Static box obstacle
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self._obstacle_x, 0.0, self._obstacle_height),
                wp.quat_identity(),
            ),
            hx=0.5,
            hy=1.0,
            hz=self._obstacle_height,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=1.0, ke=self._ke, kd=self._kd_contact, kf=self._kf,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )

    def simulate_and_check(self) -> dict:
        """Run simulation and return stability metrics."""
        z_values = []
        x_values = []
        y_values = []

        body_q = self.current_state.body_q.numpy()
        z_values.append(float(body_q[0, 2]))
        x_values.append(float(body_q[0, 0]))
        y_values.append(float(body_q[0, 1]))

        total_steps = self.clock.total_sim_steps
        has_nan = False

        for _ in range(total_steps):
            self._single_physics_step(0)
            wp.synchronize()
            body_q = self.current_state.body_q.numpy()
            z = float(body_q[0, 2])
            x = float(body_q[0, 0])
            y = float(body_q[0, 1])

            if math.isnan(z) or math.isnan(x) or math.isnan(y) or abs(z) > 100:
                has_nan = True
                break

            z_values.append(z)
            x_values.append(x)
            y_values.append(y)

        z_min = min(z_values)
        z_max = max(z_values)
        x_final = x_values[-1]
        y_max_abs = max(abs(y) for y in y_values)
        # y_max not part of stability predicate — initial_yaw perturbation makes lateral motion expected
        stable = (not has_nan
                  and z_min > 0.05
                  and z_max < 2.0
                  and x_final > self._obstacle_x + 1.0)

        return {
            "stable": stable,
            "has_nan": has_nan,
            "z_min": round(z_min, 4),
            "z_max": round(z_max, 4),
            "x_final": round(x_final, 4),
            "y_max_abs": round(y_max_abs, 4),
            "num_steps": len(z_values),
        }


def find_max_stable_dt(run_one, dt_probes, *, label: str) -> tuple[float, list[dict]]:
    results = []
    lo = None
    for dt in dt_probes:
        t0 = time.perf_counter()
        metrics = run_one(dt)
        elapsed = time.perf_counter() - t0
        metrics["dt"] = dt
        metrics["time_s"] = round(elapsed, 2)
        results.append(metrics)
        status = "STABLE" if metrics["stable"] else "UNSTABLE"
        print(
            f"  [{label}] probe dt={dt:.5f} | {status:8s} | "
            f"z=[{metrics['z_min']:.3f}, {metrics['z_max']:.3f}] "
            f"| x_final={metrics['x_final']:.3f} | {elapsed:.1f}s"
        )
        if metrics["stable"]:
            lo = dt
            break

    if lo is None:
        return 0.0, results

    unstable_above = [r["dt"] for r in results if not r["stable"] and r["dt"] > lo]
    hi = min(unstable_above) if unstable_above else lo * 2.0
    tol = lo * 0.1
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        t0 = time.perf_counter()
        metrics = run_one(mid)
        elapsed = time.perf_counter() - t0
        metrics["dt"] = mid
        metrics["time_s"] = round(elapsed, 2)
        results.append(metrics)
        status = "STABLE" if metrics["stable"] else "UNSTABLE"
        print(
            f"  [{label}] bisect dt={mid:.5f} | {status:8s} | "
            f"x_final={metrics['x_final']:.3f} | {elapsed:.1f}s"
        )
        if metrics["stable"]:
            lo = mid
        else:
            hi = mid
    return lo, results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument(
        "--num-trials", type=int, default=1,
        help="Number of perturbed trials to run (default: 1 — nominal config only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for perturbations")
    args = parser.parse_args()

    print(f"Semi-Implicit obstacle dt sweep — {args.num_trials} trial(s)")
    print(f"  nominal obstacle: x={OBSTACLE_X}, height={OBSTACLE_HEIGHT*2:.2f}m, "
          f"wheel_vel={WHEEL_VEL} rad/s")
    if args.num_trials > 1:
        print(f"  perturbations (seed={args.seed}):")
        print(f"    obstacle_height ~ U{OBSTACLE_HEIGHT_RANGE}")
        print(f"    obstacle_x      ~ U{OBSTACLE_X_RANGE}")
        print(f"    wheel_vel       ~ U{WHEEL_VEL_RANGE}")
        print(f"    initial_yaw     ~ U{INITIAL_YAW_RANGE}")
    print(f"  params: k_d={K_D}, mu={MU}, ke={KE}, kf={KF}")
    print()

    render_config = RenderingConfig(
        vis_type="null", target_fps=30, usd_file=None, start_paused=False,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    def make_run_one(trial_params: dict):
        def run_one(dt):
            sim_config = SimulationConfig(
                duration_seconds=DURATION,
                target_timestep_seconds=dt,
                num_worlds=1,
            )
            exec_config = ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1)
            engine_config = SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1)
            try:
                sim = HelhestObstacleSim(
                    sim_config, render_config, exec_config, engine_config, logging_config,
                    k_p=K_P, k_d=K_D, mu=MU, ke=KE, kd_contact=KD_CONTACT, kf=KF,
                    wheel_vel=trial_params["wheel_vel"],
                    obstacle_x=trial_params["obstacle_x"],
                    obstacle_height=trial_params["obstacle_height"],
                    initial_yaw=trial_params["initial_yaw"],
                )
                return sim.simulate_and_check()
            except Exception as e:
                return {
                    "stable": False, "has_nan": True,
                    "z_min": 0, "z_max": 0, "x_final": 0, "y_max_abs": 0,
                    "num_steps": 0, "exception": str(e)[:200],
                }
        return run_one

    rng = np.random.default_rng(args.seed)
    trials = []
    for i in range(args.num_trials):
        if args.num_trials == 1:
            trial_params = {
                "obstacle_height": OBSTACLE_HEIGHT,
                "obstacle_x": OBSTACLE_X,
                "wheel_vel": WHEEL_VEL,
                "initial_yaw": 0.0,
            }
        else:
            trial_params = {
                "obstacle_height": float(rng.uniform(*OBSTACLE_HEIGHT_RANGE)),
                "obstacle_x": float(rng.uniform(*OBSTACLE_X_RANGE)),
                "wheel_vel": float(rng.uniform(*WHEEL_VEL_RANGE)),
                "initial_yaw": float(rng.uniform(*INITIAL_YAW_RANGE)),
            }
        label = f"trial {i + 1}/{args.num_trials}"
        print(f"\n=== {label} ===")
        for k, v in trial_params.items():
            print(f"    {k} = {v:.4f}")
        t0 = time.perf_counter()
        dt_max, search_results = find_max_stable_dt(
            make_run_one(trial_params), DT_PROBES, label=label,
        )
        elapsed = time.perf_counter() - t0
        print(f"  -> max_stable_dt = {dt_max:.5f}  ({elapsed:.1f}s)")
        trials.append({
            "config": trial_params,
            "max_stable_dt": dt_max,
            "total_time_s": round(elapsed, 2),
            "search": search_results,
        })

    dt_maxes = [t["max_stable_dt"] for t in trials if t["max_stable_dt"] > 0]
    if dt_maxes:
        print("\n=== summary ===")
        print(f"  n={len(dt_maxes)}/{args.num_trials} trials with dt_max > 0")
        print(f"  median = {float(np.median(dt_maxes)):.5f}")
        print(f"  IQR    = [{float(np.quantile(dt_maxes, 0.25)):.5f}, "
              f"{float(np.quantile(dt_maxes, 0.75)):.5f}]")
        print(f"  min    = {min(dt_maxes):.5f}")
        print(f"  max    = {max(dt_maxes):.5f}")

    if args.save:
        output = {
            "simulator": "Semi-Implicit",
            "experiment": "obstacle_dt_sweep",
            "nominal": {
                "obstacle_x": OBSTACLE_X, "obstacle_height": OBSTACLE_HEIGHT,
                "wheel_vel": WHEEL_VEL,
            },
            "perturbation_ranges": {
                "obstacle_height": OBSTACLE_HEIGHT_RANGE,
                "obstacle_x": OBSTACLE_X_RANGE,
                "wheel_vel": WHEEL_VEL_RANGE,
                "initial_yaw": INITIAL_YAW_RANGE,
            },
            "num_trials": args.num_trials,
            "seed": args.seed,
            "calibrated_params": {
                "k_p": K_P, "k_d": K_D, "mu": MU,
                "ke": KE, "kd_contact": KD_CONTACT, "kf": KF,
            },
            "ramp_time": RAMP_TIME,
            "duration": DURATION,
            "max_stable_dt": trials[0]["max_stable_dt"] if len(trials) == 1 else None,
            "max_stable_dt_median": float(np.median(dt_maxes)) if dt_maxes else 0.0,
            "max_stable_dt_iqr": (
                [float(np.quantile(dt_maxes, 0.25)), float(np.quantile(dt_maxes, 0.75))]
                if dt_maxes else [0.0, 0.0]
            ),
            "trials": trials,
        }
        save_path = pathlib.Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
