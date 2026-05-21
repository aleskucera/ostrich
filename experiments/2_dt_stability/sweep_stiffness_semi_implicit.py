"""Sweep contact stiffness vs max-stable-dt for Semi-Implicit.

For each k_e in a log-spaced range, binary-search the largest stable dt
on the obstacle scene. Expected scaling from penalty-CFL theory:

    dt_max  ~  2 * sqrt(m_eff / k_eff)    i.e.   slope -1/2 on log-log.

k_d (contact damping) is scaled with sqrt(k_e) to keep damping ratio fixed.

Usage:
    python experiments/2_dt_stability/sweep_stiffness_semi_implicit.py \\
        --save results/sweep_stiffness_semi_implicit.json
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

# Fixed (non-stiffness) params — match sweep_semi_implicit.py
K_P = 0.0
K_D_SERVO = 400.0
MU = 0.02
KF = 1500.0

# Reference stiffness at which k_d is normalized.
# k_d(ke) = KD_CONTACT_REF * sqrt(ke / KE_REF)  keeps damping ratio constant.
KE_REF = 8000.0
KD_CONTACT_REF = 3000.0

# Obstacle + drive (match sweep_semi_implicit.py)
OBSTACLE_X = 2.0
OBSTACLE_HEIGHT = 0.1
WHEEL_VEL = 4.0
RAMP_TIME = 1.0
DURATION = 8.0

# Stiffness values to sweep (log-spaced, one decade either side of reference)
KE_VALUES = [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5, 3.0e5]

# dt probe values — descending. First stable one anchors binary search.
DT_PROBES = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 5.0e-5, 2.0e-5, 1.0e-5]


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
def increment_step_kernel(step_idx: wp.array(dtype=wp.int32)):
    step_idx[0] = step_idx[0] + 1


class HelhestObstacleSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: SemiImplicitEngineConfig,
        logging_config: LoggingConfig,
        ke: float,
        kd_contact: float,
    ):
        self._ke = ke
        self._kd_contact = kd_contact
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        num_dofs = 9
        total_steps = self.clock.total_sim_steps
        ctrl_np = np.zeros((total_steps, num_dofs), dtype=np.float32)
        for i in range(total_steps):
            t = (i + 1) * self.clock.dt
            ramp = min(t / RAMP_TIME, 1.0)
            wv = WHEEL_VEL * ramp
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
            set_control_from_sequence_kernel,
            dim=9,
            inputs=[
                self._ctrl_step_idx,
                self._ctrl_sequence,
                self.control.joint_target_vel,
                self._ctrl_total_steps,
            ],
        )
        wp.launch(increment_step_kernel, dim=1, inputs=[self._ctrl_step_idx])

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=MU, ke=self._ke, kd=self._kd_contact, kf=KF,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=K_P,
            k_d=K_D_SERVO,
            friction_left_right=MU,
            friction_rear=MU,
            ke=self._ke,
            kd=self._kd_contact,
            kf=KF,
        )
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((OBSTACLE_X, 0.0, OBSTACLE_HEIGHT), wp.quat_identity()),
            hx=0.5, hy=1.0, hz=OBSTACLE_HEIGHT,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=1.0, ke=self._ke, kd=self._kd_contact, kf=KF,
            ),
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )

    def simulate_and_check(self) -> dict:
        z_vals, x_vals, y_vals = [], [], []
        q0 = self.current_state.body_q.numpy()
        z_vals.append(float(q0[0, 2]))
        x_vals.append(float(q0[0, 0]))
        y_vals.append(float(q0[0, 1]))

        has_nan = False
        for _ in range(self.clock.total_sim_steps):
            self._single_physics_step(0)
            wp.synchronize()
            q = self.current_state.body_q.numpy()
            z, x, y = float(q[0, 2]), float(q[0, 0]), float(q[0, 1])
            if math.isnan(z) or math.isnan(x) or math.isnan(y) or abs(z) > 100:
                has_nan = True
                break
            z_vals.append(z); x_vals.append(x); y_vals.append(y)

        z_min, z_max = min(z_vals), max(z_vals)
        x_final = x_vals[-1]
        y_max_abs = max(abs(v) for v in y_vals)
        stable = (not has_nan and z_min > 0.05 and z_max < 1.0
                  and y_max_abs < 0.5 and x_final > OBSTACLE_X)

        return {
            "stable": stable, "has_nan": has_nan,
            "z_min": round(z_min, 4), "z_max": round(z_max, 4),
            "x_final": round(x_final, 4), "y_max_abs": round(y_max_abs, 4),
            "num_steps": len(z_vals),
        }


def run_one(dt: float, ke: float, kd_contact: float) -> dict:
    sim_config = SimulationConfig(
        duration_seconds=DURATION, target_timestep_seconds=dt, num_worlds=1,
    )
    render_config = RenderingConfig(
        vis_type="null", target_fps=30, usd_file=None, start_paused=False,
    )
    exec_config = ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1)
    engine_config = SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1)
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)
    try:
        sim = HelhestObstacleSim(
            sim_config, render_config, exec_config, engine_config, logging_config,
            ke=ke, kd_contact=kd_contact,
        )
        return sim.simulate_and_check()
    except Exception as e:
        return {
            "stable": False, "has_nan": True,
            "z_min": 0, "z_max": 0, "x_final": 0, "y_max_abs": 0,
            "num_steps": 0, "exception": str(e)[:200],
        }


def find_max_stable_dt(ke: float, kd_contact: float) -> tuple[float, list[dict]]:
    """Probe downward then bisect. Returns (max_stable_dt, per-trial results)."""
    results = []
    lo = None
    for dt in DT_PROBES:
        t0 = time.perf_counter()
        m = run_one(dt, ke, kd_contact)
        elapsed = time.perf_counter() - t0
        m["dt"] = dt
        m["time_s"] = round(elapsed, 2)
        results.append(m)
        tag = "STABLE" if m["stable"] else "UNSTABLE"
        print(f"    probe dt={dt:.5g} | {tag:8s} | x_final={m['x_final']:.3f} | {elapsed:.1f}s")
        if m["stable"]:
            lo = dt
            break

    if lo is None:
        return 0.0, results

    hi_candidates = [r["dt"] for r in results if not r["stable"] and r["dt"] > lo]
    hi = min(hi_candidates) if hi_candidates else lo * 2.0

    tol = lo * 0.1
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        t0 = time.perf_counter()
        m = run_one(mid, ke, kd_contact)
        elapsed = time.perf_counter() - t0
        m["dt"] = mid
        m["time_s"] = round(elapsed, 2)
        results.append(m)
        tag = "STABLE" if m["stable"] else "UNSTABLE"
        print(f"    bisect dt={mid:.5g} | {tag:8s} | x_final={m['x_final']:.3f} | {elapsed:.1f}s")
        if m["stable"]:
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
        "--ke-values",
        type=float,
        nargs="+",
        default=None,
        help="Override default k_e sweep values",
    )
    args = parser.parse_args()

    ke_values = args.ke_values or KE_VALUES

    print("Semi-Implicit stiffness sweep — dt_max(k_e) on obstacle scene")
    print(f"  k_e values: {ke_values}")
    print(f"  k_d(k_e) = {KD_CONTACT_REF} * sqrt(k_e / {KE_REF})")
    print(f"  obstacle: x={OBSTACLE_X}, height={OBSTACLE_HEIGHT*2:.2f}m; "
          f"drive {WHEEL_VEL} rad/s, ramp {RAMP_TIME}s, duration {DURATION}s")
    print()

    per_ke = []
    for ke in ke_values:
        kd = KD_CONTACT_REF * math.sqrt(ke / KE_REF)
        print(f"[k_e={ke:.3g}, k_d={kd:.3g}]")
        t_outer = time.perf_counter()
        dt_max, trial_results = find_max_stable_dt(ke, kd)
        dt_pred = 2.0 * math.sqrt(5.5 / ke)  # rough: wheel mass only, no joint/damping
        print(f"  -> dt_max = {dt_max:.5g}   (naive CFL ~{dt_pred:.3g})   "
              f"[{time.perf_counter() - t_outer:.1f}s]\n")
        per_ke.append({
            "ke": ke, "kd_contact": kd,
            "max_stable_dt": dt_max,
            "naive_cfl_dt": dt_pred,
            "trials": trial_results,
        })

    # Log-log slope fit (only over points with dt_max > 0)
    xs = np.log([p["ke"] for p in per_ke if p["max_stable_dt"] > 0])
    ys = np.log([p["max_stable_dt"] for p in per_ke if p["max_stable_dt"] > 0])
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    print(f"Log-log slope of dt_max vs k_e:  {slope:+.3f}   (theory: -0.5)")

    if args.save:
        out = {
            "simulator": "Semi-Implicit",
            "experiment": "stiffness_dt_scaling",
            "fixed_params": {
                "mu": MU, "kf": KF,
                "k_p": K_P, "k_d_servo": K_D_SERVO,
                "wheel_vel": WHEEL_VEL, "ramp_time": RAMP_TIME, "duration": DURATION,
                "obstacle_x": OBSTACLE_X, "obstacle_height": OBSTACLE_HEIGHT,
            },
            "kd_scaling": f"k_d = {KD_CONTACT_REF} * sqrt(k_e / {KE_REF})",
            "theory": "dt_max ~ 2 * sqrt(m_eff / k_eff), slope -1/2 on log-log",
            "loglog_slope": slope,
            "per_ke": per_ke,
        }
        path = pathlib.Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
