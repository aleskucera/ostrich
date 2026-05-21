"""MuJoCo dt stability sweep — obstacle traversal with calibrated params.

Sweeps dt to find the maximum stable timestep for obstacle traversal.
Physics params are fixed from the parameter_sweep calibration.

Usage:
    python examples/comparison_gradient/helhest/dt_sweep/sweep_mujoco.py
    python examples/comparison_gradient/helhest/dt_sweep/sweep_mujoco.py \
        --save results/sweep_mujoco.json
"""
import argparse
import json
import math
import pathlib
import time

import mujoco
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Calibrated params from 1_sim_to_real combined sweep (implicitfast)
KV = 4000.0
MU = 0.2
OBSTACLE_MU = 1.0

# Obstacle config (nominal — trials perturb these)
OBSTACLE_X = 2.0
OBSTACLE_HEIGHT = 0.08  # half-height (full step 0.12m ≈ 28% of wheel radius)
WHEEL_VEL = 4.0
RAMP_TIME = 1.0  # seconds to ramp from 0 to WHEEL_VEL
DURATION = 8.0

# Perturbation ranges
OBSTACLE_HEIGHT_RANGE = (0.07, 0.09)
OBSTACLE_X_RANGE = (1.5, 2.5)
WHEEL_VEL_RANGE = (3.5, 4.5)
INITIAL_YAW_RANGE = (-0.1, 0.1)

DT_PROBES = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

HELHEST_OBSTACLE_XML = """<mujoco model="helhest_obstacle">
  <option gravity="0 0 -9.81" timestep="{dt}"
          solver="Newton" iterations="50" ls_iterations="50"
          cone="pyramidal" impratio="1.0" integrator="implicitfast"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="{mu} 0.1 0.01"
          solref="{solref_timeconst} 1.0" solimp="0.9 0.95 0.001 0.5 2"
          condim="6"/>

    <!-- Obstacle box -->
    <geom name="obstacle" type="box"
          pos="{obstacle_x} 0 {obstacle_height}"
          size="0.5 1.0 {obstacle_height}"
          friction="{obstacle_mu} 0.1 0.01"
          solref="{solref_timeconst} 1.0" solimp="0.9 0.95 0.001 0.5 2"
          condim="6"/>

    <body name="chassis" pos="0 0 0.37" quat="{chassis_qw} 0 0 {chassis_qz}">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0"
                diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>

      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{mu} 0.1 0.01"
              solref="{solref_timeconst} 1.0" solimp="0.9 0.95 0.001 0.5 2"
              condim="6"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{mu} 0.1 0.01"
              solref="{solref_timeconst} 1.0" solimp="0.9 0.95 0.001 0.5 2"
              condim="6"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{mu} 0.1 0.01"
              solref="{solref_timeconst} 1.0" solimp="0.9 0.95 0.001 0.5 2"
              condim="6"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_act" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>"""


def simulate_and_check(
    dt, kv, mu, obstacle_mu, obstacle_x, obstacle_height, wheel_vel, duration, initial_yaw=0.0
) -> dict:
    """Run one MuJoCo simulation and return stability metrics."""
    qw = math.cos(initial_yaw / 2.0)
    qz = math.sin(initial_yaw / 2.0)
    xml = HELHEST_OBSTACLE_XML.format(
        dt=dt,
        kv=kv,
        mu=mu,
        obstacle_mu=obstacle_mu,
        obstacle_x=obstacle_x,
        obstacle_height=obstacle_height,
        chassis_qw=qw,
        chassis_qz=qz,
        solref_timeconst=0.005,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    T = int(duration / dt)
    z_values = [float(data.qpos[2])]
    x_values = [float(data.qpos[0])]
    y_values = [float(data.qpos[1])]
    has_nan = False

    for step in range(T):
        t = (step + 1) * dt
        ramp = min(t / RAMP_TIME, 1.0)
        wv = wheel_vel * ramp
        data.ctrl[:] = [wv, wv, wv]
        mujoco.mj_step(model, data)

        z = float(data.qpos[2])
        x = float(data.qpos[0])
        y = float(data.qpos[1])

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
    stable = not has_nan and z_min > 0.05 and z_max < 2.0 and x_final > obstacle_x + 1.0

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
        "--num-trials",
        type=int,
        default=1,
        help="Number of perturbed trials to run (default: 1 — nominal config only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for perturbations")
    args = parser.parse_args()

    print(f"MuJoCo obstacle dt sweep — {args.num_trials} trial(s)")
    print(
        f"  nominal obstacle: x={OBSTACLE_X}, height={OBSTACLE_HEIGHT*2:.2f}m, "
        f"wheel_vel={WHEEL_VEL} rad/s"
    )
    if args.num_trials > 1:
        print(f"  perturbations (seed={args.seed}):")
        print(f"    obstacle_height ~ U{OBSTACLE_HEIGHT_RANGE}")
        print(f"    obstacle_x      ~ U{OBSTACLE_X_RANGE}")
        print(f"    wheel_vel       ~ U{WHEEL_VEL_RANGE}")
        print(f"    initial_yaw     ~ U{INITIAL_YAW_RANGE}")
    print(f"  params: kv={KV}, mu={MU}")
    print()

    def make_run_one(trial_params: dict):
        def run_one(dt):
            return simulate_and_check(
                dt=dt,
                kv=KV,
                mu=MU,
                obstacle_mu=OBSTACLE_MU,
                obstacle_x=trial_params["obstacle_x"],
                obstacle_height=trial_params["obstacle_height"],
                wheel_vel=trial_params["wheel_vel"],
                duration=DURATION,
                initial_yaw=trial_params["initial_yaw"],
            )

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
            make_run_one(trial_params),
            DT_PROBES,
            label=label,
        )
        elapsed = time.perf_counter() - t0
        print(f"  -> max_stable_dt = {dt_max:.5f}  ({elapsed:.1f}s)")
        trials.append(
            {
                "config": trial_params,
                "max_stable_dt": dt_max,
                "total_time_s": round(elapsed, 2),
                "search": search_results,
            }
        )

    dt_maxes = [t["max_stable_dt"] for t in trials if t["max_stable_dt"] > 0]
    if dt_maxes:
        print("\n=== summary ===")
        print(f"  n={len(dt_maxes)}/{args.num_trials} trials with dt_max > 0")
        print(f"  median = {float(np.median(dt_maxes)):.5f}")
        print(
            f"  IQR    = [{float(np.quantile(dt_maxes, 0.25)):.5f}, "
            f"{float(np.quantile(dt_maxes, 0.75)):.5f}]"
        )
        print(f"  min    = {min(dt_maxes):.5f}")
        print(f"  max    = {max(dt_maxes):.5f}")

    if args.save:
        output = {
            "simulator": "MuJoCo",
            "experiment": "obstacle_dt_sweep",
            "nominal": {
                "obstacle_x": OBSTACLE_X,
                "obstacle_height": OBSTACLE_HEIGHT,
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
            "calibrated_params": {"kv": KV, "mu": MU},
            "duration": DURATION,
            "max_stable_dt": trials[0]["max_stable_dt"] if len(trials) == 1 else None,
            "max_stable_dt_median": float(np.median(dt_maxes)) if dt_maxes else 0.0,
            "max_stable_dt_iqr": (
                [float(np.quantile(dt_maxes, 0.25)), float(np.quantile(dt_maxes, 0.75))]
                if dt_maxes
                else [0.0, 0.0]
            ),
            "trials": trials,
        }
        save_path = pathlib.Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
