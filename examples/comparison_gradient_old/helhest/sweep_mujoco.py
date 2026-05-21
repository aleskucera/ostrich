"""Hyperparameter sweep for MuJoCo helhest — minimize trajectory error vs ground truth.

Usage:
    python examples/comparison_gradient/helhest/sweep_mujoco.py \
        --ground-truth results/helhest_chrono.json \
        --save results/sweep_mujoco.json
"""
import argparse
import itertools
import json
import pathlib
import time

import mujoco
import numpy as np

DURATION = 3.0
TARGET_CTRL = [1.0, 6.0, 0.0]

HELHEST_XML_TEMPLATE = """<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{dt}"
          solver="{solver}" iterations="{iterations}" ls_iterations="{ls_iterations}"
          cone="{cone}" impratio="{impratio}" integrator="{integrator}"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="{ground_friction} {ground_torsional} {ground_rolling}"
          solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
          condim="{condim}"/>

    <body name="chassis" pos="0 0 0.37">
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
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{rear_friction} {rear_torsional} {rear_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_act" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>"""


def simulate(params: dict) -> list:
    """Run forward simulation, return xy trajectory as list of [x, y]."""
    xml = HELHEST_XML_TEMPLATE.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = params["dt"]
    T = int(DURATION / dt)
    traj = []

    for step in range(T):
        data.ctrl[:] = TARGET_CTRL
        mujoco.mj_step(model, data)
        pos = data.qpos[:3].copy()
        traj.append([float(pos[0]), float(pos[1])])

    return traj


def trajectory_error(traj_sim: np.ndarray, traj_gt: np.ndarray) -> float:
    """Mean L2 distance between two trajectories, resampled to common length."""
    n_sim = len(traj_sim)
    n_gt = len(traj_gt)
    # Resample both to the shorter length via linear interpolation
    n = min(n_sim, n_gt, 500)  # cap at 500 points for speed
    t_sim = np.linspace(0, 1, n_sim)
    t_gt = np.linspace(0, 1, n_gt)
    t_common = np.linspace(0, 1, n)

    sim_x = np.interp(t_common, t_sim, traj_sim[:, 0])
    sim_y = np.interp(t_common, t_sim, traj_sim[:, 1])
    gt_x = np.interp(t_common, t_gt, traj_gt[:, 0])
    gt_y = np.interp(t_common, t_gt, traj_gt[:, 1])

    return float(np.mean(np.sqrt((sim_x - gt_x)**2 + (sim_y - gt_y)**2)))


def build_sweep_configs():
    """Build list of parameter dicts for the sweep."""
    # Stage 1: coarse sweep over most impactful params
    configs = []

    base = dict(
        solver="Newton", iterations=10, ls_iterations=10,
        cone="pyramidal", integrator="Euler", impratio=1.0,
        ground_torsional=0.1, ground_rolling=0.01,
        front_torsional=0.1, front_rolling=0.01,
        rear_torsional=0.1, rear_rolling=0.01,
        solref0=0.02, solref1=1.0,
        solimp0=0.9, solimp1=0.95, solimp2=0.001,
        condim=3,
    )

    # Coarse grid: dt × kv × friction
    for dt in [0.001, 0.002, 0.005]:
        for kv in [50, 100, 150, 200]:
            for front_friction in [0.5, 0.7, 1.0, 1.5]:
                for rear_friction in [0.2, 0.35, 0.5]:
                    cfg = {**base, "dt": dt, "kv": kv,
                           "front_friction": front_friction,
                           "rear_friction": rear_friction,
                           "ground_friction": front_friction}
                    configs.append(cfg)

    return configs


def build_fine_sweep(best_cfg: dict):
    """Fine sweep around the best coarse config, adding MuJoCo-specific params."""
    configs = []

    # Fine dt/kv/friction around best
    dt_best = best_cfg["dt"]
    kv_best = best_cfg["kv"]
    ff_best = best_cfg["front_friction"]
    rf_best = best_cfg["rear_friction"]

    for dt in [dt_best * 0.5, dt_best, dt_best * 2]:
        for kv in [kv_best * 0.8, kv_best, kv_best * 1.2]:
            for ff_mult in [0.9, 1.0, 1.1]:
                cfg = {**best_cfg, "dt": dt, "kv": kv,
                       "front_friction": ff_best * ff_mult,
                       "rear_friction": rf_best * ff_mult,
                       "ground_friction": ff_best * ff_mult}

                # Also sweep contact solver params
                for solref0 in [0.005, 0.02, 0.05]:
                    for condim in [3, 4]:
                        for cone in ["pyramidal", "elliptic"]:
                            cfg2 = {**cfg, "solref0": solref0,
                                    "condim": condim, "cone": cone}
                            configs.append(cfg2)

    return configs


def run_sweep(configs, traj_gt, label=""):
    """Run all configs, return sorted results."""
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        try:
            traj = simulate(cfg)
            traj_np = np.array(traj)
            err = trajectory_error(traj_np, traj_gt)
            results.append({"params": cfg, "error": err,
                            "final_xy": traj[-1]})
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  {label} [{i+1}/{n}] err={err:.4f} "
                      f"dt={cfg['dt']} kv={cfg['kv']} "
                      f"ff={cfg['front_friction']} rf={cfg['rear_friction']}")
        except Exception as e:
            results.append({"params": cfg, "error": float("inf"),
                            "final_xy": [0, 0], "exception": str(e)})

    results.sort(key=lambda r: r["error"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True,
                        help="Path to ground truth trajectory JSON")
    parser.add_argument("--save", metavar="PATH",
                        help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="Print top N results (default: 10)")
    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth) as f:
        gt = json.load(f)
    traj_gt = np.array(gt["target_trajectory"])
    print(f"Ground truth: {gt.get('simulator', '?')}, "
          f"dt={gt['dt']}, T={gt['T']}, "
          f"final=({traj_gt[-1, 0]:.3f}, {traj_gt[-1, 1]:.3f})")

    # Stage 1: coarse sweep
    print(f"\n=== Stage 1: Coarse sweep ===")
    coarse_configs = build_sweep_configs()
    print(f"Running {len(coarse_configs)} configurations...")
    t0 = time.perf_counter()
    coarse_results = run_sweep(coarse_configs, traj_gt, "coarse")
    t_coarse = time.perf_counter() - t0
    print(f"Coarse sweep done in {t_coarse:.1f}s")

    print(f"\nTop {args.top} coarse results:")
    for i, r in enumerate(coarse_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} kv={p['kv']} "
              f"ff={p['front_friction']} rf={p['rear_friction']} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    # Stage 2: fine sweep around best coarse config
    print(f"\n=== Stage 2: Fine sweep ===")
    best_coarse = coarse_results[0]["params"]
    fine_configs = build_fine_sweep(best_coarse)
    print(f"Running {len(fine_configs)} configurations...")
    t0 = time.perf_counter()
    fine_results = run_sweep(fine_configs, traj_gt, "fine")
    t_fine = time.perf_counter() - t0
    print(f"Fine sweep done in {t_fine:.1f}s")

    print(f"\nTop {args.top} fine results:")
    for i, r in enumerate(fine_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} kv={p['kv']} "
              f"ff={p['front_friction']:.2f} rf={p['rear_friction']:.2f} "
              f"solref0={p['solref0']} condim={p['condim']} cone={p['cone']} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    # Save
    if args.save:
        best = fine_results[0]
        output = {
            "simulator": "MuJoCo",
            "ground_truth": args.ground_truth,
            "best_params": best["params"],
            "best_error": best["error"],
            "best_final_xy": best["final_xy"],
            "coarse_sweep_size": len(coarse_configs),
            "fine_sweep_size": len(fine_configs),
            "top_10_coarse": [{"error": r["error"], "params": r["params"],
                               "final_xy": r["final_xy"]}
                              for r in coarse_results[:10]],
            "top_10_fine": [{"error": r["error"], "params": r["params"],
                             "final_xy": r["final_xy"]}
                            for r in fine_results[:10]],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
