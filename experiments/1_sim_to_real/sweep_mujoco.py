"""Hyperparameter sweep for MuJoCo helhest — minimize trajectory error vs real robot.

Usage:
    python examples/comparison_accuracy/helhest/sweep_mujoco.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json

    python examples/comparison_accuracy/helhest/sweep_mujoco.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json \
        --save results/sweep_mujoco_14_46_18.json
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

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


def simulate(params: dict, target_ctrl: list, duration: float,
             wheel_timeseries: list[dict] | None = None) -> list:
    """Run forward simulation, return xy trajectory as list of [x, y]."""
    xml = HELHEST_XML_TEMPLATE.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = params["dt"]
    T = int(duration / dt)
    traj = []

    if wheel_timeseries:
        ts_t = np.array([p["t"] for p in wheel_timeseries], dtype=np.float64)
        ts_left = np.array([p["left"] for p in wheel_timeseries], dtype=np.float64)
        ts_right = np.array([p["right"] for p in wheel_timeseries], dtype=np.float64)
        ts_rear = np.array([p["rear"] for p in wheel_timeseries], dtype=np.float64)

    for step in range(T):
        if wheel_timeseries:
            t = (step + 1) * dt
            data.ctrl[0] = np.interp(t, ts_t, ts_left)
            data.ctrl[1] = np.interp(t, ts_t, ts_right)
            data.ctrl[2] = np.interp(t, ts_t, ts_rear)
        else:
            data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)
        traj.append([float(data.qpos[0]), float(data.qpos[1])])

    return traj


def trajectory_error(traj_sim, traj_gt, sim_duration=None, gt_duration=None) -> float:
    """Mean L2 distance, resampled on common physical time if durations given."""
    sim_np = np.asarray(traj_sim)
    gt_np = np.asarray(traj_gt)
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


BASE_PARAMS = dict(
    solver="Newton", iterations=50, ls_iterations=50,
    cone="pyramidal", integrator="implicitfast", impratio=1.0,
    ground_torsional=0.1, ground_rolling=0.01,
    front_torsional=0.1, front_rolling=0.01,
    rear_torsional=0.1, rear_rolling=0.01,
    solref0=0.005, solref1=1.0,
    solimp0=0.9, solimp1=0.95, solimp2=0.001,
    condim=3,
)


def build_configs(dt_values, kv_values, mu_values):
    configs = []
    for dt in dt_values:
        for kv in kv_values:
            for mu in mu_values:
                cfg = {**BASE_PARAMS, "dt": dt, "kv": kv,
                       "front_friction": mu,
                       "rear_friction": mu,
                       "ground_friction": mu}
                configs.append(cfg)
    return configs


def run_sweep(configs, gt_data_list, label=""):
    """Run all configs against all trajectories, return sorted by combined error."""
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        errors = []
        trajectories = {}
        for gt_entry in gt_data_list:
            bag_name = gt_entry["gt_data"].get("bag_name", "?")
            try:
                traj = simulate(cfg, gt_entry["target_ctrl"], gt_entry["duration"],
                                gt_entry["wheel_ts"])
                traj_np = np.array(traj)
                sim_dur = len(traj) * cfg["dt"]
                err = trajectory_error(traj_np, np.array(gt_entry["traj_gt"]),
                                       sim_duration=sim_dur,
                                       gt_duration=gt_entry["duration"])
                errors.append(err)
                trajectories[bag_name] = {"trajectory": traj, "error": err}
            except Exception as e:
                errors.append(float("inf"))
                trajectories[bag_name] = {"error": float("inf"), "exception": str(e)[:200]}

        combined_err = float(np.mean(errors))
        results.append({"params": cfg, "error": combined_err,
                        "per_trajectory": trajectories})

        if (i + 1) % 50 == 0 or i == 0:
            err_strs = " + ".join(f"{e:.4f}" for e in errors)
            print(f"  {label} [{i+1}/{n}] err={combined_err:.4f} ({err_strs}) "
                  f"dt={cfg['dt']} kv={cfg['kv']} mu={cfg['front_friction']}")

    results.sort(key=lambda r: r["error"])
    return results


def load_ground_truth(path):
    """Load ground truth from extract_rosbag.py output."""
    with open(path) as f:
        gt = json.load(f)

    duration = gt["trajectory"].get("constant_speed_duration_s", gt["trajectory"]["duration_s"])
    traj_xy = [[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration]
    target_ctrl = gt["target_ctrl_rad_s"]
    wheel_ts = None
    if "wheel_velocities" in gt and "timeseries" in gt["wheel_velocities"]:
        wheel_ts = gt["wheel_velocities"]["timeseries"]

    return np.array(traj_xy), target_ctrl, duration, gt, wheel_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, nargs="+",
                        help="Path(s) to extracted rosbag JSON(s)")
    parser.add_argument("--save", metavar="PATH", help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dt", type=float, nargs="+", default=[0.001, 0.002, 0.005],
                        help="Timestep values to sweep")
    parser.add_argument("--kv", type=float, nargs="+", default=[500, 1000, 1500, 2000],
                        help="Velocity servo gain values to sweep")
    parser.add_argument("--mu", type=float, nargs="+", default=[0.1, 0.2, 0.35, 0.5, 0.7, 1.0],
                        help="Friction coefficient values to sweep")
    parser.add_argument("--cmd-vel-bag", type=str, nargs="+", default=None,
                        help="Rosbag dir(s); use /cmd_vel (diff-drive kinematic) as "
                             "wheel target. Must align 1:1 with --ground-truth paths.")
    args = parser.parse_args()

    cmd_vel_bags = args.cmd_vel_bag
    if cmd_vel_bags and len(cmd_vel_bags) != len(args.ground_truth):
        parser.error("--cmd-vel-bag must have same count as --ground-truth")

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
        gt_data_list.append({
            "path": gt_path,
            "traj_gt": traj_gt.tolist(),
            "target_ctrl": target_ctrl,
            "duration": duration,
            "gt_data": gt_data,
            "wheel_ts": wheel_ts,
        })
        print(f"Ground truth: {gt_data.get('bag_name', '?')}")
        print(f"  duration={duration:.1f}s, {len(traj_gt)} points, "
              f"final=({traj_gt[-1, 0]:.3f}, {traj_gt[-1, 1]:.3f})")
        if wheel_ts:
            print(f"  Using time-varying wheel velocities ({len(wheel_ts)} points)")

    configs = build_configs(args.dt, args.kv, args.mu)
    print(f"\n=== Sweep: {len(configs)} configs x {len(gt_data_list)} trajectories "
          f"(dt={args.dt}, kv={args.kv}, mu={args.mu}) ===")

    t0 = time.perf_counter()
    results = run_sweep(configs, gt_data_list, "sweep")
    elapsed = time.perf_counter() - t0
    print(f"Sweep done in {elapsed:.1f}s")

    print(f"\nTop {args.top} results:")
    for i, r in enumerate(results[:args.top]):
        p = r["params"]
        per_traj = r["per_trajectory"]
        errs = " + ".join(f"{v['error']:.4f}" for v in per_traj.values())
        print(f"  {i+1}. err={r['error']:.4f} ({errs}) | "
              f"dt={p['dt']} kv={p['kv']} mu={p['front_friction']}")

    if args.save:
        best = results[0]
        output = {
            "simulator": "MuJoCo",
            "ground_truth": args.ground_truth,
            "ground_truth_source": "real_robot",
            "best_params": best["params"],
            "best_error": best["error"],
            "best_per_trajectory": best["per_trajectory"],
            "sweep_size": len(configs),
            "top_10": [{"error": r["error"], "params": r["params"]}
                       for r in results[:10]],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
