"""Hyperparameter sweep for TinyDiffSim helhest — minimize trajectory error vs real robot.

TinyDiffSim uses explicit Euler integration, so KV is limited before instability.
It only supports sphere collision (no cylinders), and has a single global friction.

Usage:
    python examples/comparison_accuracy/helhest/sweep_tinydiffsim.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json

    python examples/comparison_accuracy/helhest/sweep_tinydiffsim.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json \
        --save results/sweep_tinydiffsim_14_46_18.json
"""
import argparse
import json
import pathlib
import tempfile
import time

import numpy as np
import pytinydiffsim_ad as pd

NU = 3

_HELHEST_URDF = """\
<?xml version="1.0"?>
<robot name="helhest_sphere_wheels">

  <link name="base_link">
    <collision>
      <origin xyz="-0.047 0 0"/>
      <geometry><box size="0.26 0.6 0.18"/></geometry>
    </collision>
    <inertial>
      <mass value="85"/>
      <origin xyz="-0.047 0 0"/>
      <inertia ixx="0.6213" ixy="0" ixz="0" iyy="0.1583" iyz="0" izz="0.677"/>
    </inertial>
  </link>

  <link name="left_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <link name="right_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <link name="rear_wheel">
    <collision>
      <geometry><sphere radius="0.36"/></geometry>
    </collision>
    <inertial>
      <mass value="5.5"/>
      <inertia ixx="0.20045" ixy="0" ixz="0" iyy="0.20045" iyz="0" izz="0.3888"/>
    </inertial>
  </link>

  <joint name="left_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.36 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.36 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="rear_wheel_j" type="continuous">
    <parent link="base_link"/>
    <child link="rear_wheel"/>
    <origin xyz="-0.697 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

</robot>
"""

WHEEL_QD = [6, 7, 8]
WHEEL_TAU = [0, 1, 2]


def _load_robot(world):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(_HELHEST_URDF)
        urdf_path = f.name
    parser = pd.TinyUrdfParser()
    urdf_structs = parser.load_urdf(urdf_path)
    mb = pd.TinyMultiBody(True)
    pd.UrdfToMultiBody2().convert2(urdf_structs, world, mb)
    return mb


def _make_ground(world):
    urdf = pd.TinyUrdfStructures()
    bl = pd.TinyUrdfLink()
    bl.link_name = "ground"
    ine = pd.TinyUrdfInertial()
    ine.mass = pd.ADDouble(0.0)
    ine.inertia_xxyyzz = pd.Vector3(0.0, 0.0, 0.0)
    bl.urdf_inertial = ine
    col = pd.TinyUrdfCollision()
    col.geometry.geom_type = pd.PLANE_TYPE
    bl.urdf_collision_shapes = [col]
    urdf.base_links = [bl]
    mb = pd.TinyMultiBody(False)
    pd.UrdfToMultiBody2().convert2(urdf, world, mb)
    return mb


def simulate(dt, kv, friction, target_ctrl, duration, wheel_timeseries=None):
    """Run forward simulation, return xy trajectory as list of [x, y]."""
    T = int(duration / dt)

    world = pd.TinyWorld()
    world.gravity = pd.Vector3(0.0, 0.0, -9.81)
    world.friction = pd.ADDouble(friction)

    ground_mb = _make_ground(world)
    robot_mb = _load_robot(world)

    robot_mb.q[0] = pd.ADDouble(0.0)
    robot_mb.q[1] = pd.ADDouble(0.0)
    robot_mb.q[2] = pd.ADDouble(0.0)
    robot_mb.q[3] = pd.ADDouble(1.0)
    robot_mb.q[4] = pd.ADDouble(0.0)
    robot_mb.q[5] = pd.ADDouble(0.0)
    robot_mb.q[6] = pd.ADDouble(0.36)

    mbs = [ground_mb, robot_mb]
    dispatcher = world.get_collision_dispatcher()
    solver = pd.TinyMultiBodyConstraintSolver()
    dt_ad = pd.ADDouble(dt)

    if wheel_timeseries:
        ts_t = np.array([p["t"] for p in wheel_timeseries], dtype=np.float64)
        ts_left = np.array([p["left"] for p in wheel_timeseries], dtype=np.float64)
        ts_right = np.array([p["right"] for p in wheel_timeseries], dtype=np.float64)
        ts_rear = np.array([p["rear"] for p in wheel_timeseries], dtype=np.float64)

    traj = [[0.0, 0.0]]

    for step in range(T):
        if wheel_timeseries:
            t = (step + 1) * dt
            ctrl = [
                float(np.interp(t, ts_t, ts_left)),
                float(np.interp(t, ts_t, ts_right)),
                float(np.interp(t, ts_t, ts_rear)),
            ]
        else:
            ctrl = target_ctrl

        for i in range(NU):
            robot_mb.tau[WHEEL_TAU[i]] = (
                pd.ADDouble(ctrl[i]) - robot_mb.qd[WHEEL_QD[i]]
            ) * pd.ADDouble(kv)

        pd.forward_kinematics(robot_mb, robot_mb.q, robot_mb.qd)
        pd.forward_dynamics(robot_mb, world.gravity)
        contacts = world.compute_contacts_multi_body(mbs, dispatcher)
        flat_contacts = [c for sublist in contacts for c in sublist]
        solver.resolve_collision(flat_contacts, dt_ad)
        pd.integrate_euler(robot_mb, dt_ad)

        x = robot_mb.q[4].value()
        y = robot_mb.q[5].value()

        # Detect instability
        if abs(x) > 100 or abs(y) > 100:
            return None

        traj.append([x, y])

    return traj


def trajectory_error(traj_sim, traj_gt):
    sim_np = np.array(traj_sim)
    gt_np = np.array(traj_gt)
    n_sim, n_gt = len(sim_np), len(gt_np)
    n = min(n_sim, n_gt, 500)

    t_sim = np.linspace(0, 1, n_sim)
    t_gt = np.linspace(0, 1, n_gt)
    t_common = np.linspace(0, 1, n)

    sim_x = np.interp(t_common, t_sim, sim_np[:, 0])
    sim_y = np.interp(t_common, t_sim, sim_np[:, 1])
    gt_x = np.interp(t_common, t_gt, gt_np[:, 0])
    gt_y = np.interp(t_common, t_gt, gt_np[:, 1])

    return float(np.mean(np.sqrt((sim_x - gt_x)**2 + (sim_y - gt_y)**2)))


def run_sweep(configs, gt_data_list, label=""):
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        t0 = time.perf_counter()
        errors = []
        trajectories = {}

        for gt_entry in gt_data_list:
            bag_name = gt_entry["gt_data"].get("bag_name", "?")
            try:
                traj = simulate(target_ctrl=gt_entry["target_ctrl"],
                                duration=gt_entry["duration"],
                                wheel_timeseries=gt_entry["wheel_ts"], **cfg)
                if traj is None:
                    raise RuntimeError("unstable")
                err = trajectory_error(traj, gt_entry["traj_gt"])
                errors.append(err)
                trajectories[bag_name] = {"trajectory": traj, "error": err}
            except Exception as e:
                errors.append(float("inf"))
                trajectories[bag_name] = {"error": float("inf"), "exception": str(e)[:200]}

        elapsed = time.perf_counter() - t0
        combined_err = float(np.mean(errors))
        results.append({"params": cfg, "error": combined_err, "per_trajectory": trajectories})

        if (i + 1) % 10 == 0 or i == 0:
            err_strs = " + ".join(f"{e:.4f}" for e in errors)
            print(f"  {label} [{i+1}/{n}] err={combined_err:.4f} ({err_strs}) "
                  f"dt={cfg['dt']} kv={cfg['kv']} fr={cfg['friction']} ({elapsed:.1f}s)")

    results.sort(key=lambda r: r["error"])
    return results


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


def build_coarse_configs():
    configs = []
    # Explicit Euler: dt must be small, kv limited
    for dt in [0.001, 0.002, 0.005]:
        for kv in [20.0, 30.0, 50.0, 80.0, 120.0]:
            for friction in [0.1, 0.2, 0.35, 0.5, 0.7]:
                configs.append(dict(dt=dt, kv=kv, friction=friction))
    return configs


def build_fine_configs(best):
    configs = []
    dt_b = best["dt"]
    kv_b = best["kv"]
    fr_b = best["friction"]

    for dt in [dt_b * 0.5, dt_b, dt_b * 2.0]:
        for kv in [kv_b * 0.8, kv_b * 0.9, kv_b, kv_b * 1.1, kv_b * 1.2]:
            for fr_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
                configs.append(dict(dt=dt, kv=kv, friction=fr_b * fr_mult))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, nargs="+",
                        help="Path(s) to extracted rosbag JSON(s)")
    parser.add_argument("--save", metavar="PATH", help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10, help="Print top N results")
    parser.add_argument("--dt", type=float, nargs="+", help="Timestep values to sweep")
    parser.add_argument("--kv", type=float, nargs="+", help="Velocity servo gain values to sweep")
    parser.add_argument("--friction", "--mu", type=float, nargs="+", help="Friction coefficient values to sweep")
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

    # Mode 1: Provided grid sweep
    if args.dt and args.kv and args.friction:
        configs = []
        for dt in args.dt:
            for kv in args.kv:
                for friction in args.friction:
                    configs.append(dict(dt=dt, kv=kv, friction=friction))
        
        print(f"\n=== Grid sweep: {len(configs)} configs ===")
        t0 = time.perf_counter()
        results = run_sweep(configs, gt_data_list, "sweep")
        elapsed = time.perf_counter() - t0
        print(f"Sweep done in {elapsed:.1f}s")
        
        coarse_configs = configs # for metadata
        fine_configs = []

    # Mode 2: Two-stage coarse/fine sweep
    else:
        print(f"\n=== Stage 1: Coarse sweep ===")
        coarse_configs = build_coarse_configs()
        print(f"Running {len(coarse_configs)} configs x {len(gt_data_list)} trajectories...")
        t0 = time.perf_counter()
        coarse_results = run_sweep(coarse_configs, gt_data_list, "coarse")
        t_coarse = time.perf_counter() - t0
        print(f"Coarse sweep done in {t_coarse:.1f}s")

        print(f"\nTop {args.top} coarse results:")
        for i, r in enumerate(coarse_results[:args.top]):
            p = r["params"]
            per_traj = r["per_trajectory"]
            errs = " + ".join(f"{v['error']:.4f}" for v in per_traj.values())
            print(f"  {i+1}. err={r['error']:.4f} ({errs}) | "
                  f"dt={p['dt']} kv={p['kv']} fr={p['friction']}")

        print(f"\n=== Stage 2: Fine sweep ===")
        best_coarse = coarse_results[0]["params"]
        fine_configs = build_fine_configs(best_coarse)
        print(f"Running {len(fine_configs)} configs x {len(gt_data_list)} trajectories...")
        t0 = time.perf_counter()
        results = run_sweep(fine_configs, gt_data_list, "fine")
        t_fine = time.perf_counter() - t0
        print(f"Fine sweep done in {t_fine:.1f}s")

    print(f"\nTop {args.top} results:")
    for i, r in enumerate(results[:args.top]):
        p = r["params"]
        per_traj = r["per_trajectory"]
        errs = " + ".join(f"{v['error']:.4f}" for v in per_traj.values())
        print(f"  {i+1}. err={r['error']:.4f} ({errs}) | "
              f"dt={p['dt']} kv={p['kv']:.1f} fr={p['friction']:.3f}")

    if args.save:
        best = results[0]
        output = {
            "simulator": "TinyDiffSim",
            "ground_truth": args.ground_truth,
            "ground_truth_source": "real_robot",
            "best_params": best["params"],
            "best_error": best["error"],
            "best_per_trajectory": best["per_trajectory"],
            "coarse_sweep_size": len(coarse_configs),
            "fine_sweep_size": len(fine_configs),
            "top_10": [{"error": r["error"], "params": r["params"]}
                       for r in results[:10]
                       if r["error"] != float("inf")],
        }
        if not fine_configs:
            # Special case for grid sweep
            output["sweep_size"] = len(coarse_configs)

        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
