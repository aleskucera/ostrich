"""Hyperparameter sweep for TinyDiffSim helhest — minimize trajectory error vs ground truth.

TinyDiffSim uses explicit Euler integration, so KV is limited to ~30-50
before instability. It also only supports sphere collision (no cylinders),
and has a single global friction parameter.

Sweepable parameters:
  - dt (timestep)
  - KV (velocity servo gain)
  - friction (global world friction)

Usage:
    python examples/comparison_gradient/helhest/sweep_tinydiffsim.py \
        --ground-truth results/helhest_chrono.json \
        --save results/sweep_tinydiffsim.json
"""
import argparse
import json
import pathlib
import tempfile
import time

import numpy as np
import pytinydiffsim_ad as pd

DURATION = 3.0
TARGET_CTRL = [1.0, 6.0, 0.0]
NU = 3

# Helhest URDF with sphere collision shapes for wheels
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


def simulate(dt, kv, friction):
    """Run forward simulation, return xy trajectory as list of [x, y]."""
    T = int(DURATION / dt)

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

    traj = [[0.0, 0.0]]

    for step in range(T):
        for i in range(NU):
            robot_mb.tau[WHEEL_TAU[i]] = (
                pd.ADDouble(TARGET_CTRL[i]) - robot_mb.qd[WHEEL_QD[i]]
            ) * pd.ADDouble(kv)

        pd.forward_kinematics(robot_mb, robot_mb.q, robot_mb.qd)
        pd.forward_dynamics(robot_mb, world.gravity)
        contacts = world.compute_contacts_multi_body(mbs, dispatcher)
        flat_contacts = [c for sublist in contacts for c in sublist]
        solver.resolve_collision(flat_contacts, dt_ad)
        pd.integrate_euler(robot_mb, dt_ad)

        x = robot_mb.q[4].value()
        y = robot_mb.q[5].value()
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


def run_sweep(configs, traj_gt, label=""):
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        try:
            traj = simulate(**cfg)
            err = trajectory_error(traj, traj_gt)
            results.append({"params": cfg, "error": err, "final_xy": traj[-1]})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {label} [{i+1}/{n}] err={err:.4f} "
                      f"dt={cfg['dt']} kv={cfg['kv']} fr={cfg['friction']}")
        except Exception as e:
            results.append({"params": cfg, "error": float("inf"),
                            "final_xy": [0, 0], "exception": str(e)})
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {label} [{i+1}/{n}] FAILED: {e}")

    results.sort(key=lambda r: r["error"])
    return results


def build_coarse_configs():
    configs = []
    # Explicit Euler limits: dt<=0.005, KV<=50
    for dt in [0.001, 0.002, 0.005]:
        for kv in [10.0, 20.0, 30.0, 40.0, 50.0]:
            for friction in [0.3, 0.5, 0.7, 1.0, 1.5]:
                configs.append(dict(dt=dt, kv=kv, friction=friction))
    return configs


def build_fine_configs(best):
    configs = []
    dt_b = best["dt"]
    kv_b = best["kv"]
    fr_b = best["friction"]

    for dt in [dt_b * 0.5, dt_b, dt_b * 2.0]:
        for kv in [kv_b * 0.8, kv_b * 0.9, kv_b, kv_b * 1.1, kv_b * 1.2]:
            for fr_mult in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
                configs.append(dict(
                    dt=dt, kv=kv, friction=fr_b * fr_mult,
                ))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True,
                        help="Path to ground truth trajectory JSON")
    parser.add_argument("--save", metavar="PATH",
                        help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="Print top N results (default: 10)")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        gt = json.load(f)
    traj_gt = gt["target_trajectory"]
    gt_np = np.array(traj_gt)
    print(f"Ground truth: {gt.get('simulator', '?')}, "
          f"dt={gt['dt']}, T={gt['T']}, "
          f"final=({gt_np[-1, 0]:.3f}, {gt_np[-1, 1]:.3f})")

    # Stage 1: coarse sweep
    print(f"\n=== Stage 1: Coarse sweep ===")
    coarse_configs = build_coarse_configs()
    print(f"Running {len(coarse_configs)} configurations...")
    t0 = time.perf_counter()
    coarse_results = run_sweep(coarse_configs, traj_gt, "coarse")
    t_coarse = time.perf_counter() - t0
    print(f"Coarse sweep done in {t_coarse:.1f}s")

    print(f"\nTop {args.top} coarse results:")
    for i, r in enumerate(coarse_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} kv={p['kv']} fr={p['friction']} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    # Stage 2: fine sweep
    print(f"\n=== Stage 2: Fine sweep ===")
    best_coarse = coarse_results[0]["params"]
    fine_configs = build_fine_configs(best_coarse)
    print(f"Running {len(fine_configs)} configurations...")
    t0 = time.perf_counter()
    fine_results = run_sweep(fine_configs, traj_gt, "fine")
    t_fine = time.perf_counter() - t0
    print(f"Fine sweep done in {t_fine:.1f}s")

    print(f"\nTop {args.top} fine results:")
    for i, r in enumerate(fine_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} kv={p['kv']:.1f} fr={p['friction']:.3f} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    if args.save:
        best = fine_results[0]
        output = {
            "simulator": "TinyDiffSim",
            "ground_truth": args.ground_truth,
            "best_params": best["params"],
            "best_error": best["error"],
            "best_final_xy": best["final_xy"],
            "coarse_sweep_size": len(coarse_configs),
            "fine_sweep_size": len(fine_configs),
            "top_10_coarse": [{"error": r["error"], "params": r["params"],
                               "final_xy": r["final_xy"]}
                              for r in coarse_results[:10]
                              if r["error"] != float("inf")],
            "top_10_fine": [{"error": r["error"], "params": r["params"],
                             "final_xy": r["final_xy"]}
                            for r in fine_results[:10]
                            if r["error"] != float("inf")],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
