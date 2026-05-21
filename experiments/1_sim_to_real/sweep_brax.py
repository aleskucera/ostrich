"""Hyperparameter sweep for Brax helhest — minimize trajectory error vs real robot.

Brax has no cylinder collision geometry, so wheels are approximated as spheres
(same approximation as TinyDiffSim). Uses the positional pipeline by default,
which is the most stable pipeline on this robot (see brax_sweep.py notes).

Usage:
    python experiments/1_sim_to_real/sweep_brax.py \
        --ground-truth ../data/right_turn_b.json ../data/acceleration.json \
        --dt 0.005 0.01 --kv 100 150 200 --mu 0.3 0.5 0.7 \
        --save results/sweep_brax.json
"""
import argparse
import json
import os
import pathlib
import time
import warnings

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import brax.io.mjcf as mjcf

BASE_XML = """<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom name="ground" type="plane" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.677"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="sphere" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_vel"  joint="left_wheel_j"  kv="150"/>
    <velocity name="right_vel" joint="right_wheel_j" kv="150"/>
    <velocity name="rear_vel"  joint="rear_wheel_j"  kv="150"/>
  </actuator>
</mujoco>"""


def patch_sys(sys, dt, kv, mu):
    kv_arr = jnp.array(kv, jnp.float32)
    mu_arr = jnp.full_like(sys.geom_friction[:, 0], jnp.float32(mu))
    return sys.replace(
        opt=sys.opt.replace(timestep=jnp.array(dt, jnp.float32)),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(kv_arr),
        actuator_biasprm=sys.actuator_biasprm.at[:, 2].set(-kv_arr),
        geom_friction=sys.geom_friction.at[:, 0].set(mu_arr),
    )


_rollout_cache: dict = {}


def get_rollout_fn(pipeline, T):
    key = (id(pipeline), T)
    if key in _rollout_cache:
        return _rollout_cache[key]

    def rollout(sys, ctrl_seq):
        q0 = jnp.zeros(sys.q_size()).at[2].set(0.37).at[3].set(1.0)
        qd0 = jnp.zeros(sys.qd_size())
        state = pipeline.init(sys, q0, qd0)

        def step(state, ctrl):
            state = pipeline.step(sys, state, ctrl)
            return state, state.x.pos[0, :2]

        _, xy = jax.lax.scan(step, state, ctrl_seq)
        return xy  # (T, 2)

    fn = jax.jit(rollout)
    _rollout_cache[key] = fn
    return fn


def build_ctrl_seq(T, dt, target_ctrl, wheel_timeseries):
    if wheel_timeseries:
        ts_t = np.array([p["t"] for p in wheel_timeseries], dtype=np.float64)
        ts_l = np.array([p["left"] for p in wheel_timeseries], dtype=np.float64)
        ts_r = np.array([p["right"] for p in wheel_timeseries], dtype=np.float64)
        ts_re = np.array([p["rear"] for p in wheel_timeseries], dtype=np.float64)
        t = (np.arange(T) + 1) * dt
        seq = np.stack([np.interp(t, ts_t, ts_l),
                        np.interp(t, ts_t, ts_r),
                        np.interp(t, ts_t, ts_re)], axis=-1)
    else:
        seq = np.tile(np.asarray(target_ctrl, dtype=np.float32), (T, 1))
    return jnp.asarray(seq, dtype=jnp.float32)


def simulate(pipeline, sys_base, dt, kv, mu, target_ctrl, duration, wheel_timeseries):
    T = int(duration / dt)
    sys = patch_sys(sys_base, dt=dt, kv=kv, mu=mu)
    ctrl_seq = build_ctrl_seq(T, dt, target_ctrl, wheel_timeseries)
    rollout = get_rollout_fn(pipeline, T)
    xy = rollout(sys, ctrl_seq)
    xy.block_until_ready()
    xy_np = np.asarray(xy)
    if not np.all(np.isfinite(xy_np)) or np.max(np.abs(xy_np)) > 100:
        return None
    return xy_np.tolist()


def trajectory_error(traj_sim, traj_gt, sim_duration=None, gt_duration=None):
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
    return np.array(traj_xy), target_ctrl, duration, gt, wheel_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, nargs="+")
    parser.add_argument("--save", metavar="PATH")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--pipeline", choices=["positional", "generalized", "spring"],
                        default="positional")
    parser.add_argument("--dt", type=float, nargs="+", default=[0.005, 0.01])
    parser.add_argument("--kv", type=float, nargs="+", default=[100, 150, 200])
    parser.add_argument("--mu", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = parser.parse_args()

    if args.pipeline == "positional":
        import brax.positional.pipeline as pipe
    elif args.pipeline == "generalized":
        import brax.generalized.pipeline as pipe
    else:
        import brax.spring.pipeline as pipe

    print(f"Loading Brax model (pipeline={args.pipeline})...", flush=True)
    sys_base = mjcf.loads(BASE_XML)

    gt_data_list = []
    for gt_path in args.ground_truth:
        traj_gt, target_ctrl, duration, gt_data, wheel_ts = load_ground_truth(gt_path)
        gt_data_list.append(dict(
            path=gt_path, traj_gt=traj_gt.tolist(), target_ctrl=target_ctrl,
            duration=duration, gt_data=gt_data, wheel_ts=wheel_ts,
        ))
        print(f"Ground truth: {gt_data.get('bag_name', '?')}")
        print(f"  duration={duration:.1f}s, {len(traj_gt)} points, "
              f"final=({traj_gt[-1, 0]:.3f}, {traj_gt[-1, 1]:.3f})")

    configs = [dict(dt=dt, kv=kv, mu=mu)
               for dt in args.dt for kv in args.kv for mu in args.mu]
    print(f"\n=== Sweep: {len(configs)} configs x {len(gt_data_list)} trajectories ===")

    results = []
    t_start = time.perf_counter()
    for i, cfg in enumerate(configs):
        errors = []
        trajectories = {}
        for gt_entry in gt_data_list:
            bag_name = gt_entry["gt_data"].get("bag_name", "?")
            try:
                traj = simulate(pipe, sys_base, cfg["dt"], cfg["kv"], cfg["mu"],
                                gt_entry["target_ctrl"], gt_entry["duration"],
                                gt_entry["wheel_ts"])
                if traj is None:
                    raise RuntimeError("unstable")
                sim_dur = len(traj) * cfg["dt"]
                err = trajectory_error(traj, gt_entry["traj_gt"],
                                       sim_duration=sim_dur,
                                       gt_duration=gt_entry["duration"])
                errors.append(err)
                trajectories[bag_name] = {"trajectory": traj, "error": err}
            except Exception as e:
                errors.append(float("inf"))
                trajectories[bag_name] = {"error": float("inf"), "exception": str(e)[:200]}
        combined = float(np.mean(errors))
        results.append({"params": cfg, "error": combined, "per_trajectory": trajectories})
        if (i + 1) % 5 == 0 or i == 0:
            err_strs = " + ".join(f"{e:.4f}" for e in errors)
            print(f"  [{i+1}/{len(configs)}] err={combined:.4f} ({err_strs}) | "
                  f"dt={cfg['dt']} kv={cfg['kv']} mu={cfg['mu']}")
    elapsed = time.perf_counter() - t_start
    print(f"Sweep done in {elapsed:.1f}s")

    results.sort(key=lambda r: r["error"])
    print(f"\nTop {args.top} results:")
    for i, r in enumerate(results[:args.top]):
        p = r["params"]
        errs = " + ".join(f"{v['error']:.4f}" for v in r["per_trajectory"].values())
        print(f"  {i+1}. err={r['error']:.4f} ({errs}) | "
              f"dt={p['dt']} kv={p['kv']} mu={p['mu']}")

    if args.save:
        best = results[0]
        output = {
            "simulator": "Brax",
            "pipeline": args.pipeline,
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
