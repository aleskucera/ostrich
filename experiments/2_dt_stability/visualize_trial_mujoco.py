"""Visualize a single MuJoCo trial from sweep_mujoco as trajectory plots.

Regenerates the same perturbed config as `sweep_mujoco.py --seed SEED --num-trials N`,
runs the sim at a chosen dt, and plots chassis x(t), z(t), wheel angular velocities,
and top-down (x,y) path with the obstacle footprint.

Usage:
    python visualize_trial_mujoco.py --seed 0 --num-trials 10 --trial 1
    python visualize_trial_mujoco.py --trial 1 --dt 0.0005 --save trial1.png
"""
import argparse
import math
import pathlib

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from sweep_mujoco import (
    HELHEST_OBSTACLE_XML, KV, MU, OBSTACLE_MU, RAMP_TIME, DURATION,
    OBSTACLE_HEIGHT_RANGE, OBSTACLE_X_RANGE, WHEEL_VEL_RANGE, INITIAL_YAW_RANGE,
)


def sample_trial_params(seed: int, num_trials: int, trial: int) -> dict:
    """Replay the RNG used in sweep_mujoco.main() to get trial `trial` (1-indexed)."""
    rng = np.random.default_rng(seed)
    for _ in range(trial - 1):
        rng.uniform(*OBSTACLE_HEIGHT_RANGE)
        rng.uniform(*OBSTACLE_X_RANGE)
        rng.uniform(*WHEEL_VEL_RANGE)
        rng.uniform(*INITIAL_YAW_RANGE)
    return {
        "obstacle_height": float(rng.uniform(*OBSTACLE_HEIGHT_RANGE)),
        "obstacle_x": float(rng.uniform(*OBSTACLE_X_RANGE)),
        "wheel_vel": float(rng.uniform(*WHEEL_VEL_RANGE)),
        "initial_yaw": float(rng.uniform(*INITIAL_YAW_RANGE)),
    }


def simulate_record(dt: float, trial_params: dict, duration: float) -> dict:
    qw = math.cos(trial_params["initial_yaw"] / 2.0)
    qz = math.sin(trial_params["initial_yaw"] / 2.0)
    xml = HELHEST_OBSTACLE_XML.format(
        dt=dt, kv=KV, mu=MU, obstacle_mu=OBSTACLE_MU,
        obstacle_x=trial_params["obstacle_x"],
        obstacle_height=trial_params["obstacle_height"],
        chassis_qw=qw, chassis_qz=qz,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    n_steps = int(duration / dt)
    t = np.zeros(n_steps + 1)
    x = np.zeros(n_steps + 1); y = np.zeros(n_steps + 1); z = np.zeros(n_steps + 1)
    wL = np.zeros(n_steps + 1); wR = np.zeros(n_steps + 1); wRear = np.zeros(n_steps + 1)

    jid = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in ("left_wheel_j", "right_wheel_j", "rear_wheel_j")}
    qvel_adr = {name: model.jnt_dofadr[jid[name]] for name in jid}

    def snapshot(k, cur_t):
        t[k] = cur_t
        x[k] = data.qpos[0]; y[k] = data.qpos[1]; z[k] = data.qpos[2]
        wL[k]   = data.qvel[qvel_adr["left_wheel_j"]]
        wR[k]   = data.qvel[qvel_adr["right_wheel_j"]]
        wRear[k] = data.qvel[qvel_adr["rear_wheel_j"]]

    snapshot(0, 0.0)

    blow_up_step = None
    for step in range(n_steps):
        cur_t = (step + 1) * dt
        ramp = min(cur_t / RAMP_TIME, 1.0)
        wv = trial_params["wheel_vel"] * ramp
        data.ctrl[:] = [wv, wv, wv]
        mujoco.mj_step(model, data)
        if (not np.isfinite(data.qpos[:3]).all()) or abs(float(data.qpos[2])) > 100:
            blow_up_step = step
            break
        snapshot(step + 1, cur_t)

    last = blow_up_step if blow_up_step is not None else n_steps
    return {
        "t": t[: last + 1], "x": x[: last + 1], "y": y[: last + 1], "z": z[: last + 1],
        "wL": wL[: last + 1], "wR": wR[: last + 1], "wRear": wRear[: last + 1],
        "blew_up": blow_up_step is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Must match the sweep you want to reproduce")
    parser.add_argument("--trial", type=int, default=1, help="1-indexed trial number")
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    params = sample_trial_params(args.seed, args.num_trials, args.trial)
    print(f"Trial {args.trial}/{args.num_trials}  (seed={args.seed}, dt={args.dt})")
    for k, v in params.items():
        print(f"  {k} = {v:.4f}")

    out = simulate_record(args.dt, params, args.duration)
    status = "BLEW UP" if out["blew_up"] else "finished"
    print(f"  -> sim {status}: t=[0, {out['t'][-1]:.3f}], "
          f"x=[{out['x'].min():.3f}, {out['x'].max():.3f}], "
          f"z=[{out['z'].min():.3f}, {out['z'].max():.3f}]")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    ax = axes[0, 0]
    ax.plot(out["t"], out["x"], color="#2196F3", label="chassis x")
    ax.plot(out["t"], out["x"] + 0.36, color="#2196F3", ls=":", lw=1.0,
            label="front wheel rim (chassis + R)")
    face_x = params["obstacle_x"] - 0.5  # half-size of obstacle in x
    ax.axhline(face_x, color="k", ls="--", lw=1,
               label=f'obstacle front face x={face_x:.2f}')
    ax.axhline(params["obstacle_x"], color="gray", ls=":", lw=1, alpha=0.6,
               label=f'obstacle center x={params["obstacle_x"]:.2f}')
    ax.set_xlabel("time (s)"); ax.set_ylabel("x (m)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("forward progress")

    ax = axes[0, 1]
    ax.plot(out["t"], out["z"], color="#E91E63")
    ax.axhline(0.37, color="k", ls=":", lw=1, alpha=0.5, label="rest z=0.37")
    ax.set_xlabel("time (s)"); ax.set_ylabel("chassis z (m)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("chassis height")

    ax = axes[1, 0]
    ax.plot(out["t"], out["wL"],    label="left",  color="#2196F3")
    ax.plot(out["t"], out["wR"],    label="right", color="#E91E63")
    ax.plot(out["t"], out["wRear"], label="rear",  color="#4CAF50")
    ax.axhline(params["wheel_vel"], color="k", ls="--", lw=1, alpha=0.5,
               label=f'cmd={params["wheel_vel"]:.2f}')
    ax.set_xlabel("time (s)"); ax.set_ylabel("wheel ω (rad/s)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2); ax.set_title("wheel angular velocities")

    ax = axes[1, 1]
    ax.plot(out["x"], out["y"], color="#607D8B", lw=1.5)
    ax.plot(out["x"][0], out["y"][0], "go", label="start")
    ax.plot(out["x"][-1], out["y"][-1], "r^", label="end")
    ox = params["obstacle_x"]; oh = params["obstacle_height"]
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((ox - 0.5, -1.0), 1.0, 2.0,
                           facecolor="gray", alpha=0.35, edgecolor="k", linewidth=0.8))
    ax.text(ox, 1.05, f"obstacle (2h={2*oh:.2f}m)", ha="center", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.3)
    ax.set_aspect("equal"); ax.legend(fontsize=9)
    ax.set_title("top-down path")

    fig.suptitle(
        f"MuJoCo trial {args.trial}  dt={args.dt}  "
        f"2h_obs={2*params['obstacle_height']:.2f}m  ω_cmd={params['wheel_vel']:.2f}  "
        f"yaw0={params['initial_yaw']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()

    if args.save:
        out_path = pathlib.Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
