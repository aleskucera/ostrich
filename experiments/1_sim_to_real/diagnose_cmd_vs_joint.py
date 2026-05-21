"""Diagnostic: compare /cmd_vel-derived vs /joint_states-derived wheel targets.

Runs MuJoCo (with Exp 1 calibrated params) on the turn bag twice:
  1. Target = /joint_states per-wheel velocities (what we currently use).
  2. Target = /cmd_vel (vx, wz) converted to per-wheel via diff-drive kinematics.

Prints combined L2 error for each and plots both sim trajectories over the GT.

Usage:
    python experiments/1_sim_to_real/diagnose_cmd_vs_joint.py \
        --bag data/rosbags/nuc/helhest_2026_04_10-14_46_18 \
        --gt experiments/data/right_turn_b.json
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from sweep_mujoco import simulate, trajectory_error, BASE_PARAMS

# Exp 1 calibrated params
BEST = {**BASE_PARAMS, "dt": 0.001, "kv": 4000.0,
        "front_friction": 0.2, "rear_friction": 0.2, "ground_friction": 0.2}

# Effective kinematic geometry fitted from cmd_vel vs joint_states (LS fit on
# bag helhest_2026_04_10-14_46_18). Physical URDF values are r=0.36, track=0.72;
# fitted values absorb motor lag/slip so cmd_vel → wheel targets match measurement.
WHEEL_RADIUS = 0.5741
HALF_TRACK = 0.384  # track_fit / 2 = 0.7680 / 2


def load_cmd_vel_timeseries(bag_dir):
    """Read /cmd_vel as list of (t, vx, wz).

    Accepts either a rosbag directory (reads /cmd_vel) or a JSON file produced
    by fit_joy_to_cmdvel.py --predict-bag (synthesized cmd_vel for bags that
    lack the topic).
    """
    import pathlib as _pl
    p = _pl.Path(bag_dir)
    if p.is_file() and p.suffix == ".json":
        import json as _json
        data = _json.loads(p.read_text())
        return [(float(c["t"]), float(c["vx"]), float(c["wz"]))
                for c in data["cmd_vel"]]

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.LATEST)

    msgs = []
    with Reader(bag_dir) as r:
        t_first = None
        for conn, t_ns, raw in r.messages():
            if conn.topic != "/cmd_vel":
                continue
            if t_first is None:
                t_first = t_ns
            m = ts.deserialize_cdr(raw, conn.msgtype)
            twist = m.twist if hasattr(m, "twist") else m
            msgs.append((
                (t_ns - t_first) * 1e-9,
                float(twist.linear.x),
                float(twist.angular.z),
            ))
    return msgs


def cmd_vel_to_wheel_ts(cmd_msgs, t_ref_ts):
    """Convert (t, vx, wz) cmd_vel stream to per-wheel timeseries aligned with t_ref."""
    # Normalize cmd_vel to start at t=0 like joint_states timeseries does.
    if not cmd_msgs:
        return []
    t_arr = np.array([m[0] for m in cmd_msgs])
    vx_arr = np.array([m[1] for m in cmd_msgs])
    wz_arr = np.array([m[2] for m in cmd_msgs])

    # Resample at same times as reference joint_states timeseries for direct comparison.
    t_ref = np.array([p["t"] for p in t_ref_ts])
    vx = np.interp(t_ref, t_arr, vx_arr)
    wz = np.interp(t_ref, t_arr, wz_arr)

    left = (vx - wz * HALF_TRACK) / WHEEL_RADIUS
    right = (vx + wz * HALF_TRACK) / WHEEL_RADIUS
    rear = vx / WHEEL_RADIUS

    return [
        {"t": float(t_ref[i]), "left": float(left[i]),
         "right": float(right[i]), "rear": float(rear[i])}
        for i in range(len(t_ref))
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", type=pathlib.Path, required=True,
                    help="Path to rosbag directory (contains metadata.yaml)")
    ap.add_argument("--gt", type=pathlib.Path, required=True,
                    help="Extracted rosbag JSON (e.g. right_turn_b.json)")
    ap.add_argument("--save", type=str, default="results/diag_cmd_vs_joint.png")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())
    duration = gt["trajectory"].get("constant_speed_duration_s",
                                    gt["trajectory"]["duration_s"])
    traj_gt = np.array(
        [[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration]
    )

    joint_ts = gt["wheel_velocities"]["timeseries"]

    cmd_msgs = load_cmd_vel_timeseries(args.bag)
    print(f"Loaded {len(cmd_msgs)} /cmd_vel msgs, {len(joint_ts)} /joint_states samples")
    cmd_ts = cmd_vel_to_wheel_ts(cmd_msgs, joint_ts)

    target_ctrl = gt["target_ctrl_rad_s"]

    runs = {
        "joint_states": joint_ts,
        "cmd_vel (kinematic)": cmd_ts,
    }
    results = {}
    for label, wts in runs.items():
        traj = simulate(BEST, target_ctrl, duration, wts)
        traj_np = np.array(traj)
        sim_dur = len(traj) * BEST["dt"]
        err = trajectory_error(traj_np, traj_gt,
                               sim_duration=sim_dur, gt_duration=duration)
        results[label] = (traj_np, err)
        print(f"  {label:28s} → combined L2 error = {err:.4f} m")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(traj_gt[:, 1], traj_gt[:, 0], "k--", lw=1.8, label="real robot")
    colors = {"joint_states": "#2196F3", "cmd_vel (kinematic)": "#E91E63"}
    for label, (traj, err) in results.items():
        ax.plot(traj[:, 1], traj[:, 0], color=colors[label], lw=1.3,
                label=f"{label}  (err={err:.3f}\\,m)")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("cmd_vel vs joint_states as MuJoCo target")

    out = pathlib.Path(args.save)
    if not out.is_absolute():
        out = pathlib.Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
