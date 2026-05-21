"""Plot /cmd_vel (converted to per-wheel velocities) vs /joint_states for each wheel.

Usage:
    python experiments/data/plot_cmd_vs_joint_states.py \
        --bag helhest_2026_04_10-14_46_18

The Apr 10 bags include /cmd_vel; Apr 13 bags do not.
"""
import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: install rosbags (`uv pip install mcap rosbags`)", file=sys.stderr)
    sys.exit(1)

ROSBAG_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "rosbags" / "nuc"
OUTPUT_DIR = pathlib.Path(__file__).parent

WHEEL_RADIUS = 0.36
TRACK_WIDTH = 0.72


def read_bag(bag_path: pathlib.Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    cmd, js = [], []
    with Reader(str(bag_path)) as reader:
        for conn, ts, raw in reader.messages():
            t = ts / 1e9
            if conn.topic == "/cmd_vel":
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                cmd.append((t, m.twist.linear.x, m.twist.angular.z))
            elif conn.topic == "/joint_states":
                m = typestore.deserialize_cdr(raw, conn.msgtype)
                names = list(m.name)
                vels = [float(v) for v in m.velocity]
                js.append((t, names, vels))
    return cmd, js


def cmd_to_wheels(vx, wz):
    """Differential drive: convert (vx, wz) to (left, right, rear) rad/s."""
    v_l = (vx - wz * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    v_r = (vx + wz * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
    v_rear = (v_l + v_r) / 2.0
    return v_l, v_r, v_rear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="helhest_2026_04_10-14_46_18")
    ap.add_argument("--bag-dir", type=pathlib.Path, default=ROSBAG_DIR)
    ap.add_argument("--output", type=pathlib.Path,
                    default=OUTPUT_DIR / "cmd_vs_joint_states.png")
    ap.add_argument("--r-fit", type=float, default=None,
                    help="fitted wheel radius (m); if omitted, fit from this bag")
    ap.add_argument("--track-fit", type=float, default=None,
                    help="fitted track width (m); if omitted, fit from this bag")
    args = ap.parse_args()

    bag_path = args.bag_dir / args.bag
    cmd, js = read_bag(bag_path)
    if not cmd:
        print(f"ERROR: no /cmd_vel in {args.bag}")
        sys.exit(1)
    if not js:
        print(f"ERROR: no /joint_states in {args.bag}")
        sys.exit(1)

    t0 = min(cmd[0][0], js[0][0])

    # Commanded wheel velocities (URDF geometry)
    t_cmd = np.array([c[0] - t0 for c in cmd])
    cmd_wheels = np.array([cmd_to_wheels(c[1], c[2]) for c in cmd])  # (N, 3)

    # Fitted geometry: solve a=1/r, b=track/(2r) by LS on this bag if not given
    vx = np.array([c[1] for c in cmd])
    wz = np.array([c[2] for c in cmd])
    names_js = js[0][1]
    li, ri = names_js.index("left_wheel_j"), names_js.index("right_wheel_j")
    t_js_abs = np.array([m[0] for m in js])
    v_l = np.array([m[2][li] for m in js])
    v_r = np.array([m[2][ri] for m in js])
    mask = (t_js_abs >= cmd[0][0]) & (t_js_abs <= cmd[-1][0])
    vx_i = np.interp(t_js_abs[mask], np.array([c[0] for c in cmd]), vx)
    wz_i = np.interp(t_js_abs[mask], np.array([c[0] for c in cmd]), wz)
    A = np.vstack([np.column_stack([vx_i, -wz_i]),
                   np.column_stack([vx_i,  wz_i])])
    y = np.concatenate([v_l[mask], v_r[mask]])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r_fit = args.r_fit if args.r_fit is not None else 1.0 / a
    track_fit = args.track_fit if args.track_fit is not None else 2.0 * b * r_fit
    print(f"Fitted: r={r_fit:.4f} m, track={track_fit:.4f} m")

    def cmd_fit(vx, wz):
        vl = (vx - wz * track_fit / 2.0) / r_fit
        vr = (vx + wz * track_fit / 2.0) / r_fit
        return vl, vr, (vl + vr) / 2.0

    cmd_wheels_fit = np.array([cmd_fit(c[1], c[2]) for c in cmd])

    # Measured wheel velocities
    names = js[0][1]
    l_idx = names.index("left_wheel_j")
    r_idx = names.index("right_wheel_j")
    rr_idx = names.index("rear_wheel_j")
    t_js = np.array([m[0] - t0 for m in js])
    meas = np.array([[m[2][l_idx], m[2][r_idx], m[2][rr_idx]] for m in js])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ["left_wheel_j", "right_wheel_j", "rear_wheel_j"]
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(t_cmd, cmd_wheels[:, i], "--", color="tab:red",
                label="cmd (URDF r=0.36, track=0.72)", linewidth=1.2)
        ax.plot(t_cmd, cmd_wheels_fit[:, i], "--", color="tab:green",
                label=f"cmd (fitted r={r_fit:.3f}, track={track_fit:.3f})",
                linewidth=1.2)
        ax.plot(t_js, meas[:, i], "-", color="tab:blue",
                label="measured (/joint_states)", linewidth=1.0)
        ax.set_ylabel(f"{lab}\n[rad/s]")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"cmd_vel vs joint_states — {args.bag}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")

    # Print tracking error summary
    for i, lab in enumerate(labels):
        cmd_interp = np.interp(t_js, t_cmd, cmd_wheels[:, i])
        err = meas[:, i] - cmd_interp
        print(f"  {lab}: mean_err={err.mean():+.3f}  rms={np.sqrt((err**2).mean()):.3f} rad/s")


if __name__ == "__main__":
    main()
