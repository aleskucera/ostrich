"""Fit effective wheel radius and track width from cmd_vel vs joint_states.

Model (differential drive):
    v_left  = (vx - wz * track/2) / r
    v_right = (vx + wz * track/2) / r

Let a = 1/r, b = track/(2r). Then linear in (a, b):
    v_left  =  a*vx - b*wz
    v_right =  a*vx + b*wz

Solve by least squares over all joint_states timestamps (cmd_vel interpolated).
"""
import argparse
import pathlib
import sys

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: install rosbags (`uv pip install mcap rosbags`)", file=sys.stderr)
    sys.exit(1)

ROSBAG_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "rosbags" / "nuc"


def read_bag(bag_path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    cmd, js = [], []
    with Reader(str(bag_path)) as reader:
        for conn, t_ns, raw in reader.messages():
            t = t_ns / 1e9
            if conn.topic == "/cmd_vel":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                cmd.append((t, m.twist.linear.x, m.twist.angular.z))
            elif conn.topic == "/joint_states":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                js.append((t, list(m.name), [float(v) for v in m.velocity]))
    return cmd, js


def fit_bag(bag_path):
    cmd, js = read_bag(bag_path)
    if not cmd or not js:
        return None

    t_cmd = np.array([c[0] for c in cmd])
    vx = np.array([c[1] for c in cmd])
    wz = np.array([c[2] for c in cmd])
    t_js = np.array([m[0] for m in js])

    names = js[0][1]
    li, ri = names.index("left_wheel_j"), names.index("right_wheel_j")
    v_l = np.array([m[2][li] for m in js])
    v_r = np.array([m[2][ri] for m in js])

    # Interp cmd onto js timestamps, clip to cmd window
    mask = (t_js >= t_cmd[0]) & (t_js <= t_cmd[-1])
    vx_i = np.interp(t_js[mask], t_cmd, vx)
    wz_i = np.interp(t_js[mask], t_cmd, wz)
    vl, vr = v_l[mask], v_r[mask]

    # Stack: [vx, -wz; vx, +wz] @ [a; b] = [vl; vr]
    A = np.vstack([
        np.column_stack([vx_i, -wz_i]),
        np.column_stack([vx_i,  wz_i]),
    ])
    y = np.concatenate([vl, vr])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)

    r = 1.0 / a
    track = 2.0 * b * r

    pred = A @ np.array([a, b])
    resid = y - pred
    rms = float(np.sqrt((resid ** 2).mean()))

    # Compare vs URDF defaults
    r0, track0 = 0.36, 0.72
    pred0_l = (vx_i - wz_i * track0 / 2) / r0
    pred0_r = (vx_i + wz_i * track0 / 2) / r0
    rms0 = float(np.sqrt(((vl - pred0_l) ** 2).mean() + ((vr - pred0_r) ** 2).mean()) / np.sqrt(2))

    return {"r": r, "track": track, "rms": rms, "rms_urdf": rms0, "n": len(vl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=None, help="single bag; else all bags with /cmd_vel")
    ap.add_argument("--bag-dir", type=pathlib.Path, default=ROSBAG_DIR)
    args = ap.parse_args()

    if args.bag:
        bags = [args.bag_dir / args.bag]
    else:
        bags = sorted(d for d in args.bag_dir.iterdir() if d.is_dir())

    per_bag = []
    for b in bags:
        res = fit_bag(b)
        if res is None:
            print(f"{b.name}: no cmd_vel/joint_states — skip")
            continue
        print(f"{b.name}: r={res['r']:.4f}m  track={res['track']:.4f}m  "
              f"rms={res['rms']:.3f}  (URDF rms={res['rms_urdf']:.3f})  n={res['n']}")
        per_bag.append(res)

    if len(per_bag) > 1:
        rs = np.array([p["r"] for p in per_bag])
        ts = np.array([p["track"] for p in per_bag])
        print(f"\nmean: r={rs.mean():.4f}m (±{rs.std():.4f})  "
              f"track={ts.mean():.4f}m (±{ts.std():.4f})")


if __name__ == "__main__":
    main()
