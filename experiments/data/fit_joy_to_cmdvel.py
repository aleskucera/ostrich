"""Fit a linear map /joy.axes -> (vx, wz) from bags that have both topics.

Apr 10 bags have both /joy (~50 Hz) and /cmd_vel (~10 Hz). Apr 13 bags have /joy only.
We fit:
    vx =  sum_i a_i * axes[i] + a0
    wz =  sum_i b_i * axes[i] + b0
by least squares on interpolated samples.

Usage:
    python experiments/data/fit_joy_to_cmdvel.py                 # fit + report
    python experiments/data/fit_joy_to_cmdvel.py --predict-bag \\
        helhest_2026_04_13-09_56_50                              # also synthesize cmd_vel
"""
import argparse
import json
import pathlib
import sys

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: install rosbags", file=sys.stderr)
    sys.exit(1)

ROSBAG_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "rosbags" / "nuc"
OUTPUT_DIR = pathlib.Path(__file__).parent


def read_joy_cmd(bag_path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    joy, cmd = [], []
    with Reader(str(bag_path)) as r:
        for conn, t_ns, raw in r.messages():
            t = t_ns / 1e9
            if conn.topic == "/joy":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                joy.append((t, np.asarray(m.axes, dtype=np.float64)))
            elif conn.topic == "/cmd_vel":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                cmd.append((t, float(m.twist.linear.x), float(m.twist.angular.z)))
    return joy, cmd


def assemble_training(bags):
    """Return X (N, naxes+1) and Y (N, 2) sampled at /cmd_vel times."""
    Xs, Ys = [], []
    for bag in bags:
        joy, cmd = read_joy_cmd(bag)
        if not joy or not cmd:
            print(f"  skip {bag.name}: joy={len(joy)} cmd={len(cmd)}")
            continue
        t_joy = np.array([j[0] for j in joy])
        axes = np.stack([j[1] for j in joy])  # (Nj, naxes)
        naxes = axes.shape[1]
        t_cmd = np.array([c[0] for c in cmd])
        vx = np.array([c[1] for c in cmd])
        wz = np.array([c[2] for c in cmd])
        # Restrict cmd to joy window, then interp each axis at cmd times
        mask = (t_cmd >= t_joy[0]) & (t_cmd <= t_joy[-1])
        if not mask.any():
            continue
        t = t_cmd[mask]
        feat = np.column_stack([
            *(np.interp(t, t_joy, axes[:, i]) for i in range(naxes)),
            np.ones_like(t),
        ])
        Xs.append(feat)
        Ys.append(np.column_stack([vx[mask], wz[mask]]))
        print(f"  {bag.name}: {feat.shape[0]} samples, {naxes} axes")
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir", type=pathlib.Path, default=ROSBAG_DIR)
    ap.add_argument("--train-bags", nargs="+", default=None,
                    help="bag names to train on (default: all with cmd_vel>0)")
    ap.add_argument("--predict-bag", default=None,
                    help="bag to synthesize /cmd_vel for (saves JSON)")
    ap.add_argument("--save", type=pathlib.Path,
                    default=OUTPUT_DIR / "joy_to_cmdvel.json")
    args = ap.parse_args()

    if args.train_bags:
        train_dirs = [args.bag_dir / b for b in args.train_bags]
    else:
        train_dirs = []
        for d in sorted(args.bag_dir.iterdir()):
            if not d.is_dir():
                continue
            with Reader(str(d)) as r:
                topics = {c.topic for c in r.connections}
            if "/joy" in topics and "/cmd_vel" in topics:
                train_dirs.append(d)

    print(f"Training on {len(train_dirs)} bags:")
    X, Y = assemble_training(train_dirs)
    if X is None:
        print("No training data"); sys.exit(1)

    # LS fit: X @ W = Y   (W has shape (naxes+1, 2))
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ W
    rms = np.sqrt(((Y - pred) ** 2).mean(axis=0))
    print(f"\nFit: naxes={X.shape[1]-1}, N={X.shape[0]} samples")
    print(f"  W (axes -> vx,wz):")
    for i in range(X.shape[1] - 1):
        print(f"    ax{i}: vx={W[i,0]:+.4f}  wz={W[i,1]:+.4f}")
    print(f"    bias: vx={W[-1,0]:+.4f}  wz={W[-1,1]:+.4f}")
    print(f"  rms: vx={rms[0]:.4f} m/s  wz={rms[1]:.4f} rad/s")

    args.save.write_text(json.dumps({
        "W": W.tolist(),
        "rms_vx": float(rms[0]),
        "rms_wz": float(rms[1]),
        "train_bags": [b.name for b in train_dirs],
    }, indent=2))
    print(f"Saved coefficients -> {args.save}")

    if args.predict_bag:
        bag = args.bag_dir / args.predict_bag
        joy, _ = read_joy_cmd(bag)
        if not joy:
            print(f"No /joy in {args.predict_bag}"); sys.exit(1)
        t = np.array([j[0] for j in joy])
        axes = np.stack([j[1] for j in joy])
        feat = np.column_stack([axes, np.ones(len(t))])
        pred = feat @ W
        out = {
            "bag_name": args.predict_bag,
            "source": "joy->cmd_vel fit",
            "cmd_vel": [
                {"t": round(float(t[i] - t[0]), 4),
                 "vx": round(float(pred[i, 0]), 4),
                 "wz": round(float(pred[i, 1]), 4)}
                for i in range(len(t))
            ],
        }
        out_path = OUTPUT_DIR / f"{args.predict_bag}_synth_cmdvel.json"
        out_path.write_text(json.dumps(out))
        print(f"Predicted cmd_vel -> {out_path}  ({len(t)} samples)")


if __name__ == "__main__":
    main()
