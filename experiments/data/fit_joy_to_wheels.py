"""Fit /joy.axes directly to per-wheel velocities from /joint_states.

Avoids the /cmd_vel intermediate: regression target is measured wheel velocity
so the learned map absorbs motor lag, slip, and controller nonlinearities.

    v_left  = sum_j W_l[j] * axes[j] + b_l
    v_right = ...
    v_rear  = ...

Apr 10 bags (which have both /joy and /joint_states) are used for training.
A target bag (e.g. Apr 13 acceleration) is then mapped to a wheel timeseries
in the same format as gt["wheel_velocities"]["timeseries"].

Usage:
    python experiments/data/fit_joy_to_wheels.py \\
        --predict-bag helhest_2026_04_13-09_56_50
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


def read_joy_joints(bag_path):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    joy, js = [], []
    with Reader(str(bag_path)) as r:
        for conn, t_ns, raw in r.messages():
            t = t_ns / 1e9
            if conn.topic == "/joy":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                joy.append((t, np.asarray(m.axes, dtype=np.float64)))
            elif conn.topic == "/joint_states":
                m = ts.deserialize_cdr(raw, conn.msgtype)
                js.append((t, list(m.name), [float(v) for v in m.velocity]))
    return joy, js


LPF_TAU = 0.5  # motor lag time constant [s]


def _lowpass(t, x, tau):
    """First-order low-pass, sample-rate aware."""
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        dt = max(t[i] - t[i - 1], 1e-4)
        a = dt / (tau + dt)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def assemble(bags):
    """Return X (N, 2*naxes+1), Y (N, 3) sampled at /joint_states times.

    Features are the raw joy axes plus a low-pass-filtered copy (tau=LPF_TAU),
    so the regression can capture the motor/wheel ramp-up lag without being
    nonlinear in the parameters.
    """
    Xs, Ys = [], []
    for bag in bags:
        joy, js = read_joy_joints(bag)
        if not joy or not js:
            continue
        t_joy = np.array([j[0] for j in joy])
        axes = np.stack([j[1] for j in joy])
        naxes = axes.shape[1]
        axes_lp = np.column_stack([
            _lowpass(t_joy, axes[:, i], LPF_TAU) for i in range(naxes)
        ])

        names = js[0][1]
        li, ri, rr = (names.index("left_wheel_j"), names.index("right_wheel_j"),
                      names.index("rear_wheel_j"))
        t_js = np.array([m[0] for m in js])
        v = np.array([[m[2][li], m[2][ri], m[2][rr]] for m in js])

        mask = (t_js >= t_joy[0]) & (t_js <= t_joy[-1])
        if not mask.any():
            continue
        t = t_js[mask]
        feat = np.column_stack([
            *(np.interp(t, t_joy, axes[:, i]) for i in range(naxes)),
            *(np.interp(t, t_joy, axes_lp[:, i]) for i in range(naxes)),
            np.ones_like(t),
        ])
        Xs.append(feat)
        Ys.append(v[mask])
        print(f"  {bag.name}: {feat.shape[0]} samples")
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir", type=pathlib.Path, default=ROSBAG_DIR)
    ap.add_argument("--train-bags", nargs="+", default=None,
                    help="default: all Apr 10 bags (they have /joy + /joint_states)")
    ap.add_argument("--predict-bag", default=None,
                    help="bag name to synthesize wheel timeseries for")
    ap.add_argument("--align-to", type=pathlib.Path, default=None,
                    help="GT JSON; auto-align synth t0 via cross-correlation")
    ap.add_argument("--save-coeffs", type=pathlib.Path,
                    default=OUTPUT_DIR / "joy_to_wheels.json")
    args = ap.parse_args()

    if args.train_bags:
        train = [args.bag_dir / b for b in args.train_bags]
    else:
        # Prefer Apr 10 bags which have both topics
        train = [d for d in sorted(args.bag_dir.iterdir())
                 if d.is_dir() and "2026_04_10" in d.name]

    print(f"Training on {len(train)} bags:")
    X, Y = assemble(train)
    if X is None:
        print("No training data"); sys.exit(1)

    # Enforce differential-drive structure: v_left = alpha - beta,
    # v_right = alpha + beta, where alpha (forward) depends on throttle axes
    # and beta (turn) depends on the steering axis only. Without this, a
    # direction-biased training distribution leaks throttle -> turn.
    naxes_raw = (X.shape[1] - 1) // 2
    # Include raw + low-pass-filtered copies of each axis group.
    TURN_AXES = [3, 3 + naxes_raw]              # steering raw + lp
    FORWARD_AXES = [2, 4, 2 + naxes_raw, 4 + naxes_raw]  # trigger + throttle + lp

    def subset(idxs):
        return np.column_stack([*(X[:, i] for i in idxs), X[:, -1]])  # + bias

    y_sum = 0.5 * (Y[:, 0] + Y[:, 1])
    y_diff = 0.5 * (Y[:, 1] - Y[:, 0])
    X_fwd = subset(FORWARD_AXES)
    X_turn = subset(TURN_AXES)
    w_alpha_s, *_ = np.linalg.lstsq(X_fwd, y_sum, rcond=None)
    # Beta (turn) is fit without intercept — a constant turn bias would
    # otherwise cause unconditional drift whenever joysticks are neutral.
    X_turn_nobias = np.column_stack([X[:, i] for i in TURN_AXES])
    w_beta_only,  *_ = np.linalg.lstsq(X_turn_nobias, y_diff, rcond=None)
    w_beta_s = np.concatenate([w_beta_only, [0.0]])
    w_rear_s,  *_ = np.linalg.lstsq(X_fwd, Y[:, 2], rcond=None)

    # Expand back to full coefficient vectors over all features + bias.
    nfeat = X.shape[1]
    def expand(w_sub, idxs):
        out = np.zeros(nfeat)
        for k, i in enumerate(idxs):
            out[i] = w_sub[k]
        out[-1] = w_sub[-1]
        return out
    w_alpha = expand(w_alpha_s, FORWARD_AXES)
    w_beta  = expand(w_beta_s,  TURN_AXES)
    w_rear  = expand(w_rear_s,  FORWARD_AXES)
    W = np.column_stack([w_alpha - w_beta, w_alpha + w_beta, w_rear])
    pred = X @ W
    rms = np.sqrt(((Y - pred) ** 2).mean(axis=0))
    print(f"\nFit: naxes={X.shape[1]-1}, N={X.shape[0]}")
    for i in range(X.shape[1] - 1):
        print(f"  ax{i}: L={W[i,0]:+.4f}  R={W[i,1]:+.4f}  Rear={W[i,2]:+.4f}")
    print(f"  bias: L={W[-1,0]:+.4f}  R={W[-1,1]:+.4f}  Rear={W[-1,2]:+.4f}")
    print(f"  rms (rad/s): L={rms[0]:.3f}  R={rms[1]:.3f}  Rear={rms[2]:.3f}")

    args.save_coeffs.write_text(json.dumps({
        "W": W.tolist(), "rms": rms.tolist(),
        "train_bags": [b.name for b in train],
    }, indent=2))
    print(f"Saved -> {args.save_coeffs}")

    if args.predict_bag:
        bag = args.bag_dir / args.predict_bag
        joy, js = read_joy_joints(bag)
        if not joy:
            print(f"No /joy in {args.predict_bag}"); sys.exit(1)
        t = np.array([j[0] for j in joy])
        axes = np.stack([j[1] for j in joy])
        n = axes.shape[1]
        axes_lp = np.column_stack([_lowpass(t, axes[:, i], LPF_TAU) for i in range(n)])
        t0 = js[0][0] if js else t[0]
        t_rel = t - t0
        pred = np.column_stack([axes, axes_lp, np.ones(len(t))]) @ W

        # Optionally align to target GT timeseries via cross-correlation
        # of the left-wheel signal. This accounts for manual retiming of
        # GT JSONs whose t=0 was shifted from the raw bag.
        if args.align_to is not None:
            gt = json.loads(args.align_to.read_text())
            gt_ts = gt["wheel_velocities"]["timeseries"]
            t_gt = np.array([p["t"] for p in gt_ts])
            l_gt = np.array([p["left"] for p in gt_ts])
            # Resample both to a common 50 Hz grid covering the union
            hz = 50.0
            t_lo = min(t_rel[0], t_gt[0])
            t_hi = max(t_rel[-1], t_gt[-1])
            grid = np.arange(t_lo, t_hi, 1.0 / hz)
            s_synth = np.interp(grid, t_rel, pred[:, 0], left=0.0, right=0.0)
            s_meas = np.interp(grid, t_gt, l_gt, left=0.0, right=0.0)
            corr = np.correlate(s_meas - s_meas.mean(),
                                s_synth - s_synth.mean(), mode="full")
            lag = np.argmax(corr) - (len(s_synth) - 1)
            shift = lag / hz  # seconds to add to synth times
            t_rel = t_rel + shift
            print(f"  align: shift={shift:+.3f}s (peak lag {lag} samples)")

        out = {
            "bag_name": args.predict_bag,
            "source": "joy->wheels linear fit",
            "wheel_velocities": {
                "timeseries": [
                    {"t": round(float(t_rel[i]), 4),
                     "left": round(float(pred[i, 0]), 4),
                     "right": round(float(pred[i, 1]), 4),
                     "rear": round(float(pred[i, 2]), 4)}
                    for i in range(len(t))
                ],
            },
        }
        out_path = OUTPUT_DIR / f"{args.predict_bag}_synth_wheels.json"
        out_path.write_text(json.dumps(out))
        print(f"Predicted wheel ts -> {out_path}  ({len(t)} samples)")

        # Compare vs measured joint_states on the same bag
        if js:
            names = js[0][1]
            li, ri, rr = (names.index("left_wheel_j"), names.index("right_wheel_j"),
                          names.index("rear_wheel_j"))
            t_js = np.array([m[0] for m in js])
            v = np.array([[m[2][li], m[2][ri], m[2][rr]] for m in js])
            mask = (t_js >= t[0]) & (t_js <= t[-1])
            if mask.any():
                ip = np.column_stack([
                    np.interp(t_js[mask], t, pred[:, 0]),
                    np.interp(t_js[mask], t, pred[:, 1]),
                    np.interp(t_js[mask], t, pred[:, 2]),
                ])
                err = v[mask] - ip
                r = np.sqrt((err ** 2).mean(axis=0))
                print(f"  vs measured /joint_states on this bag: "
                      f"rms L={r[0]:.3f}  R={r[1]:.3f}  Rear={r[2]:.3f}")


if __name__ == "__main__":
    main()
