"""Export recovered wheel-velocity control at 10 Hz for hardware replay.

Reads results/axion.json, picks a trial (default: lowest best_loss), expands
the K-knot spline at the native dt (0.1 s = 10 Hz), applies skid-steer
coupling (rear = (L+R)/2), and writes:

  - results/recovered_control.csv        columns: t,left,right,rear  (rad/s)
  - results/recovered_control.json       GT-compatible schema {t, lrr}

Usage:
    python experiments/3_gradient_quality_box/export_recovered_control.py
    python experiments/3_gradient_quality_box/export_recovered_control.py --trial 1
    python experiments/3_gradient_quality_box/export_recovered_control.py --trial all
"""
import argparse
import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    return W


def expand_trial(trial, W):
    params = np.asarray(trial["final_params"])   # [K, 2]
    lr = W @ params                              # [T, 2]
    rear = 0.5 * (lr[:, 0] + lr[:, 1])
    return np.column_stack([lr[:, 0], lr[:, 1], rear])   # [T, 3]


def write_outputs(t, lrr, stem):
    csv_path = RES / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "left", "right", "rear"])
        for ti, row in zip(t, lrr):
            w.writerow([f"{ti:.4f}", f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}"])

    json_path = RES / f"{stem}.json"
    json_path.write_text(json.dumps({
        "rate_hz": 10.0,
        "dt": float(t[1] - t[0]) if len(t) > 1 else 0.1,
        "control": {"t": t.tolist(), "lrr": lrr.tolist()},
    }, indent=2))
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="best",
                    help="trial index, 'best' (default, lowest best_loss), or 'all'")
    ap.add_argument("--source", default=str(RES / "axion.json"))
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.source).read_text())
    K, dt, horizon_s = d["K"], d["dt"], d["horizon_s"]
    T = int(round(horizon_s / dt))
    assert abs(dt - 0.1) < 1e-9, f"expected dt=0.1 (10 Hz), got dt={dt}"
    W = make_interp_matrix(T, K)
    t = np.arange(T) * dt

    if args.trial == "best":
        best = min(range(len(d["trials"])), key=lambda i: d["trials"][i]["best_loss"])
        print(f"best trial: index={best}, seed={d['trials'][best]['seed']}, "
              f"loss={d['trials'][best]['best_loss']:.4f}")
        lrr = expand_trial(d["trials"][best], W)
        write_outputs(t, lrr, "recovered_control")
    elif args.trial == "all":
        for i, trial in enumerate(d["trials"]):
            print(f"trial {i} seed={trial['seed']} loss={trial['best_loss']:.4f}")
            lrr = expand_trial(trial, W)
            write_outputs(t, lrr, f"recovered_control_seed{trial['seed']}")
    else:
        i = int(args.trial)
        trial = d["trials"][i]
        print(f"trial {i} seed={trial['seed']} loss={trial['best_loss']:.4f}")
        lrr = expand_trial(trial, W)
        write_outputs(t, lrr, f"recovered_control_seed{trial['seed']}")

    print(f"\n{T} samples @ 10 Hz, horizon={horizon_s}s, columns: [t, left, right, rear] (rad/s)")


if __name__ == "__main__":
    main()
