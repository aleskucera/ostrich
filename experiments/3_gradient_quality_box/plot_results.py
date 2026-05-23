"""Plot loss-vs-iter curves for the box gradient-quality experiment.

Loads results/<engine>.json files (any present) and overlays them: one band
per engine (median + min/max across trials), so the gradient quality of
different engines on the same box trajectory can be compared at a glance.

Usage:
    python experiments/3_gradient_quality_box/plot_results.py
    python experiments/3_gradient_quality_box/plot_results.py --save other.png
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
COLORS = {"Axion": "#2196F3", "MuJoCo": "#E91E63",
          "Semi-Implicit": "#FF9800", "MJX": "#4CAF50"}


def load_engines():
    out = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        d = json.load(open(p))
        sim = d.get("simulator", p.stem)
        out[sim] = d
    return out


def trial_curve(losses, n_grid):
    """Resample a single trial's loss curve to a common iter grid (linear)."""
    L = np.asarray(losses, dtype=float)
    x_src = np.linspace(0, 1, len(L))
    x_dst = np.linspace(0, 1, n_grid)
    return np.interp(x_dst, x_src, L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=str(RESULTS_DIR / "gradient_quality_box.png"))
    args = ap.parse_args()

    engines = load_engines()
    if not engines:
        print(f"No results in {RESULTS_DIR} — run optimize_*.py first.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    n_grid = max(50, max(d["iterations"] for d in engines.values()))
    x = np.linspace(0, 1, n_grid)

    for name, d in engines.items():
        curves = np.stack([trial_curve(t["losses"], n_grid) for t in d["trials"]])
        med = np.median(curves, axis=0)
        lo = curves.min(axis=0); hi = curves.max(axis=0)
        c = COLORS.get(name, "tab:gray")
        ax.fill_between(x * d["iterations"], lo, hi, color=c, alpha=0.20, zorder=2)
        ax.plot(x * d["iterations"], med, "-", color=c, lw=2.2, label=name, zorder=3)
        ax.plot(x * d["iterations"], lo, "-", color=c, lw=0.8, alpha=0.6, zorder=2)
        ax.plot(x * d["iterations"], hi, "-", color=c, lw=0.8, alpha=0.6, zorder=2)

    ax.set_xlabel("optimization iteration")
    ax.set_ylabel(r"loss  $\langle \|\Delta xy\|^2 \rangle$ [m$^2$]")
    ax.set_yscale("log")
    ax.set_title("Gradient quality on the box scene — K-knot wheel-velocity spline "
                 "fit to real trajectory\n"
                 "(median + [min, max] band over trials, same GT, calibrated physics)")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=140, bbox_inches="tight")
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
