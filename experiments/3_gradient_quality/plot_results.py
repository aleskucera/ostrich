"""Plot Experiment 3: gradient-quality convergence (wall-clock only).

Single panel: running-best RMSE vs wall-clock. Median line + IQR band across
N trials with different perturbed initial guesses.

Usage:
    python experiments/3_gradient_quality/plot_results.py
    python experiments/3_gradient_quality/plot_results.py --save results/gradient_quality.png
    GT_STEM=acceleration python experiments/3_gradient_quality/plot_results.py
"""
import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STYLES = {
    "Axion":         {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MuJoCo":        {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "MJX":           {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "TinyDiffSim":   {"color": "#607D8B", "marker": "D", "lw": 1.8, "zorder": 3},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 2},
}
LABELS = {
    "Axion":         r"\textbf{Axion}",
    "MJX":           "MJX",
    "MuJoCo":        "MuJoCo",
    "TinyDiffSim":   "TinyDiffSim",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = ["Axion", "MJX", "TinyDiffSim", "Semi-Implicit"]

N_GRID = 80  # points in the common wall-clock grid


def _trial_curves(trial):
    """Return (cum_s, rmse_best) for one trial."""
    rmse = np.array(trial.get("rmse_m", []), dtype=float)
    times_ms = np.array(trial.get("time_ms", []), dtype=float)
    if len(rmse) == 0:
        return np.array([]), np.array([])
    rmse_best = np.minimum.accumulate(rmse)
    cum_s = np.cumsum(times_ms) / 1000.0
    return cum_s, rmse_best


def load_sim(path):
    """Load a sim's JSON (multi-trial or flat single-run) and return list of (cum_s, rmse_best)."""
    d = json.loads(path.read_text())
    if "trials" in d:
        trials = d["trials"]
    else:
        # Back-compat: flat single-run file
        trials = [d]
    curves = [_trial_curves(t) for t in trials]
    return d.get("simulator", path.stem), [c for c in curves if len(c[0]) > 0]


def _aggregate_on_grid(curves, n_grid=N_GRID):
    """Interpolate each trial's (cum_s, rmse_best) onto a common log time grid.

    Returns (t_grid, median, q1, q3). Grid spans [max(cum_s[0]), min(cum_s[-1])]
    so all trials have data at every point.
    """
    t_lo = max(c[0][0] for c in curves)
    t_hi = min(c[0][-1] for c in curves)
    if t_hi <= t_lo:
        # Degenerate; fall back to min-time range
        t_lo = min(c[0][0] for c in curves)
        t_hi = max(c[0][-1] for c in curves)
    t_grid = np.geomspace(t_lo, t_hi, n_grid)

    interp = []
    for cum_s, rmse_best in curves:
        y = np.interp(t_grid, cum_s, rmse_best)
        interp.append(y)
    arr = np.stack(interp, axis=0)  # (n_trials, n_grid)
    median = np.median(arr, axis=0)
    q1 = np.quantile(arr, 0.25, axis=0)
    q3 = np.quantile(arr, 0.75, axis=0)
    return t_grid, median, q1, q3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    gt_stem = os.environ.get("GT_STEM", "")
    suffix = f"_{gt_stem}" if gt_stem else ""

    name_map = {
        "axion": "Axion", "mjx": "MJX",
        "semi_implicit": "Semi-Implicit", "tinydiffsim": "TinyDiffSim",
    }
    sims = {}
    for stem, sim_name in name_map.items():
        path = RESULTS_DIR / f"{stem}{suffix}.json"
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        _, curves = load_sim(path)
        if curves:
            sims[sim_name] = curves
            print(f"  [load] {sim_name}: {len(curves)} trial(s)")

    if not sims:
        print("No results found. Run run_experiment.sh first.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 2.6))

    for sim in SIM_ORDER:
        if sim not in sims:
            continue
        curves = sims[sim]
        st = STYLES[sim]

        if len(curves) == 1:
            cum_s, rmse_best = curves[0]
            ax.plot(
                cum_s, rmse_best,
                color=st["color"], marker=st["marker"], linewidth=st["lw"],
                markersize=4, markevery=max(1, len(cum_s) // 15),
                label=LABELS[sim], zorder=st["zorder"],
            )
            continue

        t_grid, median, q1, q3 = _aggregate_on_grid(curves)
        ax.fill_between(
            t_grid, q1, q3,
            color=st["color"], alpha=0.18, linewidth=0, zorder=st["zorder"] - 1,
        )
        ax.plot(
            t_grid, median,
            color=st["color"], marker=st["marker"], linewidth=st["lw"],
            markersize=4, markevery=max(1, len(t_grid) // 12),
            label=LABELS[sim], zorder=st["zorder"],
        )

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Running-best RMSE (m)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.35, linewidth=0.6)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=len(handles), fontsize=11,
        frameon=False, columnspacing=1.5, handlelength=1.5,
    )

    # --- Save ---
    save_arg = args.save or f"gradient_quality{suffix}.png"
    out = RESULTS_DIR / save_arg if "/" not in save_arg else pathlib.Path(save_arg)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / f"gradient_quality{suffix}.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
