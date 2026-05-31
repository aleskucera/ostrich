"""Plot box gradient-quality convergence (wall-clock x-axis, paper style).

Mirrors experiments/3_gradient_quality/plot_results.py: single panel of
running-best loss vs wall-clock seconds (log-log), median line + IQR band
across N trials per engine. Reads results/<engine>.json files saved by
optimize_axion.py / optimize_mjx.py / optimize_semi_implicit.py.

Engines with fewer than --min-iters iterations are skipped (treats stale
sanity-test JSONs as not-yet-rerun for production).

Caveat: the per-trial wall_s is a single total, so per-iter time is
approximated as ``min(wall_s) / iterations`` per engine (the warmest trial's
average — picks up warm-state per-iter cost without including the one-time
JIT compile or CUDA-graph capture that the *first* trial pays once per
process). Within an engine all trials use the same per-iter estimate so the
loss IQR band reflects only loss-curve variance, not wall-clock variance.
Absolute x-axis times therefore *exclude* the one-time compile (78 s MJX,
~60 s Axion, ~20 min SI cold capture) — read them as "amortised warm cost".
For exact per-iter timings the runner scripts would need to save time_ms
per iter like ``experiments/3_gradient_quality/optimize_*.py`` already do.

Usage:
    python experiments/3_gradient_quality_box/plot_results.py
    python experiments/3_gradient_quality_box/plot_results.py --save other.png
"""
import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paper_style as ps  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

ps.apply()

_LW = ps.LINEWIDTH
STYLES = {
    "Axion":         {"color": ps.COLORS["Axion"],       "marker": "o", "lw": _LW, "zorder": 5},
    "MJX":           {"color": ps.COLORS["MJX"],         "marker": "s", "lw": _LW, "zorder": 4},
    "Semi-Implicit": {"color": ps.COLORS["Semi-Implicit"], "marker": "^", "lw": _LW, "zorder": 3},
    "MuJoCo":        {"color": ps.COLORS["MuJoCo"],      "marker": "s", "lw": _LW, "zorder": 4},
    "TinyDiffSim":   {"color": ps.COLORS["TinyDiffSim"], "marker": "D", "lw": _LW, "zorder": 2},
}
LABELS = {
    "Axion":         r"\textbf{Ostrich}",
    "MJX":           "MJX",
    "MuJoCo":        "MuJoCo",
    "Semi-Implicit": "Semi-Impl.",
    "TinyDiffSim":   "TinyDiffSim",
}
SIM_ORDER = ["Axion", "MJX", "MuJoCo", "Semi-Implicit", "TinyDiffSim"]

N_GRID = 80


def _trial_curve(trial, per_iter_s):
    """Return (cum_wall_s, running_best_loss) for one trial.

    per_iter_s: warm-state per-iter cost shared across all trials of the engine
    (computed once in main as min(wall_s) / iters across trials, so trials
    that paid one-time cold-compile/CUDA-graph-capture costs don't drag the
    x-axis to the right and make the loss band fan out artificially).
    """
    losses = np.asarray(trial["losses"], dtype=float)
    if len(losses) == 0:
        return np.array([]), np.array([])
    running_best = np.minimum.accumulate(losses)
    # cum_wall_s[i] = wall-clock at end of iter i, so it starts at per_iter_s,
    # not 0 — that keeps the log-x plot from blowing up at iter 0.
    cum_wall_s = (np.arange(len(losses)) + 1) * per_iter_s
    return cum_wall_s, running_best


def _aggregate_on_grid(curves, n_grid=N_GRID):
    """Interpolate each trial onto a common log time grid; return median+IQR."""
    t_lo = max(c[0][0] for c in curves)
    t_hi = min(c[0][-1] for c in curves)
    if t_hi <= t_lo:
        t_lo = min(c[0][0] for c in curves)
        t_hi = max(c[0][-1] for c in curves)
    t_grid = np.geomspace(t_lo, t_hi, n_grid)
    interp = np.stack([np.interp(t_grid, cum, best) for cum, best in curves])
    return t_grid, np.median(interp, axis=0), np.quantile(interp, 0.25, axis=0), \
           np.quantile(interp, 0.75, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=None, help="output filename (defaults to "
                    "results/gradient_quality_box.png + paper dir if present)")
    ap.add_argument("--min-iters", type=int, default=10,
                    help="skip engine JSONs with fewer iters than this "
                         "(filters out sanity-test JSONs).")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    engines = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        d = json.load(open(p))
        sim = d.get("simulator", p.stem)
        if d.get("iterations", 0) < args.min_iters:
            print(f"  [skip] {p.name} ({sim}): only {d.get('iterations')} iters "
                  f"(< {args.min_iters}) — looks like a sanity run, not production.")
            continue
        # Use the warmest trial's per-iter time as the uniform x-axis estimate
        # for every trial of this engine. Otherwise a single cold-compile trial
        # (e.g. Axion trial 1 on dasenka pays ~66s for axion module compilation
        # before its first warm iter at ~0.5s) stretches its curve far to the
        # right while warm trials sit at the left — interpolating those onto a
        # common log grid fans the loss band out into an inflated wedge.
        warm_wall_s = min(t["wall_s"] for t in d["trials"])
        per_iter_s = warm_wall_s / d["iterations"]
        curves = []
        for t in d["trials"]:
            cum, best = _trial_curve(t, per_iter_s)
            if len(cum) > 0:
                curves.append((cum, best))
        if not curves:
            continue
        engines[sim] = {"curves": curves, "iterations": d["iterations"],
                        "num_trials": d["num_trials"], "per_iter_s": per_iter_s}
        print(f"  [load] {sim}: {len(curves)} trial(s), {d['iterations']} iters, "
              f"warm per-iter ≈ {per_iter_s:.3g}s, "
              f"final best = {min(best[-1] for _, best in curves):.4f}")

    if not engines:
        print("No production results in {RESULTS_DIR} — run optimize_*.py first.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for sim in SIM_ORDER:
        if sim not in engines:
            continue
        st = STYLES[sim]
        curves = engines[sim]["curves"]
        if len(curves) == 1:
            cum, best = curves[0]
            ax.plot(cum, best, color=st["color"], marker=st["marker"],
                    linewidth=st["lw"], markersize=4,
                    markevery=max(1, len(cum) // 12),
                    label=LABELS[sim], zorder=st["zorder"])
            continue
        t_grid, median, q1, q3 = _aggregate_on_grid(curves)
        ax.fill_between(t_grid, q1, q3, color=st["color"], alpha=0.18,
                        linewidth=0, zorder=st["zorder"] - 1)
        ax.plot(t_grid, median, color=st["color"], marker=st["marker"],
                linewidth=st["lw"], markersize=4,
                markevery=max(1, len(t_grid) // 12),
                label=LABELS[sim], zorder=st["zorder"])

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(r"Running-best loss  $\langle \|\Delta xy\|^2 \rangle$ (m$^2$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35, linewidth=0.6)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=len(handles),
              fontsize=11, frameon=False, columnspacing=1.5, handlelength=1.5)

    out = pathlib.Path(args.save) if args.save else RESULTS_DIR / "gradient_quality_box.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "gradient_quality_box.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
