"""Paper-style 3-engine convergence plot for 3_gradient_quality_box2.

Mirrors experiments/3_gradient_quality_box/plot_results.py: single-panel
running-best loss vs wall-clock seconds (log-log), median line + IQR band
across N trials per engine. Reads results/<engine>*.json files saved by
optimize_axion.py / optimize_mjx.py / optimize_semi_implicit.py.

Specifically picks `axion_all_fixes.json`, `mjx_all_fixes.json`, etc. when
present (the final tuned runs) and falls back to `<engine>.json` otherwise.

Caveat (inherited from box1): the per-trial wall_s is a single total, so
per-iter time is approximated as ``min(wall_s) / iterations`` per engine
(the warmest trial's average). Within an engine all trials use the same
per-iter estimate so the loss IQR band reflects only loss-curve variance,
not wall-clock variance. Absolute x-axis times exclude one-time JIT compile
or CUDA-graph capture costs.

Usage:
    python experiments/3_gradient_quality_box2/plot_convergence.py
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update({
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
})

STYLES = {
    "Axion":         {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MJX":           {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 3},
}
LABELS = {
    "Axion":         r"\textbf{Ostrich}",
    "MJX":           "MJX",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = ["Axion", "MJX", "Semi-Implicit"]

N_GRID = 80


def _trial_curve(trial, per_iter_s, max_iters=None):
    """Return (cum_wall_s, running_best_loss) for one trial.

    max_iters: if set, truncate the loss curve at this many iters.
    """
    losses = np.asarray(trial["losses"], dtype=float)
    if max_iters is not None:
        losses = losses[:max_iters]
    if len(losses) == 0:
        return np.array([]), np.array([])
    running_best = np.minimum.accumulate(losses)
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
    return (
        t_grid,
        np.median(interp, axis=0),
        np.quantile(interp, 0.25, axis=0),
        np.quantile(interp, 0.75, axis=0),
    )


def _pick_json_for_engine(engine_key):
    """Prefer ``<engine>_all_fixes.json`` (the final tuned runs); fall back
    to ``<engine>.json`` if the fix-tagged file isn't there yet."""
    preferred = RESULTS_DIR / f"{engine_key}_all_fixes.json"
    if preferred.is_file():
        return preferred
    fallback = RESULTS_DIR / f"{engine_key}.json"
    if fallback.is_file():
        return fallback
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", default=str(RESULTS_DIR / "convergence_box2.png"))
    ap.add_argument("--min-iters", type=int, default=10,
                    help="skip engine JSONs with fewer iters than this "
                    "(filters out sanity-test JSONs).")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="truncate each loss curve at this many iters "
                    "(useful for showing only the early descent regime).")
    ap.add_argument("--engines", nargs="+", default=None,
                    choices=SIM_ORDER,
                    help="subset of engines to plot (default: all available). "
                    "e.g. --engines Axion MJX")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    sim_order = args.engines if args.engines else SIM_ORDER

    # Engine key (file prefix) ↔ "simulator" string in the JSON.
    engine_file_keys = {
        "Axion": "axion",
        "MJX": "mjx",
        "Semi-Implicit": "semi_implicit",
    }

    engines = {}
    for sim in sim_order:
        path = _pick_json_for_engine(engine_file_keys[sim])
        if path is None:
            print(f"  [skip] {sim}: no <engine>.json or <engine>_all_fixes.json found")
            continue
        d = json.load(open(path))
        if d.get("iterations", 0) < args.min_iters:
            print(f"  [skip] {path.name} ({sim}): only {d.get('iterations')} iters "
                  f"(< {args.min_iters}) — sanity run, not production.")
            continue
        # Use warmest-trial per-iter time as the uniform x-axis estimate.
        warm_wall_s = min(t["wall_s"] for t in d["trials"])
        per_iter_s = warm_wall_s / d["iterations"]
        eff_iters = d["iterations"] if args.max_iters is None \
                    else min(args.max_iters, d["iterations"])
        curves = []
        for t in d["trials"]:
            cum, best = _trial_curve(t, per_iter_s, max_iters=args.max_iters)
            if len(cum) > 0:
                curves.append((cum, best))
        if not curves:
            continue
        engines[sim] = {"curves": curves, "iterations": eff_iters,
                        "num_trials": d["num_trials"], "per_iter_s": per_iter_s,
                        "json": path.name}
        print(f"  [load] {sim}: {len(curves)} trial(s), {eff_iters} iters "
              f"(of {d['iterations']} total), "
              f"warm per-iter ≈ {per_iter_s:.3g}s, "
              f"final best (median) = "
              f"{np.median([c[1][-1] for c in curves]):.4f}  (from {path.name})")

    if not engines:
        print(f"No production results in {RESULTS_DIR} — run optimize_*.py first.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for sim in sim_order:
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
    ax.set_ylabel(r"Running-best loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35, linewidth=0.6)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=len(handles),
              fontsize=11, frameon=False, columnspacing=1.5, handlelength=1.5)

    out = pathlib.Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "convergence_box2.png"
        plt.savefig(out_paper, dpi=200, bbox_inches="tight")
        print(f"Saved {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
