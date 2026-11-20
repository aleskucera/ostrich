"""Paper-style 3-engine convergence plot for 3_gradient_quality_box2.

Mirrors experiments/3_gradient_quality_box/plot_results.py: single-panel
running-best loss vs wall-clock seconds (log-log), median line + IQR band
across N trials per engine. Reads results/<engine>*.json files saved by
optimize_ostrich.py / optimize_mjx.py / optimize_semi_implicit.py.

Specifically picks `ostrich_all_fixes.json`, `mjx_all_fixes.json`, etc. when
present (the final tuned runs) and falls back to `<engine>.json` otherwise.

Per-iteration time comes from PER_ITER_S below: an exclusive measurement
(one job, one GPU) fitted over 5 and 15 iterations, which separates the
marginal per-iteration cost from each engine's one-time setup and EXCLUDES
that setup from the x-axis. This replaces the old min(wall_s)/iterations
estimate, which took the single fastest trial and amortised setup into every
iteration (badly wrong for Semi-Implicit, whose setup is ~29 min per trial).
Within an engine all trials share one per-iter estimate, so the IQR band
reflects loss-curve variance only, not wall-clock variance.

Usage:
    python experiments/3_gradient_quality_box2/plot_convergence.py
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
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "ostrich_paper" / "figures"

# savefig crops with bbox_inches="tight", so what LaTeX scales to
# \columnwidth is the CROPPED width, which itself moves with the font sizes.
# DRAWN_IN is that cropped width, measured from the emitted PNG and pinned
# here; the script prints the measured value and the achieved printed text
# size on every run, so a drift is visible immediately.
FIG_W = 7.0
DRAWN_IN = 6.23
S = ps.apply(drawn_in=DRAWN_IN)

STYLES = {
    "Ostrich":       {"color": ps.COLORS["Ostrich"], "marker": ps.MARKERS["Ostrich"],
                      "zorder": 5},
    "MJX":           {"color": ps.COLORS["MJX"], "marker": ps.MARKERS["MJX"],
                      "zorder": 4},
    "Semi-Implicit": {"color": ps.COLORS["Semi-Implicit"],
                      "marker": ps.MARKERS["Semi-Implicit"], "zorder": 3},
}
LABELS = {
    "Ostrich":         r"\textbf{Ostrich}",
    "MJX":           "MJX",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = ["Ostrich", "MJX", "Semi-Implicit"]

# Paper figure pinning (RA-L revision). Explicit rather than glob-preferred so
# the figure is reproducible from a named result set.
PAPER_JSON = {
    "Ostrich":       "ostrich_postfix_vjp.json",
    "MJX":           "mjx_2ms_lr0.3_final.json",   # 2 ms, swept lr = 0.3
    "Semi-Implicit": "semi_implicit_all_fixes.json",
}

# Warm per-iteration seconds, measured exclusively (one job alone on one GPU)
# as marginal = (wall_15 - wall_5) / 10. Setup is EXCLUDED from the x-axis;
# it is 1.2 s (Ostrich), 84 s (MJX), 1731 s (Semi-Implicit).
PER_ITER_S = {"Ostrich": 0.552, "MJX": 116.393, "Semi-Implicit": 2.559}

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
    for tag in ("_postfix_vjp", "_postfix"):
        cand = RESULTS_DIR / f"{engine_key}{tag}.json"
        if cand.exists():
            return cand
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
    ap.add_argument("--fig-h", type=float, default=2.6,
                    help="canvas height in inches; the printed height is "
                         "3.40 in / (cropped width / cropped height)")
    ap.add_argument("--min-iters", type=int, default=10,
                    help="skip engine JSONs with fewer iters than this "
                    "(filters out sanity-test JSONs).")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="truncate each loss curve at this many iters "
                    "(useful for showing only the early descent regime).")
    ap.add_argument("--engines", nargs="+", default=None,
                    choices=SIM_ORDER,
                    help="subset of engines to plot (default: all available). "
                    "e.g. --engines Ostrich MJX")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    sim_order = args.engines if args.engines else SIM_ORDER

    # Engine key (file prefix) ↔ "simulator" string in the JSON.
    engine_file_keys = {
        "Ostrich": "ostrich",
        "MJX": "mjx",
        "Semi-Implicit": "semi_implicit",
    }

    engines = {}
    for sim in sim_order:
        pinned = RESULTS_DIR / PAPER_JSON.get(sim, "")
        path = pinned if pinned.is_file() \
               else _pick_json_for_engine(engine_file_keys[sim])
        if path is None:
            print(f"  [skip] {sim}: no <engine>.json or <engine>_all_fixes.json found")
            continue
        d = json.load(open(path))
        if d.get("iterations", 0) < args.min_iters:
            print(f"  [skip] {path.name} ({sim}): only {d.get('iterations')} iters "
                  f"(< {args.min_iters}) — sanity run, not production.")
            continue
        # Exclusive measured marginal per-iteration time (setup excluded).
        per_iter_s = PER_ITER_S.get(sim)
        if per_iter_s is None:
            per_iter_s = min(t["wall_s"] for t in d["trials"]) / d["iterations"]
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

    fig, ax = plt.subplots(figsize=(FIG_W, args.fig_h))

    for sim in sim_order:
        if sim not in engines:
            continue
        st = STYLES[sim]
        curves = engines[sim]["curves"]
        if len(curves) == 1:
            cum, best = curves[0]
            ax.plot(cum, best, color=st["color"], marker=st["marker"],
                    linewidth=ps.LW * S, markersize=ps.MS * S,
                    markevery=max(1, len(cum) // 12),
                    label=LABELS[sim], zorder=st["zorder"])
            continue
        t_grid, median, q1, q3 = _aggregate_on_grid(curves)
        ax.fill_between(t_grid, q1, q3, color=st["color"], alpha=0.18,
                        linewidth=0, zorder=st["zorder"] - 1)
        ax.plot(t_grid, median, color=st["color"], marker=st["marker"],
                linewidth=ps.LW * S, markersize=ps.MS * S,
                markevery=max(1, len(t_grid) // 12),
                label=LABELS[sim], zorder=st["zorder"])

    ax.set_xlabel("Wall-clock time [s]")
    ax.set_ylabel(r"Running-best loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=ps.GRID_MAJOR["alpha"],
            linewidth=ps.GRID_MAJOR["lw"] * S)
    ax.grid(True, which="minor", alpha=ps.GRID_MINOR["alpha"],
            linewidth=ps.GRID_MINOR["lw"] * S)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=len(handles),
              fontsize=ps.PRINT["legend"] * S, frameon=False,
              columnspacing=1.5, handlelength=1.5)

    out = pathlib.Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    _px = plt.imread(out)
    _w, _h = _px.shape[1] / 200.0, _px.shape[0] / 200.0
    print(f"\nSaved {out}  (cropped canvas {_w:.2f} x {_h:.2f} in -> printed "
          f"{ps.COLUMN_IN:.2f} x {ps.COLUMN_IN * _h / _w:.2f} in; DRAWN_IN is "
          f"{DRAWN_IN:.2f}, printed label {ps.PRINT['label'] * DRAWN_IN / _w:.2f} pt)")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "convergence_box2.png"
        plt.savefig(out_paper, dpi=200, bbox_inches="tight")
        print(f"Saved {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
