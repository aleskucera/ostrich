"""Paper-style loss-curve plot for 3_gradient_quality_box2.

Single figure overlaying TWO curves (median + IQR band across N trials):
  - "actual loss at iter k" — what the optimiser reports each iter,
                              wobbles because Adam overshoots and contact
                              gradients are noisy
  - "best-iter loss"         — running min, what would be deployed when
                              snapshotting at min-loss params

Both shown on the same log-y axis so the gap between "what Adam ends up at"
and "what we actually deploy" is visible at a glance. Style matches
experiments/3_gradient_quality_box/plot_results.py (LaTeX serif, IQR band,
log scales).

Usage:
    python experiments/3_gradient_quality_box2/plot_loss_curves.py
    python experiments/3_gradient_quality_box2/plot_loss_curves.py \
        --json results/mjx_all_fixes.json --save mjx_loss_curves.png
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "ostrich_paper" / "figures"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Match the paper's existing engine colours.
ENGINE_COLOR = {
    "Ostrich":         "#2196F3",
    "MJX":           "#E91E63",
    "Semi-Implicit": "#FF9800",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(RESULTS_DIR / "ostrich_all_fixes.json"))
    ap.add_argument("--save", default=str(RESULTS_DIR / "loss_curves.png"))
    args = ap.parse_args()

    d = json.load(open(args.json))
    sim = d.get("simulator", "Ostrich")
    trials = d["trials"]
    N = len(trials)
    n_iters = len(trials[0]["losses"])

    raw = np.array([t["losses"] for t in trials], dtype=np.float64)
    best = np.minimum.accumulate(raw, axis=1)
    iters = np.arange(n_iters)

    # Median + IQR (robust to outlier trials)
    raw_med = np.median(raw, axis=0)
    raw_q1 = np.percentile(raw, 25, axis=0)
    raw_q3 = np.percentile(raw, 75, axis=0)
    best_med = np.median(best, axis=0)
    best_q1 = np.percentile(best, 25, axis=0)
    best_q3 = np.percentile(best, 75, axis=0)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    base_color = ENGINE_COLOR.get(sim, "#2196F3")

    # Actual loss — faded, dashed, on top of the engine colour but lighter
    c_raw = "0.55"  # neutral gray so it doesn't compete with the bold best-iter line
    ax.fill_between(iters, raw_q1, raw_q3,
                    color=c_raw, alpha=0.15, zorder=2, linewidth=0)
    ax.plot(iters, raw_med, color=c_raw, linestyle="--", linewidth=1.5,
            marker="s", markersize=3.5, markevery=max(1, n_iters // 15),
            markerfacecolor="white", markeredgewidth=0.8,
            label=r"actual loss at iter $k$ (median, IQR)",
            zorder=3)

    # Running-best — engine colour, solid, the headline curve
    ax.fill_between(iters, best_q1, best_q3,
                    color=base_color, alpha=0.22, zorder=4, linewidth=0)
    ax.plot(iters, best_med, color=base_color, linewidth=2.2,
            marker="o", markersize=4, markevery=max(1, n_iters // 15),
            label=r"running-best loss (median, IQR) " \
                  r"\textit{[what gets deployed]}",
            zorder=5)

    # Axes / styling
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss")
    ax.set_yscale("log")
    ax.set_xlim(0, n_iters - 1)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which="major", alpha=0.35, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)

    # Inline "gap" annotation: arrow between final actual and final best.
    fa = raw_med[-1]; fb = best_med[-1]
    if fa > fb * 1.05:
        ax.annotate("", xy=(n_iters - 1.5, fb), xytext=(n_iters - 1.5, fa),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.0,
                                     shrinkA=2, shrinkB=2),
                    zorder=6)
        ax.text(n_iters - 2.5, np.sqrt(fa * fb),
                rf"${(fa/fb):.1f}\times$", color="0.2", fontsize=9,
                ha="right", va="center", fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False)

    fig.suptitle(rf"\textbf{{{sim}}} — final-pose optimisation, "
                 rf"$N={N}$ trials", fontsize=11, y=1.00)
    fig.tight_layout()

    out = pathlib.Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        sim_suffix = sim.lower().replace(" ", "_").replace("-", "_")
        out_paper = paper_dir / f"box2_loss_curves_{sim_suffix}.png"
        plt.savefig(out_paper, dpi=200, bbox_inches="tight")
        print(f"Saved {out_paper}")

    print()
    print(f"=== Summary ({N} trials × {n_iters} iters) — robust statistics ===")
    print(f"Actual loss at iter {n_iters-1}: "
          f"median={raw_med[-1]:.4f}  IQR=[{raw_q1[-1]:.4f}, {raw_q3[-1]:.4f}]")
    print(f"Best-iter loss at iter {n_iters-1}:   "
          f"median={best_med[-1]:.4f}  IQR=[{best_q1[-1]:.4f}, {best_q3[-1]:.4f}]")
    gap = raw_med[-1] / best_med[-1]
    print(f"Median ratio (actual/best): {gap:.2f}x")


if __name__ == "__main__":
    main()
