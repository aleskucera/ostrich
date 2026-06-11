"""Plot N-trial robust comparison for 3_gradient_quality_box2.

Reads results/<engine>.json saved by optimize_*.py. One row of panels
per engine, three columns: success/failure scatter (pos vs terminal-speed),
control-jerk histogram, and a summary annotation. Single page so the
reader sees the three engines side-by-side.

Style mirrors experiments/4_scalability_box/plot_results.py (LaTeX serif,
engine colors/markers).

Usage:
    python experiments/3_gradient_quality_box2/plot_results.py
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
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STYLES = {
    "Ostrich":         {"color": "#2196F3", "marker": "o"},
    "MJX":           {"color": "#E91E63", "marker": "s"},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^"},
}
LABELS = {
    "Ostrich":         r"\textbf{Ostrich}",
    "MJX":           "MJX",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = list(STYLES.keys())

POS_SUCCESS = 0.2     # m — matches optimize_*.py's metrics["success"]
VEL_SUCCESS = 0.3     # m/s


def load_results():
    out = {}
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        sim = d.get("simulator")
        if sim is not None and sim in STYLES:
            out[sim] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=str(RESULTS_DIR / "robust_comparison.png"))
    args = ap.parse_args()

    results = load_results()
    if not results:
        print(f"No results in {RESULTS_DIR}. Run run_experiment.sh first.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    ax_scatter, ax_box_err, ax_box_jerk = axes

    # Panel 1: scatter pos_error vs terminal_speed, color/marker per engine,
    # red shaded "success" quadrant (lower-left).
    ax_scatter.axhspan(0, VEL_SUCCESS, xmin=0, xmax=POS_SUCCESS / 1.0,
                       color="green", alpha=0.06, zorder=0)
    ax_scatter.axvline(POS_SUCCESS, color="green", linestyle="--",
                       linewidth=0.8, alpha=0.6, zorder=1)
    ax_scatter.axhline(VEL_SUCCESS, color="green", linestyle="--",
                       linewidth=0.8, alpha=0.6, zorder=1)
    ax_scatter.text(POS_SUCCESS * 1.05, VEL_SUCCESS * 1.05, "success",
                    color="green", fontsize=8, ha="left", va="bottom", alpha=0.7,
                    rotation=0)

    # Per-engine box plot data
    pos_errors_by_engine = {}
    jerks_by_engine = {}

    for sim in SIM_ORDER:
        if sim not in results:
            continue
        st = STYLES[sim]
        trials = results[sim]["trials"]
        pos = np.array([t["final_metrics"]["pos_error_m"] for t in trials])
        vel = np.array([t["final_metrics"]["terminal_speed_mps"] for t in trials])
        jerk = np.array([t["final_metrics"]["control_jerk"] for t in trials])
        pos_errors_by_engine[sim] = pos
        jerks_by_engine[sim] = jerk

        ax_scatter.scatter(pos, vel,
                            color=st["color"], marker=st["marker"],
                            s=42, alpha=0.75, edgecolors="black",
                            linewidths=0.4, label=LABELS[sim], zorder=4)

    ax_scatter.set_xlabel("Final position error (m)")
    ax_scatter.set_ylabel("Terminal speed (m/s)")
    ax_scatter.set_xscale("log"); ax_scatter.set_yscale("log")
    ax_scatter.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax_scatter.grid(True, which="minor", alpha=0.1, linewidth=0.4)
    ax_scatter.legend(loc="lower right", fontsize=8, frameon=False)
    ax_scatter.set_title("Final-pose error vs terminal speed", fontsize=10)

    # Panel 2: box plot of pos_error per engine
    sims_present = [s for s in SIM_ORDER if s in pos_errors_by_engine]
    positions = np.arange(len(sims_present))
    bp = ax_box_err.boxplot([pos_errors_by_engine[s] for s in sims_present],
                             positions=positions, widths=0.55, patch_artist=True,
                             medianprops={"color": "black", "linewidth": 1.2},
                             flierprops={"marker": "o", "markersize": 3})
    for patch, sim in zip(bp["boxes"], sims_present):
        patch.set_facecolor(STYLES[sim]["color"])
        patch.set_alpha(0.65)
        patch.set_edgecolor("black"); patch.set_linewidth(0.6)
    ax_box_err.axhline(POS_SUCCESS, color="green", linestyle="--",
                       linewidth=0.8, alpha=0.6, zorder=1)
    ax_box_err.text(len(sims_present) - 0.5, POS_SUCCESS * 1.05,
                    rf"success threshold ({POS_SUCCESS}\,m)",
                    fontsize=7.5, color="green", alpha=0.8,
                    ha="right", va="bottom")
    ax_box_err.set_xticks(positions)
    ax_box_err.set_xticklabels([LABELS[s] for s in sims_present])
    ax_box_err.set_ylabel("Final position error (m)")
    ax_box_err.set_yscale("log")
    ax_box_err.grid(True, axis="y", which="major", alpha=0.3, linewidth=0.5)
    ax_box_err.set_title("Pos-error distribution (N trials)", fontsize=10)

    # Panel 3: box plot of control jerk per engine
    bp2 = ax_box_jerk.boxplot([jerks_by_engine[s] for s in sims_present],
                               positions=positions, widths=0.55, patch_artist=True,
                               medianprops={"color": "black", "linewidth": 1.2},
                               flierprops={"marker": "o", "markersize": 3})
    for patch, sim in zip(bp2["boxes"], sims_present):
        patch.set_facecolor(STYLES[sim]["color"])
        patch.set_alpha(0.65)
        patch.set_edgecolor("black"); patch.set_linewidth(0.6)
    ax_box_jerk.set_xticks(positions)
    ax_box_jerk.set_xticklabels([LABELS[s] for s in sims_present])
    ax_box_jerk.set_ylabel(r"Control jerk $\Sigma_t\|\Delta u\|$")
    ax_box_jerk.set_yscale("log")
    ax_box_jerk.grid(True, axis="y", which="major", alpha=0.3, linewidth=0.5)
    ax_box_jerk.set_title("Control smoothness (lower = smoother)", fontsize=10)

    # Print summary table to stdout (handy for caption text)
    print(f"{'engine':>14s}  {'N':>4s}  {'success':>8s}  {'med pos':>9s}  "
          f"{'med vel':>9s}  {'med jerk':>9s}")
    print("-" * 70)
    for sim in sims_present:
        d = results[sim]
        agg = d.get("aggregate", {})
        N = d["num_trials"]
        sr = agg.get("success_rate", float("nan"))
        mp = agg.get("median_pos_error_m", float("nan"))
        mv = agg.get("median_terminal_speed_mps", float("nan"))
        mj_jerk = float(np.median(jerks_by_engine[sim]))
        print(f"{sim:>14s}  {N:>4d}  {sr:>7.0%}   {mp:>9.3f}  "
              f"{mv:>9.3f}  {mj_jerk:>9.2f}")

    fig.suptitle("Random-IC final-pose task: per-engine robust statistics",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = pathlib.Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        for fname in ("box_robust_comparison.png", "robust_comparison_box.png"):
            out_paper = paper_dir / fname
            plt.savefig(out_paper, dpi=300, bbox_inches="tight")
            print(f"Saved {out_paper}")


if __name__ == "__main__":
    main()
