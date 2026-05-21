"""Plot maximum stable timestep per simulator — horizontal bar chart with ×N vs Axion.

Same layout as plot_stability_horizontal.py but annotates each bar with a
multiplier showing how many times larger Axion's max stable dt is compared
to that simulator.  The ×N column is placed outside the axes area using a
blended transform so it never overlaps with value labels.

Usage:
    python examples/comparison/stacked_boxes/plot_results.py
    python examples/comparison/stacked_boxes/plot_results.py --show
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STYLES = {
    "Semi-Implicit": {"color": "#607D8B"},
    "Genesis":       {"color": "#4CAF50"},
    "MuJoCo":        {"color": "#E91E63"},
    "MJX":           {"color": "#FF5722"},
    "TinyDiffSim":   {"color": "#9C27B0"},
    "Dojo":          {"color": "#FF9800"},
    "Axion":         {"color": "#2196F3"},
}
SIM_ORDER = list(STYLES.keys())

LABELS = {
    "Semi-Implicit": "Semi-Implicit",
    "Genesis":       "Genesis",
    "MuJoCo":        "MuJoCo",
    "MJX":           "MJX",
    "TinyDiffSim":   "TinyDiffSim",
    "Dojo":          "Dojo",
    "Axion":         r"\textbf{Axion}",
}

AXION_COLOR = "#2196F3"


def load_thresholds() -> dict[str, dict]:
    """Load all stacked_boxes result files that contain binary-search threshold data."""
    out = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if "max_stable_dt" not in data:
            continue
        sim = data.get("simulator", path.stem)
        out[sim] = data
    return out


def _fmt(val: float) -> str:
    if val < 0.001:
        exp = int(np.floor(np.log10(val)))
        mantissa = val / 10 ** exp
        return rf"{mantissa:.2f} \times 10^{{{exp}}}"
    return f"{val:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    thresholds = load_thresholds()
    if not thresholds:
        print("No threshold results found. Run the benchmark scripts first.")
        return

    if "Axion" not in thresholds:
        print("Axion results not found — cannot compute ×N ratios.")
        return

    sims = [s for s in SIM_ORDER if s in thresholds]
    if not sims:
        sims = sorted(thresholds.keys())

    # Separate simulators that are never stable (max_stable_dt == 0)
    never_stable = {s for s in sims if thresholds[s]["max_stable_dt"] == 0}
    stable_sims = [s for s in sims if s not in never_stable]

    # Sort ascending so the strongest simulator (Axion) is at the top
    # Never-stable simulators go at the bottom
    never_stable_sorted = sorted(never_stable)
    sims = never_stable_sorted + sorted(stable_sims, key=lambda s: thresholds[s]["max_stable_dt"])

    axion_dt = thresholds["Axion"]["max_stable_dt"]
    colors   = [STYLES.get(s, {"color": "gray"})["color"] for s in sims]
    values   = [thresholds[s]["max_stable_dt"] for s in sims]
    hit_max  = [thresholds[s].get("hit_max", False) for s in sims]
    ylabels  = [LABELS.get(s, s) for s in sims]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    y = np.arange(len(sims))
    # Plot bars only for stable simulators (value > 0)
    bar_values = [v if v > 0 else 0 for v in values]
    bars = ax.barh(y, bar_values, color=colors, height=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"Max stable $\Delta t$ (s)")
    ax.grid(True, axis="x", which="both", alpha=0.25, zorder=0, linewidth=0.5)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(-0.6, len(sims) - 0.4)

    xmax = max(v for v in values if v > 0)
    ax.set_xlim(right=xmax * 4)

    right_xfm = blended_transform_factory(ax.transAxes, ax.transData)

    for i, (sim, val, hm) in enumerate(zip(sims, values, hit_max)):
        cy = y[i]

        if val == 0:
            # Never-stable: show "N/A" label and ×∞
            ax.text(ax.get_xlim()[0] * 1.5, cy, r"\textit{never stable}",
                    va="center", ha="left", fontsize=7, color="0.45")
            ax.text(
                1.04, cy, r"$\times\infty$",
                va="center", ha="left", fontsize=7,
                color=AXION_COLOR, fontweight="bold",
                transform=right_xfm, clip_on=False,
            )
        else:
            # Value label to the right of the bar
            val_label = f"${_fmt(val)}$" + (r"$^+$" if hm else "")
            ax.text(val * 1.5, cy, val_label, va="center", ha="left", fontsize=7)

            if sim != "Axion":
                ratio = axion_dt / val
                ratio_str = (
                    f"$\\times{ratio:.0f}$" if ratio >= 10 else f"$\\times{ratio:.1f}$"
                )
                ax.text(
                    1.04, cy, ratio_str,
                    va="center", ha="left", fontsize=7,
                    color=AXION_COLOR, fontweight="bold",
                    transform=right_xfm, clip_on=False,
                )

    ax.text(
        1.04, 1.01, r"vs \textbf{Axion}",
        va="bottom", ha="left", fontsize=6,
        color="gray", transform=ax.transAxes, clip_on=False,
    )

    if any(hit_max):
        ax.text(
            0.98, 0.02,
            r"${}^+$ search limit; true threshold may be higher",
            transform=ax.transAxes,
            fontsize=6, ha="right", va="bottom", color="gray",
        )

    # Reserve right margin in the figure for the ×N column
    plt.tight_layout(pad=0.4, rect=(0, 0, 0.76, 1))

    out = RESULTS_DIR / "stacked_boxes.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
