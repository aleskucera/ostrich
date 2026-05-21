"""Plot per-iteration timing for the ball-throw gradient optimization benchmark.

Usage:
    python examples/comparison/ball_throw/plot_results.py
    python examples/comparison/ball_throw/plot_results.py --show
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
    "Nimble":       {"color": "#795548"},
    "Dojo":         {"color": "#4CAF50"},
    "Genesis":      {"color": "#9C27B0"},
    "MuJoCo-FD":    {"color": "#E91E63"},
    "MJX-jacfwd":   {"color": "#FF5722"},
    "MJX-grad":     {"color": "#FF8A65"},
    "TinyDiffSim":  {"color": "#607D8B"},
    "Brax":         {"color": "#009688"},
    "Axion":        {"color": "#2196F3"},
}
LABELS = {
    "Nimble":       "Nimble",
    "Dojo":         "Dojo",
    "Genesis":      "Genesis",
    "MuJoCo-FD":    "MuJoCo-FD",
    "MJX-jacfwd":   "MJX-jacfwd",
    "MJX-grad":     "MJX-grad",
    "TinyDiffSim":  "TinyDiffSim",
    "Brax":         "Brax",
    "Axion":        r"\textbf{Axion}",
}
SIM_ORDER = list(STYLES.keys())
AXION_COLOR = "#2196F3"


def load_results() -> dict:
    out = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if "time_ms" not in data:
            continue
        sim = data.get("simulator", path.stem)
        out[sim] = data
    return out


def _median(data: dict) -> float:
    t = np.array(data["time_ms"])
    t = t[3:] if len(t) > 3 else t
    return float(np.median(t))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No results found. Run the benchmark scripts first.")
        return
    if "Axion" not in results:
        print("Axion results not found — cannot compute ×N ratios.")
        return

    # Sort ascending by median timing (fastest = Axion at top)
    sims = sorted(results.keys(), key=lambda s: _median(results[s]))

    axion_median = _median(results["Axion"])
    colors  = [STYLES.get(s, {"color": "gray"})["color"] for s in sims]
    ylabels = [LABELS.get(s, s) for s in sims]

    medians, p25, p75 = [], [], []
    for sim in sims:
        t = np.array(results[sim]["time_ms"])
        t = t[3:] if len(t) > 3 else t
        medians.append(float(np.median(t)))
        p25.append(float(np.percentile(t, 25)))
        p75.append(float(np.percentile(t, 75)))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    y = np.arange(len(sims))
    bars = ax.barh(y, medians, color=colors, height=0.5, zorder=3)
    ax.errorbar(
        medians, y,
        xerr=[np.array(medians) - np.array(p25), np.array(p75) - np.array(medians)],
        fmt="none", color="black", capsize=2, linewidth=0.6, zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"ms\,/\,iter (median $\pm$ IQR)")
    ax.grid(True, axis="x", which="both", alpha=0.25, zorder=0, linewidth=0.5)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(-0.6, len(sims) - 0.4)
    ax.set_xlim(right=max(medians) * 4)

    right_xfm = blended_transform_factory(ax.transAxes, ax.transData)

    for bar, sim, val, m25, m75 in zip(bars, sims, medians, p25, p75):
        cy = bar.get_y() + bar.get_height() / 2
        label = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
        ax.text(val * 1.5, cy, label, va="center", ha="left", fontsize=7)

        if sim != "Axion":
            ratio = val / axion_median
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

    plt.tight_layout(pad=0.4, rect=(0, 0, 0.76, 1))
    out = RESULTS_DIR / "timing.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
