"""dt-vs-error overlay for the box scene — paper style.

Loads results/accuracy_vs_dt.json (produced by run_accuracy_vs_dt.py) and
plots combined error vs dt for each engine. Visual rules:

  - Stable + finite point  -> filled engine marker (o/s/^ per engine).
                              No distinction by error magnitude — the
                              threshold line + shaded band above show the
                              reader where "usable" ends.
  - Crashed / NaN          -> red X (single "Crashed / NaN" legend entry),
                              plotted at BLOWUP_DISPLAY at the top so it
                              clearly reads as "off the scale".

Style mirrors experiments/4_scalability_box/plot_results.py (LaTeX serif,
log-log, OOM-band-style threshold visualisation, bottom legend).

Usage:
    python experiments/2_dt_stability_box/plot_dt_vs_error.py
    python experiments/2_dt_stability_box/plot_dt_vs_error.py --json other.json
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import blended_transform_factory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "ostrich_paper" / "figures"

plt.rcParams.update(
    {
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
    }
)

STYLES = {
    "Ostrich": {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MuJoCo": {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 3},
}
LABELS = {
    "Ostrich": r"\textbf{Ostrich}",
    "MuJoCo": "MuJoCo",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = list(STYLES.keys())

# Fixed y-value where all crashed/NaN X markers are plotted. Putting them
# at one row keeps the plot uncluttered — actual error values for crashed
# runs aren't meaningful (anywhere from 1 m to NaN), only the dt at which
# they failed is. Sits above the threshold band so failures read as "off
# the usable range" without flying off the top of the plot.
CRASHED_MARKER_Y = 10.0

# Manual layout for the "Nx larger usable dt" annotation. Edit any of these
# to nudge the arrow and label independently. Coordinates are in data space
# (x = dt in seconds, y = error in m). Set a value to None to fall back to
# the auto default.
#
# Vertical distance between arrow and text: two ways to control it —
#   * ``label_y``    — absolute text y in metres (wins if not None)
#   * ``label_gap``  — text y = arrow_y * label_gap (only used if
#                      label_y is None). On the log y-axis, label_gap < 1
#                      pushes the text below the arrow (smaller y), > 1
#                      pushes it above. 0.75 = a quarter-decade below.
SPEEDUP_LABEL_CFG: dict = {
    "arrow_y": 0.05,  # y-position (m) of the horizontal arrow segment
    "label_x": None,  # x-position (s) of the text — default centers it
    "label_y": None,  # ABSOLUTE text y (m). If None, uses label_gap.
    "label_gap": 0.9,  # text y = arrow_y * label_gap. <1 = below, >1 = above
    "rotation": 0,  # text rotation in degrees
    "fontsize": 10,  # text font size
}

# Manual layout for the "accuracy threshold (X m)" red label. ``x`` is in
# axes coords (0 = left edge, 1 = right edge). ``y_gap`` multiplies the
# threshold value to position the text (1.15 = a hair above the line). Set
# ``y`` to an absolute y (m) to override y_gap. ``ha`` is matplotlib's
# horizontal alignment of the text relative to its anchor.
THRESHOLD_LABEL_CFG: dict = {
    "x": 0.25,  # axes coord (0..1) — 0.012 = just inside the left spine
    "y": None,  # ABSOLUTE y in metres; if None, uses y_gap * threshold
    "y_gap": 1.15,  # text y = threshold * y_gap (only used if y is None)
    "ha": "left",  # "left" | "center" | "right"
    "va": "bottom",
    "fontsize": 10,
}


def _is_crashed(row):
    err = row["mean_combined_with_yaw"]
    return (not row.get("all_stable", True)) or (not np.isfinite(err))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(RESULTS_DIR / "accuracy_vs_dt.json"))
    ap.add_argument("--save", default=str(RESULTS_DIR / "dt_vs_error.png"))
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Accuracy threshold (m) for the red line + shaded band + "
        "speedup ratio annotation. Default 0.2 m (~one wheel-radius). "
        "Affects visualisation only — does not re-run the sweep.",
    )
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    threshold = args.threshold
    results = data["results"]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for sim in SIM_ORDER:
        if sim not in results:
            continue
        rows = sorted(results[sim], key=lambda r: r["dt"])
        st = STYLES[sim]

        # Line connecting stable+finite points only (so it doesn't shoot off
        # when the engine crashes at the upper end of dt).
        stable_dts = [r["dt"] for r in rows if not _is_crashed(r)]
        stable_errs = [r["mean_combined_with_yaw"] for r in rows if not _is_crashed(r)]
        if stable_dts:
            ax.plot(
                stable_dts,
                stable_errs,
                color=st["color"],
                marker=st["marker"],
                linewidth=st["lw"],
                markersize=6,
                label=LABELS[sim],
                zorder=st["zorder"],
            )

        # Engine-colored X markers for crashed/NaN, all sitting on a fixed
        # row at CRASHED_MARKER_Y so the plot stays compact. A dashed
        # engine-colored segment from the last stable point up to the first
        # X (and across to subsequent X markers, if any) attributes the
        # crashes to the right engine without color-coding the X itself.
        crashed_dts = [r["dt"] for r in rows if _is_crashed(r)]
        if crashed_dts and stable_dts:
            xs = [stable_dts[-1]] + list(crashed_dts)
            ys = [stable_errs[-1]] + [CRASHED_MARKER_Y] * len(crashed_dts)
            ax.plot(xs, ys, color=st["color"], linestyle="--", linewidth=1.2, alpha=0.55, zorder=2)
            ax.plot(
                crashed_dts,
                [CRASHED_MARKER_Y] * len(crashed_dts),
                "x",
                color=st["color"],
                markersize=10,
                markeredgewidth=2.2,
                zorder=6,
            )
        elif crashed_dts:
            ax.plot(
                crashed_dts,
                [CRASHED_MARKER_Y] * len(crashed_dts),
                "x",
                color=st["color"],
                markersize=10,
                markeredgewidth=2.2,
                zorder=6,
            )

    # Threshold visualisation: solid red line + shaded band above + label.
    ax.axhline(threshold, color="red", linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhspan(threshold, CRASHED_MARKER_Y * 3, color="red", alpha=0.05, zorder=0)
    # Threshold label — position fully configurable via THRESHOLD_LABEL_CFG.
    th_x = THRESHOLD_LABEL_CFG.get("x", 0.012)
    th_y = THRESHOLD_LABEL_CFG.get("y") or (threshold * THRESHOLD_LABEL_CFG.get("y_gap", 1.15))
    ax.text(
        th_x,
        th_y,
        rf"accuracy threshold (${threshold}$\,m)",
        fontsize=THRESHOLD_LABEL_CFG.get("fontsize", 10),
        color="red",
        alpha=0.9,
        ha=THRESHOLD_LABEL_CFG.get("ha", "left"),
        va=THRESHOLD_LABEL_CFG.get("va", "bottom"),
        transform=blended_transform_factory(ax.transAxes, ax.transData),
    )

    # "Nx larger usable dt" arrow between MuJoCo's and Ostrich's largest usable
    # (stable AND under-threshold) dts.
    if "Ostrich" in results and "MuJoCo" in results:

        def _max_usable_dt(rows):
            usable = [
                r["dt"]
                for r in rows
                if not _is_crashed(r) and r["mean_combined_with_yaw"] <= threshold
            ]
            return max(usable) if usable else None

        mj_max = _max_usable_dt(results["MuJoCo"])
        ax_max = _max_usable_dt(results["Ostrich"])
        if mj_max and ax_max:
            ratio = ax_max / mj_max
            # Auto defaults, applied wherever SPEEDUP_LABEL_CFG entry is None.
            arrow_y = SPEEDUP_LABEL_CFG.get("arrow_y") or 0.055
            label_x = SPEEDUP_LABEL_CFG.get("label_x") or np.sqrt(mj_max * ax_max)
            label_gap = SPEEDUP_LABEL_CFG.get("label_gap", 0.75)
            label_y = SPEEDUP_LABEL_CFG.get("label_y") or arrow_y * label_gap
            rotation = SPEEDUP_LABEL_CFG.get("rotation", 0)
            fontsize = SPEEDUP_LABEL_CFG.get("fontsize", 10)
            ax.annotate(
                "",
                xy=(ax_max, arrow_y),
                xytext=(mj_max, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.3, shrinkA=2, shrinkB=2),
                zorder=7,
            )
            ax.text(
                label_x,
                label_y,
                rf"$\sim{ratio:.0f}\times$ larger usable $h$",
                ha="center",
                va="top",
                fontsize=fontsize,
                fontweight="bold",
                rotation=rotation,
                zorder=7,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"timestep $h$ (s)")
    ax.set_ylabel("Combined error (position + yaw) (m)")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5), numticks=12))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", alpha=0.35, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
    ax.set_ylim(3e-2, CRASHED_MARKER_Y * 3)

    # Legend at the bottom with a separate "Crashed / NaN" entry. X colour
    # in the plot is per-engine (so the reader can tell which engine crashed
    # without needing colour-coded crashes in the legend) — the legend entry
    # uses a neutral gray X to convey marker shape only.
    ax.plot([], [], "x", color="0.4", markersize=10, markeredgewidth=2.2, label="Crashed / NaN")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        columnspacing=1.8,
    )

    out = pathlib.Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}  (data: {args.json})")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        # Save under both filenames the paper might reference (the older
        # box_<name>.png convention and the newer <name>_box.png).
        for fname in ("box_dt_stability.png", "dt_stability_box.png"):
            out_paper = paper_dir / fname
            plt.savefig(out_paper, dpi=300, bbox_inches="tight")
            print(f"Saved {out_paper}")


if __name__ == "__main__":
    main()
