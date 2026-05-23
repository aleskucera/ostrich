"""dt-vs-error overlay for the box scene — fully reproducible.

Loads results/accuracy_vs_dt.json (produced by run_accuracy_vs_dt.py) and
plots combined error vs dt for each engine, with markers categorising:
  - filled circle  = stable AND under the accuracy threshold (usable),
  - open circle    = stable but error above the threshold ("runs, not usable"),
  - X marker       = unstable / NaN / didn't pass the box (broken).

There are NO hand-copied numbers in this script.

Usage:
    python experiments/2_dt_stability_box/plot_dt_vs_error.py
    python experiments/2_dt_stability_box/plot_dt_vs_error.py --json other.json
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

COLORS = {"Axion": "#2196F3", "MuJoCo": "#E91E63", "Semi-Implicit": "#FF9800"}
# A small finite floor for log-scale plotting of NaN-but-finite blow-ups.
BLOWUP_DISPLAY = 1e3


def categorise(row, threshold):
    err = row["mean_combined_with_yaw"]
    stable = row["all_stable"]
    if not np.isfinite(err):
        return "broken", BLOWUP_DISPLAY
    if not stable:
        # finite but failed the stability criterion (e.g. didn't cross box).
        # Plot it where its error would be, but use the "broken" marker so the
        # reader sees it doesn't count as usable.
        return "broken", min(err, BLOWUP_DISPLAY)
    if err > threshold:
        return "degraded", err
    return "usable", err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(RESULTS_DIR / "accuracy_vs_dt.json"))
    ap.add_argument("--save", default=str(RESULTS_DIR / "dt_vs_error.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    threshold = data["accuracy_threshold_m"]
    results = data["results"]

    fig, ax = plt.subplots(figsize=(10, 5.7))

    for name, rows in results.items():
        color = COLORS.get(name, "tab:gray")
        # Sorted by dt for the connecting line.
        rows = sorted(rows, key=lambda r: r["dt"])
        dts = np.array([r["dt"] for r in rows])
        errs_for_line = np.array([
            min(r["mean_combined_with_yaw"], BLOWUP_DISPLAY)
            if np.isfinite(r["mean_combined_with_yaw"]) else BLOWUP_DISPLAY
            for r in rows])
        ax.plot(dts, errs_for_line, "-", color=color, alpha=0.45, lw=1.4, zorder=2)

        # Categorise + scatter per category, with a single legend entry per engine
        # for the "usable" marker.
        usable_x, usable_y, deg_x, deg_y, brk_x, brk_y = [], [], [], [], [], []
        for r in rows:
            cat, y = categorise(r, threshold)
            x = r["dt"]
            (usable_x if cat == "usable" else deg_x if cat == "degraded" else brk_x).append(x)
            (usable_y if cat == "usable" else deg_y if cat == "degraded" else brk_y).append(y)

        if usable_x:
            ax.plot(usable_x, usable_y, "o", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=f"{name} — usable", zorder=4)
        if deg_x:
            ax.plot(deg_x, deg_y, "o", color="white", markersize=10,
                    markeredgecolor=color, markeredgewidth=2.0,
                    label=f"{name} — runs, err > {threshold} m", zorder=4)
        if brk_x:
            ax.plot(brk_x, brk_y, "X", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.7,
                    label=f"{name} — broken / NaN", zorder=5)

    # Accuracy threshold line + annotation
    ax.axhline(threshold, color="gray", ls="--", lw=1, alpha=0.6, zorder=1)
    ax.text(ax.get_xlim()[0] * 1.4 if ax.get_xlim()[0] > 0 else 1.2e-4,
            threshold * 1.18, f"usable threshold ({threshold} m)",
            fontsize=9, color="gray")

    # "Order-of-magnitude" annotation between the engines' usable ranges.
    if "Axion" in results and "MuJoCo" in results:
        mj_usable_dts = [r["dt"] for r in results["MuJoCo"]
                         if categorise(r, threshold)[0] == "usable"]
        ax_usable_dts = [r["dt"] for r in results["Axion"]
                         if categorise(r, threshold)[0] == "usable"]
        if mj_usable_dts and ax_usable_dts:
            mj_max = max(mj_usable_dts)
            ax_max = max(ax_usable_dts)
            ratio = ax_max / mj_max
            ax.annotate("", xy=(ax_max, 0.013), xytext=(mj_max, 0.013),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
            ax.text(np.sqrt(mj_max * ax_max), 0.0095,
                    rf"~{ratio:.0f}× larger usable $\Delta t$",
                    ha="center", fontsize=10.5, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"timestep $\Delta t$ [s]")
    ax.set_ylabel("combined error (position + yaw) [m]")
    ax.set_title("Accuracy vs dt on the box scene — real-data drive, "
                 "tuned best params per engine")
    ax.grid(which="both", alpha=0.22)
    ax.set_ylim(2e-3, 3e3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=140, bbox_inches="tight")
    print(f"Saved {args.save}  (data: {args.json})")


if __name__ == "__main__":
    main()
