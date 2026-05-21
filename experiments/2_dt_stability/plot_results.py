"""Plot Experiment 2: dt stability (bar chart) + accuracy vs dt curve.

Left panel: max stable dt per simulator on obstacle scene (from 2_dt_stability).
Right panel: error vs dt on flat-ground trajectories (from 1_sim_to_real dt sweeps).

Usage:
    python experiments/2_dt_stability/plot_results.py
    python experiments/2_dt_stability/plot_results.py --save results/dt_stability.png
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import blended_transform_factory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
SIM_TO_REAL_DIR = pathlib.Path(__file__).parent.parent / "1_sim_to_real" / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

STYLES = {
    "Axion": {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MuJoCo": {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 3},
    "TinyDiffSim": {"color": "#607D8B", "marker": "D", "lw": 1.8, "zorder": 2},
}
LABELS = {
    "Axion": r"\textbf{Axion}",
    "MuJoCo": "MuJoCo",
    "Semi-Implicit": "Semi-Impl.",
    "TinyDiffSim": "TinyDiffSim",
}
SIM_ORDER = list(STYLES.keys())

ACCURACY_THRESHOLD = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # --- Load dt_stability (max-stable-dt on obstacle) ---
    stability = {}
    for sim_file, sim_name in [
        ("sweep_axion.json", "Axion"),
        ("sweep_mujoco.json", "MuJoCo"),
        ("sweep_semi_implicit.json", "Semi-Implicit"),
    ]:
        path = RESULTS_DIR / sim_file
        if path.exists():
            with open(path) as f:
                stability[sim_name] = json.load(f)

    # --- Load dt-only sweeps from 1_sim_to_real ---
    dt_sweeps = {}
    for sim_file, sim_name in [
        ("sweep_axion_dt.json", "Axion"),
        ("sweep_mujoco_dt.json", "MuJoCo"),
        ("sweep_semi_implicit_dt.json", "Semi-Implicit"),
        ("sweep_tinydiffsim_dt.json", "TinyDiffSim"),
    ]:
        path = SIM_TO_REAL_DIR / sim_file
        if path.exists():
            with open(path) as f:
                dt_sweeps[sim_name] = json.load(f)

    fig, (ax_bar, ax_curve) = plt.subplots(
        1,
        2,
        figsize=(8.5, 3.4),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig.subplots_adjust(wspace=0.35)

    # ---------- Left: Max stable dt bar chart (vertical) ----------
    sims = [s for s in SIM_ORDER if s in stability]

    def _stability_stats(entry: dict) -> tuple[float, float, float]:
        """Return (height, err_lo, err_hi). Bar = min (worst-case stable dt);
        upper whisker spans up to max (best-case stable dt in the sweep)."""
        trials = entry.get("trials")
        if not trials:
            return entry.get("max_stable_dt", 0.0) or 0.0, 0.0, 0.0
        dts = [t["max_stable_dt"] for t in trials if t["max_stable_dt"] > 0]
        if not dts:
            return 0.0, 0.0, 0.0
        dts_min = min(dts)
        dts_max = max(dts)
        return dts_min, 0.0, max(dts_max - dts_min, 0.0)

    heights = []
    err_lo = []
    err_hi = []
    for s in sims:
        h, el, eh = _stability_stats(stability[s])
        heights.append(h)
        err_lo.append(el)
        err_hi.append(eh)
    colors = [STYLES[s]["color"] for s in sims]
    x_pos = np.arange(len(sims))

    has_errs = any(el > 0 or eh > 0 for el, eh in zip(err_lo, err_hi))

    bars = ax_bar.bar(
        x_pos,
        heights,
        color=colors,
        width=0.6,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    if has_errs:
        ax_bar.errorbar(
            x_pos,
            heights,
            yerr=[err_lo, err_hi],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.0,
            zorder=4,
        )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([LABELS[s] for s in sims])
    ax_bar.set_ylabel(r"Max stable $\Delta t$ (s)")
    ax_bar.set_yscale("log")
    ax_bar.grid(True, axis="y", which="both", alpha=0.35, linewidth=0.6, zorder=0)
    ax_bar.set_xlim(-0.5, len(sims) - 0.5)

    for bar, dt, eh in zip(bars, heights, err_hi):
        cx = bar.get_x() + bar.get_width() / 2
        label = "unstable" if dt <= 0 else rf"${dt * 1000:.4g}$\,ms"
        top = max(dt + eh, 1e-5)
        ax_bar.text(cx, top * 1.25, label, va="bottom", ha="center", fontsize=10)

    if heights:
        top_bound = max((h + eh) for h, eh in zip(heights, err_hi))
        ax_bar.set_ylim(top=top_bound * 8)

    # ---------- Right: Error vs dt ----------
    for sim in SIM_ORDER:
        if sim not in dt_sweeps:
            continue
        top = dt_sweeps[sim].get("top_10", [])
        pts = sorted(
            [(r["params"]["dt"], r["error"]) for r in top],
            key=lambda p: p[0],
        )
        if not pts:
            continue
        dts = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        style = STYLES[sim]
        ax_curve.plot(
            dts,
            errs,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=5,
            label=LABELS[sim],
            zorder=style["zorder"],
        )

        # Mark max stable dt from Exp 2 as vertical dashed line (worst-case: min across trials)
        if sim in stability:
            entry = stability[sim]
            trials = entry.get("trials") or []
            dts = [t["max_stable_dt"] for t in trials if t.get("max_stable_dt", 0) > 0]
            stable_dt = min(dts) if dts else (entry.get("max_stable_dt") or 0)
            if stable_dt > 0:
                ax_curve.axvline(
                    stable_dt,
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.8,
                    zorder=2,
                )
                ax_curve.text(
                    stable_dt * 1.08,
                    0.6,
                    rf"max stable $\Delta t$",
                    rotation=270,
                    fontsize=10,
                    color=style["color"],
                    alpha=0.95,
                    ha="left",
                    va="bottom",
                    transform=blended_transform_factory(ax_curve.transData, ax_curve.transAxes),
                )

    ax_curve.set_xlabel(r"$\Delta t$ (s)")
    ax_curve.set_ylabel(r"Combined $L_2$ error (m)")
    ax_curve.set_xscale("log")
    ax_curve.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_curve.grid(True, which="both", alpha=0.35, linewidth=0.6)

    # Accuracy threshold
    ax_curve.axhline(
        ACCURACY_THRESHOLD,
        color="red",
        linestyle="-",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )
    ax_curve.text(
        0.85,
        ACCURACY_THRESHOLD * 1.04,
        rf"threshold ({ACCURACY_THRESHOLD}\,m)",
        fontsize=10,
        color="red",
        alpha=1.0,
        ha="right",
        va="bottom",
        transform=ax_curve.get_yaxis_transform(),
    )

    handles, labels = ax_curve.get_legend_handles_labels()
    ax_curve.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=13,
        framealpha=0.85,
        columnspacing=1.2,
        handlelength=1.5,
    )

    # --- Save ---
    save_arg = args.save or "dt_stability.png"
    out = RESULTS_DIR / save_arg if "/" not in save_arg else pathlib.Path(save_arg)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "dt_stability.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
