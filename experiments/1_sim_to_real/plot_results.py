"""Plot parameter sweep results: turn XY, acceleration X(t), error bar chart.

Usage:
    python experiments/1_sim_to_real/plot_results.py
    python experiments/1_sim_to_real/plot_results.py --save results/sim_to_real.png
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SIM_COLORS = {
    "Axion": "#2196F3",
    "MuJoCo": "#E91E63",
    "Semi-Implicit": "#FF9800",
    "TinyDiffSim": "#607D8B",
    "Dojo": "#4CAF50",
}

SIM_ORDER = ["Axion", "MuJoCo", "Semi-Implicit", "TinyDiffSim", "Dojo"]

# Simulators above this error are considered failed (excluded from Experiment 2+)
ACCURACY_THRESHOLD = 1.0  # meters


def load_ground_truth(name):
    path = DATA_DIR / f"{name}.json"
    with open(path) as f:
        gt = json.load(f)
    duration = gt["trajectory"].get("constant_speed_duration_s", gt["trajectory"]["duration_s"])
    pts = [p for p in gt["trajectory"]["points"] if p["t"] <= duration]
    xy = np.array([[p["x"], p["y"]] for p in pts])
    t = np.array([p["t"] for p in pts])
    return xy, t


def load_sweep_results():
    """Load all available sweep results."""
    results = {}
    for path in sorted(RESULTS_DIR.glob("sweep_*.json")):
        with open(path) as f:
            d = json.load(f)
        sim = d["simulator"]

        # Prefer combined sweep results (multi-trajectory)
        if sim in results and "best_per_trajectory" not in results[sim]:
            continue  # keep combined, skip single
        if (
            sim in results
            and "best_per_trajectory" in results[sim]
            and "best_per_trajectory" not in d
        ):
            continue  # already have combined, skip single

        results[sim] = d
    return results


def get_trajectory(sweep_data, bag_name):
    """Extract trajectory for a specific bag from sweep results."""
    if "best_per_trajectory" in sweep_data:
        per_traj = sweep_data["best_per_trajectory"]
        for key, val in per_traj.items():
            if bag_name in key:
                traj = val.get("trajectory")
                if traj:
                    return np.array(traj)
    return None


def get_dt_and_steps(sweep_data, bag_name):
    """Get dt and number of steps for a trajectory."""
    bp = sweep_data.get("best_params", {})
    fp = sweep_data.get("fixed_params", {})
    dt = bp.get("dt", fp.get("dt", None))

    traj = get_trajectory(sweep_data, bag_name)
    steps = len(traj) if traj is not None else 0

    return dt, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Load ground truths
    gt_turn_xy, gt_turn_t = load_ground_truth("right_turn_b")
    gt_accel_xy, gt_accel_t = load_ground_truth("acceleration")

    # Load sweep results
    sweeps = load_sweep_results()

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.2), gridspec_kw={"width_ratios": [2, 2, 1.9]})
    ax_turn, ax_accel, ax_bar = axes

    # --- Panel 1: Turn XY trajectory (rotated: plot y on horizontal axis) ---
    # Real trajectory extends ~1.2m in x and ~3m in -y; swapping the axes
    # makes the panel naturally landscape-shaped.
    ax_turn.plot(
        gt_turn_xy[:, 1], gt_turn_xy[:, 0], "k--", linewidth=1.6, label="Real robot", zorder=10
    )
    # Direction arrows along real trajectory
    for frac in (0.25, 0.5, 0.75):
        idx = int(frac * (len(gt_turn_xy) - 1))
        ax_turn.annotate(
            "",
            xy=(gt_turn_xy[idx + 1, 1], gt_turn_xy[idx + 1, 0]),
            xytext=(gt_turn_xy[idx, 1], gt_turn_xy[idx, 0]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.7,
                            mutation_scale=18),
            zorder=12,
        )

    excluded_from_plot = []  # sims with err > threshold but no trajectory to plot

    for sim_name in SIM_ORDER:
        if sim_name not in sweeps:
            continue
        err = sweeps[sim_name]["best_error"]
        failed = err > ACCURACY_THRESHOLD
        traj = get_trajectory(sweeps[sim_name], "turn")
        if traj is None:
            traj = get_trajectory(sweeps[sim_name], "14_46_18")
        if traj is None:
            if failed:
                excluded_from_plot.append(sim_name)
            continue
        color = SIM_COLORS[sim_name]
        ax_turn.plot(traj[:, 1], traj[:, 0], color=color, linewidth=1.1, label=(r"\textbf{Axion}" if sim_name == "Axion" else sim_name))
        for frac in (0.25, 0.5, 0.75):
            idx = int(frac * (len(traj) - 1))
            ax_turn.annotate(
                "",
                xy=(traj[idx + 1, 1], traj[idx + 1, 0]),
                xytext=(traj[idx, 1], traj[idx, 0]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=9,
            )

    # Add dimmed legend entries for excluded simulators
    for sim_name in excluded_from_plot:
        ax_turn.plot(
            [],
            [],
            color=SIM_COLORS[sim_name],
            linewidth=1.1,
            alpha=0.4,
            label=(r"\textbf{Axion}" if sim_name == "Axion" else sim_name) + " (excluded)",
        )

    ax_turn.set_xlabel("y (m)")
    ax_turn.set_ylabel("x (m)")
    ax_turn.grid(True, alpha=0.3)

    # --- Panel 2: Acceleration x(t) ---
    ax_accel.plot(
        gt_accel_t, gt_accel_xy[:, 0], "k--",
        linewidth=1.6, label="Real robot", zorder=10,
    )

    excluded_accel = []
    for sim_name in SIM_ORDER:
        if sim_name not in sweeps:
            continue
        err = sweeps[sim_name]["best_error"]
        failed = err > ACCURACY_THRESHOLD
        traj = get_trajectory(sweeps[sim_name], "accel")
        if traj is None:
            traj = get_trajectory(sweeps[sim_name], "09_56")
        if traj is None:
            if failed:
                excluded_accel.append(sim_name)
            continue
        color = SIM_COLORS[sim_name]
        bp = sweeps[sim_name].get("best_params", {})
        fp = sweeps[sim_name].get("fixed_params", {})
        dt = bp.get("dt", fp.get("dt", 0.01))
        traj_t = np.arange(len(traj)) * dt
        ax_accel.plot(traj_t, traj[:, 0], color=color, linewidth=1.1, label=(r"\textbf{Axion}" if sim_name == "Axion" else sim_name))

    for sim_name in excluded_accel:
        ax_accel.plot(
            [],
            [],
            color=SIM_COLORS[sim_name],
            linewidth=1.1,
            label=f"\u0336".join(sim_name) + "\u0336" + " (err > threshold)",
        )

    ax_accel.set_xlabel("time (s)")
    ax_accel.set_ylabel("x (m)")
    ax_accel.grid(True, alpha=0.3)

    # --- Panel 3: Error bar chart with dt annotation ---
    sim_names = []
    sim_errors = []
    sim_dts = []
    sim_steps_list = []
    bar_colors = []

    for sim_name in SIM_ORDER:
        if sim_name not in sweeps:
            continue
        err = sweeps[sim_name]["best_error"]
        bp = sweeps[sim_name].get("best_params", {})
        fp = sweeps[sim_name].get("fixed_params", {})
        dt = bp.get("dt", fp.get("dt", "?"))

        # Get total steps (sum across trajectories)
        total_steps = 0
        if "best_per_trajectory" in sweeps[sim_name]:
            for key, val in sweeps[sim_name]["best_per_trajectory"].items():
                traj = val.get("trajectory")
                if traj:
                    total_steps += len(traj)

        sim_names.append(sim_name)
        sim_errors.append(err)
        sim_dts.append(dt)
        sim_steps_list.append(total_steps)
        bar_colors.append(SIM_COLORS[sim_name])

    # Reorder bars per requested display order (best first at the top)
    BAR_ORDER = ["Axion", "MuJoCo", "TinyDiffSim", "Semi-Implicit", "Dojo"]
    rank = {n: i for i, n in enumerate(BAR_ORDER)}
    order = sorted(range(len(sim_names)), key=lambda i: rank.get(sim_names[i], 99),
                   reverse=True)  # reverse: barh draws bottom-up, so first becomes top
    sim_names = [sim_names[i] for i in order]
    sim_errors = [sim_errors[i] for i in order]
    sim_dts = [sim_dts[i] for i in order]
    sim_steps_list = [sim_steps_list[i] for i in order]
    bar_colors = [bar_colors[i] for i in order]

    y_pos = np.arange(len(sim_names))
    # Fail mask: bars over threshold are hatched + grayer
    fail_mask = [e > ACCURACY_THRESHOLD for e in sim_errors]
    # Mark excluded simulators with "(excluded)" suffix
    def _fmt(n, failed):
        suf = " (excluded)" if failed else ""
        return rf"\textbf{{Axion}}{suf}" if n == "Axion" else n + suf
    display_names = [_fmt(n, f) for n, f in zip(sim_names, fail_mask)]
    hatches = ["///" if failed else "" for failed in fail_mask]
    alphas = [0.4 if failed else 1.0 for failed in fail_mask]

    bars = ax_bar.barh(
        y_pos, sim_errors, color=bar_colors, height=0.5, edgecolor="black", linewidth=0.8, zorder=3
    )
    for bar, hatch, alpha in zip(bars, hatches, alphas):
        bar.set_hatch(hatch)
        bar.set_alpha(alpha)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(display_names)
    ax_bar.set_xlabel("Combined $L_2$ error (m)")
    ax_bar.set_title("Accuracy (lower is better)", pad=20)
    ax_bar.grid(True, axis="x", alpha=0.3, zorder=0)
    ax_bar.set_ylim(-0.5, len(sim_names) - 0.5)

    # Threshold label above the top bar
    ax_bar.text(
        ACCURACY_THRESHOLD,
        len(sim_names) - 0.45,
        rf" threshold ({ACCURACY_THRESHOLD}\,m)",
        color="red",
        fontsize=11,
        va="bottom",
        ha="left",
    )

    # Annotate bars with dt and steps
    for bar, err, dt in zip(bars, sim_errors, sim_dts):
        cy = bar.get_y() + bar.get_height() / 2
        label = rf" {err:.2f}  ($\Delta t={dt}$)"
        ax_bar.text(err + 0.08, cy, label, va="center", ha="left", fontsize=11)

    ax_bar.set_xlim(right=max(sim_errors) * 1.9)

    plt.tight_layout()

    # Draw threshold line in segments so it doesn't cross bar annotation text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skip_ranges = []
    for t in ax_bar.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=renderer).transformed(ax_bar.transData.inverted())
        x0, x1 = sorted([bb.x0, bb.x1])
        y0, y1 = sorted([bb.y0, bb.y1])
        if x0 <= ACCURACY_THRESHOLD <= x1:
            pad = 0.05
            skip_ranges.append((y0 - pad, y1 + pad))
    skip_ranges.sort()
    y_lo, y_hi = ax_bar.get_ylim()
    segments = []
    cursor = y_lo
    for s0, s1 in skip_ranges:
        if s0 > cursor:
            segments.append((cursor, min(s0, y_hi)))
        cursor = max(cursor, s1)
        if cursor >= y_hi:
            break
    if cursor < y_hi:
        segments.append((cursor, y_hi))
    for s0, s1 in segments:
        ax_bar.plot(
            [ACCURACY_THRESHOLD, ACCURACY_THRESHOLD],
            [s0, s1],
            color="red",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
    # Shared legend under the two trajectory panels (spans ~left 2/3 of figure)
    handles, labels = ax_turn.get_legend_handles_labels()
    # Get the midpoint x-coord of the first two panels in figure coords
    bbox_turn = ax_turn.get_position()
    bbox_accel = ax_accel.get_position()
    x_center = (bbox_turn.x0 + bbox_accel.x1) / 2.0
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(x_center, 0.04),
        ncol=len(labels),
        fontsize=13,
        frameon=False,
    )
    plt.subplots_adjust(bottom=0.22)

    save_arg = args.save or "sim_to_real.png"
    save_path = RESULTS_DIR / save_arg if "/" not in save_arg else pathlib.Path(save_arg)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "sim_to_real.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
