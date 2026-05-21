"""Plot control stability results.

Two modes:
  threshold  — horizontal bar chart of max stable Δt per simulator
               (from binary_search results)
  trajectory — 2-panel figure: joint angle vs time + gain sweep

Usage:
    python examples/comparison/control_stability/plot_results.py
    python examples/comparison/control_stability/plot_results.py --mode threshold
    python examples/comparison/control_stability/plot_results.py --mode trajectory --show
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.transforms import blended_transform_factory
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# ─── threshold mode style ────────────────────────────────────────────────────

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

THRESHOLD_STYLES = {
    "Semi-Implicit": {"color": "#607D8B"},
    "Genesis":       {"color": "#4CAF50"},
    "MJX":           {"color": "#FF5722"},
    "MuJoCo":        {"color": "#E91E63"},
    "TinyDiffSim":   {"color": "#9C27B0"},
    "Dojo":          {"color": "#FF9800"},
    "Axion":         {"color": "#2196F3"},
}
THRESHOLD_SIM_ORDER = list(THRESHOLD_STYLES.keys())

THRESHOLD_LABELS = {
    "Semi-Implicit": "Semi-Implicit",
    "Genesis":       "Genesis",
    "MJX":           "MJX",
    "MuJoCo":        "MuJoCo",
    "TinyDiffSim":   "TinyDiffSim",
    "Dojo":          "Dojo",
    "Axion":         r"\textbf{Axion}",
}

AXION_COLOR = "#2196F3"

_NAME_MAP = {
    "Axion-Implicit": "Axion",
    "Axion":          "Axion",
    "MuJoCo":         "MuJoCo",
    "MJX":            "MJX",
    "Genesis":        "Genesis",
    "Semi-Implicit":  "Semi-Implicit",
    "TinyDiffSim":    "TinyDiffSim",
    "Dojo":           "Dojo",
}
_FILE_MAP = {
    "axion":          "Axion",
    "mujoco":         "MuJoCo",
    "mjx":            "MJX",
    "genesis":        "Genesis",
    "semi_implicit":  "Semi-Implicit",
    "tinydiffsim":    "TinyDiffSim",
    "dojo":           "Dojo",
}

# ─── trajectory mode style ───────────────────────────────────────────────────

TRAJ_STYLES = {
    "Axion-Implicit": {"color": "#2196F3", "marker": "o", "lw": 2.5, "zorder": 5},
    "MuJoCo":         {"color": "#E91E63", "marker": "x", "lw": 1.8, "zorder": 3},
}
TRAJ_SIM_ORDER = list(TRAJ_STYLES.keys())

_TRAJ_FILE_MAP = {
    "axion_implicit_dt":   ("Axion-Implicit", "dt_sweep"),
    "axion_implicit_gain": ("Axion-Implicit", "gain_sweep"),
    "mujoco_dt":           ("MuJoCo",         "dt_sweep"),
    "mujoco_gain":         ("MuJoCo",         "gain_sweep"),
}

CLIP_DEG = 200.0


# ─── loaders ─────────────────────────────────────────────────────────────────

def load_thresholds() -> dict:
    out = {}
    for path in sorted(RESULTS_DIR.glob("*_threshold.json")):
        data = json.loads(path.read_text())
        raw = data.get("simulator")
        sim = _NAME_MAP.get(raw)
        if sim is None:
            for frag, key in _FILE_MAP.items():
                if frag in path.stem:
                    sim = key
                    break
        if sim and "max_stable_dt" in data:
            out[sim] = data
    return out


def load_trajectory_results() -> tuple:
    dt_data, gain_data = {}, {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        stem = path.stem
        if stem not in _TRAJ_FILE_MAP:
            continue
        sim, exp = _TRAJ_FILE_MAP[stem]
        data = json.loads(path.read_text())
        if exp == "dt_sweep":
            dt_data[sim] = data
        else:
            gain_data[sim] = data
    return dt_data, gain_data


# ─── threshold plot ───────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    if val < 0.001:
        exp = int(np.floor(np.log10(val)))
        mantissa = val / 10 ** exp
        return rf"{mantissa:.2f} \times 10^{{{exp}}}"
    return f"{val:.3f}"


def plot_threshold(show: bool):
    thresholds = load_thresholds()
    if not thresholds:
        print("No threshold results found. Run each simulator with --experiment binary_search first.")
        return

    if "Axion" not in thresholds:
        print("Axion results not found — cannot compute ×N ratios.")
        return

    sims = [s for s in THRESHOLD_SIM_ORDER if s in thresholds]
    if not sims:
        sims = sorted(thresholds.keys())

    # Sort ascending so the strongest simulator (Axion) is at the top
    sims = sorted(sims, key=lambda s: thresholds[s]["max_stable_dt"])

    axion_dt = thresholds["Axion"]["max_stable_dt"]
    colors   = [THRESHOLD_STYLES.get(s, {"color": "gray"})["color"] for s in sims]
    values   = [thresholds[s]["max_stable_dt"] for s in sims]
    hit_max  = [thresholds[s].get("hit_max", False) for s in sims]
    ylabels  = [THRESHOLD_LABELS.get(s, s) for s in sims]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    y = np.arange(len(sims))
    bars = ax.barh(y, values, color=colors, height=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"Max stable $\Delta t$ (s)")
    ax.grid(True, axis="x", which="both", alpha=0.25, zorder=0, linewidth=0.5)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(-0.6, len(sims) - 0.4)

    xmax = max(values)
    ax.set_xlim(right=xmax * 4)

    right_xfm = blended_transform_factory(ax.transAxes, ax.transData)

    for bar, sim, val, hm in zip(bars, sims, values, hit_max):
        cy = bar.get_y() + bar.get_height() / 2

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

    plt.tight_layout(pad=0.4, rect=(0, 0, 0.76, 1))
    out = RESULTS_DIR / "pendulum_control_threshold.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    if show:
        plt.show()


# ─── trajectory plot ──────────────────────────────────────────────────────────

def plot_timeseries(ax, dt_data: dict, target_dt: float = 0.05):
    for sim in TRAJ_SIM_ORDER:
        if sim not in dt_data:
            continue
        runs = {r["dt"]: r for r in dt_data[sim]["runs"]}
        available = sorted(runs.keys())
        dt = min(available, key=lambda d: abs(d - target_dt))
        run = runs[dt]
        style = TRAJ_STYLES[sim]

        t = np.array(run["time"])
        q = np.array([v if v is not None else float("nan")
                      for v in run["joint_angle"]])
        q_deg = np.degrees(q)
        every = max(1, len(t) // 20)

        ax.plot(t, q_deg,
                color=style["color"], marker=style["marker"],
                markevery=every, markersize=5,
                linewidth=style["lw"], label=sim, zorder=style["zorder"])

        if not run["stable"]:
            idx = next((i for i, v in enumerate(q_deg)
                        if np.isfinite(v) and abs(v) > 28.6), len(q) - 1)
            ann_y = min(max(q_deg[idx], -CLIP_DEG + 20), CLIP_DEG - 20)
            ax.annotate(
                "UNSTABLE\n(clipped)",
                xy=(t[idx], ann_y),
                xytext=(t[idx] + 0.15, ann_y - 40),
                fontsize=7, color=style["color"],
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.2),
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1,
               label="Target $\\theta = 0°$", zorder=1)
    ax.set_ylim(-CLIP_DEG, CLIP_DEG)
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Joint angle (deg)")
    ax.set_title(
        f"Pendulum Control — Angle vs Time"
        f"\n($\\Delta t={target_dt}$ s, $k_p=500$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_gain_sweep(ax, gain_data: dict):
    for sim in TRAJ_SIM_ORDER:
        if sim not in gain_data:
            continue
        style = TRAJ_STYLES[sim]
        kp_vals, max_angles = [], []
        for run in gain_data[sim]["runs"]:
            valid = [abs(a) for a in run["joint_angle"] if a is not None]
            kp_vals.append(run["kp"])
            raw = max(valid) if valid else float("nan")
            max_angles.append(min(raw, np.radians(CLIP_DEG)))

        ax.plot(kp_vals, np.degrees(max_angles),
                color=style["color"], marker=style["marker"],
                markersize=6, linewidth=style["lw"],
                label=sim, zorder=style["zorder"])

    ax.axhline(np.degrees(0.5), color="gray", linestyle=":",
               linewidth=1, label="Instability threshold (0.5 rad)")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0, top=CLIP_DEG)
    ax.set_xlabel("Proportional gain $k_p$ (N·m/rad)")
    ax.set_ylabel("Max $|\\theta|$ reached (deg, clipped)")
    ax.set_title("Gain Sweep — Stability vs $k_p$\n($\\Delta t=0.05$ s)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_trajectory(show: bool):
    dt_data, gain_data = load_trajectory_results()
    if not dt_data and not gain_data:
        print("No control_stability trajectory results found. Run the benchmark scripts first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Control Stability: Axion Implicit Servo vs MuJoCo Explicit PD\n"
        "(single pendulum, 1 kg, 1 m, displaced 60°, $\\Delta t = 0.05$ s)",
        fontsize=12, fontweight="bold",
    )

    if dt_data:
        plot_timeseries(axes[0], dt_data)
    else:
        axes[0].text(0.5, 0.5, "No dt_sweep data", ha="center",
                     va="center", transform=axes[0].transAxes, color="gray")

    if gain_data:
        plot_gain_sweep(axes[1], gain_data)
    else:
        axes[1].text(0.5, 0.5, "No gain_sweep data", ha="center",
                     va="center", transform=axes[1].transAxes, color="gray")

    plt.tight_layout()
    out = RESULTS_DIR / "pendulum_control.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")

    if show:
        plt.show()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trajectory", "threshold"], default="threshold")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.mode == "threshold":
        plot_threshold(args.show)
    else:
        plot_trajectory(args.show)


if __name__ == "__main__":
    main()
