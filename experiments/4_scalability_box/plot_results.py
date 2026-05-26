"""Scalability of helhest_junior box optimization vs number of worlds.

Two-panel figure: median time per iteration + peak GPU memory, log-log,
across Axion / MJX / Semi-Implicit. Same style as
experiments/4_scalability/plot_results.py (paper LaTeX serif).

Reads results/<engine>_<N>.json files (axion_<N>.json, mjx_grad_<N>.json,
semi_implicit_<N>.json) and only powers-of-two N for a clean log-scale grid.

Usage:
    python experiments/4_scalability_box/plot_results.py
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import blended_transform_factory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

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
    "Axion": {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MJX-grad": {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 3},
}
LABELS = {
    "Axion": r"\textbf{Axion}",
    "MJX-grad": "MJX",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = list(STYLES.keys())

GPU_MEM_LIMIT_MB = 24 * 1024  # 24 GB (RTX 3090 on dasenka)

# -----------------------------------------------------------------------------
# Manual slope-label positions for the memory panel. Edit these to taste —
# no other plot code needs to change. Coordinates are in data space (x =
# num_worlds, y = MB). Rotation is in degrees, anchored at (x, y).
# Auto-defaults (geometric midpoint of the dashed extrapolation, rotation
# matching the visual line slope) are used for any engine NOT listed here.
# Set ``"text"`` to override the auto-formatted "X.X GB/world" label.
# -----------------------------------------------------------------------------
SLOPE_LABEL_POSITIONS: dict[str, dict] = {
    "MJX-grad": {"x": 3.6, "y": 9000, "rotation": 57},
    "Semi-Implicit": {"x": 340, "y": 4000, "rotation": 55},
    "Axion": {"x": 5000, "y": 3300, "rotation": 55},
}
HIDE_SLOPE_LABEL: set[str] = set()  # add engine names here to suppress label


def load_results() -> dict[str, dict[int, dict]]:
    """Returns {simulator: {num_worlds: data}}."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        sim = data.get("simulator")
        nw = data.get("num_worlds")
        if sim is None or nw is None or sim not in STYLES:
            continue
        if nw < 1 or (nw & (nw - 1)) != 0:
            continue  # powers of two only
        out.setdefault(sim, {})[nw] = data
    return out


def _segment_display_angle_deg(ax, x0, y0, x1, y1):
    """Angle in degrees of segment (x0,y0)->(x1,y1) as it appears on screen,
    accounting for log scales and axes aspect ratio."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


def _add_oom_extrapolation(ax_mem, sim_data, sim_name, slope_fit_min_n=None):
    """Linear-fit memory growth, extend a dashed line along the fitted slope
    up to the observed OOM N (= 2x last successful), and place an X marker
    where the line ends (or at the 24 GB limit, whichever is lower).

    Slope label is rotated to match the actual displayed slope of the
    dashed extrapolation line (computed in display coordinates so it
    accounts for log scales and the axes aspect ratio).
    """
    if sim_name not in sim_data:
        return None
    d = sim_data[sim_name]
    mems = np.array([m for m in d["mems"] if m is not None])
    mem_worlds = np.array([w for w, m in zip(d["worlds"], d["mems"]) if m is not None])
    if len(mem_worlds) < 2:
        return None
    if slope_fit_min_n is not None:
        fit_mask = mem_worlds >= slope_fit_min_n
        if fit_mask.sum() < 2:
            fit_mask = slice(None)
        coeffs = np.polyfit(mem_worlds[fit_mask], mems[fit_mask], 1)
    else:
        coeffs = np.polyfit(mem_worlds, mems, 1)
    slope_mb = float(coeffs[0])
    color = STYLES[sim_name]["color"]

    oom_w = int(mem_worlds[-1]) * 2  # observed OOM = next power of 2
    extrap_x = np.array([mem_worlds[-1], oom_w])
    extrap_y = np.polyval(coeffs, extrap_x)
    ax_mem.plot(extrap_x, extrap_y, color=color, linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)
    # X at observed OOM, clipped to GPU limit so it always sits on or below
    # the limit line.
    oom_y = float(min(extrap_y[-1], GPU_MEM_LIMIT_MB))
    ax_mem.plot(oom_w, oom_y, "x", color="red", markersize=9, markeredgewidth=2.0, zorder=6)

    # Slope label. Default: at geometric-mean midpoint of the dashed
    # extrapolation, rotated to match the visual slope. Override per-engine
    # via SLOPE_LABEL_POSITIONS at top of file.
    if sim_name in HIDE_SLOPE_LABEL:
        return oom_w
    if slope_mb >= 1024:
        auto_label = rf"${slope_mb / 1024:.1f}$\,GB/world"
    elif slope_mb >= 10:
        auto_label = rf"${slope_mb:.0f}$\,MB/world"
    else:
        auto_label = rf"${slope_mb:.1f}$\,MB/world"
    auto_x = float(np.sqrt(mem_worlds[-1] * oom_w))
    auto_y = float(np.polyval(coeffs, auto_x))
    auto_rot = _segment_display_angle_deg(ax_mem, mem_worlds[-1], mems[-1], oom_w, extrap_y[-1])
    cfg = SLOPE_LABEL_POSITIONS.get(sim_name, {})
    ax_mem.text(
        cfg.get("x", auto_x),
        cfg.get("y", auto_y),
        cfg.get("text", auto_label),
        fontsize=cfg.get("fontsize", 8),
        color=color,
        alpha=0.95,
        ha=cfg.get("ha", "center"),
        va=cfg.get("va", "bottom"),
        rotation=cfg.get("rotation", auto_rot),
        rotation_mode="anchor",
        zorder=7,
    )
    return oom_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    all_results = load_results()
    if not all_results:
        print("No results found. Run run_sweep.sh first.")
        return

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.subplots_adjust(wspace=0.30)

    sim_data = {}
    for sim in SIM_ORDER:
        if sim not in all_results:
            continue
        by_worlds = all_results[sim]
        worlds = sorted(by_worlds.keys())
        style = STYLES[sim]
        times = [by_worlds[n]["median_time_ms"] for n in worlds]
        mems_raw = [
            by_worlds[n].get("peak_gpu_mb_nvml_absolute") or by_worlds[n].get("peak_gpu_mb")
            for n in worlds
        ]
        sim_data[sim] = {"worlds": worlds, "times": times, "mems": mems_raw}

        ax_time.plot(
            worlds,
            times,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=5,
            label=LABELS[sim],
            zorder=style["zorder"],
        )

        mem_worlds = [w for w, m in zip(worlds, mems_raw) if m is not None]
        mems = [m for m in mems_raw if m is not None]
        if mems:
            ax_mem.plot(
                mem_worlds,
                mems,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["lw"],
                markersize=5,
                label=LABELS[sim],
                zorder=style["zorder"],
            )

    # 24 GB GPU limit line + OOM shading
    ax_mem.axhline(GPU_MEM_LIMIT_MB, color="red", linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)

    # OOM extrapolation + memory-panel markers per engine. Axion's memory
    # curve is flat below ~512 worlds (GPU underutilised), so we fit slope
    # only on N>=512 — otherwise the flat-region pre-knee points pull the
    # per-world slope down by ~50x.
    # Slope labels are placed *along* each engine's dashed extrapolation,
    # rotated to match the displayed slope (see _segment_display_angle_deg).
    oom_info = {}
    oom_info["MJX-grad"] = _add_oom_extrapolation(ax_mem, sim_data, "MJX-grad")
    oom_info["Semi-Implicit"] = _add_oom_extrapolation(ax_mem, sim_data, "Semi-Implicit")
    oom_info["Axion"] = _add_oom_extrapolation(ax_mem, sim_data, "Axion", slope_fit_min_n=512)

    # Time-panel OOM markers: place X at the y-value where each engine's
    # time curve last had data, then extend with a short dashed segment to
    # the OOM x-coord (so the X visually continues the line).
    for sim_name, oom_w in oom_info.items():
        if oom_w is None or sim_name not in sim_data:
            continue
        d = sim_data[sim_name]
        worlds = d["worlds"]
        times = d["times"]
        color = STYLES[sim_name]["color"]
        last_w = worlds[-1]
        last_t = times[-1]
        # Linear extrapolation of time-curve slope in log-log to the OOM N.
        if len(worlds) >= 2:
            log_slope = (np.log(times[-1]) - np.log(times[-2])) / (
                np.log(worlds[-1]) - np.log(worlds[-2])
            )
            log_t_oom = np.log(last_t) + log_slope * (np.log(oom_w) - np.log(last_w))
            t_oom = np.exp(log_t_oom)
        else:
            t_oom = last_t
        ax_time.plot(
            [last_w, oom_w],
            [last_t, t_oom],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            zorder=2,
        )
        ax_time.plot(oom_w, t_oom, "x", color="red", markersize=9, markeredgewidth=2.0, zorder=6)

    for ax in (ax_time, ax_mem):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of worlds")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 5), numticks=12))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="major", alpha=0.35, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)

    ax_time.set_ylabel("Median time per iteration (ms)")
    ax_mem.set_ylabel("Peak GPU memory (MB)")
    # 24 GB limit text — inside the shaded OOM band so it doesn't conflict
    # with the MJX (red) slope label at top-left.
    ax_mem.text(
        0.5,
        GPU_MEM_LIMIT_MB * 1.6,
        r"24\,GB GPU limit",
        fontsize=8,
        color="red",
        alpha=0.85,
        ha="center",
        va="bottom",
        transform=blended_transform_factory(ax_mem.transAxes, ax_mem.transData),
    )
    ax_mem.axhspan(GPU_MEM_LIMIT_MB, ax_mem.get_ylim()[1] * 2, color="red", alpha=0.05, zorder=0)

    ax_time.plot([], [], "x", color="red", markersize=9, markeredgewidth=2.0, label="Out of memory")
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        columnspacing=1.8,
    )

    out = RESULTS_DIR / "scalability_box.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")
    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "scalability_box.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
