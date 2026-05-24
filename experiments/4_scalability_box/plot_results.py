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

plt.rcParams.update({
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
})

STYLES = {
    "Axion":         {"color": "#2196F3", "marker": "o", "lw": 2.0, "zorder": 5},
    "MJX-grad":      {"color": "#E91E63", "marker": "s", "lw": 1.8, "zorder": 4},
    "Semi-Implicit": {"color": "#FF9800", "marker": "^", "lw": 1.8, "zorder": 3},
}
LABELS = {
    "Axion":         r"\textbf{Axion}",
    "MJX-grad":      "MJX-grad",
    "Semi-Implicit": "Semi-Impl.",
}
SIM_ORDER = list(STYLES.keys())

GPU_MEM_LIMIT_MB = 24 * 1024  # 24 GB (RTX 3090 on dasenka)


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


def _add_oom_extrapolation(ax_mem, sim_data, sim_name, ann_y_factor=7.0,
                            ann_x_factor=6.0, ann_rotation=52, slope_fit_min_n=None):
    """Linear-fit memory growth, draw dashed extrapolation + OOM x marker.

    ``slope_fit_min_n``: if set, only use data points with N >= this value
    for the linear fit. Useful for engines like Axion whose memory is flat
    until a GPU-saturation knee — fitting through the flat region would
    underestimate the per-world slope.
    """
    if sim_name not in sim_data:
        return
    d = sim_data[sim_name]
    mems = np.array([m for m in d["mems"] if m is not None])
    mem_worlds = np.array([w for w, m in zip(d["worlds"], d["mems"]) if m is not None])
    if len(mem_worlds) < 2:
        return
    if slope_fit_min_n is not None:
        fit_mask = mem_worlds >= slope_fit_min_n
        if fit_mask.sum() < 2:
            fit_mask = slice(None)
        coeffs = np.polyfit(mem_worlds[fit_mask], mems[fit_mask], 1)
    else:
        coeffs = np.polyfit(mem_worlds, mems, 1)
    slope_gb = coeffs[0] / 1024
    color = STYLES[sim_name]["color"]
    mid_idx = len(mem_worlds) // 2
    ax_mem.text(mem_worlds[mid_idx] * ann_x_factor,
                mems[mid_idx] * ann_y_factor,
                rf"$\sim{slope_gb:.2f}$\,GB/world",
                fontsize=7, color=color, alpha=0.8,
                rotation=ann_rotation, rotation_mode="anchor")
    extrap_worlds = np.array([mem_worlds[-1], 32, 64, 128, 256, 512, 1024,
                               2048, 4096, 8192, 16384, 32768])
    extrap_worlds = extrap_worlds[extrap_worlds > mem_worlds[-1]]
    if len(extrap_worlds) > 0:
        extrap_worlds = np.concatenate([[mem_worlds[-1]], extrap_worlds])
        extrap_mems = np.polyval(coeffs, extrap_worlds)
        ax_mem.plot(extrap_worlds, extrap_mems, color=color,
                    linestyle="--", linewidth=1.2, alpha=0.5, zorder=2)
        oom_w = int(mem_worlds[-1]) * 2
        ax_mem.plot(oom_w, GPU_MEM_LIMIT_MB, "x",
                    color="red", markersize=8, markeredgewidth=2.0, zorder=6)
        return oom_w
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    all_results = load_results()
    if not all_results:
        print("No results found. Run run_sweep.sh first.")
        return

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.35)

    sim_data = {}
    for sim in SIM_ORDER:
        if sim not in all_results:
            continue
        by_worlds = all_results[sim]
        worlds = sorted(by_worlds.keys())
        style = STYLES[sim]
        times = [by_worlds[n]["median_time_ms"] for n in worlds]
        mems_raw = [by_worlds[n].get("peak_gpu_mb_nvml_absolute")
                    or by_worlds[n].get("peak_gpu_mb")
                    for n in worlds]
        sim_data[sim] = {"worlds": worlds, "times": times, "mems": mems_raw}

        ax_time.plot(worlds, times, color=style["color"], marker=style["marker"],
                     linewidth=style["lw"], markersize=5,
                     label=LABELS[sim], zorder=style["zorder"])

        mem_worlds = [w for w, m in zip(worlds, mems_raw) if m is not None]
        mems = [m for m in mems_raw if m is not None]
        if mems:
            ax_mem.plot(mem_worlds, mems, color=style["color"], marker=style["marker"],
                        linewidth=style["lw"], markersize=5,
                        label=LABELS[sim], zorder=style["zorder"])

    # 24 GB GPU limit + shading
    ax_mem.axhline(GPU_MEM_LIMIT_MB, color="red", linestyle="-", linewidth=0.8,
                   alpha=0.7, zorder=1)
    ax_mem.text(0.99, GPU_MEM_LIMIT_MB * 1.3, r"24\,GB GPU limit",
                fontsize=8, color="red", alpha=0.8, ha="right",
                transform=blended_transform_factory(ax_mem.transAxes, ax_mem.transData))

    # OOM extrapolation + markers per failing engine (different annotation
    # placements to avoid label overlap). Axion's memory curve is flat below
    # ~512 worlds (GPU underutilised), so we fit the slope on N>=512 only —
    # otherwise the flat-region pre-knee points pull the per-world slope down
    # by ~50x. Confirmed wall: Axion OOM at 16384 on a 24 GB 3090.
    oom_mjx = _add_oom_extrapolation(ax_mem, sim_data, "MJX-grad",
                                      ann_x_factor=6.0, ann_y_factor=7.0,
                                      ann_rotation=52)
    oom_si = _add_oom_extrapolation(ax_mem, sim_data, "Semi-Implicit",
                                     ann_x_factor=3.0, ann_y_factor=2.5,
                                     ann_rotation=40)
    oom_axion = _add_oom_extrapolation(ax_mem, sim_data, "Axion",
                                        ann_x_factor=0.18, ann_y_factor=4.0,
                                        ann_rotation=52, slope_fit_min_n=512)

    # OOM markers on time panel too (same world count as memory failure)
    for sim_name, oom_w in [("MJX-grad", oom_mjx), ("Semi-Implicit", oom_si),
                            ("Axion", oom_axion)]:
        if oom_w is None or sim_name not in sim_data:
            continue
        median_t = np.median(sim_data[sim_name]["times"])
        ax_time.plot(oom_w, median_t, "x", color="red",
                     markersize=8, markeredgewidth=2.0, zorder=6)

    # Axion-vs-MJX speedup double-arrow at the smallest common world count
    if "Axion" in sim_data and "MJX-grad" in sim_data:
        ax_d = sim_data["Axion"]; mx_d = sim_data["MJX-grad"]
        common = sorted(set(ax_d["worlds"]) & set(mx_d["worlds"]))
        if common:
            w = common[0]
            ax_t = ax_d["times"][ax_d["worlds"].index(w)]
            mx_t = mx_d["times"][mx_d["worlds"].index(w)]
            ratio = mx_t / ax_t
            ax_time.annotate("", xy=(w, mx_t), xytext=(w, ax_t),
                             arrowprops=dict(arrowstyle="<->", color="0.3",
                                             lw=1.2, shrinkA=3, shrinkB=3))
            mid_y = np.sqrt(ax_t * mx_t)
            ax_time.text(w * 1.6, mid_y, rf"${ratio:.0f}\times$",
                         fontsize=10, color="0.3", fontweight="bold",
                         va="center", ha="left")

    # Axion knee (GPU saturated)
    _knee_w = None
    if "Axion" in sim_data:
        ax_d = sim_data["Axion"]
        # Heuristic knee: world count where time/world starts growing.
        # Pick 512 (matches flat exp 4) if present, else the inflection.
        for candidate in (512, 256, 1024):
            if candidate in ax_d["worlds"]:
                _knee_w = candidate
                break
        if _knee_w is not None:
            for ax in (ax_time, ax_mem):
                ax.axvline(_knee_w, color="#2196F3", linestyle="--",
                           linewidth=1.2, alpha=0.75)

    for ax in (ax_time, ax_mem):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Number of worlds")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.35, linewidth=0.6)

    ax_time.set_ylabel("Median time per iteration (ms)")
    ax_mem.set_ylabel("Peak GPU memory (MB)")
    ax_mem.axhspan(GPU_MEM_LIMIT_MB, ax_mem.get_ylim()[1], color="red",
                   alpha=0.05, zorder=0)

    if _knee_w is not None:
        for ax in (ax_time, ax_mem):
            ax.text(_knee_w - 100, 0.13, "GPU saturated",
                    rotation=90, fontsize=8, color="#2196F3",
                    ha="right", va="bottom",
                    transform=blended_transform_factory(ax.transData, ax.transAxes))

    ax_time.plot([], [], "x", color="red", markersize=8, markeredgewidth=2.0,
                 label="Out of Memory")
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.22), frameon=False, columnspacing=1.5)

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
