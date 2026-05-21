"""Plot scalability of Helhest optimization: time and memory vs number of worlds.

Usage:
    python examples/comparison/helhest_scalability/plot_results.py
    python examples/comparison/helhest_scalability/plot_results.py --show
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.transforms import blended_transform_factory

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[3] / ".." / "axion_paper" / "figures"

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
    "MJX-grad": {"color": "#E91E63", "marker": "o", "lw": 1.8, "zorder": 3},
}
LABELS = {
    "Axion": r"\textbf{Axion}",
    "MJX-grad": "MJX-grad",
}
SIM_ORDER = list(STYLES.keys())

GPU_MEM_LIMIT_MB = 24 * 1024  # 24 GB


def load_results() -> dict[str, dict[int, dict]]:
    """Returns {simulator: {num_worlds: data}}."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("scalability_"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        sim = data.get("simulator")
        nw = data.get("num_worlds")
        if sim is None or nw is None:
            continue
        if sim not in STYLES:
            continue
        # Keep only powers of two for a clean log-scale sweep.
        if nw < 1 or (nw & (nw - 1)) != 0:
            continue
        out.setdefault(sim, {})[nw] = data
    return out


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

    # Collect data for annotations
    sim_data = {}
    for sim in SIM_ORDER:
        if sim not in all_results:
            continue
        data_by_worlds = all_results[sim]
        worlds = sorted(data_by_worlds.keys())
        style = STYLES[sim]
        label = LABELS[sim]

        times = [data_by_worlds[n]["median_time_ms"] for n in worlds]
        # Prefer the NVML-measured total GPU memory (includes JAX pool,
        # activations, fragmentation) when available; fall back to the
        # JAX-reported single-tensor peak for older runs.
        mems_raw = [
            data_by_worlds[n].get("peak_gpu_mb_nvml_absolute")
            or data_by_worlds[n].get("peak_gpu_mb")
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
            label=label,
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
                label=label,
                zorder=style["zorder"],
            )

    # --- Memory panel: GPU limit line + OOM shading + MJX extrapolation ---
    ax_mem.axhline(GPU_MEM_LIMIT_MB, color="red", linestyle="-", linewidth=0.8, alpha=0.7, zorder=1)
    ax_mem.text(
        0.99,
        GPU_MEM_LIMIT_MB * 1.3,
        r"24\,GB GPU limit",
        fontsize=8,
        color="red",
        alpha=0.8,
        ha="right",
        transform=blended_transform_factory(ax_mem.transAxes, ax_mem.transData),
    )

    if "MJX-grad" in sim_data:
        mjx = sim_data["MJX-grad"]
        mjx_worlds = np.array(mjx["worlds"])
        mjx_mems = np.array([m for m in mjx["mems"] if m is not None])
        mjx_mem_worlds = np.array([w for w, m in zip(mjx["worlds"], mjx["mems"]) if m is not None])

        # Linear extrapolation of MJX memory
        if len(mjx_mem_worlds) >= 2:
            coeffs = np.polyfit(mjx_mem_worlds, mjx_mems, 1)
            slope_gb = coeffs[0] / 1024  # MB/world -> GB/world

            # Annotate the slope along the MJX line
            mid_idx = len(mjx_mem_worlds) // 2
            mid_w = mjx_mem_worlds[mid_idx]
            mid_m = mjx_mems[mid_idx]
            ax_mem.text(
                mid_w * 6.0,
                mid_m * 7.0,
                rf"$\sim{slope_gb:.1f}$\,GB/world",
                fontsize=7,
                color="#E91E63",
                alpha=0.8,
                rotation=52,
                rotation_mode="anchor",
            )
            extrap_worlds = np.array([mjx_mem_worlds[-1], 32, 64, 128, 256, 512, 1024])
            extrap_worlds = extrap_worlds[extrap_worlds > mjx_mem_worlds[-1]]
            if len(extrap_worlds) > 0:
                extrap_worlds = np.concatenate([[mjx_mem_worlds[-1]], extrap_worlds])
                extrap_mems = np.polyval(coeffs, extrap_worlds)
                ax_mem.plot(
                    extrap_worlds,
                    extrap_mems,
                    color="#E91E63",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.5,
                    zorder=2,
                )

                # OOM marker at first observed OOM world (2x last successful).
                # Place at the 24 GB limit line: the failure is total-memory
                # (activations + tensors + buffers) exceeding the device, not
                # just the single largest tensor reported by peak_gpu_mb.
                oom_w = int(mjx_mem_worlds[-1]) * 2
                ax_mem.plot(
                    oom_w,
                    GPU_MEM_LIMIT_MB,
                    "x",
                    color="red",
                    markersize=8,
                    markeredgewidth=2.0,
                    zorder=6,
                )

        # Single OOM marker on time plot at first OOM world
        if len(mjx_worlds) >= 2:
            median_mjx_time = np.median(mjx["times"])
            # First world count where MJX actually OOM'd (observed).
            oom_world = int(mjx_mem_worlds[-1]) * 2
            ax_time.plot(
                oom_world,
                median_mjx_time,
                "x",
                color="red",
                markersize=8,
                markeredgewidth=2.0,
                zorder=6,
            )

    # --- Speedup annotations: double-headed arrow ---
    if "Axion" in sim_data and "MJX-grad" in sim_data:
        axion = sim_data["Axion"]
        mjx = sim_data["MJX-grad"]

        common = sorted(set(axion["worlds"]) & set(mjx["worlds"]))
        if common:
            w = common[0]
            ax_t = axion["times"][axion["worlds"].index(w)]
            mx_t = mjx["times"][mjx["worlds"].index(w)]
            ratio = mx_t / ax_t

            # Double-headed arrow between the two points
            ax_time.annotate(
                "",
                xy=(w, mx_t),
                xytext=(w, ax_t),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2, shrinkA=3, shrinkB=3),
            )
            # Label at geometric midpoint
            mid_y = np.sqrt(ax_t * mx_t)
            ax_time.text(
                w * 1.6,
                mid_y,
                rf"${ratio:.0f}\times$",
                fontsize=10,
                color="0.3",
                fontweight="bold",
                va="center",
                ha="left",
            )

    # --- Knee point annotation ---
    if "Axion" in sim_data:
        axion = sim_data["Axion"]
        # Knee is around 256-512 worlds where per-world cost starts growing
        knee_w = 512
        if knee_w in axion["worlds"]:
            knee_idx = axion["worlds"].index(knee_w)
            knee_t = axion["times"][knee_idx]
            knee_m = axion["mems"][knee_idx]

            for ax, val in [(ax_time, knee_t), (ax_mem, knee_m)]:
                ax.axvline(knee_w, color="#2196F3", linestyle="--", linewidth=1.2, alpha=0.75)

            _knee_w = knee_w  # save for later placement

    for ax in (ax_time, ax_mem):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of worlds")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.35, linewidth=0.6)

    ax_time.set_ylabel("Median time per iteration (ms)")
    ax_mem.set_ylabel("Peak GPU memory (MB)")

    # Shade OOM region on memory plot
    ax_mem.axhspan(GPU_MEM_LIMIT_MB, ax_mem.get_ylim()[1], color="red", alpha=0.05, zorder=0)

    # Place "GPU saturated" text next to the dashed line
    if "Axion" in sim_data and "_knee_w" in dir():
        for ax in (ax_time, ax_mem):
            ax.text(
                _knee_w - 100,
                0.13,
                "GPU saturated",
                rotation=90,
                fontsize=8,
                color="#2196F3",
                ha="right",
                va="bottom",
                transform=blended_transform_factory(ax.transData, ax.transAxes),
            )

    # Add OOM marker to legend
    oom_handle = ax_time.plot(
        [], [], "x", color="red", markersize=8, markeredgewidth=2.0, label="Out of Memory"
    )[0]
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        columnspacing=1.5,
    )

    out = RESULTS_DIR / "scalability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved to {out}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "scalability_helhest.png"
        plt.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
