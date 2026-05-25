import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "benchmarks/data/hybrid"

SCENARIO_FILES = {
    "fall": DATA_DIR / "fall_hybrid_benchmark.h5",
    "pendulum": DATA_DIR / "pendulum_hybrid_benchmark_larger_data.h5",
}
PLOT_FILES = {
    "fall": DATA_DIR / "fall_hybrid_convergence.pdf",
    "pendulum": DATA_DIR / "pendulum_hybrid_convergence.pdf",
}

HYBRID_KEYS = ["res", "mae"]
HYBRID_NAMES = {"mae": "Hybrid (MAE)", "res": "Hybrid (Res)"}
COLORS = {
    "axion": "tab:gray",
    "mae": "tab:blue",
    "res": "tab:orange",
}


def load_data(filepath: pathlib.Path) -> tuple[dict, dict, int, list]:
    iter_lists = {"axion": []}
    res_lists = {"axion": []}

    with h5py.File(filepath, "r") as f:
        n_passes = int(f.attrs["n_passes"])
        max_nr_iters = int(f.attrs.get("max_nr_iters", 16))

        pass0 = f["pass_0"]
        present_keys = [k for k in HYBRID_KEYS if f"hybrid_{k}_iters" in pass0]
        for k in present_keys:
            iter_lists[k] = []
            res_lists[k] = []

        for i in range(n_passes):
            grp = f[f"pass_{i}"]
            iter_lists["axion"].append(grp["axion_iters"][:])
            res_lists["axion"].append(grp["axion_residuals"][:])
            for k in present_keys:
                iter_lists[k].append(grp[f"hybrid_{k}_iters"][:])
                res_lists[k].append(grp[f"hybrid_{k}_residuals"][:])

    flat_iters = {label: np.concatenate(arrs) for label, arrs in iter_lists.items()}
    flat_res = {label: np.concatenate(arrs, axis=0) for label, arrs in res_lists.items()}
    return flat_iters, flat_res, max_nr_iters, present_keys


def convergence_probability(iter_counts: np.ndarray, max_iters: int) -> np.ndarray:
    return np.array([np.mean(iter_counts <= k) for k in range(1, max_iters + 1)])


def print_summary(flat_iters: dict, present_keys: list) -> None:
    max_iters = max(v.max() for v in flat_iters.values())
    print(
        f"\n  {'Engine':<16}  {'Median':>8}  {'Mean':>8}  {'Max':>6}  "
        f"{'% at max':>10}  {'% <= 8':>10}"
    )
    print("  " + "-" * 62)
    for label in ["axion"] + present_keys:
        arr = flat_iters[label]
        name = "Newton" if label == "axion" else HYBRID_NAMES.get(label, label)
        pct_max = 100.0 * np.mean(arr == max_iters)
        pct_le8 = 100.0 * np.mean(arr <= 8)
        print(
            f"  {name:<16}  {np.median(arr):>8.1f}  {np.mean(arr):>8.2f}  "
            f"{arr.max():>6d}  {pct_max:>9.1f}%  {pct_le8:>9.1f}%"
        )
    print()


class LineIQRHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        color = orig_handle.get_color()
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=color,
            alpha=0.3,
            edgecolor="none",
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + height * 0.5, ydescent + height * 0.5],
            color=color,
            linewidth=2.0,
            solid_capstyle="butt",
            transform=trans,
        )
        return [rect, line]


def _plot_aggregated(ax, res_matrix: np.ndarray, color: str, label: str):
    k_vals = np.arange(1, res_matrix.shape[1] + 1)
    median = np.nanmedian(res_matrix, axis=0)
    p25 = np.nanpercentile(res_matrix, 25, axis=0)
    p75 = np.nanpercentile(res_matrix, 75, axis=0)
    (line,) = ax.plot(k_vals, median, color=color, linewidth=2.0, label=label)
    ax.fill_between(k_vals, p25, p75, color=color, alpha=0.2)
    return line


def plot_scenario(
    flat_iters: dict,
    flat_res: dict,
    max_iters: int,
    present_keys: list,
    scenario: str,
    out_path: pathlib.Path,
    show_steps: int | None = None,
) -> None:
    labels = ["axion"] + present_keys
    n_show = min(show_steps, max_iters) if show_steps is not None else max_iters
    k_vals = np.arange(1, n_show + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    agg_lines = []
    conv_lines = []

    for label in labels:
        name = "Newton" if label == "axion" else HYBRID_NAMES.get(label, label)
        color = COLORS.get(label, "tab:green")
        res = flat_res[label]

        line = _plot_aggregated(axes[0], res[:, :n_show], color=color, label=name)
        agg_lines.append(line)

        prob = convergence_probability(flat_iters[label], max_iters)
        (cline,) = axes[1].plot(
            k_vals, prob[:n_show] * 100.0, color=color, linewidth=2.0, label=name
        )
        conv_lines.append(cline)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Newton Step [-]", fontsize=14)
    axes[0].set_ylabel("Residue Norm [-]", fontsize=14)
    axes[0].set_xticks(k_vals)
    axes[0].legend(handles=agg_lines, handler_map={Line2D: LineIQRHandler()}, fontsize=14)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].tick_params(axis="both", which="minor", labelsize=14)

    axes[1].set_xlabel("Newton Step [-]", fontsize=14)
    axes[1].set_ylabel("Steps converged within k iterations (%)", fontsize=14)
    axes[1].set_xlim(1, n_show)
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(k_vals)
    axes[1].legend(handles=conv_lines, fontsize=14)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].tick_params(axis="both", which="major", labelsize=14)
    axes[1].tick_params(axis="both", which="minor", labelsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid NR convergence analysis")
    parser.add_argument("--scenario", choices=["fall", "pendulum", "both"], default="both")
    parser.add_argument(
        "--show-steps",
        type=int,
        default=None,
        metavar="N",
        help="truncate plots to the first N Newton steps (default: show all)",
    )
    args = parser.parse_args()

    scenarios = (
        list(SCENARIO_FILES.items())
        if args.scenario == "both"
        else [(args.scenario, SCENARIO_FILES[args.scenario])]
    )

    for name, hdf5_path in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {name.upper()}")

        if not hdf5_path.exists():
            print(f"  HDF5 not found: {hdf5_path}")
            print(f"  Run gnn_benchmark_hybrid.py --scenario {name} first.")
            continue

        print(f"  Loading {hdf5_path}…")
        flat_iters, flat_res, max_iters, present_keys = load_data(hdf5_path)
        print(f"  {len(flat_iters['axion'])} simulation steps across all passes")
        print_summary(flat_iters, present_keys)
        plot_scenario(
            flat_iters,
            flat_res,
            max_iters,
            present_keys,
            name,
            PLOT_FILES[name],
            show_steps=args.show_steps,
        )


if __name__ == "__main__":
    main()
