"""Generate the paper figure for terrain traversal: 3-panel layout.

Left: Example seed xy trajectory progression
Middle: RMSE convergence envelope across all seeds
Right: Best RMSE histogram

Usage:
    python -m examples.terrain_traversal.plot_paper \
        --batch-dir results/terrain_batch --seed 11
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[2] / ".." / "axion_paper" / "figures"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AXION_COLOR = "#2196F3"
AXION_LIGHT = "#90CAF9"
TARGET_COLOR = "#E91E63"


def smooth(x, window=7):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed to use for the example trajectory panel")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    batch_dir = pathlib.Path(args.batch_dir)

    # Load example seed
    example_path = batch_dir / f"seed_{args.seed}.json"
    if not example_path.exists():
        print(f"Not found: {example_path}")
        return
    example = json.loads(example_path.read_text())

    # Load all seeds
    files = sorted(batch_dir.glob("seed_*.json"))
    all_data = [json.loads(f.read_text()) for f in files]
    num_seeds = len(all_data)

    if num_seeds == 0:
        print("No results found.")
        return

    # --- Figure ---
    fig, (ax_traj, ax_conv) = plt.subplots(
        1, 2, figsize=(11.0, 3.8),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.35},
    )

    # ===== Panel 1: Example trajectory (init, intermediate, best, target) =====
    trajs = example.get("trajectories", {})
    target = example.get("target_trajectory")
    rmse = np.array(example["rmse_m"])
    best_iters = example.get("best_iters", [])

    # Pick 3 trajectories: init (iter 0), one intermediate, and the best
    best_iter = int(np.argmin(rmse))
    best_rmse = rmse[best_iter]
    init_rmse = rmse[0]

    # Intermediate: pick the iteration closest to 50% between init and best RMSE
    target_rmse = (init_rmse + best_rmse) / 2.0
    mid_iter = int(np.argmin(np.abs(rmse - target_rmse)))
    mid_rmse = rmse[mid_iter]

    INTERMEDIATE_COLOR = "#64B5F6"

    # Init
    t0 = trajs.get("0")
    if t0:
        ax_traj.plot(t0["x"], t0["y"], color="black", linewidth=1.5, alpha=0.5,
                     label=f"iter 0 ({init_rmse:.2f}\\,m)")

    # Intermediate
    t_mid = trajs.get(str(mid_iter))
    if t_mid:
        ax_traj.plot(t_mid["x"], t_mid["y"], color=INTERMEDIATE_COLOR, linewidth=1.8, alpha=0.7,
                     label=f"iter {mid_iter} ({mid_rmse:.2f}\\,m)")

    # Best
    t_best = trajs.get(str(best_iter))
    if t_best:
        ax_traj.plot(t_best["x"], t_best["y"], color=AXION_COLOR, linewidth=2.5,
                     label=f"iter {best_iter} ({best_rmse:.2f}\\,m)")

    # Target
    ax_traj.plot(target["x"], target["y"], color=TARGET_COLOR, linewidth=2.5,
                 linestyle="--", label="target")
    ax_traj.plot(target["x"][0], target["y"][0], "o", color=TARGET_COLOR, markersize=5, zorder=5)
    if t_best:
        ax_traj.plot(t_best["x"][0], t_best["y"][0], "o", color=AXION_COLOR, markersize=5, zorder=5)

    ax_traj.set_xlabel("$x$ (m)")
    ax_traj.set_ylabel("$y$ (m)")
    ax_traj.set_aspect("auto")
    ax_traj.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_traj.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                   ncol=2, fontsize=13, framealpha=0.85, columnspacing=1.2,
                   handlelength=1.5)

    # ===== Panel 2: Convergence envelope =====
    all_rmse = np.array([d["rmse_m"] for d in all_data])
    iters = np.array(all_data[0]["iterations"])

    median_rmse = np.median(all_rmse, axis=0)
    p25 = np.percentile(all_rmse, 25, axis=0)
    p75 = np.percentile(all_rmse, 75, axis=0)

    ax_conv.fill_between(iters, p25, p75, color=AXION_LIGHT, alpha=0.5, label="IQR")
    ax_conv.plot(iters, median_rmse, color=AXION_COLOR, linewidth=2.0, label="median")

    # Running best median
    running_best = np.minimum.accumulate(median_rmse)
    ax_conv.plot(iters, running_best, color="gray", linewidth=1.0, linestyle="--",
                 alpha=0.7, label="best median")

    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("RMSE (m)")
    ax_conv.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_conv.legend(loc="upper right", fontsize=10)
    ax_conv.set_title(f"{num_seeds} random terrains", fontsize=14)

    all_best = np.array([min(d["rmse_m"]) for d in all_data])

    plt.tight_layout(pad=0.5)

    # Save
    out_local = RESULTS_DIR / "terrain_paper.png"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_local}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "terrain_traversal.png"
        fig.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    # Summary
    print(f"  Seeds: {num_seeds}")
    print(f"  Best RMSE: {np.median(all_best):.3f}m median, "
          f"{np.mean(all_best):.3f} +/- {np.std(all_best):.3f}m")
    print(f"  100% converged below {np.max(all_best):.3f}m")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
