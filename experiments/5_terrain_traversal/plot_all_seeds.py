"""Generate top-down xy trajectory plots for all seeds in a batch directory.

Saves each seed as a separate PNG for quick visual comparison.

Usage:
    python -m examples.terrain_traversal.plot_all_seeds \
        --batch-dir results/terrain_batch
    python -m examples.terrain_traversal.plot_all_seeds \
        --batch-dir results/terrain_batch --out-dir results/terrain_previews
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AXION_COLOR = "#2196F3"
TARGET_COLOR = "#E91E63"


def plot_seed(data, out_path):
    """Plot a single seed's xy trajectory (init, best, target)."""
    trajs = data.get("trajectories", {})
    target = data.get("target_trajectory")
    rmse = np.array(data["rmse_m"])
    best_iters = data.get("best_iters", [])
    seed = data.get("seed", "?")

    if not trajs or not target:
        return

    fig, ax = plt.subplots(figsize=(5, 5))

    # Init trajectory
    t0 = trajs.get("0")
    if t0:
        ax.plot(t0["x"], t0["y"], color="black", linewidth=1.5, alpha=0.4,
                label=f"init ({rmse[0]:.2f}\\,m)")

    # Best trajectory
    best_iter = int(np.argmin(rmse))
    best_rmse = rmse[best_iter]
    t_best = trajs.get(str(best_iter))
    if t_best:
        ax.plot(t_best["x"], t_best["y"], color=AXION_COLOR, linewidth=2.5,
                label=f"iter {best_iter} ({best_rmse:.2f}\\,m)")

    # Target
    ax.plot(target["x"], target["y"], color=TARGET_COLOR, linewidth=2.5,
            linestyle="--", label="target")

    ax.plot(target["x"][0], target["y"][0], "o", color=TARGET_COLOR, markersize=5, zorder=5)

    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", framealpha=0.85)
    ax.set_title(f"Seed {seed}  (best {best_rmse:.3f}\\,m @ iter {best_iter})", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: <batch-dir>/previews)")
    args = parser.parse_args()

    batch_dir = pathlib.Path(args.batch_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else batch_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(batch_dir.glob("seed_*.json"))
    if not files:
        print(f"No seed_*.json files in {batch_dir}")
        return

    for f in files:
        data = json.loads(f.read_text())
        seed = data.get("seed", f.stem)
        out_path = out_dir / f"seed_{seed}.png"
        plot_seed(data, out_path)
        best_rmse = min(data["rmse_m"])
        print(f"  Seed {seed:>3}: best={best_rmse:.3f}m -> {out_path}")

    print(f"\n  {len(files)} previews saved to {out_dir}/")


if __name__ == "__main__":
    main()
