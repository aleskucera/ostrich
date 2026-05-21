"""Plot terrain traversal results.

Supports single-seed JSONs, batch directories, and batch summary files.

Usage:
    # Single seed result:
    python -m examples.terrain_traversal.plot --json results/terrain_batch/seed_1.json

    # Batch directory (aggregate all seeds):
    python -m examples.terrain_traversal.plot --batch-dir results/terrain_batch

    # Pick one seed from batch to show xy trajectory:
    python -m examples.terrain_traversal.plot --batch-dir results/terrain_batch --seed 1
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
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AXION_COLOR = "#2196F3"
AXION_LIGHT = "#90CAF9"
TARGET_COLOR = "#E91E63"


def smooth(x, window=7):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_single(data, show=False, out_name="terrain_traversal.png"):
    """Plot single-seed result: trajectory progression + RMSE convergence.

    Uses best_iters (monotonically improving iterations) when available,
    so only increasingly better trajectories are shown.
    """
    iters = np.array(data["iterations"])
    rmse = np.array(data["rmse_m"])
    loss = np.array(data["loss"])
    time_ms = np.array(data["time_ms"])

    trajs = data.get("trajectories", {})
    target = data.get("target_trajectory")
    best_iters = data.get("best_iters", [])

    if not trajs or not target:
        print("No trajectory snapshots in JSON — re-run the benchmark.")
        return

    fig, (ax_traj, ax_rmse) = plt.subplots(
        1, 2, figsize=(10.0, 5.0),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.4},
    )

    # ===== Left: Trajectory progression =====
    # Select which iterations to show: use best_iters (monotonically improving)
    # Pick ~5-6 evenly spaced from best_iters to avoid clutter
    if best_iters and len(best_iters) > 1:
        # Always include first and last best
        if len(best_iters) <= 6:
            show_iters = best_iters
        else:
            indices = np.linspace(0, len(best_iters) - 1, 6, dtype=int)
            show_iters = [best_iters[i] for i in indices]
            # Ensure first and last are included
            if show_iters[0] != best_iters[0]:
                show_iters[0] = best_iters[0]
            if show_iters[-1] != best_iters[-1]:
                show_iters[-1] = best_iters[-1]
    else:
        # Fallback: use available trajectory keys
        show_iters = sorted(int(k) for k in trajs.keys())
        if len(show_iters) > 6:
            indices = np.linspace(0, len(show_iters) - 1, 6, dtype=int)
            show_iters = [show_iters[i] for i in indices]

    # Filter to iterations that actually have trajectory data
    show_iters = [i for i in show_iters if str(i) in trajs]

    if not show_iters:
        print("No trajectory data for selected iterations.")
        return

    last_iter = show_iters[-1]
    cmap = plt.get_cmap("Blues")

    # First iteration (init) in black
    t0 = trajs[str(show_iters[0])]
    ax_traj.plot(t0["x"], t0["y"],
                 color="black", linewidth=1.5, alpha=0.5,
                 label=f"iter {show_iters[0]} (init)")

    # Intermediate iterations in fading blues
    for idx, it in enumerate(show_iters[1:-1]):
        t = trajs[str(it)]
        frac = 0.3 + 0.5 * ((idx + 1) / max(len(show_iters) - 1, 1))
        it_rmse = rmse[it] if it < len(rmse) else 0
        ax_traj.plot(t["x"], t["y"],
                     color=cmap(frac), linewidth=1.2, alpha=0.6,
                     label=f"iter {it} ({it_rmse:.2f}\\,m)")

    # Final best iteration
    t_final = trajs[str(last_iter)]
    last_rmse = rmse[last_iter] if last_iter < len(rmse) else 0
    ax_traj.plot(t_final["x"], t_final["y"],
                 color=AXION_COLOR, linewidth=2.5,
                 label=f"iter {last_iter} ({last_rmse:.2f}\\,m)")

    # Target
    ax_traj.plot(target["x"], target["y"],
                 color=TARGET_COLOR, linewidth=2.5, linestyle="--", label="target")

    ax_traj.plot(target["x"][0], target["y"][0],
                 "o", color=TARGET_COLOR, markersize=6, zorder=5)
    ax_traj.plot(t_final["x"][0], t_final["y"][0],
                 "o", color=AXION_COLOR, markersize=6, zorder=5)

    seed_label = data.get("seed", "?")
    ax_traj.set_title(f"seed {seed_label}", fontsize=16)
    ax_traj.set_xlabel("$x$ (m)")
    ax_traj.set_ylabel("$y$ (m)")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_traj.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
                   ncol=3, framealpha=0.85, columnspacing=1.0, fontsize=14)

    # ===== Right: RMSE convergence =====
    time_steady = time_ms[1:]
    median_time = np.median(time_steady)
    best_rmse = float(np.min(rmse))
    best_iter_idx = int(np.argmin(rmse))

    ax_rmse.plot(iters, rmse, color=AXION_LIGHT, linewidth=1.0, alpha=0.7)
    ax_rmse.plot(iters, smooth(rmse), color=AXION_COLOR, linewidth=2.0)
    ax_rmse.axhline(best_rmse, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_rmse.plot(best_iter_idx, best_rmse, "v", color=AXION_COLOR, markersize=8, zorder=5)

    # Mark best_iters on the RMSE curve
    if best_iters:
        bi_rmses = [rmse[i] for i in best_iters if i < len(rmse)]
        bi_iters = [i for i in best_iters if i < len(rmse)]
        ax_rmse.plot(bi_iters, bi_rmses, "o", color=TARGET_COLOR,
                     markersize=4, zorder=4, alpha=0.6)

    ax_rmse.set_xlabel("Iteration")
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_rmse.text(0.97, 0.95, f"best = {best_rmse:.2f}\\,m (iter {best_iter_idx})",
                 transform=ax_rmse.transAxes, ha="right", va="top", fontsize=14, color="gray")
    ax_rmse.text(0.97, 0.87, f"median iter time = {median_time:.0f}\\,ms",
                 transform=ax_rmse.transAxes, ha="right", va="top", fontsize=14, color=AXION_COLOR)

    plt.tight_layout(pad=0.4)

    out_local = RESULTS_DIR / out_name
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_local}")

    _save_to_paper(fig, out_name)

    p25 = np.percentile(time_steady, 25)
    p75 = np.percentile(time_steady, 75)
    print(f"  Seed: {seed_label}")
    print(f"  Iterations: {len(iters)}")
    print(f"  Median iter time (excl. iter 0): {median_time:.0f} ms  IQR: [{p25:.0f}, {p75:.0f}]")
    print(f"  Best RMSE: {best_rmse:.3f} m (iter {best_iter_idx})")
    if best_iters:
        print(f"  Improvements: {len(best_iters)} (iters: {best_iters})")

    if show:
        plt.show()


def plot_batch_dir(batch_dir, show=False):
    """Plot aggregate results from a directory of per-seed JSONs."""
    files = sorted(pathlib.Path(batch_dir).glob("seed_*.json"))
    if not files:
        print(f"No seed_*.json files found in {batch_dir}")
        return

    all_data = []
    for f in files:
        all_data.append(json.loads(f.read_text()))

    num_seeds = len(all_data)
    all_rmse = np.array([d["rmse_m"] for d in all_data])
    all_best = np.array([min(d["rmse_m"]) for d in all_data])
    all_final = np.array([d["rmse_m"][-1] for d in all_data])
    iters = np.array(all_data[0]["iterations"])

    fig, (ax_conv, ax_hist) = plt.subplots(
        1, 2, figsize=(10.0, 5.0),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.4},
    )

    # ===== Left: Convergence envelope =====
    median_rmse = np.median(all_rmse, axis=0)
    p25 = np.percentile(all_rmse, 25, axis=0)
    p75 = np.percentile(all_rmse, 75, axis=0)
    p_min = np.min(all_rmse, axis=0)
    p_max = np.max(all_rmse, axis=0)

    ax_conv.fill_between(iters, p_min, p_max, color=AXION_LIGHT, alpha=0.3, label="min--max")
    ax_conv.fill_between(iters, p25, p75, color=AXION_LIGHT, alpha=0.5, label="IQR")
    ax_conv.plot(iters, median_rmse, color=AXION_COLOR, linewidth=2.0, label="median")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("RMSE (m)")
    ax_conv.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_conv.legend(loc="upper right", fontsize=14)
    ax_conv.set_title(f"{num_seeds} random terrains", fontsize=16)

    # ===== Right: Best RMSE histogram =====
    ax_hist.hist(all_best, bins=max(num_seeds // 3, 10), color=AXION_COLOR, alpha=0.8,
                 edgecolor="white", linewidth=0.5)
    ax_hist.axvline(np.median(all_best), color=TARGET_COLOR, linewidth=2, linestyle="--",
                    label=f"median = {np.median(all_best):.2f}\\,m")
    ax_hist.set_xlabel("Best RMSE (m)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_hist.legend(loc="upper right", fontsize=14)

    plt.tight_layout(pad=0.4)

    out_local = RESULTS_DIR / "terrain_batch.png"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_local}")

    _save_to_paper(fig, "terrain_batch.png")

    print(f"  Seeds: {num_seeds}")
    print(f"  Best  RMSE: {np.median(all_best):.3f}m median, "
          f"{np.mean(all_best):.3f} +/- {np.std(all_best):.3f}m")
    print(f"  Final RMSE: {np.median(all_final):.3f}m median, "
          f"{np.mean(all_final):.3f} +/- {np.std(all_final):.3f}m")
    print(f"  100% converged below {np.max(all_best):.3f}m")

    if show:
        plt.show()


def _save_to_paper(fig, filename):
    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / filename
        fig.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--json", type=str, default=None, help="Single result JSON")
    parser.add_argument("--batch-dir", type=str, default=None,
                        help="Directory of per-seed JSONs (from run_batch.sh)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Plot a specific seed from --batch-dir")
    args = parser.parse_args()

    if args.batch_dir:
        if args.seed is not None:
            json_path = pathlib.Path(args.batch_dir) / f"seed_{args.seed}.json"
            if not json_path.exists():
                print(f"Not found: {json_path}")
                return
            data = json.loads(json_path.read_text())
            plot_single(data, show=args.show, out_name=f"terrain_seed_{args.seed}.png")
        else:
            plot_batch_dir(args.batch_dir, show=args.show)
    elif args.json:
        data = json.loads(pathlib.Path(args.json).read_text())
        plot_single(data, show=args.show)
    else:
        print("Provide --json or --batch-dir")


if __name__ == "__main__":
    main()
