"""Plot terrain traversal: RMSE convergence + 2D trajectory progression.

Usage:
    python examples/comparison_gradient/helhest/plot_terrain.py --json results/terrain_traversal.json
    python examples/comparison_gradient/helhest/plot_terrain.py --json results/terrain_traversal.json --show
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PAPER_DIR = pathlib.Path(__file__).resolve().parents[3] / ".." / "axion_paper" / "figures"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    json_path = pathlib.Path(args.json) if args.json else RESULTS_DIR / "terrain_traversal.json"
    data = json.loads(json_path.read_text())

    iters = np.array(data["iterations"])
    rmse = np.array(data["rmse_m"])
    loss = np.array(data["loss"])
    time_ms = np.array(data["time_ms"])

    trajs = data.get("trajectories", {})
    target = data.get("target_trajectory")
    has_trajs = bool(trajs) and bool(target)

    if not has_trajs:
        print("No trajectory snapshots in JSON — re-run the benchmark.")
        return

    # --- Figure: two subplots side by side ---
    fig, (ax_traj, ax_rmse) = plt.subplots(
        1, 2, figsize=(10.0, 5.0),
        gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.4},
    )

    # ===== Left: Trajectory progression (xy plane) =====
    snap_iters = sorted(int(k) for k in trajs.keys() if int(k) != 30)
    last_iter = snap_iters[-1]
    cmap = plt.get_cmap("Blues")
    n_intermediate = len(snap_iters) - 1

    # Iter 0: initial guess in black
    t0 = trajs[str(snap_iters[0])]
    ax_traj.plot(t0["x"], t0["y"],
                 color="black", linewidth=1.5, alpha=0.5,
                 label=f"iter {snap_iters[0]} (init)")

    # Intermediate iterations in fading blues
    for idx, it in enumerate(snap_iters[1:-1]):
        t = trajs[str(it)]
        frac = 0.3 + 0.45 * (idx / max(len(snap_iters) - 3, 1))
        ax_traj.plot(t["x"], t["y"],
                     color=cmap(frac), linewidth=1.2, alpha=0.6,
                     label=f"iter {it}")

    # Final optimized trajectory
    t_final = trajs[str(last_iter)]
    ax_traj.plot(t_final["x"], t_final["y"],
                 color=AXION_COLOR, linewidth=2.5, label=f"iter {last_iter}")

    ax_traj.plot(target["x"], target["y"],
                 color=TARGET_COLOR, linewidth=2.5, linestyle="--", label="target")

    ax_traj.plot(target["x"][0], target["y"][0],
                 "o", color=TARGET_COLOR, markersize=6, zorder=5)
    ax_traj.plot(t_final["x"][0], t_final["y"][0],
                 "o", color=AXION_COLOR, markersize=6, zorder=5)

    ax_traj.set_xlabel("$x$ (m)")
    ax_traj.set_ylabel("$y$ (m)")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_traj.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, framealpha=0.85, columnspacing=1.0, fontsize=16)

    # ===== Right: RMSE convergence =====
    time_steady = time_ms[1:]
    median_time = np.median(time_steady)
    best_rmse = float(np.min(rmse))
    best_iter = int(np.argmin(rmse))

    ax_rmse.plot(iters, rmse, color=AXION_LIGHT, linewidth=1.0, alpha=0.7)
    ax_rmse.plot(iters, smooth(rmse), color=AXION_COLOR, linewidth=2.0)
    ax_rmse.axhline(best_rmse, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_rmse.plot(best_iter, best_rmse, "v", color=AXION_COLOR, markersize=8, zorder=5)
    ax_rmse.set_xlabel("Iteration")
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.grid(True, which="major", alpha=0.25, linewidth=0.5)
    ax_rmse.text(0.97, 0.95, f"best = {best_rmse:.2f}\\,m (iter {best_iter})",
                 transform=ax_rmse.transAxes, ha="right", va="top", fontsize=14, color="gray")
    ax_rmse.text(0.97, 0.87, f"median iter time = {median_time:.0f}\\,ms",
                 transform=ax_rmse.transAxes, ha="right", va="top", fontsize=14, color=AXION_COLOR)

    plt.tight_layout(pad=0.4)

    # Save
    out_local = RESULTS_DIR / "terrain_traversal.png"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local, dpi=300, bbox_inches="tight")
    print(f"Saved to {out_local}")

    paper_dir = PAPER_DIR.resolve()
    if paper_dir.is_dir():
        out_paper = paper_dir / "terrain_traversal.png"
        fig.savefig(out_paper, dpi=300, bbox_inches="tight")
        print(f"Saved to {out_paper}")

    # Summary
    p25 = np.percentile(time_steady, 25)
    p75 = np.percentile(time_steady, 75)
    print(f"  Iterations: {len(iters)}")
    print(f"  Median iter time (excl. iter 0): {median_time:.0f} ms  IQR: [{p25:.0f}, {p75:.0f}]")
    print(f"  Best RMSE: {best_rmse:.3f} m (iter {best_iter})")
    print(f"  Final loss (median last 10): {float(np.median(loss[-10:])):.4f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
