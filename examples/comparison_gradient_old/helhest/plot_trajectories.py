"""Plot target trajectories from all helhest simulators in xy plane.

Usage:
    python examples/comparison_gradient/helhest/plot_trajectories.py results/*.json
    python examples/comparison_gradient/helhest/plot_trajectories.py results/*.json --output results/trajectories.png
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def main():
    parser = argparse.ArgumentParser(description="Plot helhest target trajectories")
    parser.add_argument("files", nargs="+", help="JSON files with target_trajectory field")
    parser.add_argument("--output", "-o", default="helhest_trajectories.png",
                        help="Output PNG path (default: helhest_trajectories.png)")
    args = parser.parse_args()

    data = {}
    for path in args.files:
        with open(path) as f:
            d = json.load(f)
        if "target_trajectory" not in d:
            print(f"Skipping {path}: no target_trajectory field")
            continue
        name = d.get("simulator", pathlib.Path(path).stem)
        data[name] = {
            "traj": np.array(d["target_trajectory"]),
            "dt": d["dt"],
            "T": d["T"],
        }

    if not data:
        print("No valid trajectory files found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: all trajectories
    ax = axes[0]
    for i, (name, d) in enumerate(data.items()):
        color = COLORS[i % len(COLORS)]
        traj = d["traj"]
        label = f'{name} (dt={d["dt"]}, T={d["T"]})'
        ax.plot(traj[:, 0], traj[:, 1], label=label, linewidth=0.8, color=color)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", markersize=4, color=color)
        ax.plot(traj[0, 0], traj[0, 1], "s", markersize=3, color=color, alpha=0.5)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title("All simulators", fontsize=13)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Right: auto-exclude outliers (final position > 3x median distance from origin)
    finals = {name: np.linalg.norm(d["traj"][-1]) for name, d in data.items()}
    median_dist = np.median(list(finals.values()))
    exclude = {name for name, dist in finals.items() if dist > 3 * median_dist} if len(data) > 2 else set()

    ax = axes[1]
    for i, (name, d) in enumerate(data.items()):
        if name in exclude:
            continue
        color = COLORS[i % len(COLORS)]
        traj = d["traj"]
        label = f'{name} (dt={d["dt"]}, T={d["T"]})'
        ax.plot(traj[:, 0], traj[:, 1], label=label, linewidth=0.8, color=color)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", markersize=4, color=color)
        ax.plot(traj[0, 0], traj[0, 1], "s", markersize=3, color=color, alpha=0.5)

    excluded_str = f" (excl. {', '.join(exclude)})" if exclude else ""
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(f"Zoomed{excluded_str}", fontsize=13)
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Helhest Target Trajectories — xy plane, ctrl=(1.0, 6.0, 0.0)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")

    # Print summary table
    print(f"\n{'Simulator':<20} {'dt':>8} {'T':>6} {'dur(s)':>7} {'final_x':>9} {'final_y':>9}")
    print("-" * 65)
    for name, d in data.items():
        traj = d["traj"]
        dur = d["dt"] * d["T"]
        print(f"{name:<20} {d['dt']:>8.4f} {d['T']:>6} {dur:>7.1f} {traj[-1,0]:>9.3f} {traj[-1,1]:>9.3f}")


if __name__ == "__main__":
    main()
