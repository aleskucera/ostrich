"""Plot synth wheel velocities (from /joy fit) vs measured /joint_states."""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

DATA = pathlib.Path(__file__).parent


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else "helhest_2026_04_13-09_56_50"
    synth = json.loads((DATA / f"{bag}_synth_wheels.json").read_text())
    gt = json.loads((DATA / "acceleration.json").read_text())

    s = synth["wheel_velocities"]["timeseries"]
    t_s = np.array([p["t"] for p in s])
    pred = np.array([[p["left"], p["right"], p["rear"]] for p in s])

    m = gt["wheel_velocities"]["timeseries"]
    t_m = np.array([p["t"] for p in m])
    meas = np.array([[p["left"], p["right"], p["rear"]] for p in m])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, lab) in enumerate(zip(axes,
            ["left_wheel_j", "right_wheel_j", "rear_wheel_j"])):
        ax.plot(t_s, pred[:, i], "--", color="tab:green",
                label="synth (joy->wheels fit)", lw=1.3)
        ax.plot(t_m, meas[:, i], "-", color="tab:blue",
                label="measured /joint_states", lw=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_ylabel(f"{lab}\n[rad/s]")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"/joy->wheels direct fit vs /joint_states — {bag}")
    fig.tight_layout()
    out = DATA / f"synth_wheels_vs_joint_states_{bag}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
