"""Verify joy->cmd_vel fit: plot synth-cmd_vel wheel targets vs /joint_states."""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "1_sim_to_real"))
from diagnose_cmd_vs_joint import (
    WHEEL_RADIUS,
    HALF_TRACK,
    load_cmd_vel_timeseries,
)

DATA = pathlib.Path(__file__).parent


def main():
    synth_path = DATA / "helhest_2026_04_13-09_56_50_synth_cmdvel.json"
    gt_path = DATA / "acceleration.json"
    out_path = DATA / "synth_cmdvel_vs_joint_states.png"

    cmd = load_cmd_vel_timeseries(synth_path)
    t_cmd = np.array([c[0] for c in cmd])
    vx = np.array([c[1] for c in cmd])
    wz = np.array([c[2] for c in cmd])
    v_l = (vx - wz * HALF_TRACK) / WHEEL_RADIUS
    v_r = (vx + wz * HALF_TRACK) / WHEEL_RADIUS
    v_rear = vx / WHEEL_RADIUS

    gt = json.loads(gt_path.read_text())
    ts = gt["wheel_velocities"]["timeseries"]
    t_js = np.array([p["t"] for p in ts])
    meas = np.array([[p["left"], p["right"], p["rear"]] for p in ts])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, lab, pred) in enumerate(zip(axes,
            ["left_wheel_j", "right_wheel_j", "rear_wheel_j"],
            [v_l, v_r, v_rear])):
        ax.plot(t_cmd, pred, "--", color="tab:green",
                label="cmd (synth from /joy + fitted kin.)", lw=1.3)
        ax.plot(t_js, meas[:, i], "-", color="tab:blue",
                label="measured /joint_states", lw=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.set_ylabel(f"{lab}\n[rad/s]")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Synth /cmd_vel (from /joy) vs /joint_states — 09_56_50 (acceleration)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    for lab, pred in zip(["left", "right", "rear"], [v_l, v_r, v_rear]):
        ip = np.interp(t_js, t_cmd, pred)
        col = {"left": 0, "right": 1, "rear": 2}[lab]
        err = meas[:, col] - ip
        print(f"  {lab}: mean_err={err.mean():+.3f}  rms={np.sqrt((err**2).mean()):.3f} rad/s")


if __name__ == "__main__":
    main()
