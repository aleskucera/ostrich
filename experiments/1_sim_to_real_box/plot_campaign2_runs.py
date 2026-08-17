"""Per-run overview figures for the campaign-2 runs.

One figure per run, four panels:
  1. top-down XY: real vs engines, fitted pallet (with yaw), start marker
  2. climb profile z(t)
  3. recorded wheel commands [L, R, rear]
  4. ground speed: real (smoothed) vs engines

    python experiments/1_sim_to_real_box/plot_campaign2_runs.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

COLORS = {"ostrich": "#3562D6", "mujoco": "#C43131", "semi_implicit": "#3B8C4E"}
LABELS = {"ostrich": "Ostrich", "mujoco": "MuJoCo", "semi_implicit": "SemiImplicit"}


def smooth_speed(t, x, y):
    """Ground speed via 20 ms resample + 0.6 s boxcar (kills pose jitter)."""
    tg = np.arange(t[0], t[-1], 0.02)
    xg, yg = np.interp(tg, t, x), np.interp(tg, t, y)
    k = 31
    ker = np.ones(k) / k
    xs, ys = np.convolve(xg, ker, "same"), np.convolve(yg, ker, "same")
    spd = np.hypot(np.gradient(xs, 0.02), np.gradient(ys, 0.02))
    # boxcar edges are biased; trim half a kernel on each side
    return tg[k // 2:-k // 2], spd[k // 2:-k // 2]


def main():
    res = json.load(open(RESULTS_DIR / "eval_campaign2.json"))
    engines = [e for e in ("ostrich", "mujoco", "semi_implicit") if e in res]
    runs = list(next(iter(res.values()))["pass2"])

    for run in runs:
        gt = json.load(open(DATA_DIR / f"{run}.json"))
        rt = np.array(gt["real"]["t"])
        rx, ry, rz = (np.array(gt["real"][k]) for k in ("x", "y", "z"))
        ct = np.array(gt["control"]["t"])
        lrr = np.array(gt["control"]["lrr"])

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        ax_xy, ax_z = axes[0]
        ax_cmd, ax_v = axes[1]

        # --- top-down ---
        box = gt["box"]
        cx, cy = box["center"][:2]
        hx, hy = box["half_extents"][:2]
        yaw = box.get("yaw", 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
                            [-hx, -hy]])
        cc = corners @ R.T + [cx, cy]
        ax_xy.fill(cc[:, 0], cc[:, 1], color="#B25E1C", alpha=0.25, zorder=1)
        ax_xy.plot(cc[:, 0], cc[:, 1], "-", c="#B25E1C", lw=1.5,
                   label=f"pallet (yaw {np.degrees(yaw):.1f}°)")
        ax_xy.plot(rx, ry, "-", c="k", lw=2.0, label="real", zorder=5)
        ax_xy.plot(rx[0], ry[0], "o", c="k", ms=7, zorder=6)
        ax_xy.annotate("start", (rx[0], ry[0]), textcoords="offset points",
                       xytext=(6, 8), fontsize=8)
        for eng in engines:
            traj = res[eng]["pass2_traj"][run]
            sim = np.array(traj["sim_rel"])
            st = np.array(traj["sim_t_aligned"])
            err = res[eng]["pass2"][run]["combined_with_yaw"]
            lab = f"{LABELS[eng]} ({err:.2f} m)" if np.isfinite(err) else \
                f"{LABELS[eng]} (diverged)"
            ax_xy.plot(sim[:, 0], sim[:, 1], c=COLORS[eng], lw=1.2, alpha=0.9,
                       label=lab)
            ax_z.plot(st, sim[:, 2], c=COLORS[eng], lw=1.2, alpha=0.9)
            tv, sv = smooth_speed(st, sim[:, 0], sim[:, 1])
            ax_v.plot(tv, sv, c=COLORS[eng], lw=1.2, alpha=0.9)
        ax_xy.set_xlim(min(rx.min(), cc[:, 0].min()) - 0.3,
                       max(rx.max(), cc[:, 0].max()) + 0.3)
        ax_xy.set_ylim(min(ry.min(), cc[:, 1].min()) - 0.3,
                       max(ry.max(), cc[:, 1].max()) + 0.3)
        ax_xy.set_aspect("equal")
        ax_xy.set_title("top-down [m]")
        ax_xy.legend(fontsize=8, loc="best")

        # --- climb profile ---
        ax_z.plot(rt, rz, "-", c="k", lw=2.0, label="real")
        ax_z.axhline(2 * box["half_extents"][2], color="#B25E1C", ls="--",
                     lw=1, label="pallet height")
        ax_z.set_ylim(-0.06, 0.25)
        ax_z.set_xlim(0, rt[-1])
        ax_z.set_title("climb profile z(t) [m]")
        ax_z.legend(fontsize=8)

        # --- commands ---
        for i, (name, cl) in enumerate(zip(("left", "right", "rear"),
                                           ("#666666", "#999999", "#bbbbbb"))):
            ax_cmd.plot(ct, lrr[:, i], c=cl, lw=1.0, label=name)
        ax_cmd.set_xlim(0, rt[-1])
        ax_cmd.set_title("recorded wheel commands [rad/s]")
        ax_cmd.set_xlabel("t [s]")
        ax_cmd.legend(fontsize=8)

        # --- ground speed ---
        tv, rv = smooth_speed(rt, rx, ry)
        ax_v.plot(tv, rv, "-", c="k", lw=2.0, label="real")
        cmd_v = np.interp(tv, ct, np.abs(lrr[:, :2]).mean(axis=1)) * 0.35
        ax_v.plot(tv, cmd_v, ":", c="#888888", lw=1.2, label="command x R")
        ax_v.set_ylim(0, max(0.6, rv.max() + 0.1))
        ax_v.set_xlim(0, rt[-1])
        ax_v.set_title("ground speed [m/s]")
        ax_v.set_xlabel("t [s]")
        ax_v.legend(fontsize=8)

        fig.suptitle(f"{run}: campaign-2 run, engines at frozen campaign-1 "
                     f"params (dur {gt['duration_s']:.1f} s)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = RESULTS_DIR / f"{run}_overview.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
