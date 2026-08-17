"""Plot the campaign-2 generalization results: per-run sim-vs-real overlays.

Consumes results/eval_campaign2.json (written by eval_campaign2.py) and the
GT JSONs. One figure: rows = runs, cols = [top-down XY, x(t), z(t)], with the
real trajectory in black and one colored line per engine (pass 2, i.e. after
motor-scale calibration where applicable).

    python experiments/1_sim_to_real_box/plot_campaign2.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

COLORS = {"ostrich": "#3562D6", "mujoco": "#C43131", "semi_implicit": "#3B8C4E"}
LABELS = {"ostrich": "Ostrich", "mujoco": "MuJoCo", "semi_implicit": "SemiImplicit"}


def main():
    res = json.load(open(RESULTS_DIR / "eval_campaign2.json"))
    engines = list(res)
    runs = list(next(iter(res.values()))["pass2"])

    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 3.2 * len(runs)))
    axes = np.atleast_2d(axes)
    for i, run in enumerate(runs):
        gt = json.load(open(DATA_DIR / f"{run}.json"))
        rt = np.array(gt["real"]["t"])
        rx, ry, rz = (np.array(gt["real"][k]) for k in ("x", "y", "z"))
        ax_xy, ax_x, ax_z = axes[i]

        box = gt["box"]
        cx, cy = box["center"][:2]
        hx, hy = box["half_extents"][:2]
        yaw = box.get("yaw", 0.0)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
        cc = corners @ R.T + [cx, cy]
        ax_xy.plot(cc[:, 0], cc[:, 1], "-", c="#B25E1C", lw=1.5, label="pallet")

        ax_xy.plot(rx, ry, "-", c="k", lw=1.5, label="real")
        ax_x.plot(rt, rx, "-", c="k", lw=1.5)
        ax_z.plot(rt, rz, "-", c="k", lw=1.5)

        for eng in engines:
            traj = res[eng]["pass2_traj"][run]
            sim = np.array(traj["sim_rel"])
            st = np.array(traj["sim_t_aligned"])
            err = res[eng]["pass2"][run]["combined_with_yaw"]
            kw = dict(c=COLORS[eng], lw=1.0, alpha=0.9)
            ax_xy.plot(sim[:, 0], sim[:, 1],
                       label=f"{LABELS[eng]} ({err:.3f} m)", **kw)
            ax_x.plot(st, sim[:, 0], **kw)
            ax_z.plot(st, sim[:, 2], **kw)

        ax_xy.set_aspect("equal")
        # clip axes to the real trajectory + pallet extent so a diverging
        # engine (SI launches meters into the air) can't destroy the scale
        ax_xy.set_xlim(min(rx.min(), cc[:, 0].min()) - 0.3,
                       max(rx.max(), cc[:, 0].max()) + 0.3)
        ax_xy.set_ylim(min(ry.min(), cc[:, 1].min()) - 0.3,
                       max(ry.max(), cc[:, 1].max()) + 0.3)
        ax_x.set_ylim(rx.min() - 0.5, rx.max() + 0.5)
        ax_z.set_ylim(-0.05, 0.25)
        ax_xy.set_title(f"{run}: top-down", fontsize=10)
        ax_xy.legend(fontsize=7, loc="best")
        ax_x.set_title("x(t)", fontsize=10)
        ax_z.set_title("z(t)", fontsize=10)
        for ax in (ax_x, ax_z):
            ax.set_xlim(0, rt[-1])
        ax_xy.set_ylabel(run)

    scales = ", ".join(f"{LABELS[e]} cmd_scale={res[e]['cmd_scale']:.3f}"
                       for e in engines)
    fig.suptitle(f"Campaign-2 generalization (campaign-1 params frozen; {scales})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "eval_campaign2.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # summary table
    print(f"\n{'run':<10}" + "".join(f"{LABELS[e]:>14}" for e in engines))
    for run in runs:
        print(f"{run:<10}" + "".join(
            f"{res[e]['pass2'][run]['combined_with_yaw']:>14.3f}" for e in engines))
    print(f"{'mean':<10}" + "".join(
        f"{res[e]['mean_combined_with_yaw']['pass2']:>14.3f}" for e in engines))


if __name__ == "__main__":
    main()
