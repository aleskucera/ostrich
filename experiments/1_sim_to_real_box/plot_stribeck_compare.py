"""Visual comparison: constant-mu best vs Stribeck best vs real.

Rows = runs (two turn runs, the 2 m/s run, one straight run);
cols = top-down XY (with fitted pallet) and heading yaw(t).

    .venv/bin/python experiments/1_sim_to_real_box/plot_stribeck_compare.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_campaign2 as ec
from common_box import DATA_DIR, RESULTS_DIR, load_gt
import examples.helhest_junior.replay_real as rr

RUNS = ["ostrich10", "ostrich11", "ostrich13", "ostrich1"]
CONFIGS = {
    "constant-mu": dict(k_p=10000.0, mu_front=0.8, mu_rear=0.8,
                        mu_long_front=0.8, mu_long_rear=1.2),
    "Stribeck": dict(k_p=4000.0, mu_front=0.4, mu_rear=0.4,
                     mu_long_front=0.8, mu_long_rear=1.2,
                     mu_stiction_scale=4.0, v_stribeck=0.3),
}
COLORS = {"constant-mu": "#9DB4E8", "Stribeck": "#C43131"}

PARAMS = {}
_orig_init = rr.HelhestJuniorReplaySimulator.__init__


def _patched_init(self, *a, **kw):
    kw.update(PARAMS)
    _orig_init(self, *a, **kw)


rr.HelhestJuniorReplaySimulator.__init__ = _patched_init


def main():
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(13, 3.4 * len(RUNS)))
    for i, run in enumerate(RUNS):
        gt = load_gt(DATA_DIR / f"{run}.json")
        ax_xy, ax_yaw = axes[i]
        rt = gt["real"]["t"]
        rx, ry = gt["real"]["x"], gt["real"]["y"]
        ryaw = np.degrees(gt["real"]["yaw_rel"])

        box = gt["box"]
        cx, cy = box["center"][:2]
        hx, hy = box["half_extents"][:2]
        yb = box.get("yaw", 0.0)
        c, s = np.cos(yb), np.sin(yb)
        R = np.array([[c, -s], [s, c]])
        cc = (np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
                        [-hx, -hy]]) @ R.T) + [cx, cy]
        ax_xy.fill(cc[:, 0], cc[:, 1], color="#B25E1C", alpha=0.2)
        ax_xy.plot(cc[:, 0], cc[:, 1], "-", c="#B25E1C", lw=1.4, label="pallet")

        ax_xy.plot(rx, ry, "-", c="k", lw=2.0, label="real", zorder=5)
        ax_yaw.plot(rt, ryaw, "-", c="k", lw=2.0)

        for name, cfg in CONFIGS.items():
            PARAMS.clear()
            PARAMS.update(cfg)
            sc = ec._score_run(*ec.run_ostrich(gt, 0.937), gt)
            sim = sc["sim_rel"]
            err = sc["combined_with_yaw"]
            ax_xy.plot(sim[:, 0], sim[:, 1], c=COLORS[name], lw=1.5,
                       label=f"{name} ({err:.2f} m)")
            ax_yaw.plot(sc["real_t_used"],
                        np.degrees(sc["sim_yaw_rel_on_real_t"]),
                        c=COLORS[name], lw=1.5)
            print(f"{run} {name}: {err:.3f} m, yaw {sc['yaw_rmse_deg']:.2f} deg")

        ax_xy.set_aspect("equal")
        ax_xy.set_xlim(min(rx.min(), cc[:, 0].min()) - 0.4,
                       max(rx.max(), cc[:, 0].max()) + 0.4)
        ax_xy.set_ylim(min(ry.min(), cc[:, 1].min()) - 0.4,
                       max(ry.max(), cc[:, 1].max()) + 0.4)
        ax_xy.set_title(f"{run}: top-down [m]", fontsize=10)
        ax_xy.legend(fontsize=8, loc="best")
        ax_yaw.set_title("heading [deg]", fontsize=10)
        ax_yaw.set_xlim(0, rt[-1])
        ax_yaw.axhline(0, color="#dddddd", lw=0.8)
        if i == len(RUNS) - 1:
            ax_yaw.set_xlabel("t [s]")

    fig.suptitle("Constant-mu vs Stribeck mu(v) vs real "
                 "(identified configs, held-out and train runs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "stribeck_compare.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
