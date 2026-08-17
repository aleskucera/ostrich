"""Campaign-3 version of the paper's box_sim_to_real figure — SAME layout as
plot_paper_panels.make_fig1 (xy, z-rise, bar; serif, shared bottom legend),
with the representative held-out sample from the 14-run dataset.

Sample: ostrich9 (joint-median held-out run under both engines' identified
configs). Engines: Ostrich (identified constant-mu), MuJoCo (c3-identified =
frozen c1 with rear/tor from the c3 grid), Semi-Implicit (frozen c1; shown on
this run where it survives, off-scale in the bar with its divergence note).

    .venv/bin/python experiments/1_sim_to_real_box/paper_fig_boxc3.py
"""
import json
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
from plot_best14 import _run_mj_c3

SAMPLE = "ostrich9"
OSTRICH_CFG = dict(k_p=10000.0, mu_front=0.6, mu_rear=0.6,
                   mu_long_front=0.8, mu_long_rear=1.2)
# all-14 means at identified configs; SI diverges (6/14) at its frozen config
BAR = {"Ostrich": (0.206, 0.05), "MuJoCo": (0.330, 0.002)}
SI_NOTE = r"diverges on 6/14 runs"

SIM_COLORS = {"Ostrich": "#2196F3", "MuJoCo": "#E91E63",
              "Semi-Implicit": "#FF9800"}
SIM_ORDER = ["Ostrich", "MuJoCo", "Semi-Implicit"]

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_orig_init = rr.HelhestJuniorReplaySimulator.__init__


def _patched_init(self, *a, **kw):
    kw.update(OSTRICH_CFG)
    _orig_init(self, *a, **kw)


rr.HelhestJuniorReplaySimulator.__init__ = _patched_init


def _display(name):
    if name == "Ostrich":
        return r"\textbf{Ostrich}"
    return {"Semi-Implicit": "Semi-Impl."}.get(name, name)


def load_trajs(gt):
    trajs = {}
    so = ec._score_run(*ec.run_ostrich(gt, 0.937), gt)
    trajs["Ostrich"] = (np.asarray(so["sim_rel"]),
                        np.asarray(so["sim_t_aligned"]))
    sm = ec._score_run(*_run_mj_c3(gt), gt)
    trajs["MuJoCo"] = (np.asarray(sm["sim_rel"]),
                       np.asarray(sm["sim_t_aligned"]))
    ss = ec._score_run(*ec.run_semi_implicit(gt, 0.8963), gt)
    trajs["Semi-Implicit"] = (np.asarray(ss["sim_rel"]),
                              np.asarray(ss["sim_t_aligned"]))
    return trajs


def panel_xy(ax, trajs, gt):
    box = gt["box"]
    cx, cy = box["center"][:2]
    hx, hy = box["half_extents"][:2]
    yb = box.get("yaw", 0.0)
    c, sn = np.cos(yb), np.sin(yb)
    R = np.array([[c, -sn], [sn, c]])
    cc = (np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
                    [-hx, -hy]]) @ R.T) + [cx, cy]
    ax.fill(cc[:, 0], cc[:, 1], color="gray", alpha=0.18, zorder=1)
    ax.plot(cc[:, 0], cc[:, 1], color="dimgray", lw=0.7, ls="--", zorder=1)
    ax.text(cx, cy + hy - 0.10, "obstacle", ha="center", va="top",
            fontsize=10, color="dimgray", style="italic", zorder=2)

    real_x = np.asarray(gt["real"]["x"])
    real_y = np.asarray(gt["real"]["y"])
    ax.plot(real_x, real_y, "k--", lw=1.6, label="Real robot", zorder=10)
    for sim in SIM_ORDER:
        sr, _ = trajs[sim]
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(sr[:, 0], sr[:, 1], "-", color=SIM_COLORS[sim], lw=1.1,
                label=_display(sim), zorder=zord)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-0.1, max(real_x.max(), cc[:, 0].max()) + 0.3)
    ax.set_ylim(-1.3, 0.7)
    ax.grid(True, alpha=0.3)


def panel_z(ax, trajs, gt):
    t_lo, t_hi = 0.5, float(np.asarray(gt["real"]["t"])[-1])
    real_t = np.asarray(gt["real"]["t"])
    real_z = np.asarray(gt["real"]["z"])
    m = (real_t >= t_lo) & (real_t <= t_hi)
    ax.plot(real_t[m], real_z[m], "k--", lw=1.6, label="Real robot", zorder=10)
    for sim in SIM_ORDER:
        sr, st = trajs[sim]
        sel = (st >= t_lo) & (st <= t_hi)
        z = sr[:, 2]
        bl = (st >= t_lo) & (st <= 2.0)
        baseline = float(np.mean(z[bl])) if bl.any() else 0.0
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(st[sel], z[sel] - baseline, "-", color=SIM_COLORS[sim],
                lw=1.1, label=_display(sim), zorder=zord)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"base $z$ rise (m)")
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-0.07, 0.24)
    ax.grid(True, alpha=0.3)


def panel_bar(ax):
    sims = ["Semi-Implicit", "MuJoCo", "Ostrich"]  # bottom-up -> Ostrich top
    xmax = 0.62
    y_pos = np.arange(len(sims))
    for y, sim in zip(y_pos, sims):
        if sim in BAR:
            err, dt = BAR[sim]
            ax.barh(y, err, color=SIM_COLORS[sim], height=0.5,
                    edgecolor="black", linewidth=0.8, zorder=3)
            ax.text(err + 0.02, y, rf" {err:.3f}  ($\Delta t={dt}$\,s)",
                    va="center", ha="left", fontsize=11)
        else:
            ax.barh(y, xmax, color=SIM_COLORS[sim], height=0.5,
                    edgecolor="black", linewidth=0.8, zorder=3,
                    hatch="//", alpha=0.55, clip_on=True)
            ax.text(0.02, y, SI_NOTE, va="center", ha="left", fontsize=10,
                    zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_display(s) for s in sims])
    ax.set_xlabel(r"Combined $L_2$ error (m)")
    ax.set_title("Accuracy over 14 runs (lower is better)", pad=18)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.set_ylim(-0.5, len(sims) - 0.5)
    ax.set_xlim(0, xmax)


def main():
    gt = load_gt(DATA_DIR / f"{SAMPLE}.json")
    trajs = load_trajs(gt)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.2),
                             gridspec_kw={"width_ratios": [2, 2, 1.9],
                                          "wspace": 0.32})
    ax_xy, ax_z, ax_bar = axes
    panel_xy(ax_xy, trajs, gt)
    panel_z(ax_z, trajs, gt)
    panel_bar(ax_bar)

    handles, labels = ax_xy.get_legend_handles_labels()
    bbox_xy = ax_xy.get_position()
    bbox_z = ax_z.get_position()
    x_center = (bbox_xy.x0 + bbox_z.x1) / 2.0
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(x_center, 0.04),
               ncol=len(labels), fontsize=13, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    out = RESULTS_DIR / "paper_panels" / "box_sim_to_real_c3.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
