"""Paper-ready figures for the box-obstacle experiments.

Mirrors the old `experiments/1_sim_to_real/plot_results.py` layout:
one wide multi-panel PNG per figure, shared legend at the bottom, serif fonts,
top/right spines off. Pairs with a separately-rendered scene viz PNG in LaTeX
via the same minipage pattern as the existing Fig 2 in the paper.

Outputs:
  results/paper_panels/box_sim_to_real.png   -- 3 panels (xy, z, bar)
  results/paper_panels/box_dt_stability.png  -- 1 panel (dt sweep)
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP1 = pathlib.Path(__file__).resolve().parent
EXP2 = EXP1.parent / "2_dt_stability_box"
OUT = EXP1 / "results" / "paper_panels"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXP1))

from common_box import DATA_DIR, load_gt  # noqa: E402

SIM_COLORS = {"Axion": "#2196F3", "MuJoCo": "#E91E63", "Semi-Implicit": "#FF9800",
              "Brax": "#4CAF50"}
SIM_ORDER = ["Axion", "MuJoCo", "Semi-Implicit"]  # benchmarked engines
# Brax is NOT shown here: it forward-tracks the box fine (spring, ~0.074 m), so it
# is excluded on GRADIENT grounds (footnote), not forward accuracy.
THRESHOLD = 0.5
RUN = "18_10_33"
RUN_KEY = f"run_2026_05_20-{RUN}"

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


def load_sweeps():
    out = {}
    for p in sorted((EXP1 / "results").glob("sweep_*.json")):
        d = json.load(open(p))
        out[d["simulator"]] = d
    return out


def load_dt_data():
    return json.load(open(EXP2 / "results" / "accuracy_vs_dt.json"))


def _display(name):
    if name == "Axion":
        return r"\textbf{Ostrich}"
    return {"Semi-Implicit": "Semi-Impl."}.get(name, name)


# ------------------------------- Figure 1 ------------------------------------

def panel_xy(ax, sweeps, gt):
    box = gt["box"]
    bx0 = box["center"][0] - box["half_extents"][0]
    by0 = box["center"][1] - box["half_extents"][1]
    bw = 2 * box["half_extents"][0]
    bh = 2 * box["half_extents"][1]
    ax.add_patch(plt.Rectangle((bx0, by0), bw, bh, color="gray", alpha=0.18,
                               zorder=1, ec="dimgray", lw=0.7, ls="--"))
    ax.text(bx0 + bw / 2, by0 + bh - 0.10, "obstacle", ha="center", va="top",
            fontsize=10, color="dimgray", style="italic", zorder=2)

    real_x = np.asarray(gt["real"]["x"])
    real_y = np.asarray(gt["real"]["y"])
    ax.plot(real_x, real_y, "k--", lw=1.6, label="Real robot", zorder=10)

    for sim in SIM_ORDER:
        if sim not in sweeps:
            continue
        per_run = sweeps[sim].get("best_per_run", {})
        if RUN_KEY not in per_run:
            continue
        sr = np.asarray(per_run[RUN_KEY]["sim_rel"])
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(sr[:, 0], sr[:, 1], "-", color=SIM_COLORS[sim], lw=1.1,
                label=_display(sim), zorder=zord)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(-0.1, 4.5); ax.set_ylim(-0.7, 0.7)
    ax.grid(True, alpha=0.3)


def panel_z(ax, sweeps, gt):
    t_lo, t_hi = 1.5, 9.0
    real_t = np.asarray(gt["real"]["t"])
    real_z = np.asarray(gt["real"]["z"])
    m = (real_t >= t_lo) & (real_t <= t_hi)
    ax.plot(real_t[m], real_z[m], "k--", lw=1.6, label="Real robot", zorder=10)

    for sim in SIM_ORDER:
        if sim not in sweeps:
            continue
        per_run = sweeps[sim].get("best_per_run", {})
        if RUN_KEY not in per_run:
            continue
        sr = np.asarray(per_run[RUN_KEY]["sim_rel"])
        st = np.asarray(per_run[RUN_KEY]["sim_t_aligned"])
        sel = (st >= t_lo) & (st <= t_hi)
        z = sr[:, 2]
        bl = (st >= t_lo) & (st <= 2.5)
        baseline = float(np.mean(z[bl])) if bl.any() else 0.0
        zord = 9 if sim == "MuJoCo" else 5
        ax.plot(st[sel], z[sel] - baseline, "-", color=SIM_COLORS[sim],
                lw=1.1, label=_display(sim), zorder=zord)

    ax.set_xlabel("time (s)"); ax.set_ylabel("prism $z$ rise (m)")
    ax.set_xlim(t_lo, t_hi)
    ax.grid(True, alpha=0.3)


def panel_bar(ax, sweeps):
    BAR_ORDER = ["Axion", "MuJoCo", "Semi-Implicit"]
    sims = [s for s in BAR_ORDER if s in sweeps]
    errs = [sweeps[s]["best_error"] for s in sims]
    dts = [sweeps[s]["best_params"]["dt"] for s in sims]
    yaws = [float(np.mean([v.get("yaw_rmse_deg", np.nan)
                           for v in sweeps[s]["best_per_run"].values()]))
            for s in sims]
    # reverse so first list entry sits at the top after barh draws bottom-up
    sims = sims[::-1]; errs = errs[::-1]; dts = dts[::-1]; yaws = yaws[::-1]
    colors = [SIM_COLORS[s] for s in sims]

    y_pos = np.arange(len(sims))
    bars = ax.barh(y_pos, errs, color=colors, height=0.5,
                   edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_display(s) for s in sims])
    ax.set_xlabel(r"Combined $L_2$ error (m)")
    ax.set_title("Accuracy (lower is better)", pad=18)
    ax.grid(True, axis="x", alpha=0.3, zorder=0)
    ax.set_ylim(-0.5, len(sims) - 0.5)

    for bar, err, dt in zip(bars, errs, dts):
        cy = bar.get_y() + bar.get_height() / 2
        ax.text(err + max(errs) * 0.04, cy,
                rf" {err:.3f}  ($\Delta t={dt}$\,s)",
                va="center", ha="left", fontsize=11)

    ax.set_xlim(right=max(errs) * 2.2)


def make_fig1(sweeps, gt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.2),
                             gridspec_kw={"width_ratios": [2, 2, 1.9],
                                          "wspace": 0.32})
    ax_xy, ax_z, ax_bar = axes
    panel_xy(ax_xy, sweeps, gt)
    panel_z(ax_z, sweeps, gt)
    panel_bar(ax_bar, sweeps)

    handles, labels = ax_xy.get_legend_handles_labels()
    bbox_xy = ax_xy.get_position()
    bbox_z = ax_z.get_position()
    x_center = (bbox_xy.x0 + bbox_z.x1) / 2.0
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(x_center, 0.04),
               ncol=len(labels), fontsize=13, frameon=False)
    plt.subplots_adjust(bottom=0.22)
    out = OUT / "box_sim_to_real.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ------------------------------- Figure 2 ------------------------------------

def make_fig2(dt_data):
    results = dt_data["results"]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    ceiling = 10.0
    usable_max = {}
    for sim in SIM_ORDER:
        if sim not in results:
            continue
        c = SIM_COLORS[sim]
        rows = sorted(results[sim], key=lambda r: r["dt"])
        dts = np.array([r["dt"] for r in rows])
        errs = np.array([r["mean_combined_with_yaw"] for r in rows])
        stable = np.array([r["all_stable"] for r in rows])
        usable = stable & (errs <= THRESHOLD)
        above = stable & (errs > THRESHOLD)
        broken = ~stable

        plot_y = np.where(stable, errs, ceiling)
        ax.plot(dts, plot_y, "-", color=c, lw=1.4, alpha=0.55,
                label=_display(sim), zorder=2)
        ax.plot(dts[usable], errs[usable], "o", mfc=c, mec="black", ms=7,
                mew=0.5, ls="", zorder=4)
        if above.any():
            ax.plot(dts[above], errs[above], "o", mfc="white", mec=c, ms=7,
                    mew=1.4, ls="", zorder=4)
        if broken.any():
            ax.plot(dts[broken], np.full(broken.sum(), ceiling), "X",
                    mfc=c, mec="black", ms=10, mew=0.6, ls="", zorder=4)
        if usable.any():
            usable_max[sim] = float(dts[usable].max())

    ax.axhline(THRESHOLD, ls="--", color="dimgray", alpha=0.7, lw=1.0)
    ax.text(3.5e-3, THRESHOLD * 1.22,
            rf"usable threshold ({THRESHOLD:.1f}\,m)",
            fontsize=10, color="dimgray", ha="center", va="bottom")

    if "Axion" in usable_max and "MuJoCo" in usable_max:
        ratio = usable_max["Axion"] / usable_max["MuJoCo"]
        y_arrow = 0.030
        ax.annotate("", xy=(usable_max["Axion"] * 1.03, y_arrow),
                    xytext=(usable_max["MuJoCo"] * 0.97, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.6))
        x_mid = np.sqrt(usable_max["Axion"] * usable_max["MuJoCo"])
        ax.text(x_mid, y_arrow * 0.55,
                rf"$\mathbf{{\sim {int(round(ratio))}\times}}$ "
                rf"\textbf{{larger usable}} $\boldsymbol{{\Delta t}}$",
                ha="center", va="top", fontsize=11)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"timestep $\Delta t$ (s)")
    ax.set_ylabel(r"combined $L_2$ error (m)")
    ax.set_ylim(8e-3, 30)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.08)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.04),
               ncol=len(labels), fontsize=13, frameon=False)
    plt.subplots_adjust(bottom=0.20)

    out = OUT / "box_dt_stability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    sweeps = load_sweeps()
    dt_data = load_dt_data()
    gt = load_gt(DATA_DIR / f"{RUN_KEY}.json")
    make_fig1(sweeps, gt)
    make_fig2(dt_data)


if __name__ == "__main__":
    main()
