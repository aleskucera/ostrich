"""3-panel comparison figure for the box sim-to-real benchmark.

Reads ``results/sweep_<engine>.json`` from each engine and overlays the best
trajectory of each on the real total-station data for one representative run.

Panels:
  1) top-down XY (with box footprint)
  2) prism elevation Z vs time (the climb)
  3) combined L2 accuracy bar chart per engine

Usage:
    python experiments/1_sim_to_real_box/plot_results.py
    python experiments/1_sim_to_real_box/plot_results.py --run 18_10_33 --save my.png
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common_box import DATA_DIR, RESULTS_DIR, load_gt

SIM_COLORS = {"Axion": "#2196F3", "MuJoCo": "#E91E63",
              "Semi-Implicit": "#FF9800", "TinyDiffSim": "#607D8B", "Dojo": "#4CAF50"}
SIM_ORDER = ["Axion", "MuJoCo", "Semi-Implicit", "TinyDiffSim", "Dojo"]
ACCURACY_THRESHOLD = 0.5  # metres — box benchmark is tighter than exp-1's 1.0


def load_sweeps():
    out = {}
    for p in sorted(RESULTS_DIR.glob("sweep_*.json")):
        with open(p) as f:
            d = json.load(f)
        out[d["simulator"]] = d
    return out


def fmt_params(sim, bp):
    if sim == "Axion":
        return rf"$\mu_r$={bp['mu_rear']}, $\Delta t$={bp['dt']}"
    if sim == "MuJoCo":
        return rf"kv={bp['kv']:g}, $\mu$={bp['mu']}, $\Delta t$={bp['dt']}"
    return ", ".join(f"{k}={v}" for k, v in bp.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="18_10_33", help="run id suffix to overlay in panels 1,2")
    ap.add_argument("--save", default=str(RESULTS_DIR / "box_sim_to_real.png"))
    args = ap.parse_args()

    sweeps = load_sweeps()
    if not sweeps:
        print(f"No sweep_*.json in {RESULTS_DIR} — run sweep_axion / sweep_mujoco first.")
        return

    run_key = f"run_2026_05_20-{args.run}"
    gt = load_gt(DATA_DIR / f"{run_key}.json")
    box = gt["box"]
    real_x = np.asarray(gt["real"]["x"]); real_y = np.asarray(gt["real"]["y"])
    real_z = np.asarray(gt["real"]["z"]); real_t = np.asarray(gt["real"]["t"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                             gridspec_kw={"width_ratios": [2, 2, 2]})
    ax_xy, ax_z, ax_bar = axes

    ax_xy.plot(real_x, real_y, "k--", lw=1.8, label="Real", zorder=4)
    ax_z.plot(real_t, real_z, "k--", lw=1.8, label="Real", zorder=4)
    box_rect = plt.Rectangle(
        (box["center"][0] - box["half_extents"][0], box["center"][1] - box["half_extents"][1]),
        2 * box["half_extents"][0], 2 * box["half_extents"][1],
        color="gray", alpha=0.25, label="Box", zorder=1)
    ax_xy.add_patch(box_rect)

    for sim in [s for s in SIM_ORDER if s in sweeps]:
        d = sweeps[sim]
        per_run = d.get("best_per_run", {})
        if run_key not in per_run:
            continue
        sr = np.asarray(per_run[run_key]["sim_rel"])
        st = np.asarray(per_run[run_key]["sim_t_aligned"])
        c = SIM_COLORS.get(sim, "tab:gray")
        ax_xy.plot(sr[:, 0], sr[:, 1], "-", color=c, lw=1.5, label=sim, zorder=3)
        ax_z.plot(st, sr[:, 2], "-", color=c, lw=1.5, label=sim, zorder=3)

    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]")
    ax_xy.set_title(f"Top-down trajectory — {args.run}")
    ax_xy.axis("equal"); ax_xy.grid(alpha=0.3); ax_xy.legend(loc="best", fontsize=9)

    ax_z.set_xlabel("t [s]"); ax_z.set_ylabel("prism z rise [m]")
    ax_z.set_title(f"Climb — {args.run}")
    ax_z.grid(alpha=0.3); ax_z.legend(loc="best", fontsize=9)

    # Accuracy bars (combined L2 over the GT runs used by each sweep).
    sims = [s for s in SIM_ORDER if s in sweeps]
    errs = [sweeps[s]["best_error"] for s in sims]
    bps = [fmt_params(s, sweeps[s]["best_params"]) for s in sims]
    order = np.argsort(errs)
    sims_o = [sims[i] for i in order]; errs_o = [errs[i] for i in order]; bps_o = [bps[i] for i in order]
    y = np.arange(len(sims_o))
    colors = [SIM_COLORS.get(s, "tab:gray") for s in sims_o]
    ax_bar.barh(y, errs_o, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
    ax_bar.set_yticks(y); ax_bar.set_yticklabels(sims_o)
    ax_bar.invert_yaxis()
    ax_bar.axvline(ACCURACY_THRESHOLD, ls="--", color="red", alpha=0.6,
                   label=f"threshold ({ACCURACY_THRESHOLD} m)")
    for i, (e, bp) in enumerate(zip(errs_o, bps_o)):
        ax_bar.text(e + max(errs_o) * 0.02, i, f"{e:.3f}  ({bp})", va="center", fontsize=9)
    ax_bar.set_xlim(right=max(errs_o) * 1.7)
    ax_bar.set_xlabel("Combined 3D $L_2$ error [m]")
    ax_bar.set_title("Accuracy (lower is better)")
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.grid(axis="x", alpha=0.3, zorder=0)

    fig.suptitle(f"helhest_junior sim-to-real over a box ({len(sims)} engines)", fontsize=13)
    fig.tight_layout()
    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=140, bbox_inches="tight")
    print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
