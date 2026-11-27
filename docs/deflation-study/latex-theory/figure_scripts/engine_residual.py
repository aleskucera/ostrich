"""In-engine relative residual of the Newton linear solves at the production budget (cap 16).

Relative residual ‖b − A x‖/‖b‖ reached by the Newton linear solves of the last 80% of
steps, grouped by solver (Jacobi PCR, DPCR on the joint/control rows, DPCR on the
joint/control rows plus the 8 contact rows with the largest strength of connection),
one bar per scene, as in engine_iterations.py. Bar: median; tick: 90th percentile. The
dashed line is the PCR tolerance. The experiment itself is
test_scripts/deflation/engine_dpcr.py (forward_budget); scenes: engine_scenes.py.

Needs the CUDA device. Takes about 1 minute.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/engine_residual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root, for test_scripts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from _common import save_fig

from test_scripts.deflation.engine_dpcr import VARIANTS, forward_budget
from test_scripts.deflation.engine_scenes import SCENES

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#00b894", "#2b2b29", "#8a8980", "#e4e3dd"


def main():
    with wp.ScopedDevice("cuda:0"):
        results = {name: forward_budget(name) for name in SCENES}
    for scene, rows in results.items():
        for variant, r in rows.items():
            print(f"{scene:8s} {variant:15s} res_p50 {r['res_p50']:.3g}  res_p90 {r['res_p90']:.3g}"
                  f"  iters_p50 {r['iters_p50']:.3g}  at_cap {r['at_cap']:.3g}")
    tols = {VARIANTS[v].tol for v in VARIANTS}
    assert len(tols) == 1
    tol = tols.pop()

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(4.5, 2.9))
    x, w = np.arange(len(VARIANTS)), 0.26
    for k, (scene, color) in enumerate(zip(SCENES, (BLUE, ORANGE, AQUA))):
        xs = x + (k - 1) * (w + 0.02)
        p50 = [results[scene][v]["res_p50"] for v in VARIANTS]
        p90 = [results[scene][v]["res_p90"] for v in VARIANTS]
        ax.bar(xs, p50, width=w, color=color, label=SCENES[scene].label, zorder=2)
        ax.hlines(p90, xs - w / 2, xs + w / 2, color=INK, lw=1.2, zorder=3)
    ax.axhline(tol, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1.0)
    ax.set_xticks(x, list(VARIANTS))
    ax.set_ylabel(r"relative residual $\|b - Ax\|/\|b\|$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK, lw=1.2), plt.Line2D([], [], color=MUTED, lw=0.8, ls="--")]
    labels += ["90th percentile", f"PCR tolerance {tol:.0e}".replace("e-0", "e-")]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout()
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
