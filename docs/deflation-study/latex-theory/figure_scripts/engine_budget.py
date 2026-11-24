"""In-engine DPCR at the production budget (engine analogue of Figure 2).

Relative residual ‖b − A x‖/‖b‖ that the engine's Newton linear solves reach with a cap
of 16 PCR iterations, median and 90th percentile over the Newton solves of the last 80%
of steps, for Jacobi PCR (the engine today), DPCR on the joint/control rows, and DPCR on
the joint/control rows plus the 8 contact rows with the largest strength of connection.
The labels give the share of solves that reach the cap. The experiment itself is
test_scripts/deflation/engine_dpcr.py (forward_budget); scenes: engine_scenes.py.

Needs the CUDA device. Takes about 10 minutes.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/engine_budget.py
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

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"


def main():
    with wp.ScopedDevice("cuda:0"):
        results = {name: forward_budget(name) for name in SCENES}
    for scene, rows in results.items():
        for variant, r in rows.items():
            print(f"{scene:8s} {variant:15s} " + "  ".join(f"{k} {v:.3g}" for k, v in r.items()))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(SCENES), figsize=(6.3, 2.6), sharey=True)
    x, w = np.arange(len(VARIANTS)), 0.36
    for ax, (scene, rows) in zip(axes, results.items()):
        r = [rows[v] for v in VARIANTS]
        ax.bar(x - (w + 0.02) / 2, [q["res_p50"] for q in r], width=w, color=BLUE, label="median", zorder=2)
        ax.bar(x + (w + 0.02) / 2, [q["res_p90"] for q in r], width=w, color=ORANGE, label="90th percentile", zorder=2)
        for xi, q in zip(x, r):
            ax.text(xi, q["res_p90"] * 1.6, f"cap {100 * q['at_cap']:.0f}%", ha="center", fontsize=6.5, color=INK)
        ax.set_yscale("log")
        ax.set_title(SCENES[scene].label)
        ax.set_xticks(x, [v.replace(" + ", "\n+ ") for v in VARIANTS], fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"relative residual $\|b - Ax\|/\|b\|$")
    axes[0].set_ylim(top=1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
