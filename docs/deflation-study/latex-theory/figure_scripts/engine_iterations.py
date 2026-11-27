"""In-engine PCR iterations at the production budget (cap 16).

Median PCR iterations per Newton linear solve over the Newton solves of the last 80%
of steps, grouped by solver (Jacobi PCR, DPCR on the joint/control rows, DPCR on the
joint/control rows plus the 8 contact rows with the largest strength of connection),
one bar per scene. The dashed line is the cap. The experiment itself is
test_scripts/deflation/engine_dpcr.py (forward_budget); scenes: engine_scenes.py.

Needs the CUDA device. Takes about 1 minute.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/engine_iterations.py
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
CAP = 16


def main():
    with wp.ScopedDevice("cuda:0"):
        results = {name: forward_budget(name) for name in SCENES}
    for scene, rows in results.items():
        for variant, r in rows.items():
            print(f"{scene:8s} {variant:15s} iters_p50 {r['iters_p50']:.3g}  at_cap {r['at_cap']:.3g}")
    assert all(VARIANTS[v].max_iters == CAP for v in VARIANTS)

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    x, w = np.arange(len(VARIANTS)), 0.26
    for k, (scene, color) in enumerate(zip(SCENES, (BLUE, ORANGE, AQUA))):
        its = [results[scene][v]["iters_p50"] for v in VARIANTS]
        bars = ax.bar(x + (k - 1) * (w + 0.02), its, width=w, color=color, label=SCENES[scene].label, zorder=2)
        for b, v in zip(bars, its):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(CAP, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(x[-1] + 1.5 * w, CAP + 0.3, f"cap {CAP}", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, list(VARIANTS))
    ax.set_ylabel("PCR iterations (median)")
    ax.set_ylim(0, CAP + 3)
    ax.set_yticks(range(0, CAP + 1, 4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout()
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
