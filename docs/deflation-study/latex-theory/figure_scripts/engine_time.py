"""In-engine GPU time per Newton iteration (engine analogue of Figure 3).

Median GPU time per Newton iteration from the engine's per-component profiler (one
engine step captured as a CUDA graph and replayed, at five states along the rollout),
stacked: the preconditioner phase (for DPCR: row selection, A Z, Cholesky factor and
E⁻¹ of A_SS, diag(A_F)) and the PCR solve, at the production budget (cap 16), for one
world and for 64 worlds. The experiment itself is test_scripts/deflation/engine_dpcr.py
(forward_time); scenes: engine_scenes.py.

Needs the CUDA device. Takes about 20 minutes.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/engine_time.py
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

from test_scripts.deflation.engine_dpcr import VARIANTS, forward_time
from test_scripts.deflation.engine_scenes import SCENES

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"
WORLDS = (1, 64)


def main():
    with wp.ScopedDevice("cuda:0"):
        results = {(W, name): forward_time(name, W) for W in WORLDS for name in SCENES}
    for (W, scene), rows in results.items():
        for variant, r in rows.items():
            print(f"W={W:3d} {scene:8s} {variant:15s} " + "  ".join(f"{k} {v:.3g} ms" for k, v in r.items()))

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(len(WORLDS), len(SCENES), figsize=(6.3, 4.4))
    x = np.arange(len(VARIANTS))
    for row, W in enumerate(WORLDS):
        for col, scene in enumerate(SCENES):
            ax = axes[row, col]
            r = [results[W, scene][v] for v in VARIANTS]
            solve = np.array([q["cr_solve"] for q in r])
            setup = np.array([q["preconditioner"] for q in r])
            ax.bar(x, solve, width=0.6, color=BLUE, label="PCR solve", zorder=2)
            ax.bar(x, setup, width=0.6, bottom=solve, color=ORANGE, label="preconditioner / deflation setup",
                   zorder=2)
            for xi, total in zip(x, solve + setup):
                ax.text(xi, total, f"{total:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
            ax.set_title(f"{SCENES[scene].label}, W = {W}", fontsize=8)
            ax.set_xticks(x, [v.replace(" + ", "\n+ ") for v in VARIANTS], fontsize=7)
            ax.margins(y=0.15)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTED)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
        axes[row, 0].set_ylabel("ms per Newton iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
