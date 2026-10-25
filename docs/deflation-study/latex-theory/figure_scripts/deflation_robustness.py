"""Robustness of deflation vs the additive two-level preconditioner (step 1d).

Median PCR iterations on the held-out split against the principal angle θ between
span(Z) and the exact bottom-16 eigenvectors of the Jacobi-scaled A, for random
and spectral (neighbouring-eigenvector) errors, on both Helhest scenes. The
experiment itself is test_scripts/deflation_robustness.py (run_scene); this
script only re-plots it. The real Z sources are in deflation_z_sources.py.

Needs data/baselines/helhest_systems.npz and helhest_surface_systems.npz
(test_scripts/dump_linear_systems.py, see docs/deflation-study/PLAN.md step 0).
Takes about 30 s on CPU.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/deflation_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root, for test_scripts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _common import save_fig

from test_scripts.deflation import SCENES
from test_scripts.deflation_robustness import ANGLES, run_scene
from test_scripts.recycle_probe import pct

RED, BLUE, INK, MUTED, GRID = "#e34948", "#2a78d6", "#2b2b29", "#8a8980", "#e4e3dd"


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(SCENES), figsize=(6.3, 3.1), sharey=True)
    x = np.arange(len(ANGLES))
    for ax, (label, path) in zip(axes, SCENES.items()):
        jac, curves, _ = run_scene(label, path)
        for (kind, method), arr in curves.items():
            ax.plot(x, np.array(arr)[:, 0], marker="o", ms=2.5, lw=1.2,
                    color=RED if method == "additive" else BLUE,
                    ls="-" if kind == "spectral" else "--",
                    label=f"{method}, {kind} error")
        ax.axhline(pct(jac, 50), color=INK, ls=":", lw=1, label="Jacobi median")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x, [f"{a:g}" for a in ANGLES])
        ax.set_xlabel(r"principal angle $\theta$ to exact subspace (°)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("PCR iterations (median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save_fig(fig, __file__)


if __name__ == "__main__":
    main()
