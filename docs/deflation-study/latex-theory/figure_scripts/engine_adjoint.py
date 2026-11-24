"""In-engine adjoint gradients with DPCR (engine analogue of Figure 4).

Left three panels: relative error of trajectory gradients (random velocity losses at
the middle and last step, w.r.t. the control targets of all steps before) against the
same backward pass with dense float64 adjoint solves, median and maximum, for the
engine's adjoint solve with Jacobi PCR, with DPCR (A_SS refactored at the saved state)
and with DPCR reusing the forward pass's factor (adjoint.reuse_deflation_factor). All
at the production budget (cap 16). Right panel: the chain, step-summed gradients against
central finite differences (tight forward pass), gradients with an unusable finite
difference left out; "exact adjoint" uses dense adjoint solves. The experiments are
test_scripts/deflation/engine_dpcr.py (adjoint_errors, adjoint_fd).

Needs the CUDA device. Takes about 40 minutes.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/engine_adjoint.py
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

from test_scripts.deflation.engine_dpcr import adjoint_errors, adjoint_fd
from test_scripts.deflation.engine_scenes import SCENES

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _bars(ax, names, med, top, fmt):
    x, w = np.arange(len(names)), 0.36
    for k, (values, color, label) in enumerate(((med, BLUE, "median"), (top, ORANGE, "max"))):
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), values, width=w, color=color, label=label, zorder=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt(v), ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.set_xticks(x, [n.replace(", ", ",\n") for n in names], fontsize=6.5)
    _style(ax)


def main():
    with wp.ScopedDevice("cuda:0"):
        errors = {name: adjoint_errors(name) for name in SCENES}
        fd = adjoint_fd()
    for scene, rows in errors.items():
        for variant, r in rows.items():
            print(f"{scene:8s} {variant:22s} " + "  ".join(f"{k} {v:.3g}" for k, v in r.items()))
    for variant, e in fd.items():
        print(f"chain FD {variant:22s} n {len(e)}  p50 {np.median(e):.4f}  max {e.max():.4f}")

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(SCENES) + 1, figsize=(6.3, 2.7))
    for ax, (scene, rows) in zip(axes, errors.items()):
        names = list(rows)
        _bars(ax, names, [rows[n]["p50"] for n in names], [rows[n]["max"] for n in names], lambda v: f"{v:.0e}")
        ax.set_yscale("log")
        ax.set_title(f"{SCENES[scene].label}\nvs. exact adjoint", fontsize=7.5)
    axes[0].set_ylabel("relative gradient error")
    names = list(fd)
    _bars(axes[-1], names, [100 * np.median(fd[n]) for n in names], [100 * fd[n].max() for n in names],
          lambda v: f"{v:.1f}")
    axes[-1].set_title(f"{SCENES['chain'].label}\nvs. finite differences [%]", fontsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
