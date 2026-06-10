"""Turbo colorbar keyed to optimization-iteration number, for the paper figure.

Builds a standalone colorbar PNG (transparent + white-bg) using the SAME turbo
ramp the trajectories use, with ticks at each logged iterate labeled by its real
optimization-iteration number (from the npz `iter_indices`). Composite it next
to the render in the paper.

    python examples/helhest_junior/gradient/make_colorbar.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = pathlib.Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Exact turbo ramp used for the trajectories (mirrors RAMPS["turbo"] in
# examples/helhest/gradient/gradient_figure_to_blender.py — kept in sync here so
# this colorbar doesn't need to import that bpy-dependent module).
TURBO = [
    (0.19, 0.07, 0.55),
    (0.10, 0.50, 0.99),
    (0.10, 0.90, 0.72),
    (0.55, 0.99, 0.23),
    (0.98, 0.73, 0.10),
    (0.88, 0.14, 0.10),
]
cmap = LinearSegmentedColormap.from_list("turbo_fig", TURBO, N=256)

# Real optimization-iteration numbers of the logged iterates.
data = np.load(HERE / "data" / "junior_traj.npz", allow_pickle=True)
iter_indices = [int(i) for i in data["iter_indices"]]
n = len(iter_indices)
# Colors were assigned by RANK: iterate r -> turbo(r/(n-1)). Place a tick at each
# rank position and label it with that iterate's real iteration number.
ranks = [r / (n - 1) for r in range(n)]


def make_bar(orientation: str, figsize, fname_stem: str):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation=orientation)
    cb.set_ticks(ranks)
    cb.set_ticklabels([str(i) for i in iter_indices], fontsize=7)
    cb.set_label("optimization iteration  (early → late / best)", fontsize=9)
    cb.outline.set_linewidth(0.6)
    for transparent, suffix, fc in ((True, "", None), (False, "_white", "white")):
        fig.savefig(OUT / f"{fname_stem}{suffix}.png", transparent=transparent,
                    facecolor=fc, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


make_bar("horizontal", (7.0, 0.62), "colorbar_iteration_h")
make_bar("vertical", (0.7, 6.0), "colorbar_iteration_v")
print(f"iter_indices: {iter_indices}")
print(f"wrote colorbar_iteration_h[_white].png and colorbar_iteration_v[_white].png to {OUT}")
