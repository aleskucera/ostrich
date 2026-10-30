"""GPU time per solve of deflation (DPCR) with the real Z sources, both Helhest scenes.

Companion of deflation_z_sources.py (same sources, same iteration counts). Single
world (W = 1), median over the held-out systems. Each bar is stacked:

  solve    fixed cost + setup (A·Z, E, Cholesky, projections) + iterations × per-
           iteration time. Fixed and per-iteration times are measured on the GPU with
           the engine's PCRSolver and the deflation Warp kernels, captured as a CUDA
           graph (test_scripts/deflation_hybrid.gpu_costs). Iterations are those of
           Figure 2 (float64, tol 1e-5, cap 64), so the time is linear in them.
  Z build  producing Z on the GPU, float32, median per system:
             cross-step,    Warp kernels of test_scripts/deflation/deflation_zbuild.py: chained
             recycle-exact  Chebyshev-filtered refinement of the previous Z, with the
                            fewest filter steps whose DPCR iterations match the exact
                            eigenvectors (cross-step: once per time step, shared by the
                            systems of that step; recycle-exact: per system);
             harmonic m     Warp kernels of deflation_zbuild.py from the m Krylov vectors
                            and W = A·V of the previous solve (both come free from it);
                            m = 16 needs no eigenproblem, m = 32 a 32×32 one;
             joint/control  nothing to build;
             + 8 rows       the strength-of-connection scores and the top-8 selection
                            (eager torch, dense, an upper bound).

The dotted line is Jacobi PCR, the current ostrich solver, with the same iteration
convention. Needs data/baselines/*.npz (see docs/deflation-study/PLAN.md step 0)
and a CUDA GPU.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/deflation_z_sources_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repo root, for test_scripts

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp
from _common import save_fig
from deflation_z_sources import K_C, SOURCES, scene_iterations

from test_scripts.deflation.deflation import DT, SCENES
from test_scripts.deflation.deflation_cost import DEV, time_torch
from test_scripts.deflation.deflation_hybrid import N_JC, gpu_costs
from test_scripts.deflation.deflation_zbuild import measure_scene
from test_scripts.precond_lab import load_systems

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"
K = 16
DEFLATION_K = {"cross-step (N-1)": K, "recycle-exact (prev)": K, "harmonic m=16": K,
               "harmonic m=32": K, "joint/control": N_JC, f"strength A (all j) {K_C}": N_JC + K_C}


def scaled_gpu(A):
    """Jacobi-scaled A as a float32 GPU tensor."""
    A = A.to(DEV, torch.float32)
    D = torch.diagonal(A).rsqrt()
    return D.unsqueeze(1) * A * D.unsqueeze(0)


def zbuild_times(path, S, va):
    """ms per system to build Z for each source."""
    out = measure_scene(path)
    As = scaled_gpu(S[sorted(va, key=lambda j: S[j].n)[len(va) // 2]].A)

    def strength():
        M = As.abs()
        M.fill_diagonal_(0.0)
        torch.topk(M.max(1).values[N_JC:], K_C)

    return out | {"joint/control": 0.0, f"strength A (all j) {K_C}": time_torch(strength)}


def main():
    wp.init()
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(SCENES), figsize=(6.3, 3.1), sharey=True)
    x = np.arange(len(SOURCES))
    for ax, (label, path) in zip(axes, SCENES.items()):
        d = np.load(path, allow_pickle=True)
        S = load_systems(path=path, dtype=DT)
        va = list(range(int(0.8 * len(S)), len(S)))
        jac_its, its = scene_iterations(path)
        print(f"{label}: GPU fixed / per-iteration cost (W = 1)")
        cost = gpu_costs(d, sorted(set(DEFLATION_K.values())))
        zb = zbuild_times(path, S, va)
        solve = [cost[DEFLATION_K[s]][0] + its[s] * cost[DEFLATION_K[s]][1] for s in SOURCES]
        build = [zb[s] for s in SOURCES]
        jac = cost[0][0] + jac_its * cost[0][1]
        ax.bar(x, solve, width=0.6, color=BLUE, label="solve (setup + iterations)")
        ax.bar(x, build, width=0.6, bottom=solve, color=ORANGE, label="building $\\mathbf{Z}$")
        for xi, s, z in zip(x, solve, build):
            ax.annotate(f"{s + z:.2f}", (xi, s + z), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.axhline(jac, color=INK, ls=":", lw=1, label="Jacobi PCR (current ostrich)")
        ax.annotate(f"{jac:.2f}", (1, jac), xycoords=("axes fraction", "data"), xytext=(2, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK)
        ax.set_title({"obstacle": "obstacle scene", "surface": "terrain scene"}[label])
        ax.set_xticks(x, SOURCES.values(), fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        print(f"{label}: Jacobi {jac:.3f} ms | " + ", ".join(
            f"{s} {a:.3f}+{b:.3f}" for s, a, b in zip(SOURCES, solve, build)))
    axes[0].set_ylabel("time per solve (ms, median, $W = 1$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, __file__)


if __name__ == "__main__":
    main()
