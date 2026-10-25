"""Iterations of deflation (DPCR) with real Z sources, both Helhest scenes.

Median PCR iterations on the held-out split (last 20% of systems) against the
Jacobi PCR median, which is what the ostrich solver does today. Sources:
  * the k = 16 spectral sources from test_scripts/deflation_ab.py (z_sources);
  * joint/control: unit vectors on the 18 joint/control rows, i.e. an exact
    solve of the joint block + PCR on the Schur complement of the contact rows;
  * joint/control + K_C contact rows, also eliminated as unit vectors, picked
    by the classical AMG strength of connection max_{j≠r} |a_rj| / √(a_rr a_jj)
    (test_scripts/deflation_rowsel.py, rule "strength A (all j)"). Needs only
    the entries of A, no eigensolve.

Needs data/baselines/helhest_systems.npz and helhest_surface_systems.npz
(test_scripts/dump_linear_systems.py, see docs/deflation-study/PLAN.md step 0).

Run:
    python docs/deflation-study/latex-theory/figure_scripts/deflation_z_sources.py
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

from test_scripts.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation_ab import z_sources
from test_scripts.deflation_rowsel import sources as rowsel_sources
from test_scripts.precond_lab import jacobi_apply, load_systems, pcr
from test_scripts.recycle_probe import pct

BLUE, INK, MUTED, GRID = "#2a78d6", "#2b2b29", "#8a8980", "#e4e3dd"
K_C = 8     # strongest contact rows added to joint/control (deflation_rowsel.KCS = 4, 8, 12)
SOURCES = {"cross-step (N-1)": "cross-step", "recycle-exact (prev)": "recycle-exact",
           "harmonic m=16": "harmonic $m{=}16$",
           "harmonic m=32": "harmonic $m{=}32$", "joint/control": "joint/control",
           f"strength A (all j) {K_C}": f"joint/control + {K_C} rows"}


def scene_iterations(path):
    """Jacobi median and the deflation median for each source, held-out split."""
    S = load_systems(path=path, dtype=DT)
    va = list(range(int(0.8 * len(S)), len(S)))
    jac = pct([pcr(S[i].A, S[i].b, jacobi_apply(S[i].A))[1] for i in va], 50)
    d = np.load(path, allow_pickle=True)
    src = z_sources(S, d, va) | rowsel_sources(S, d, va)
    its = {name: pct([deflated_pcr(S[i].A, S[i].b, Z, jacobi_apply(S[i].A))[1]
                      for i, Z in src[name].items()], 50) for name in SOURCES}
    return jac, its


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, len(SCENES), figsize=(6.3, 3.1), sharey=True)
    x = np.arange(len(SOURCES))
    for ax, (label, path) in zip(axes, SCENES.items()):
        jac, its = scene_iterations(path)
        vals = [its[name] for name in SOURCES]
        ax.bar(x, vals, width=0.6, color=BLUE, label="deflation (DPCR), median")
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:g}", (xi, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
        ax.axhline(jac, color=INK, ls=":", lw=1, label="Jacobi median (current ostrich)")
        ax.set_title(label)
        ax.set_xticks(x, SOURCES.values(), fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        print(f"{label}: Jacobi {jac:g} | " + ", ".join(f"{k} {v:g}" for k, v in its.items()))
    axes[0].set_ylabel("PCR iterations (median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_fig(fig, __file__)


if __name__ == "__main__":
    main()
