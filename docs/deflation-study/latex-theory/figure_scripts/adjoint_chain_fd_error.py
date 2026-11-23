"""Gradient error against finite differences on a contact-free chain (idea 7.2).

Median and maximum relative error of the trajectory gradient for two adjoint
solves at the engine's budget (cap 16): the engine's Jacobi PCR, and DPCR with
the joint rows eliminated and diag(A_F) as the preconditioner. Scene: 12-link
chain, link masses alternating by 100, no contacts, dt 0.01, 40 steps, tight
forward pass. Gradients whose FD reference is unusable (eps 1e-2 and 1e-3
disagree by 45% or more, or FD is exactly 0) are left out. The experiment itself is
test_scripts/deflation/adjoint_articulated.py (fd_errors); this script only
re-plots it.

Needs the CUDA device. Takes about 25 minutes.

Run:
    python docs/deflation-study/latex-theory/figure_scripts/adjoint_chain_fd_error.py
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

from test_scripts.deflation.adjoint_articulated import build_chain, fd_errors
from test_scripts.deflation.adjoint_trajectory import make_engine

LINKS, MASS_RATIO, STEPS, DT = 12, 100.0, 40, 0.01
MAX_FD_SPREAD = 0.45
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#8a8980", "#e4e3dd"


def main():
    with wp.ScopedDevice("cuda:0"):
        model = build_chain(LINKS, MASS_RATIO)
        _, rows = fd_errors(model, make_engine(model, STEPS, "tight"), LINKS, STEPS, DT)
    use = [errs for _, _, spread, errs in rows if spread < MAX_FD_SPREAD]
    print(f"{len(use)} of {len(rows)} gradients have a usable FD reference")
    for name, T, spread, errs in rows:
        print(f"  {name:10s} T={T:2d} spread {spread:7.3f}  "
              + "  ".join(f"{s} {e:.3f}" for s, e in errs.items()))
    eng = 100 * np.array([e["engine"] for e in use])
    dpcr = 100 * np.array([e["joints_schur"] for e in use])

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    groups = ["Jacobi PCR\n(engine today)", "DPCR, joint rows\neliminated"]
    x, w = np.arange(len(groups)), 0.34
    for k, (label, values, color) in enumerate((("median", (np.median(eng), np.median(dpcr)), BLUE),
                                                ("max", (eng.max(), dpcr.max()), ORANGE))):
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), values, width=w, color=color, label=label, zorder=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel("gradient error vs. FD [%]", color=INK)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK)
    fig.tight_layout()
    print(save_fig(fig, __file__))


if __name__ == "__main__":
    main()
