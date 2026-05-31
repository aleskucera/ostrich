"""Shared paper-figure style for all box-experiment plots.

Goal: every plot looks like the *same type* of figure. All text is LaTeX/serif
at a single on-page size (8pt, the IEEE caption size), all engines use one
color/marker map, and lines/markers share one weight. Figures are authored at
the paper's true display width so \\includegraphics[width=\\columnwidth] does not
rescale them (8pt authored = 8pt printed), and saved at 600 dpi for crisp text.

Usage in a generator:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import paper_style as ps

    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3))
    ...
    ps.save(fig, local_path, paper_path)   # 600 dpi to both
"""
import matplotlib.pyplot as plt

# Exact widths of the ieeeconf layout, measured via \the\columnwidth (pt/72.27).
COL_W = 3.40       # \columnwidth  (single-column figures)
TEXT_W = 7.00      # \textwidth    (full-width figure*)
SIM2REAL_W = 5.18  # 0.74*\textwidth: the sim-to-real plot minipage

DPI = 600

# Engine -> color, shared across every figure. Multiple aliases map to one hue
# (MuJoCo/MJX share pink; the data key may be "Axion" but it renders as Ostrich).
COLORS = {
    "Axion": "#2196F3", "Ostrich": "#2196F3",
    "MuJoCo": "#E91E63", "MJX": "#E91E63", "MJX-grad": "#E91E63",
    "Semi-Implicit": "#FF9800",
    "TinyDiffSim": "#607D8B", "Brax": "#4CAF50",
}
MARKERS = {
    "Axion": "o", "Ostrich": "o",
    "MuJoCo": "s", "MJX": "s", "MJX-grad": "s",
    "Semi-Implicit": "^", "TinyDiffSim": "D", "Brax": "v",
}
# One weight everywhere (figures are authored 1:1, so these are printed sizes).
LINEWIDTH = 1.1
MARKERSIZE = 3.5

RCPARAMS = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,
    "legend.frameon": False,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


def apply():
    """Install the shared rcParams. Call once at the top of each generator."""
    plt.rcParams.update(RCPARAMS)


def save(fig, local_path, paper_path=None):
    """Save at the shared 600 dpi to the local results path and (if given) the
    paper figures directory."""
    fig.savefig(local_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {local_path}")
    if paper_path is not None:
        fig.savefig(paper_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved {paper_path}")
