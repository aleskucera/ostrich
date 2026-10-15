r"""Shared print-size style for the paper's figures (Figs. 2-6).

Each generator draws on its own canvas and LaTeX then scales the PNG to the
column, so a font size written in a script is NOT the size the reader sees.
This module inverts that: you declare the size you want IN PRINT and it sets
the rcParams for whatever canvas the script happens to use.

    import paper_style
    s = paper_style.apply(drawn_in=7.5)          # canvas width in inches
    ax.plot(..., lw=paper_style.LW * s, ms=paper_style.MS * s)

`s` is the factor LaTeX will divide the drawing by, so multiplying every
hand-set line width and marker size by it keeps them in print units too.

Measured from main.pdf on 2026-09-04 (pdfimages -list, printed width =
pixels / x-ppi): \columnwidth is 3.40 in and \textwidth is 7.00 in. The
printed sizes below are the ones the author approved on Fig. 5.
"""

COLUMN_IN = 3.40   # \columnwidth  (ieeeconf, letterpaper, 10 pt, conference)
TEXT_IN = 7.00     # \textwidth

# Sizes as they appear on the printed page, in points. IEEE sets captions at
# 8 pt; these sit just under that.
PRINT = {
    "base": 6.5,
    "label": 6.5,
    "title": 6.5,
    "tick": 6.0,
    "legend": 6.0,
    "annot": 5.8,
}
LW = 0.65   # curve line width, printed pt
MS = 2.50   # data marker size, printed pt
MEW = 0.42  # marker edge width, printed pt
GRID_MAJOR = dict(alpha=0.35, lw=0.30)
GRID_MINOR = dict(alpha=0.12, lw=0.20)

# palette and marker shapes shared by Figs. 2-6
BLUE, CRIMSON, ORANGE = "#2196F3", "#E91E63", "#FF9800"
COLORS = {"Ostrich": BLUE, "MuJoCo": CRIMSON, "MJX": CRIMSON,
          "Semi-Implicit": ORANGE}
MARKERS = {"Ostrich": "o", "MuJoCo": "s", "MJX": "s", "Semi-Implicit": "^"}


def scale(drawn_in, printed_in=COLUMN_IN):
    """Factor by which LaTeX shrinks a `drawn_in`-wide canvas to the page."""
    return drawn_in / printed_in


def apply(drawn_in, printed_in=COLUMN_IN, usetex=True):
    """Set rcParams for a canvas `drawn_in` wide that prints `printed_in` wide.

    Returns the scale factor, to be applied to any hand-set lw/markersize.
    """
    import matplotlib.pyplot as plt

    s = scale(drawn_in, printed_in)
    plt.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": PRINT["base"] * s,
        "axes.labelsize": PRINT["label"] * s,
        "axes.titlesize": PRINT["title"] * s,
        "xtick.labelsize": PRINT["tick"] * s,
        "ytick.labelsize": PRINT["tick"] * s,
        "legend.fontsize": PRINT["legend"] * s,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return s
