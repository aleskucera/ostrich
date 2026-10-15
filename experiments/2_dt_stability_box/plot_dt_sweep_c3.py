"""dt-vs-error figure from the c3 sweep (4 trajectories of the new SIV-A
dataset) — plateau-departure presentation.

Instead of an absolute usability threshold, each engine's plateau is the h
range where its combined error stays within DEPART_FACTOR of its own
small-h floor (the model-error floor, h-invariant). The plateau edge is
where integration error takes over. Dashed engine-colored floor segments
span each plateau; the arrow compares Ostrich's and MuJoCo's edges.

Usage:
    python experiments/2_dt_stability_box/plot_dt_sweep_c3.py
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paper_style as ps  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
DEPART_FACTOR = 2.0
CRASHED_MARKER_Y = 12.0

# Canvas is unchanged (7.5 x 3.2 in at 265 dpi -> 1987 x 848 px, the size the
# paper already includes); paper_style converts the shared PRINTED text sizes
# into this canvas's font sizes and returns the factor LaTeX shrinks it by, so
# hand-set line widths and marker sizes are given in printed points too.
FIG_W, FIG_H, DPI = 7.5, 3.2, 265
S = ps.apply(drawn_in=FIG_W)

ENGINES = {  # file key -> (label, color, marker, zorder)
    "ostrich": (r"\textbf{Ostrich}", ps.COLORS["Ostrich"], ps.MARKERS["Ostrich"], 5),
    "mujoco": ("MuJoCo", ps.COLORS["MuJoCo"], ps.MARKERS["MuJoCo"], 4),
    "semi_implicit": ("Semi-Impl.", ps.COLORS["Semi-Implicit"],
                      ps.MARKERS["Semi-Implicit"], 3),
}


SKIP = {}  # per-engine h values to exclude, e.g. {"ostrich": {0.005}}


def load(key):
    with open(RESULTS / f"dt_sweep_c3_{key}.json") as f:
        rows = json.load(f)["rows"]
    ok = [(r["h"], r["mean_combined"]) for r in rows
          if r["n_diverged"] == 0 and r["h"] not in SKIP.get(key, ())]
    crashed = [r["h"] for r in rows if r["n_diverged"] > 0]
    return sorted(ok), sorted(crashed)


def plateau_edge(ok):
    floor = min(e for _, e in ok)
    inside = [h for h, e in ok if e <= DEPART_FACTOR * floor]
    return floor, max(inside)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    edges = {}

    for key, (label, color, marker, z) in ENGINES.items():
        ok, crashed = load(key)
        hs = [h for h, _ in ok]
        errs = [e for _, e in ok]
        ax.plot(hs, errs, color=color, marker=marker, linewidth=ps.LW * S,
                markersize=ps.MS * S, label=label, zorder=z)

        floor, edge = plateau_edge(ok)
        edges[key] = edge
        # plateau-extent bracket below the curve (annotation layer, offset
        # from the data so it never hides markers)
        by = floor / 1.45
        ax.plot([hs[0], edge], [by, by], color=color, linewidth=0.85 * ps.LW * S,
                alpha=0.9, zorder=2)
        for x in (hs[0], edge):
            ax.plot([x, x], [by / 1.12, by * 1.12], color=color,
                    linewidth=0.85 * ps.LW * S, alpha=0.9, zorder=2)

        if crashed:
            xs = [hs[-1]] + crashed
            ys = [errs[-1]] + [CRASHED_MARKER_Y] * len(crashed)
            ax.plot(xs, ys, color=color, linestyle="--", linewidth=0.6 * ps.LW * S,
                    alpha=0.55, zorder=2)
            ax.plot(crashed, [CRASHED_MARKER_Y] * len(crashed), "x",
                    color=color, markersize=1.33 * ps.MS * S,
                    markeredgewidth=0.95 * ps.LW * S, zorder=6)

    # plateau-edge arrow: MuJoCo -> Ostrich
    mj, ax_edge = edges["mujoco"], edges["ostrich"]
    ratio = ax_edge / mj
    ax.set_ylim(0.026, 25)
    # The label goes ABOVE the arrow: at the shared print size it is tall
    # enough that hanging it below drops it onto the bottom spine and the x
    # tick labels. The band between the arrow and the plateau brackets is
    # empty, so it sits there instead.
    arrow_y = 0.040
    ax.annotate("", xy=(ax_edge, arrow_y), xytext=(mj, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=ps.LW * S,
                                shrinkA=2, shrinkB=2), zorder=7)
    ax.text(np.sqrt(mj * ax_edge), arrow_y * 1.23,
            rf"$\sim{ratio:.0f}\times$ larger usable $h$",
            ha="center", va="bottom", fontsize=ps.PRINT["annot"] * S,
            fontweight="bold", zorder=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"timestep $h$ [s]")
    ax.set_ylabel(r"combined pos.\,+\,yaw error [m]")
    ax.grid(True, which="major", alpha=ps.GRID_MAJOR["alpha"],
            lw=ps.GRID_MAJOR["lw"] * S)
    ax.grid(True, which="minor", alpha=ps.GRID_MINOR["alpha"],
            lw=ps.GRID_MINOR["lw"] * S)
    ax.legend(loc="upper right", ncol=1, frameon=False)
    fig.tight_layout()
    out = RESULTS / "dt_sweep_c3.png"
    fig.savefig(out, dpi=DPI)
    print(f"saved {out}, edges: {edges}, ratio {ratio:.1f}x")


if __name__ == "__main__":
    main()
