"""Compose the 5 paper panels into one preview image so the full LaTeX
figure layout can be reviewed at a glance before paper integration.

Panel (a) is a placeholder until the GL scene render is produced.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

OUT = (pathlib.Path(__file__).resolve().parent / "results" / "paper_panels")


def _placeholder(ax, label):
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, label, ha="center", va="center",
            fontsize=15, color="dimgray", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("dimgray"); s.set_linestyle("--"); s.set_linewidth(0.8)


def fig1_simtoreal():
    """Figure 1: scene + xy + z + bar (1x4 at \\textwidth)."""
    legend = mpimg.imread(OUT / "box_legend.png")
    xy = mpimg.imread(OUT / "box_xy.png")
    z = mpimg.imread(OUT / "box_z.png")
    bar = mpimg.imread(OUT / "box_bar.png")

    fig = plt.figure(figsize=(16, 3.0))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[0.2, 3.0],
        hspace=0.10, wspace=0.10,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
    )
    ax_l = fig.add_subplot(gs[0, :]); ax_l.imshow(legend); ax_l.axis("off")

    ax_a = fig.add_subplot(gs[1, 0])
    _placeholder(ax_a, "(a)  Scene viz")

    ax_b = fig.add_subplot(gs[1, 1]); ax_b.imshow(xy); ax_b.axis("off")
    ax_b.set_title("(b)  Top-down", loc="left", fontsize=10)

    ax_c = fig.add_subplot(gs[1, 2]); ax_c.imshow(z); ax_c.axis("off")
    ax_c.set_title("(c)  Climb $z$ vs $t$", loc="left", fontsize=10)

    ax_d = fig.add_subplot(gs[1, 3]); ax_d.imshow(bar); ax_d.axis("off")
    ax_d.set_title("(d)  Accuracy", loc="left", fontsize=10)

    out = OUT / "fig1_preview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def fig2_dtsweep():
    """Figure 2: standalone dt sweep (single panel at \\columnwidth)."""
    legend = mpimg.imread(OUT / "box_legend.png")
    dt = mpimg.imread(OUT / "box_dt.png")

    fig = plt.figure(figsize=(7.0, 3.5))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.18, 3.0],
        hspace=0.05,
        left=0.02, right=0.98, top=0.97, bottom=0.03,
    )
    ax_l = fig.add_subplot(gs[0, 0]); ax_l.imshow(legend); ax_l.axis("off")
    ax_e = fig.add_subplot(gs[1, 0]); ax_e.imshow(dt); ax_e.axis("off")

    out = OUT / "fig2_preview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    fig1_simtoreal()
    fig2_dtsweep()


if __name__ == "__main__":
    main()
