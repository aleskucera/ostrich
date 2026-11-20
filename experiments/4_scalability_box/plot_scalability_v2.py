r"""Revised scalability figure: throughput as the headline, memory as support.

Panel (a): optimization throughput (world-iterations/s) vs #worlds, every
engine at its best memory configuration (MJX with jax.checkpoint per step).
Panel (b): peak GPU memory (NVML) vs #worlds, with the 24 GB card ceiling.

Visual conventions (match Figs. 2-3 of the paper):
  - marker SHAPE encodes the engine (o Ostrich, s MJX, ^ Semi-Implicit);
  - solid line + filled marker = the engine's operating configuration,
    dotted line + open marker = the alternative configuration;
  - failure is drawn once, the same way in both panels: the curve's last
    successful point is continued by a short segment to an X at the first
    FAILING batch size. In (b) that X sits on the 24 GB ceiling line (which
    is why the run failed); in (a), where no throughput exists for the
    failing batch, the segment stays at the last measured height.
    No "OOM" text labels -- the caption defines the X.

The layout knobs below trade chrome (titles, labels, legend) against panel
area; the figure's printed size is fixed by \columnwidth and the aspect
ratio, so the two can only be traded, not both increased.

    .venv/bin/python experiments/4_scalability_box/plot_scalability_v2.py
"""
import argparse
import collections
import glob
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"

# palette matches Figs. 2-3: Ostrich blue, MuJoCo/MJX crimson, Semi-Implicit orange
BLUE, CRIMSON, ORANGE = "#2196F3", "#E91E63", "#FF9800"

# key: (label, short label, color, marker, is_operating_configuration)
SERIES = {
    "ostrich": (r"\textbf{Ostrich}", r"\textbf{Ostrich}", BLUE, "o", True),
    "ostrich_ckpt": (r"\textbf{Ostrich} + checkpoint", r"\textbf{Ostrich} + ckpt.",
                     BLUE, "o", False),
    "mjx_ckpt_step": ("MJX + checkpoint", "MJX + ckpt.", CRIMSON, "s", True),
    "mjx_ckpt_none": ("MJX plain BPTT", "MJX plain BPTT", CRIMSON, "s", False),
    "semi_implicit": ("Semi-Implicit", "Semi-Implicit", ORANGE, "^", True),
}
# below the panels matplotlib fills column-major, so this puts each engine's
# operating configuration on the top row and its alternative directly beneath
# it (col 1 Ostrich, col 2 MJX, col 3 Semi-Implicit).
LEGEND_ORDER = ["ostrich", "ostrich_ckpt",
                "mjx_ckpt_step", "mjx_ckpt_none",
                "semi_implicit"]

# first world count that FAILED with OOM (annotation target)
OOM = {"mjx_ckpt_none": 8, "mjx_ckpt_step": 8192, "semi_implicit": 1024, "ostrich": 16384}

CARD_MB = 24576.0  # RTX 3090, 24 GB

FONTS = {  # (base, axis label, tick, legend, annotation)
    "large": (15.0, 15.0, 14.5, 14.0, 13.5),
    "small": (13.0, 13.0, 12.0, 12.0, 11.5),
}


def load():
    data = collections.defaultdict(dict)
    for f in glob.glob(str(RES / "*.json")):
        m = re.match(r".*/([a-z_]+?)_(\d+)\.json", f)
        if not m:
            continue
        name, w = m.group(1), int(m.group(2))
        d = json.load(open(f))
        data[name][w] = d
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xticks", choices=("pow2", "plain"), default="pow2")
    ap.add_argument("--fonts", choices=("large", "small"), default="small")
    ap.add_argument("--titles", choices=("above", "tag"), default="tag",
                    help="'above': a title line over each panel; 'tag': a bare "
                         "(a)/(b) inside the panel, quantity moved to the y-label")
    ap.add_argument("--legend", choices=("below", "below1", "inside"), default="below1",
                    help="'below1': one row of short labels (half the height of "
                         "the two-row form); 'inside': panel (a)'s lower-right corner")
    ap.add_argument("--lw", type=float, default=1.3,
                    help="curve line width; the connector, marker edges and the "
                         "X stroke scale with it")
    ap.add_argument("--ms", type=float, default=5.0, help="data marker size")
    ap.add_argument("--width", type=float, default=6.8)
    ap.add_argument("--height", type=float, default=3.32)
    ap.add_argument("--out", default=str(RES / "scalability_v2_paper.png"))
    args = ap.parse_args()

    data = load()
    base, lab, tick, leg, ann = FONTS[args.fonts]
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif", "font.size": base,
        "axes.labelsize": lab, "axes.titlesize": lab,
        "xtick.labelsize": tick, "ytick.labelsize": tick,
        "legend.fontsize": leg, "axes.spines.top": False,
        "axes.spines.right": False})
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(args.width, args.height))

    # 24 GB ceiling, drawn first so the curves sit on top of it
    ax_m.axhline(CARD_MB, color="0.45", ls=(0, (5, 3)), lw=1.0, zorder=1)
    # Above the line, and out at ~40% of the width: below it the label runs
    # into the plain-BPTT curve's marker at 4 worlds, and the band above the
    # ceiling is empty between the crimson X at 8 and the orange X at 1024.
    ax_m.annotate(r"24\,GB", xy=(0.42, CARD_MB), xycoords=("axes fraction", "data"),
                  ha="center", va="bottom", fontsize=ann, color="0.35",
                  textcoords="offset points", xytext=(0, 3), zorder=2,
                  path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    handles = {}
    for key, (label, short, color, mk, operating) in SERIES.items():
        if key not in data:
            continue
        style = dict(color=color, marker=mk, ms=args.ms, lw=args.lw,
                     ls="-" if operating else ":",
                     markerfacecolor=color if operating else "white",
                     markeredgecolor=color, markeredgewidth=0.65 * args.lw,
                     zorder=5 if operating else 4)

        ws = sorted(data[key])
        thr = [w / (data[key][w]["median_time_ms"] / 1000.0) for w in ws]
        (h,) = ax_t.plot(ws, thr, label=label, **style)
        handles[key] = h

        # memory: prefer the NVML re-run series, else the NVML field
        mem_key = {"ostrich": "ostrich_nvml", "semi_implicit": "si_nvml"}.get(key, key)
        src = data.get(mem_key, data[key])
        wm = sorted(src)
        mem = [src[w].get("peak_gpu_mb_nvml") or src[w].get("peak_gpu_mb") for w in wm]
        ax_m.plot(wm, mem, **style)

        if key in OOM:
            wf = OOM[key]
            # the connector keeps the series' own line style, so a reader can
            # trace it back to the right curve where two curves of the same
            # colour nearly coincide (the two MJX configurations in (a)).
            conn = dict(ls=style["ls"], lw=0.6 * args.lw, color=color, alpha=0.75, zorder=3)
            cross = dict(marker="x", c=color, ms=args.ms * 1.33, mew=0.95 * args.lw, ls="none", zorder=7,
                         path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
            # (a) no throughput exists at wf: hold the last measured height
            ax_t.plot([ws[-1], wf], [thr[-1], thr[-1]], **conn)
            ax_t.plot([wf], [thr[-1]], **cross)
            # (b) the run died at the card ceiling: put the X there
            ax_m.plot([wm[-1], wf], [mem[-1], CARD_MB], **conn)
            ax_m.plot([wf], [CARD_MB], **cross)

    for ax in (ax_t, ax_m):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        # plain labels are wide, so they get every fifth power (1, 32, 1024,
        # 32768) and a little more room on the right.
        ax.set_xlim(*((0.55, 8.0e4) if args.xticks == "plain" else (0.6, 5.6e4)))
        step = 5 if args.xticks == "plain" else 3
        ax.xaxis.set_major_locator(ticker.FixedLocator([2 ** k for k in range(0, 16, step)]))
        ax.xaxis.set_minor_locator(ticker.FixedLocator([2 ** k for k in range(16)]))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        if args.xticks == "plain":
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(round(v))}"))
        else:
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: rf"$2^{{{int(round(v)).bit_length() - 1}}}$"))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=tuple(range(2, 10)),
                                                     numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="major", alpha=0.35, lw=0.6)
        ax.grid(True, which="minor", alpha=0.12, lw=0.4)

    ax_t.set_ylim(4e-3, 8.0e2)
    ax_m.set_ylim(1.8e2, 5.5e4)
    if args.titles == "above":
        ax_t.set_ylabel("throughput [world-it./s]")
        ax_m.set_ylabel("peak memory [MB]")
        ax_t.set_title("(a) throughput (fwd+bwd)")
        ax_m.set_title("(b) peak GPU memory")
    else:
        # the quantity moves into the y-label, so the title line -- and the
        # ~0.3 in of figure height it costs -- can go.
        ax_t.set_ylabel("throughput [world-it./s]")
        ax_m.set_ylabel("peak GPU memory [MB]")
        for ax, tag in ((ax_t, "(a)"), (ax_m, "(b)")):
            ax.text(0.025, 0.975, tag, transform=ax.transAxes, ha="left", va="top",
                    fontsize=lab, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

    ordered = [k for k in LEGEND_ORDER if k in handles]
    if args.legend == "below":
        fig.legend([handles[k] for k in ordered], [SERIES[k][0] for k in ordered],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
                   frameon=False, columnspacing=1.6, handletextpad=0.5,
                   handlelength=2.2, labelspacing=0.3, borderaxespad=0.0)
    elif args.legend == "below1":
        # one row: short labels, tight handles. Halves the legend's height.
        fig.legend([handles[k] for k in ["ostrich", "ostrich_ckpt", "mjx_ckpt_step",
                                         "mjx_ckpt_none", "semi_implicit"] if k in handles],
                   [SERIES[k][1] for k in ["ostrich", "ostrich_ckpt", "mjx_ckpt_step",
                                           "mjx_ckpt_none", "semi_implicit"] if k in handles],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                   frameon=False, columnspacing=0.9, handletextpad=0.4,
                   handlelength=1.6, borderaxespad=0.0, fontsize=leg - 1.5)
    else:
        # panel (a)'s lower right is empty (every curve rises to the right),
        # so the legend costs no data area there.
        ax_t.legend([handles[k] for k in ordered], [SERIES[k][1] for k in ordered],
                    loc="lower right", ncol=1, frameon=True, framealpha=0.88,
                    edgecolor="0.8", fancybox=False, borderpad=0.35,
                    handletextpad=0.5, handlelength=1.9, labelspacing=0.25,
                    borderaxespad=0.4).set_zorder(9)

    # explicit margins (not tight_layout) so both panels get exactly the same
    # width -- (a)'s wider y-tick labels would otherwise shrink it.
    H = args.height
    big = args.fonts == "large"
    xaxis_in = 0.60 if big else 0.53           # x tick labels + "parallel worlds"
    legend_in = {"below": 0.50 if big else 0.44,
                 "below1": 0.28 if big else 0.25,
                 "inside": 0.0}[args.legend]
    title_in = (0.33 if big else 0.29) if args.titles == "above" else 0.05
    left = (0.105 if args.fonts == "large" else 0.093) * 6.8 / args.width
    fig.subplots_adjust(left=left,
                        right=0.990 if args.xticks == "plain" else 0.995,
                        bottom=(xaxis_in + legend_in) / H,
                        top=1.0 - title_in / H,
                        wspace=0.30)

    out = pathlib.Path(args.out)
    fig.savefig(out, dpi=300)
    print(f"saved {out}  ({args.width}x{args.height} in, aspect {args.width / H:.3f})")


if __name__ == "__main__":
    main()
