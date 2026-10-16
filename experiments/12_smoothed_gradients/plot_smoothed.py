"""The smoothed losses L_sigma = L * G_sigma themselves, from the dense loss
grids of ball_landscape.py and box_landscape.py (no new simulation).

Per scenario: heatmaps at sigma = 0, 0.025, 0.05 on one colour scale with
contours, and a 1D slice through them (ball: across the bounce bands at
50 deg; box: along the push magnitude at 0 deg, through the stuck region).

Run: PYTHONPATH=. python plot_smoothed.py
Writes results/smoothed_landscapes.png.
"""

import math

import numpy as np
import torch

import ball_landscape as bl
import box_scenario as bx

SIGMAS = (0.0, 0.025, 0.05)


def smoothed_loss(L, spacing, sigma, pad):
    """L * G_sigma on the interior of a grid padded by `pad` cells."""
    if sigma == 0:
        return L[pad:-pad, pad:-pad]
    k = int(math.ceil(bl.PAD_SIGMAS * sigma / spacing))
    d = torch.arange(-k, k + 1, dtype=torch.float64) * spacing
    dx, dy = torch.meshgrid(d, d, indexing="ij")
    G = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    G = (G / G.sum())[None, None]
    out = torch.nn.functional.conv2d(L[None, None], G)[0, 0]
    trim = pad - k
    return out[trim:-trim, trim:-trim] if trim else out


SCENARIOS = (
    {
        "file": "ball_landscape.npz", "title": "Ball",
        "x": ("launch speed [m/s]", lambda t: bl.SPEED[0] + t * (bl.SPEED[1] - bl.SPEED[0])),
        "y": ("launch angle [deg]", lambda t: math.degrees(bl.ANGLE[0] + t * (bl.ANGLE[1] - bl.ANGLE[0]))),
        "slice": (0.5, "angle 50 deg"), "xfmt": "{:.1f}",
    },
    {
        "file": "box_landscape.npz", "title": "Box (Ostrich friction)",
        "x": ("push magnitude [N]", lambda t: bx.FORCE[0] + t * (bx.FORCE[1] - bx.FORCE[0])),
        "y": ("push angle [deg]", lambda t: math.degrees(bx.ANGLE[0] + t * (bx.ANGLE[1] - bx.ANGLE[0]))),
        "slice": (60 / 90, "angle 0 deg"), "xfmt": "{:.0f}",
    },
)


def main(scenarios=SCENARIOS, out="smoothed_landscapes.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    blues = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ink, muted = "#0b0b0b", "#52514e"
    line_colors = (ink, "#2a78d6", "#eb6834")

    fig, axes = plt.subplots(len(scenarios), len(SIGMAS) + 1, figsize=(17, 4.4 * len(scenarios)),
                             constrained_layout=True, squeeze=False)
    for row, sc in enumerate(scenarios):
        d = np.load(bl.RESULTS / sc["file"])
        L, axis, pad, spacing = torch.tensor(d["L"]), d["axis"], int(d["pad"]), float(d["spacing"])
        inner = axis[pad:-pad]
        ext = [inner[0], inner[-1], inner[0], inner[-1]]
        surfaces = [smoothed_loss(L, spacing, s, pad).numpy() for s in SIGMAS]
        norm = PowerNorm(0.5, vmin=0.0, vmax=float(surfaces[0].max()))
        levels = np.quantile(surfaces[0], np.linspace(0.05, 0.95, 9))
        ticks = np.linspace(0, 1, 4)
        for col, (s, Ls) in enumerate(zip(SIGMAS, surfaces)):
            ax = axes[row, col]
            im = ax.imshow(Ls.T, origin="lower", extent=ext, cmap=blues, norm=norm, aspect="equal")
            ax.contour(inner, inner, Ls.T, levels=levels, colors=muted, linewidths=0.5, alpha=0.7)
            ax.axhline(sc["slice"][0], color=line_colors[col], lw=1.2, ls="--")
            label = f"{sc['title']}: raw loss" if s == 0 else f"smoothed, sigma = {s}"
            ax.set_title(label, color=ink, fontsize=10, loc="left")
            ax.set_xticks(ticks, [sc["xfmt"].format(sc["x"][1](t)) for t in ticks])
            ax.set_yticks(ticks, [f"{sc['y'][1](t):.0f}" for t in ticks])
            ax.set_xlabel(sc["x"][0], color=muted)
            ax.set_ylabel(sc["y"][0], color=muted)
            ax.tick_params(colors=muted)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.colorbar(im, ax=axes[row, : len(SIGMAS)], shrink=0.85, label="loss")

        ax = axes[row, -1]
        j = int(np.argmin(np.abs(inner - sc["slice"][0])))
        xs = [sc["x"][1](t) for t in inner]
        for s, Ls, color in zip(SIGMAS, surfaces, line_colors):
            ax.plot(xs, Ls[:, j], color=color, lw=1.4 if s == 0 else 2, label="raw" if s == 0 else f"sigma = {s}")
        ax.set_title(f"Slice at {sc['slice'][1]}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel(sc["x"][0], color=muted)
        ax.set_ylabel("loss", color=muted)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(colors=muted)
        ax.legend(frameon=False, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Smoothed losses L_sigma = E[L(u + sigma eps)]: same colour scale per row; dashed line marks the slice",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / out, dpi=150)
    print(f"wrote {bl.RESULTS / out}")


if __name__ == "__main__":
    main()
