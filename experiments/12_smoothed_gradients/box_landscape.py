"""Box scenario: loss landscape, exact / Ostrich-style / smoothed gradient fields.

Same method as ball_landscape.py: a dense loss grid over u in [0, 1]^2
(padded by 4 sigma) gives the ground-truth smoothed gradient by
correlation; exact and Ostrich-style adjoint gradients (friction weight
held fixed, adjoint contact compliance 1e-6) on a coarse grid of points.

Run: PYTHONPATH=. python box_landscape.py   (~5 min with 4 threads)
Writes results/box_landscape.{npz,png}; main(friction="tight") writes
results/box_landscape_tight.{npz,png}.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import box_scenario as bx

OSTRICH = {"friction": "frozen", "contact_compliance": 1e-6}


def main(spacing=1 / 100, sigma=0.05, n_arrows=19, friction="ostrich"):
    sim = bx.make_sim(friction)
    name = "box_landscape" if friction == "ostrich" else f"box_landscape_{friction}"
    pad = int(math.ceil(bl.PAD_SIGMAS * sigma / spacing))
    axis = torch.arange(-pad, int(round(1 / spacing)) + 1 + pad, dtype=torch.float64) * spacing
    U0, U1 = torch.meshgrid(axis, axis, indexing="ij")
    t0 = time.time()
    # Outside [0, 1] the push is clamped to the parameter box (F >= 0 matters).
    L_grid, worst = bx.loss(sim, torch.stack([U0.flatten(), U1.flatten()], dim=1).clamp(0.0, 1.0),
                            return_residual=True)
    L_grid = L_grid.reshape(U0.shape)
    print(f"loss grid {tuple(L_grid.shape)}: {time.time() - t0:.0f} s, worst Newton residual {worst:.1e}", flush=True)
    g_smooth = bl.smoothed_gradient(L_grid, spacing, sigma)

    idx = torch.linspace(0, g_smooth.shape[1] - 1, n_arrows).round().long()
    I0, I1 = torch.meshgrid(idx, idx, indexing="ij")
    interior = axis[pad:-pad]
    u_arrows = torch.stack([interior[I0.flatten()], interior[I1.flatten()]], dim=1)
    t0 = time.time()
    _, g_exact = bx.loss_and_grad(sim, u_arrows)
    _, g_ostrich = bx.loss_and_grad(sim, u_arrows, **OSTRICH)
    print(f"adjoint gradients ({len(u_arrows)} x 2): {time.time() - t0:.0f} s", flush=True)
    g_sm = g_smooth[:, I0.flatten(), I1.flatten()].T

    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=1)
    flat = g_exact.norm(dim=1) < 1e-2 * g_sm.norm(dim=1)
    print(f"sigma = {sigma}: exact gradient ~0 (<1% of smoothed) at {flat.float().mean():.0%} of points; "
          f"cos(exact, smoothed) median {cos(g_exact, g_sm).median():.3f}; "
          f"cos(ostrich-style, exact) median {cos(g_ostrich, g_exact).median():.3f}, "
          f"|ostrich|/|exact| median {(g_ostrich.norm(dim=1) / g_exact.norm(dim=1)).median():.2f}")

    np.savez(bl.RESULTS / f"{name}.npz", axis=axis.numpy(), L=L_grid.numpy(), pad=pad, spacing=spacing,
             sigma=sigma, g_smooth=g_smooth.numpy(), u_arrows=u_arrows.numpy(), g_exact=g_exact.numpy(),
             g_ostrich=g_ostrich.numpy(), g_sm=g_sm.numpy())
    plot(axis.numpy(), L_grid.numpy(), pad, u_arrows.numpy(), g_exact.numpy(), g_ostrich.numpy(), g_sm.numpy(), sigma,
         name, friction)


def plot(axis, L, pad, u_arrows, g_exact, g_ostrich, g_sm, sigma, name="box_landscape", friction="ostrich"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ink, muted = "#0b0b0b", "#52514e"
    inner = slice(pad, -pad)
    ext = [axis[pad], axis[-pad - 1], axis[pad], axis[-pad - 1]]
    ticks = np.linspace(0, 1, 4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), constrained_layout=True)
    panels = ((None, "Loss  (x(T) - 0.5 m)^2"), (g_exact, "Exact gradient (adjoint)"),
              (g_ostrich, "Ostrich-style adjoint (w frozen)"), (g_sm, f"Smoothed gradient, sigma = {sigma}"))
    for ax, (g, title) in zip(axes, panels):
        im = ax.imshow(L[inner, inner].T, origin="lower", extent=ext, cmap=blues, aspect="equal")
        if g is not None:
            n = np.linalg.norm(g, axis=1, keepdims=True)
            zero = n[:, 0] < 1e-6
            unit = g / np.maximum(n, 1e-12)
            ax.quiver(u_arrows[~zero, 0], u_arrows[~zero, 1], -unit[~zero, 0], -unit[~zero, 1],
                      color=ink, scale=26, width=0.004, headwidth=3.5)
            ax.scatter(u_arrows[zero, 0], u_arrows[zero, 1], s=4, color=muted)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xticks(ticks, [f"{bx.FORCE[0] + t * (bx.FORCE[1] - bx.FORCE[0]):.0f}" for t in ticks])
        ax.set_yticks(ticks, [f"{math.degrees(bx.ANGLE[0] + t * (bx.ANGLE[1] - bx.ANGLE[0])):.0f}" for t in ticks])
        ax.set_xlabel("push magnitude [N]", color=muted)
        ax.set_ylabel("push angle [deg]", color=muted)
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="loss [m^2]")
    fig.suptitle(f"Pushed box (mu = 0.5, {friction} friction regularisers): arrows show the descent direction -grad L, "
                 "unit length; dots: gradient below 1e-6",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / f"{name}.png", dpi=150)
    print(f"wrote {bl.RESULTS / f'{name}.png'}")


if __name__ == "__main__":
    main()
