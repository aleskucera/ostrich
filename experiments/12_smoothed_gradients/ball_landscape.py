"""Ball scenario: loss landscape, exact and smoothed gradient fields.

A frictionless ball (normal contact only, restitution e = 0.8) is launched
from the ground with speed and angle; the loss is a terminal-state cost at
T = 1 s, |q - q*|^2 + W_V |v - v*|^2 (x, y components). The velocity term
jumps when a bounce moves across T; position alone is continuous in the
launch parameters (bounces only add kinks there). The target is the terminal
state of the launch u = (0.5, 0.5), so the global minimum is L = 0 there. Parameters are normalised to u in [0, 1]^2
(speed 1-4 m/s, angle 20-80 deg) so that one isotropic sigma makes sense.

Ground-truth smoothed gradient: in 2D, grad L_sigma(u) = E[L(u + sigma eps) eps] / sigma
is a correlation of the loss with the kernel delta / sigma^2 * G_sigma(delta),
computed on a dense loss grid that is padded by 4 sigma. This is exact up to
grid resolution and, unlike a mean of exact gradients, it sees the jumps.

Run: PYTHONPATH=. python ball_landscape.py   (from this directory; heavy: run on dasenka)
Writes results/ball_landscape.npz and results/ball_landscape.png.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

import sim2d

R, E, T_STEPS = 0.05, 0.8, 100
SPEED = (1.0, 4.0)
ANGLE = (math.radians(20.0), math.radians(80.0))
# Terminal (x, y) position and velocity of the launch u = (0.5, 0.5): 2.5 m/s, 50 deg.
TARGET_Q = torch.tensor([1.606969024216352, 0.10179862567089831])
TARGET_V = torch.tensor([1.6069690242163488, 0.2257165112366079])
W_V = 0.1  # s^2
RESULTS = Path(__file__).parent / "results"
PAD_SIGMAS = 4  # Gaussian kernel truncated at 4 sigma


def make_sim():
    return sim2d.Sim2D(sim2d.ball(1.0, R), sim2d.SimConfig(restitution=E, mu=0.0))


def launch(u):
    """u (B, 2) -> speed, angle."""
    speed = SPEED[0] + (SPEED[1] - SPEED[0]) * u[:, 0]
    angle = ANGLE[0] + (ANGLE[1] - ANGLE[0]) * u[:, 1]
    return speed, angle


def initial_state(u):
    speed, angle = launch(u)
    n = len(u)
    q0 = torch.stack([torch.zeros(n), torch.full((n,), R + 1e-4), torch.zeros(n)], dim=1)
    v0 = torch.stack([speed * torch.cos(angle), speed * torch.sin(angle), torch.zeros(n)], dim=1)
    return q0, v0


def terminal_cost(q, v):
    return ((q[:, :2] - TARGET_Q) ** 2).sum(dim=1) + W_V * ((v[:, :2] - TARGET_V) ** 2).sum(dim=1)


def loss(sim, u, chunk=4096):
    """L(u) for a batch of parameters (B,)."""
    out = []
    for u_c in torch.split(u, chunk):
        q, v = initial_state(u_c)
        for _ in range(T_STEPS):
            q, v, _ = sim.step(q, v)
        out.append(terminal_cost(q, v))
    return torch.cat(out)


def loss_and_grad(sim, u, **adjoint):
    """L(u) and its exact gradient dL/du (adjoint through the rollout)."""
    q0, v0 = initial_state(u)
    tape, q, v = sim2d.rollout(sim, q0, v0, torch.zeros(T_STEPS, len(u), 3))
    L = terminal_cost(q, v)
    zero = torch.zeros(len(u), 1)
    g_qT = torch.cat([2 * (q[:, :2] - TARGET_Q), zero], dim=1)
    g_vT = torch.cat([2 * W_V * (v[:, :2] - TARGET_V), zero], dim=1)
    _, g_v0, _ = sim2d.rollout_vjp(sim, tape, g_qT, g_vT, **adjoint)
    speed, angle = launch(u)
    c, s = torch.cos(angle), torch.sin(angle)
    g_u = torch.stack(
        [
            (g_v0[:, 0] * c + g_v0[:, 1] * s) * (SPEED[1] - SPEED[0]),
            (-g_v0[:, 0] * s + g_v0[:, 1] * c) * speed * (ANGLE[1] - ANGLE[0]),
        ],
        dim=1,
    )
    return L, g_u


def smoothed_gradient(L_grid, spacing, sigma):
    """Correlate a padded loss grid with delta / sigma^2 * G_sigma(delta).
    Returns the gradient on the interior (grid minus a 4 sigma border), (2, n, n)."""
    k = int(math.ceil(PAD_SIGMAS * sigma / spacing))
    d = torch.arange(-k, k + 1, dtype=torch.float64) * spacing
    dx, dy = torch.meshgrid(d, d, indexing="ij")
    G = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    G = G / G.sum()
    # Normalise by the discrete second moment (not sigma^2): exact for linear L.
    kernel = torch.stack([dx * G, dy * G])[:, None] / (dx**2 * G).sum()
    out = torch.nn.functional.conv2d(L_grid[None, None], kernel)
    return out[0]


def main(spacing=1 / 200, sigma=0.05, n_arrows=21):
    RESULTS.mkdir(exist_ok=True)
    sim = make_sim()
    pad = int(math.ceil(PAD_SIGMAS * sigma / spacing))
    axis = torch.arange(-pad, int(round(1 / spacing)) + 1 + pad, dtype=torch.float64) * spacing
    U0, U1 = torch.meshgrid(axis, axis, indexing="ij")
    t0 = time.time()
    L_grid = loss(sim, torch.stack([U0.flatten(), U1.flatten()], dim=1)).reshape(U0.shape)
    print(f"loss grid {tuple(L_grid.shape)}: {time.time() - t0:.0f} s")
    g_smooth = smoothed_gradient(L_grid, spacing, sigma)  # on axis[pad:-pad]

    # Exact gradients on a coarse grid of interior points, with the smoothed
    # gradient read off at the same points.
    idx = torch.linspace(0, g_smooth.shape[1] - 1, n_arrows).round().long()
    I0, I1 = torch.meshgrid(idx, idx, indexing="ij")
    interior = axis[pad:-pad]
    u_arrows = torch.stack([interior[I0.flatten()], interior[I1.flatten()]], dim=1)
    t0 = time.time()
    _, g_exact = loss_and_grad(sim, u_arrows)
    print(f"exact gradients ({len(u_arrows)}): {time.time() - t0:.0f} s")
    g_sm = g_smooth[:, I0.flatten(), I1.flatten()].T

    cos = torch.nn.functional.cosine_similarity(g_exact, g_sm, dim=1)
    ratio = g_exact.norm(dim=1) / g_sm.norm(dim=1)
    print(f"sigma = {sigma}: cos(exact, smoothed) median {cos.median():.3f}, "
          f"< 0.5 at {(cos < 0.5).float().mean():.0%} of points, < 0 at {(cos < 0).float().mean():.0%}; "
          f"|exact|/|smoothed| median {ratio.median():.2f}, max {ratio.max():.1f}")

    np.savez(RESULTS / "ball_landscape.npz", axis=axis.numpy(), L=L_grid.numpy(), pad=pad,
             spacing=spacing, sigma=sigma, g_smooth=g_smooth.numpy(), u_arrows=u_arrows.numpy(),
             g_exact=g_exact.numpy(), g_sm=g_sm.numpy())
    plot(axis.numpy(), L_grid.numpy(), pad, u_arrows.numpy(), g_exact.numpy(), g_sm.numpy(), sigma)


def plot(axis, L, pad, u_arrows, g_exact, g_sm, sigma):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Sequential blue ramp (dataviz reference palette), light = low loss.
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ink, muted = "#0b0b0b", "#52514e"
    inner = slice(pad, -pad)
    ext = [axis[pad], axis[-pad - 1], axis[pad], axis[-pad - 1]]
    speed_ticks = np.linspace(0, 1, 4)
    angle_ticks = np.linspace(0, 1, 4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, title, g in ((axes[0], "Loss  L(speed, angle)", None),
                         (axes[1], "Exact gradient (adjoint)", g_exact),
                         (axes[2], f"Smoothed gradient, sigma = {sigma}", g_sm)):
        im = ax.imshow(L[inner, inner].T, origin="lower", extent=ext, cmap=blues, aspect="equal")
        if g is not None:
            unit = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            ax.quiver(u_arrows[:, 0], u_arrows[:, 1], -unit[:, 0], -unit[:, 1],
                      color=ink, scale=28, width=0.004, headwidth=3.5)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xticks(speed_ticks, [f"{SPEED[0] + t * (SPEED[1] - SPEED[0]):.0f}" for t in speed_ticks])
        ax.set_yticks(angle_ticks, [f"{math.degrees(ANGLE[0] + t * (ANGLE[1] - ANGLE[0])):.0f}" for t in angle_ticks])
        ax.set_xlabel("launch speed [m/s]", color=muted)
        ax.set_ylabel("launch angle [deg]", color=muted)
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="loss [m^2]")
    fig.suptitle("Bouncing ball (e = 0.8, frictionless), terminal-state loss: arrows show the descent direction -grad L, unit length",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(RESULTS / "ball_landscape.png", dpi=150)
    print(f"wrote {RESULTS / 'ball_landscape.png'}")


if __name__ == "__main__":
    main()
