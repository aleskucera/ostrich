"""Corner drop: gradients of an expected loss under a bimodal outcome.

The drop position is uncertain, x0 ~ N(theta, SIGMA); the ball lands left or
right of the apex, so x(T) is bimodal. Loss L = (x(T) - X_TARGET)^2, and the
objective is E[L](theta). Its gradient splits into

    dE[L]/dtheta = E[dL/dx0]  +  sum_b (L(b+) - L(b-)) N(b; theta, SIGMA)

a within-mode term and a jump term per discontinuity b of L. Two kinds of
discontinuity appear: the physical mode switch at the apex (b = 0), and many
small numerical jumps from the time stepping (impacts and contacts switching
steps) inside each mode. The small jumps reset a locally steep slope, so the
local slope the adjoint computes differs from the within-mode trend by O(1);
halving the time step does not shrink that share.

The apex is an unstable equilibrium: a ball dropped within ~1e-3 m of it
balances there before falling off, so x(T) passes between the modes through
a very narrow, very steep but continuous layer (dx(T)/dx0 ~ 1e3-1e4, and
~1e8 at x0 = 0). Averaging exact gradients is then unbiased in principle but
heavy-tailed: almost all of the mode-switch information sits in that layer,
which samples rarely hit. The decomposition treats the layer |x0| < DELTA as
the boundary: within-mode term outside it, jump L(DELTA) - L(-DELTA).

Exact references come from a dense grid of drops (spacing GRID_STEP): the
true gradient by quadrature of L against the Gaussian's derivative (which
includes every jump), and the within-mode term by quadrature of the adjoint
gradients. Monte Carlo estimators draw x0 samples from the same grid
(nearest point): averaged exact gradients, zeroth-order (antithetic), and the
hybrid = averaged exact gradients of the samples outside the layer + the
apex jump term, with the jump size L(DELTA) - L(-DELTA) from two extra
rollouts.

Run: PYTHONPATH=. python corner_drop_study.py
Writes results/corner_drop.npz and results/corner_drop.png.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import corner_drop as cd
import sim2d

SIGMA = 0.02
X_TARGET = 1.6
GRID = (-0.25, 0.30)
GRID_STEP = 1e-4
DELTA = 0.002  # half-width of the boundary layer around the apex
THETAS = np.linspace(-0.08, 0.12, 81)
MC_THETAS = (-0.02, 0.0, 0.015, 0.04)
MC_N = (2, 8, 32, 128)
MC_REPEATS = 200


def loss_of(xT):
    return (xT - X_TARGET) ** 2


def grid_losses(sim, chunk=2048):
    """x(T), L(x0) and dL/dx0 (adjoint) on the dense grid."""
    x0 = torch.arange(GRID[0], GRID[1] + GRID_STEP / 2, GRID_STEP, dtype=torch.float64)
    xTs, Ls, gs = [], [], []
    for xc in torch.split(x0, chunk):
        q0 = torch.stack([xc, torch.full_like(xc, cd.Y0), torch.zeros_like(xc)], dim=1)
        tape, q, _ = sim2d.rollout(sim, q0, torch.zeros_like(q0), torch.zeros(cd.T_STEPS, len(xc), 3))
        g_q = torch.zeros_like(q)
        g_q[:, 0] = 2 * (q[:, 0] - X_TARGET)
        g_q0, _, _ = sim2d.rollout_vjp(sim, tape, g_q, torch.zeros_like(q))
        xTs.append(q[:, 0])
        Ls.append(loss_of(q[:, 0]))
        gs.append(g_q0[:, 0])
    return x0.numpy(), torch.cat(xTs).numpy(), torch.cat(Ls).numpy(), torch.cat(gs).numpy()


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def quadrature(x, L, dL, thetas):
    """Exact E[L], dE[L]/dtheta (Stein: E[L (x - theta)] / sigma^2) and the
    within-mode term E[dL/dx0; |x0| > DELTA], by quadrature on the grid."""
    out = {"E": [], "grad": [], "pathwise": [], "within": []}
    outside = np.abs(x) > DELTA
    fd = np.gradient(L, x)  # finite differences of the grid: include the small jumps
    for th in thetas:
        p = gaussian(x, th, SIGMA) * GRID_STEP
        out["E"].append((L * p).sum())
        out["grad"].append((L * (x - th) / SIGMA**2 * p).sum())
        out["pathwise"].append((dL * p * outside).sum())
        out["within"].append((fd * p * outside).sum())
    return {k: np.array(v) for k, v in out.items()}


def main():
    sim = cd.make_sim()
    t0 = time.time()
    x, xT, L, dL = grid_losses(sim)
    print(f"grid: {len(x)} drops with adjoint, {time.time() - t0:.0f} s", flush=True)
    jump = np.interp(DELTA, x, L) - np.interp(-DELTA, x, L)  # across the boundary layer
    q = quadrature(x, L, dL, THETAS)
    jump_term = jump * gaussian(0.0, THETAS, SIGMA)
    w_right = 0.5 * (1 + np.vectorize(math.erf)(THETAS / (SIGMA * math.sqrt(2))))
    resid = q["grad"] - q["pathwise"] - jump_term
    print(f"jump across the apex layer L({DELTA}) - L(-{DELTA}) = {jump:.3f}; decomposition residual "
          f"max {np.abs(resid).max():.3f} vs max |grad| {np.abs(q['grad']).max():.3f}")

    # Monte Carlo estimators, samples drawn from the grid (nearest point).
    rng = np.random.default_rng(0)
    mc = {}
    for th in MC_THETAS:
        true = np.interp(th, THETAS, q["grad"])
        for n in MC_N:
            eps = rng.standard_normal((MC_REPEATS, n))
            idx_p = np.clip(np.rint((th + SIGMA * eps - x[0]) / GRID_STEP).astype(int), 0, len(x) - 1)
            idx_m = np.clip(np.rint((th - SIGMA * eps - x[0]) / GRID_STEP).astype(int), 0, len(x) - 1)
            pathwise = 0.5 * (dL[idx_p] + dL[idx_m]).mean(axis=1)
            zeroth = ((L[idx_p] - L[idx_m]) * eps).mean(axis=1) / (2 * SIGMA)
            out_p, out_m = np.abs(x[idx_p]) > DELTA, np.abs(x[idx_m]) > DELTA
            within = 0.5 * (dL[idx_p] * out_p + dL[idx_m] * out_m).mean(axis=1)
            hybrid = within + jump * gaussian(0.0, th, SIGMA)
            for name, est in (("pathwise", pathwise), ("zeroth", zeroth), ("hybrid", hybrid)):
                rel = np.abs(est - true) / max(abs(true), 1e-9)
                mc[(th, n, name)] = (np.median(rel), np.quantile(rel, 0.95))
    print(f"\nrelative error vs the true gradient over {MC_REPEATS} repeats: median [95th percentile]")
    for th in MC_THETAS:
        print(f"theta = {th:+.3f} (w_right {0.5 * (1 + math.erf(th / (SIGMA * math.sqrt(2)))):.2f}, "
              f"true grad {np.interp(th, THETAS, q['grad']):+.2f})")
        for n in MC_N:
            print(f"   N = {n:3d}: " + "  ".join(f"{name} {mc[(th, n, name)][0]:6.3f} [{mc[(th, n, name)][1]:5.3f}]"
                                            for name in ("pathwise", "zeroth", "hybrid")))

    # A fan of noisy drops at one theta, for the picture.
    th_fan, n_fan = 0.01, 40
    x_fan = torch.tensor(th_fan + SIGMA * rng.standard_normal(n_fan))
    _, _, paths = cd.drop(sim, x_fan, record=True)

    np.savez(bl.RESULTS / "corner_drop.npz", x=x, xT=xT, L=L, dL=dL, thetas=THETAS, jump=jump, jump_term=jump_term,
             w_right=w_right, paths=paths.numpy(), th_fan=th_fan, sigma=SIGMA, x_target=X_TARGET,
             mc=np.array([[th, n, ["pathwise", "zeroth", "hybrid"].index(name), *v] for (th, n, name), v in mc.items()]),
             **{f"q_{k}": v for k, v in q.items()})
    plot()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(bl.RESULTS / "corner_drop.npz")
    ink, muted, rule = "#0b0b0b", "#52514e", "#c9ccd1"
    left_c, right_c, third_c = "#2a78d6", "#eb6834", "#1baf7a"
    x, xT, L, th, sigma = d["x"], d["xT"], d["L"], d["thetas"], float(d["sigma"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)

    def style(ax, title, xlabel, ylabel=None):
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=muted)
        if ylabel:
            ax.set_ylabel(ylabel, color=muted)
        ax.tick_params(colors=muted)
        ax.grid(alpha=0.3, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # (a) noisy drops, coloured by the mode they end in
    ax = axes[0, 0]
    ax.fill([-cd.W, 0, cd.W], [0, cd.H, 0], color=rule, zorder=0)
    ax.axhline(0, color=muted, lw=1)
    paths = d["paths"]
    for i in range(paths.shape[1]):
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=right_c if paths[-1, i, 0] > 0 else left_c, lw=0.8, alpha=0.6)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-0.05, 0.55)
    style(ax, f"40 drops, x0 ~ N({float(d['th_fan'])}, {sigma}) m", "x [m]", "y [m]")

    # (b) outcome distribution at that theta, with its mean
    ax = axes[0, 1]
    rng = np.random.default_rng(1)
    s = np.clip(np.rint((float(d["th_fan"]) + sigma * rng.standard_normal(20000) - x[0]) / GRID_STEP).astype(int), 0, len(x) - 1)
    out = xT[s]
    ax.hist(out[out < 0], bins=60, color=left_c, alpha=0.85, label="left mode")
    ax.hist(out[out > 0], bins=60, color=right_c, alpha=0.85, label="right mode")
    ax.axvline(out.mean(), color=ink, lw=1.5, ls="--")
    ax.text(out.mean(), ax.get_ylim()[1] * 0.92, f"  mean {out.mean():+.2f} m\n  (no ball ends here)", color=ink, fontsize=9, va="top")
    ax.axvline(X_TARGET, color=third_c, lw=1.5)
    ax.text(X_TARGET, ax.get_ylim()[1] * 0.6, " target", color=third_c, fontsize=9)
    style(ax, "Outcome x(T): 20 000 noisy drops", "x(T) [m]", "count")
    ax.legend(frameon=False, fontsize=9, loc="center left")

    # (c) loss and mode weight along x0 / theta
    ax = axes[0, 2]
    ax.plot(x, L, color=muted, lw=1, label="L(x0), no noise")
    ax.plot(th, d["q_E"], color=ink, lw=2, label=f"E[L](theta), sigma = {sigma}")
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(-0.5, 14)
    ax2 = ax.twinx()
    ax2.plot(th, d["w_right"], color=right_c, lw=1.5, ls=":")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("weight of the right mode (dotted)", color=right_c)
    ax2.tick_params(colors=right_c)
    style(ax, "Loss: raw and expected", "theta = nominal drop position [m]", "loss [m^2]")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # (d) gradient decomposition along theta
    ax = axes[1, 0]
    ax.plot(th, d["q_grad"], color=ink, lw=2.2, label="true dE[L]/dtheta")
    ax.plot(th, d["jump_term"], color=right_c, lw=1.8, label="apex mode switch: (L(d) - L(-d)) N(0; theta, sigma)")
    ax.plot(th, d["q_pathwise"], color=left_c, lw=1.8, label="local slope: averaged exact (adjoint) gradients")
    ax.plot(th, d["q_within"] - d["q_pathwise"], color=third_c, lw=1.8, label="small numerical jumps inside the modes")
    ax.plot(th, d["q_within"] + d["jump_term"], color=muted, lw=1.2, ls="--", label="sum of the three")
    ax.set_xlim(th[0], th[-1])
    style(ax, "Gradient of the expected loss and its three parts", "theta [m]", "gradient [m]")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (e), (f) Monte Carlo error vs samples at two thetas
    mc = d["mc"]
    names = ("averaged exact gradients", "zeroth-order", "hybrid")
    colors = (left_c, right_c, third_c)
    markers = ("o", "s", "^")
    for ax, t in zip(axes[1, 1:], (0.0, 0.04)):
        for k in range(3):
            rows = mc[(np.isclose(mc[:, 0], t)) & (mc[:, 2] == k)]
            ax.plot(rows[:, 1], rows[:, 3], color=colors[k], lw=2, marker=markers[k], ms=6, label=names[k])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(MC_N, [str(n) for n in MC_N])
        w = 0.5 * (1 + math.erf(t / (sigma * math.sqrt(2))))
        style(ax, f"Estimator error at theta = {t} (right-mode weight {w:.2f})", "samples N (antithetic pairs)",
              "median relative error")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Corner drop: a bimodal outcome and the gradient of its expected loss",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "corner_drop.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'corner_drop.png'}")


if __name__ == "__main__":
    main()
