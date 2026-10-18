"""Gradient benchmark: baseline estimators of the smoothed gradient vs the truth.

For each scenario, the objective is the Gaussian-smoothed loss
L_sigma(theta) = E[L(theta + sigma eps)], and the target is its gradient.
Samples are read from the scenario's dense grid (nearest node; grids.py, and
corner_drop.npz for the corner drop), so repeats are free; the truth is the
quadrature of the same grid against the Gaussian's derivative (Stein),
which includes every jump. Nodes whose rollout had an unconverged Newton
step (grids.py counts them; the wheel's are wedged contacts at bump valleys,
where the forward step has no root and the adjoint through it explodes)
take the values of the nearest converged node, so solver failures do not
count as estimator error.

Estimators (estimators.py; N antithetic pairs, orthogonal directions):
  hard         adjoint gradient at theta (no smoothing)
  first_order  mean of adjoint gradients at theta +- sigma eps
  zeroth_order (L(theta + sigma eps) - L(theta - sigma eps)) eps / (2 sigma)
  alpha_order  Suh et al. 2022, gamma = 2
  ddcg         Onoda, Parmas et al. 2026, c = 0.3
Cost in forward-rollout equivalents: a rollout 1, rollout + adjoint
ADJ_COST (measured: forward + adjoint time over forward time).

Errors: |g_hat - g| / mean_theta |g| (normalised by the scenario's gradient
scale, since |g| passes through 0 at the optimum), median and 95th percentile
over thetas x repeats; cos(g_hat, g), median and the fraction below 0 (a
step along -g_hat goes uphill).

Run: PYTHONPATH=. python bench_gradients.py
Writes results/bench_gradients.npz and results/bench_gradients.png.
"""

import math

import numpy as np
import torch

import ball_landscape as bl
import estimators as est

N_PAIRS = (2, 8, 32, 128, 512)
REPEATS = 200
GAMMA = 2.0
ESTIMATORS = ("hard", "first_order", "zeroth_order", "alpha_order", "ddcg")


def _grid2(lo, hi, n):
    a = np.linspace(lo, hi, n)
    return np.stack([m.ravel() for m in np.meshgrid(a, a, indexing="ij")], axis=1)


SCENARIOS = {
    "corner": {"sigma": 0.02, "thetas": np.linspace(-0.06, 0.08, 15)[:, None], "adj_cost": 1.4},
    "wheel": {"sigma": 0.05, "thetas": np.linspace(0.2, 0.8, 25)[:, None], "adj_cost": 1.36},
    "box": {"sigma": 0.05, "thetas": _grid2(0.2, 0.8, 7), "adj_cost": 1.86},
}


class Grid:
    """A regular grid on a box, nodes in row-major order (last axis fastest)."""

    def __init__(self, axis, L, g, unconverged=None):
        self.axis, self.L, self.g = axis, L.copy(), g.copy()
        self.d = g.shape[1]
        self.h = axis[1] - axis[0]
        mesh = np.meshgrid(*[axis] * self.d, indexing="ij")
        self.u = np.stack([m.ravel() for m in mesh], axis=1)
        self.n_replaced = 0
        if unconverged is not None and unconverged.any():
            bad, good = np.nonzero(unconverged > 0)[0], np.nonzero(unconverged == 0)[0]
            near = good[((self.u[bad, None] - self.u[None, good]) ** 2).sum(axis=-1).argmin(axis=1)]
            self.L[bad], self.g[bad] = self.L[near], self.g[near]
            self.n_replaced = len(bad)

    def index(self, u):
        """Nearest node of points u (..., d) -> flat indices (...)."""
        i = np.clip(np.rint((u - self.axis[0]) / self.h).astype(int), 0, len(self.axis) - 1)
        flat = np.zeros(i.shape[:-1], dtype=int)
        for k in range(self.d):
            flat = flat * len(self.axis) + i[..., k]
        return flat

    def truth(self, theta, sigma):
        """E[L], grad E[L] under N(theta, sigma^2 I) by quadrature on the
        nodes. The weights are normalised and the Stein moment by its discrete
        value, so a linear L gives its exact slope."""
        du = self.u - theta
        p = np.exp(-0.5 * (du**2).sum(axis=1) / sigma**2)
        p /= p.sum()
        E = (p * self.L).sum()
        grad = ((p * (self.L - E))[:, None] * du).sum(axis=0) / (p[:, None] * du**2).sum(axis=0)
        return E, grad


def load(name):
    if name == "corner":
        d = np.load(bl.RESULTS / "corner_drop.npz")
        return Grid(d["x"], d["L"], d["dL"][:, None])
    d = np.load(bl.RESULTS / f"grid_{name}.npz")
    return Grid(d["axis"], d["L"], d["g"], d["unconverged"])


def directions(R, N, d, gen):
    """estimators.directions batched over R repeats: (R, N, d)."""
    nb = math.ceil(N / d)
    Q, Rm = torch.linalg.qr(torch.randn(R, nb, d, d, generator=gen, dtype=torch.float64))
    Q = Q * torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1))[..., None, :]
    norms = torch.randn(R, nb, d, d, generator=gen, dtype=torch.float64).norm(dim=-1)
    return (Q * norms[..., None]).reshape(R, nb * d, d)[:, :N]


def estimates(grid, theta, sigma, N, gen):
    """(R, d) estimates of each estimator at theta from N pairs."""
    eps = directions(REPEATS, N, grid.d, gen)
    ip = grid.index(theta + sigma * eps.numpy())
    im = grid.index(theta - sigma * eps.numpy())
    Lp, Lm = torch.tensor(grid.L[ip]), torch.tensor(grid.L[im])
    gp, gm = torch.tensor(grid.g[ip]), torch.tensor(grid.g[im])
    hard = torch.tensor(grid.g[grid.index(theta)]).expand(REPEATS, grid.d)
    alpha, _ = est.alpha_order_from_samples(eps, Lp, Lm, gp, gm, sigma, GAMMA)
    return {
        "hard": hard,
        "first_order": est.first_order_from_samples(gp, gm),
        "zeroth_order": est.zeroth_order_from_samples(eps, Lp, Lm, sigma),
        "alpha_order": alpha,
        "ddcg": est.ddcg_from_samples(eps, Lp, Lm, gp, gm, sigma)[0],
    }


def cost(name, N, adj_cost):
    return {"hard": adj_cost, "first_order": 2 * N * adj_cost, "zeroth_order": 2 * N,
            "alpha_order": 2 * N * adj_cost, "ddcg": 2 * N * adj_cost}[name]


def bench(scen, gen):
    s = SCENARIOS[scen]
    grid = load(scen)
    true = np.array([grid.truth(th, s["sigma"])[1] for th in s["thetas"]])
    scale = np.linalg.norm(true, axis=1).mean()
    rows = []  # (estimator index, N, cost, err median, err q95, cos median, fraction cos < 0)
    for N in N_PAIRS:
        err = {k: [] for k in ESTIMATORS}
        cos = {k: [] for k in ESTIMATORS}
        for th, g in zip(s["thetas"], true):
            for k, e in estimates(grid, th, s["sigma"], N, gen).items():
                e = e.numpy()
                err[k].append(np.linalg.norm(e - g, axis=1) / scale)
                cos[k].append(e @ g / np.maximum(np.linalg.norm(e, axis=1) * np.linalg.norm(g), 1e-300))
        for i, k in enumerate(ESTIMATORS):
            e, c = np.concatenate(err[k]), np.concatenate(cos[k])
            rows.append((i, N, cost(k, N, s["adj_cost"]), np.median(e), np.quantile(e, 0.95),
                         np.median(c), np.mean(c < 0)))
    return np.array(rows), true, scale


def main():
    gen = torch.Generator().manual_seed(0)
    out = {}
    for scen in SCENARIOS:
        rows, true, scale = bench(scen, gen)
        out[f"{scen}_rows"], out[f"{scen}_true"], out[f"{scen}_scale"] = rows, true, scale
        out[f"{scen}_thetas"] = SCENARIOS[scen]["thetas"]
        print(f"\n{scen}: sigma {SCENARIOS[scen]['sigma']}, {len(true)} thetas, gradient scale {scale:.3g}, "
              f"{load(scen).n_replaced} unconverged nodes replaced")
        print(f"  {'estimator':13s} {'N':>4s} {'cost':>7s}   err median [q95]   cos median, wrong direction")
        for r in rows:
            print(f"  {ESTIMATORS[int(r[0])]:13s} {int(r[1]):4d} {r[2]:7.1f}   {r[3]:7.3f} [{r[4]:7.3f}]   "
                  f"{r[5]:+.3f}  {r[6]:.1%}")
    np.savez(bl.RESULTS / "bench_gradients.npz", estimators=np.array(ESTIMATORS), **out)
    plot(out)


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    colors = {"hard": "#8a8f98", "first_order": "#2a78d6", "zeroth_order": "#eb6834", "alpha_order": "#1baf7a",
              "ddcg": "#7d4cdb"}
    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 8), constrained_layout=True)
    for j, scen in enumerate(SCENARIOS):
        rows = out[f"{scen}_rows"]
        for i, k in enumerate(ESTIMATORS):
            r = rows[rows[:, 0] == i]
            for ax, med, band in ((axes[0, j], 3, 4), (axes[1, j], 5, 6)):
                ax.plot(r[:, 2], r[:, med], "o-", color=colors[k], lw=1.8, ms=4, label=k.replace("_", "-"))
                ax.plot(r[:, 2], r[:, band], "--", color=colors[k], lw=1, alpha=0.7)
        for ax, ylabel in ((axes[0, j], "error / gradient scale (median, dashed q95)"),
                           (axes[1, j], "cos to true gradient (median, dashed: fraction < 0)")):
            ax.set_xscale("log")
            ax.set_xlabel("cost [forward rollouts]", color=muted)
            ax.set_ylabel(ylabel, color=muted)
            ax.tick_params(colors=muted)
            ax.grid(alpha=0.3, lw=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(f"{scen}  (sigma {SCENARIOS[scen]['sigma']})", color=ink, fontsize=10, loc="left")
        axes[1, j].set_ylim(-1.05, 1.05)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.savefig(bl.RESULTS / "bench_gradients.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'bench_gradients.png'}")


if __name__ == "__main__":
    main()
