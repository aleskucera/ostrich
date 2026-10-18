"""Optimization benchmark: deterministic loss reached per compute, baselines.

Each gradient estimator of bench_gradients drives Adam on the scenario's
parameters; the cross-entropy method (CEM) is the gradient-free baseline.
The measure is the deterministic (unsmoothed) loss L(theta_k) of the current
iterate against the compute spent (forward-rollout equivalents, as in
bench_gradients); smoothing is only the tool. Losses and gradients are read
from the dense grids (nearest node), so runs are free.

Smoothing methods anneal sigma geometrically from the scenario's sigma to
SIGMA_END x sigma over the budget (with a fixed sigma they converge to the
smoothed loss's minimum, not L's); one fixed-sigma zeroth-order run is kept
as the reference. SIGMA_END keeps the final sigma at >= 2 grid spacings.

Protocol: start points uniform in the scenario's theta box (4 sigma inside the
grid), iterates projected back into it. Step size (Adam: lr = c sigma; CEM:
initial spread = c sigma) tuned per method over STEP_SCALES on TUNE_SEEDS
starts, by the median final loss; reported on separate EVAL_SEEDS starts.
Success: final loss below SUCCESS; also reported for the HARD_STARTS alone,
the starts in a different mode than the optimum.

Run: PYTHONPATH=. python bench_optimize.py
Writes results/bench_optimize.npz and results/bench_optimize.png.
"""

import numpy as np
import torch

import ball_landscape as bl
import bench_gradients as bg
import estimators as est

BUDGET = 10000  # forward-rollout equivalents per run
STEP_SCALES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TUNE_SEEDS, EVAL_SEEDS = 50, 200
SUCCESS = 0.01  # final loss below: x(T) within 0.1 m of its target (box: and upright)
# starts (u, L(u)) in a different mode than the optimum: the corner drop
# left of the apex; the wheel stalling at the ledge or not reaching it; the
# box staying put or tipping (L >= 0.2; sliding upright toward the target is
# below)
HARD_STARTS = {"corner": lambda u, L: u[:, 0] < 0.0, "wheel": lambda u, L: u[:, 0] < 0.3,
               "box": lambda u, L: L >= 0.2}
CEM_POP, CEM_ELITE = 16, 4
SIGMA_END = {"corner": 1 / 30, "wheel": 1 / 30, "box": 1 / 5}
# (estimator, pairs N, anneal sigma)
METHODS = [("hard", 0, False), ("first_order", 4, True), ("first_order", 16, True), ("zeroth_order", 4, True),
           ("zeroth_order", 16, True), ("zeroth_order", 16, False), ("alpha_order", 4, True),
           ("alpha_order", 16, True), ("cem", CEM_POP, False)]
CHECKPOINTS = np.geomspace(10, BUDGET, 25)


def bounds(scen, grid):
    s = 4 * bg.SCENARIOS[scen]["sigma"]
    return grid.axis[0] + s, grid.axis[-1] - s


def gradient(grid, kind, N, theta, sigma, gen):
    """Estimates (S, d) at the iterates theta (S, d)."""
    if kind == "hard":
        return torch.tensor(grid.g[grid.index(theta)])
    eps = bg.directions(len(theta), N, grid.d, gen)
    ip = grid.index(theta[:, None] + sigma * eps.numpy())
    im = grid.index(theta[:, None] - sigma * eps.numpy())
    Lp, Lm = torch.tensor(grid.L[ip]), torch.tensor(grid.L[im])
    if kind == "zeroth_order":
        return est.zeroth_order_from_samples(eps, Lp, Lm, sigma)
    gp, gm = torch.tensor(grid.g[ip]), torch.tensor(grid.g[im])
    if kind == "first_order":
        return est.first_order_from_samples(gp, gm)
    return est.alpha_order_from_samples(eps, Lp, Lm, gp, gm, sigma, bg.GAMMA)[0]


def run_adam(scen, grid, kind, N, anneal, c, starts, gen):
    """Loss of the iterates at the CHECKPOINTS (S, len(CHECKPOINTS))."""
    sigma0 = sigma = bg.SCENARIOS[scen]["sigma"]
    lo, hi = bounds(scen, grid)
    per_iter = bg.cost(kind, N, bg.SCENARIOS[scen]["adj_cost"])
    theta = torch.tensor(starts, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=c * sigma)
    spent, out, k = 0.0, [], 0
    while k < len(CHECKPOINTS):
        while k < len(CHECKPOINTS) and spent + per_iter > CHECKPOINTS[k]:
            out.append(grid.L[grid.index(theta.detach().numpy())])
            k += 1
        if anneal:
            sigma = sigma0 * SIGMA_END[scen] ** (spent / BUDGET)
        theta.grad = gradient(grid, kind, N, theta.detach().numpy(), sigma, gen)
        opt.step()
        with torch.no_grad():
            theta.clamp_(lo, hi)
        spent += per_iter
    return np.stack(out, axis=1)


def run_cem(scen, grid, c, starts, gen):
    """CEM on the deterministic loss: sample CEM_POP points around the mean,
    refit mean and spread to the CEM_ELITE best. The iterate is the mean."""
    lo, hi = bounds(scen, grid)
    sigma = bg.SCENARIOS[scen]["sigma"]
    mean, std = starts.copy(), np.full_like(starts, c * sigma)
    rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,), generator=gen)))
    spent, out, k = 0.0, [], 0
    while k < len(CHECKPOINTS):
        while k < len(CHECKPOINTS) and spent + CEM_POP > CHECKPOINTS[k]:
            out.append(grid.L[grid.index(mean)])
            k += 1
        x = np.clip(mean[:, None] + std[:, None] * rng.standard_normal((len(mean), CEM_POP, grid.d)), lo, hi)
        elite = np.take_along_axis(x, np.argsort(grid.L[grid.index(x)], axis=1)[:, :CEM_ELITE, None], axis=1)
        mean, std = elite.mean(axis=1), np.maximum(elite.std(axis=1), 1e-3 * sigma)
        spent += CEM_POP
    return np.stack(out, axis=1)


def run(scen, grid, kind, N, anneal, c, starts, gen):
    if kind == "cem":
        return run_cem(scen, grid, c, starts, gen)
    return run_adam(scen, grid, kind, N, anneal, c, starts, gen)


def main():
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    out = {"checkpoints": CHECKPOINTS, "methods": np.array([k if k in ("hard", "cem") else
                                                            f"{k} N={n}" + ("" if a else " fixed")
                                                            for k, n, a in METHODS])}
    for scen in bg.SCENARIOS:
        grid = bg.load(scen)
        lo, hi = bounds(scen, grid)
        tune = rng.uniform(lo, hi, (TUNE_SEEDS, grid.d))
        evals = rng.uniform(lo, hi, (EVAL_SEEDS, grid.d))
        hard = HARD_STARTS[scen](evals, grid.L[grid.index(evals)])
        print(f"\n{scen}: {EVAL_SEEDS} starts ({hard.sum()} in another mode), budget {BUDGET}, "
              f"success = final loss < {SUCCESS}")
        print(f"  {'method':24s} {'step':>5s} {'final loss median [q25, q75]':>34s} {'success':>8s} {'hard starts':>12s}")
        curves, best_c = [], []
        for (kind, N, anneal), name in zip(METHODS, out["methods"]):
            score = [np.median(run(scen, grid, kind, N, anneal, c, tune, gen)[:, -1]) for c in STEP_SCALES]
            c = STEP_SCALES[int(np.argmin(score))]
            L = run(scen, grid, kind, N, anneal, c, evals, gen)
            curves.append(L)
            best_c.append(c)
            f = L[:, -1]
            print(f"  {name:24s} {c:5.2f} {np.median(f):12.4f} [{np.quantile(f, 0.25):8.4f}, {np.quantile(f, 0.75):8.4f}]"
                  f" {np.mean(f < SUCCESS):8.0%} {np.mean(f[hard] < SUCCESS):12.0%}")
        out[f"{scen}_curves"] = np.stack(curves)  # (methods, seeds, checkpoints)
        out[f"{scen}_step"] = np.array(best_c)
        out[f"{scen}_hard_starts"] = hard
    np.savez(bl.RESULTS / "bench_optimize.npz", **out)
    plot(out)


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    colors = {"hard": "#8a8f98", "first_order": "#2a78d6", "zeroth_order": "#eb6834", "alpha_order": "#1baf7a",
              "cem": "#7d4cdb"}
    fig, axes = plt.subplots(2, len(bg.SCENARIOS), figsize=(5 * len(bg.SCENARIOS), 8), constrained_layout=True)
    for j, scen in enumerate(bg.SCENARIOS):
        for name, L in zip(out["methods"], out[f"{scen}_curves"]):
            kind = str(name).split(" ")[0]
            style = "--" if str(name).endswith("fixed") else ":" if str(name).endswith("=4") else "-"
            axes[0, j].plot(out["checkpoints"], np.median(L, axis=0), style, color=colors[kind], lw=1.8, label=str(name))
            axes[1, j].plot(out["checkpoints"], np.mean(L < SUCCESS, axis=0), style, color=colors[kind], lw=1.8)
        for ax, ylabel in ((axes[0, j], "deterministic loss of the iterate (median)"),
                           (axes[1, j], f"success rate (loss < {SUCCESS})")):
            ax.set_xscale("log")
            ax.set_xlabel("cost [forward rollouts]", color=muted)
            ax.set_ylabel(ylabel, color=muted)
            ax.tick_params(colors=muted)
            ax.grid(alpha=0.3, lw=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(scen, color=ink, fontsize=10, loc="left")
        axes[1, j].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(bl.RESULTS / "bench_optimize.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'bench_optimize.png'}")


if __name__ == "__main__":
    main()
