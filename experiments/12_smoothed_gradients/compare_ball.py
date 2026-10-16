"""Ball scenario: gradient estimators against the ground-truth smoothed gradient.

Test points are nodes of the loss grid from ball_landscape.py (u in [0.2, 0.8]^2,
so the convolution ground truth is valid up to sigma = 0.1). For each sigma,
number of antithetic pairs N and repeat, all estimators share one batched
rollout (with adjoint) of every sample point. Reported per estimator: median
over points of the relative error |g_hat - g_sigma| / |g_sigma| (mean over
repeats), and its split into bias and spread across repeats. Points are
grouped by whether the exact gradient agrees with the smoothed one
(cos > 0.9, "smooth") or not ("jumpy").

Run: PYTHONPATH=. python compare_ball.py   (after ball_landscape.py; heavy)
Writes results/compare_ball.npz and results/compare_ball.png.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import estimators as est

SIGMAS = (0.025, 0.05, 0.1)
N_PAIRS = (2, 8, 32)
REPEATS = 6
N_POINTS = 40
GAMMA = 2.0  # alpha-order bias test, in standard errors of the zeroth-order mean


def load_landscape():
    d = np.load(bl.RESULTS / "ball_landscape.npz")
    return torch.tensor(d["axis"]), torch.tensor(d["L"]), float(d["spacing"])


def truth_at(L, spacing, sigma, idx):
    """Ground-truth smoothed gradient at grid nodes idx (P, 2) -> (P, 2)."""
    g = bl.smoothed_gradient(L, spacing, sigma)
    k = int(math.ceil(bl.PAD_SIGMAS * sigma / spacing))
    return g[:, idx[:, 0] - k, idx[:, 1] - k].T


def batched_loss_grad(sim, u, chunk=4096):
    Ls, gs = [], []
    for u_c in torch.split(u, chunk):
        L, g = bl.loss_and_grad(sim, u_c)
        Ls.append(L)
        gs.append(g)
    return torch.cat(Ls), torch.cat(gs)


def main():
    axis, L, spacing = load_landscape()
    gen = torch.Generator().manual_seed(0)
    inside = torch.nonzero((axis >= 0.2) & (axis <= 0.8)).flatten()
    idx = inside[torch.randint(len(inside), (N_POINTS, 2), generator=gen)]
    u_pts = torch.stack([axis[idx[:, 0]], axis[idx[:, 1]]], dim=1)
    sim = bl.make_sim()

    _, g_hard = batched_loss_grad(sim, u_pts)
    results = {"u": u_pts.numpy(), "hard": g_hard.numpy()}
    for sigma in SIGMAS:
        truth = truth_at(L, spacing, sigma, idx)
        results[f"truth_{sigma}"] = truth.numpy()
        for n in N_PAIRS:
            t0 = time.time()
            eps = torch.stack([torch.stack([est.directions(n, 2, generator=gen) for _ in range(N_POINTS)])
                               for _ in range(REPEATS)])  # (R, P, N, 2)
            u_s = torch.cat([u_pts[None, :, None] + sigma * eps, u_pts[None, :, None] - sigma * eps], dim=2)
            Ls, gs = batched_loss_grad(sim, u_s.reshape(-1, 2))
            Ls, gs = Ls.reshape(REPEATS, N_POINTS, 2 * n), gs.reshape(REPEATS, N_POINTS, 2 * n, 2)
            Lp, Lm, gp, gm = Ls[..., :n], Ls[..., n:], gs[..., :n, :], gs[..., n:, :]
            results[f"first_{sigma}_{n}"] = est.first_order_from_samples(gp, gm).numpy()
            results[f"zeroth_{sigma}_{n}"] = est.zeroth_order_from_samples(eps, Lp, Lm, sigma).numpy()
            ao, alpha = est.alpha_order_from_samples(eps, Lp, Lm, gp, gm, sigma, GAMMA)
            results[f"alpha_{sigma}_{n}"] = ao.numpy()
            results[f"alphaw_{sigma}_{n}"] = alpha.numpy()
            print(f"sigma {sigma}, N {n}: {Ls.numel()} rollouts, {time.time() - t0:.0f} s", flush=True)
    np.savez(bl.RESULTS / "compare_ball.npz", **results)
    report(results)


def errors(results, name, sigma, n):
    """Relative error per (repeat, point), bias and spread per point."""
    truth = results[f"truth_{sigma}"]
    g = results[f"{name}_{sigma}_{n}"] if name != "hard" else results["hard"][None]
    norm = np.linalg.norm(truth, axis=-1)
    rel = np.linalg.norm(g - truth, axis=-1) / norm
    bias = np.linalg.norm(g.mean(axis=0) - truth, axis=-1) / norm
    spread = np.sqrt((np.linalg.norm(g - g.mean(axis=0), axis=-1) ** 2).mean(axis=0)) / norm
    return rel.mean(axis=0), bias, spread


def report(results):
    hard = results["hard"]
    print(f"\n{'sigma':>6} {'group':>7} {'N':>4}  " + "  ".join(f"{e:>22}" for e in ("hard", "first", "zeroth", "alpha")))
    for sigma in SIGMAS:
        truth = results[f"truth_{sigma}"]
        cos = (hard * truth).sum(-1) / (np.linalg.norm(hard, axis=-1) * np.linalg.norm(truth, axis=-1))
        for group, mask in (("smooth", cos > 0.9), ("jumpy", cos <= 0.9)):
            if not mask.any():
                continue
            for n in N_PAIRS:
                cells = []
                for name in ("hard", "first", "zeroth", "alpha"):
                    rel, bias, spread = errors(results, name, sigma, n)
                    cells.append(f"{np.median(rel[mask]):6.3f} (b {np.median(bias[mask]):5.2f} s {np.median(spread[mask]):5.2f})")
                alpha = results[f"alphaw_{sigma}_{n}"][:, mask].mean()
                print(f"{sigma:6.3f} {group:>7} {n:4d}  " + "  ".join(f"{c:>22}" for c in cells)
                      + f"   [{mask.sum()} pts, mean alpha {alpha:.2f}]")
    plot(results)


def plot(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    series = (("first", "first-order", "#2a78d6", "o"), ("zeroth", "zeroth-order", "#eb6834", "s"),
              ("alpha", "alpha-order", "#1baf7a", "^"))
    hard = results["hard"]
    fig, axes = plt.subplots(2, len(SIGMAS), figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for col, sigma in enumerate(SIGMAS):
        truth = results[f"truth_{sigma}"]
        cos = (hard * truth).sum(-1) / (np.linalg.norm(hard, axis=-1) * np.linalg.norm(truth, axis=-1))
        for row, mask in enumerate((cos > 0.9, cos <= 0.9)):
            ax = axes[row, col]
            hard_rel = np.median(errors(results, "hard", sigma, N_PAIRS[0])[0][mask])
            ax.axhline(hard_rel, color=muted, lw=1, ls="--")
            ax.text(N_PAIRS[0], hard_rel, "exact gradient", color=muted, va="bottom", ha="left", fontsize=9)
            ends = []
            for name, label, color, marker in series:
                med = [np.median(errors(results, name, sigma, n)[0][mask]) for n in N_PAIRS]
                ax.plot(N_PAIRS, med, color=color, lw=2, marker=marker, ms=7, label=label)
                ends.append([med[-1], label])
            # End labels, nudged apart in log space so they do not overlap.
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] * 1.13)
            for y, label in ends:
                ax.text(N_PAIRS[-1] * 1.15, y, label, color=ink, va="center", fontsize=9)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(N_PAIRS, [str(n) for n in N_PAIRS])
            ax.set_xlim(N_PAIRS[0] * 0.8, N_PAIRS[-1] * 3.5)
            ax.set_title(f"sigma = {sigma}  ({mask.sum()} points)", color=ink, fontsize=10, loc="left")
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(colors=muted)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("antithetic pairs N (2N rollouts)", color=muted)
    axes[0, 0].set_ylabel("exact gradient agrees (cos > 0.9)\nmedian relative error", color=muted)
    axes[1, 0].set_ylabel("exact gradient misleads (cos <= 0.9)\nmedian relative error", color=muted)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="outside upper right", ncols=3)
    fig.suptitle("Ball, terminal-state loss: error vs the ground-truth smoothed gradient",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "compare_ball.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'compare_ball.png'}")


if __name__ == "__main__":
    main()
