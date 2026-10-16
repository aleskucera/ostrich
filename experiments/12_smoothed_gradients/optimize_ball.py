"""Ball scenario: optimization progress per rollout for each gradient estimator.

Minimizes the terminal-state loss of ball_landscape.py over u = (speed, angle)
in [0, 1]^2 from random starts with Adam (lr 0.01, default betas, gradient
norm clipped at 1). Sampled estimators (first-order, zeroth-order,
alpha-order) use N_PAIRS antithetic pairs per step and run ITERS steps, a
budget of ITERS * 2 * N_PAIRS forward rollouts, with sigma fixed or annealed
geometrically; the exact gradient (adjoint) runs the same ITERS steps at one
rollout each. The deterministic loss L(u) is logged along the way (those
evaluations are not counted). First-order and alpha-order also need an
adjoint per sample, zeroth-order does not (it is computed here anyway because
everything shares one batched rollout).

Run: PYTHONPATH=. python optimize_ball.py   (heavy: ~90 min locally)
Writes results/optimize_ball.npz and results/optimize_ball.png.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import estimators as est

N_PAIRS = 8
ITERS = 125
STARTS = 16
LR = 0.01
CLIP = 1.0
GAMMA = 2.0
LOG_EVERY = 5
SIGMAS = {"0.02": (0.02, 0.02), "0.05": (0.05, 0.05), "0.1 -> 0.005": (0.1, 0.005)}
SAMPLED = ("first", "zeroth", "alpha")


class Adam:
    def __init__(self, shape, lr=LR, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, *betas, eps
        self.m, self.v, self.k = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64), 0

    def step(self, u, g):
        norm = g.norm(dim=-1, keepdim=True)
        g = g * torch.clamp(CLIP / norm.clamp(min=1e-12), max=1.0)
        self.k += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        m_hat, v_hat = self.m / (1 - self.b1**self.k), self.v / (1 - self.b2**self.k)
        return torch.clamp(u - self.lr * m_hat / (v_hat.sqrt() + self.eps), 0.0, 1.0)


def batched(fn, u, chunk=4096):
    outs = [fn(u_c) for u_c in torch.split(u, chunk)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(parts) for parts in zip(*outs))
    return torch.cat(outs)


def starts(gen):
    return 0.05 + 0.9 * torch.rand(STARTS, 2, generator=gen, dtype=torch.float64)


def optimize(loss_grad, u0, sigmas, gen, n_pairs=N_PAIRS, iters=ITERS, project=None, extra_hard=()):
    """Adam with every sampled estimator x sigma schedule from the starts u0,
    plus the exact gradient (loss_grad itself) and any `extra_hard` runs
    ((name, loss_grad) pairs, e.g. another adjoint mode). All sample points,
    the exact-gradient points and (on log iterations) the current points share
    ONE batched loss_grad call per iteration: the per-rollout overhead
    dominates at these batch sizes. `project` maps sample points into the
    parameter box before evaluation (None: no clamping)."""
    starts_n = len(u0)
    configs = [(m, name) for m in SAMPLED for name in sigmas]
    u = u0.repeat(len(configs), 1).reshape(len(configs), starts_n, 2)
    hard_u = [u0.clone() for _ in range(1 + len(extra_hard))]
    opt, hard_opt = Adam(u.shape), [Adam(u0.shape) for _ in hard_u]
    log_rollouts, log_loss, hard_loss = [], [], [[] for _ in hard_u]
    n_cfg, n_s = len(configs), starts_n * 2 * n_pairs
    proj = project or (lambda x: x)
    sched = lambda name, it: sigmas[name][0] * (sigmas[name][1] / sigmas[name][0]) ** (it / max(iters - 1, 1))
    t0 = time.time()
    for it in range(iters + 1):
        sig = torch.tensor([sched(name, min(it, iters - 1)) for _, name in configs], dtype=torch.float64)
        eps = torch.stack([torch.stack([est.directions(n_pairs, 2, generator=gen) for _ in range(starts_n)])
                           for _ in configs])  # (cfg, start, N, 2)
        s = sig[:, None, None, None]
        pts = torch.cat([u[:, :, None] + s * eps, u[:, :, None] - s * eps], dim=2)  # (cfg, start, 2N, 2)
        log = it % LOG_EVERY == 0 or it == iters
        batch = [proj(pts.reshape(-1, 2)), hard_u[0]] + ([u.reshape(-1, 2)] if log else [])
        L_all, g_all = loss_grad(torch.cat(batch))
        hard_g = [g_all[n_cfg * n_s : n_cfg * n_s + starts_n]]
        hard_loss[0].append(L_all[n_cfg * n_s : n_cfg * n_s + starts_n])
        for k, (_, lg) in enumerate(extra_hard, start=1):
            L_k, g_k = lg(hard_u[k])
            hard_loss[k].append(L_k)
            hard_g.append(g_k)
        if log:
            log_rollouts.append(it * 2 * n_pairs)
            log_loss.append(L_all[n_cfg * n_s + starts_n :].reshape(n_cfg, starts_n))
        if it == iters:
            break
        L = L_all[: n_cfg * n_s].reshape(n_cfg, starts_n, 2 * n_pairs)
        g = g_all[: n_cfg * n_s].reshape(n_cfg, starts_n, 2 * n_pairs, 2)
        Lp, Lm, gp, gm = L[..., :n_pairs], L[..., n_pairs:], g[..., :n_pairs, :], g[..., n_pairs:, :]
        grads = torch.zeros_like(u)
        for ci, (m, _) in enumerate(configs):
            if m == "first":
                grads[ci] = est.first_order_from_samples(gp[ci], gm[ci])
            elif m == "zeroth":
                grads[ci] = est.zeroth_order_from_samples(eps[ci], Lp[ci], Lm[ci], sig[ci].item())
            else:
                grads[ci] = est.alpha_order_from_samples(eps[ci], Lp[ci], Lm[ci], gp[ci], gm[ci], sig[ci].item(), GAMMA)[0]
        u = opt.step(u, grads)
        hard_u = [o.step(x, gk) for o, x, gk in zip(hard_opt, hard_u, hard_g)]
        if it % 25 == 0:
            print(f"iter {it}: {time.time() - t0:.0f} s, median L per config "
                  + " ".join(f"{x:.3f}" for x in log_loss[-1].median(dim=1).values.tolist())
                  + " | hard " + " ".join(f"{h[-1].median():.3f}" for h in hard_loss), flush=True)
    print(f"done: {time.time() - t0:.0f} s", flush=True)
    return {"configs": np.array([f"{m}|{n}" for m, n in configs]), "rollouts": np.array(log_rollouts),
            "loss": torch.stack(log_loss).numpy(), "u_final": u.numpy(),
            "hard_names": np.array(["exact"] + [n for n, _ in extra_hard]),
            "hard_rollouts": np.arange(iters + 1), "hard_loss": np.stack([torch.stack(h).numpy() for h in hard_loss]),
            "u0": u0.numpy()}


def main():
    sim = bl.make_sim()
    gen = torch.Generator().manual_seed(1)
    u0 = starts(gen)
    loss_grad = lambda u: batched(lambda x: bl.loss_and_grad(sim, x), u)
    out = optimize(loss_grad, u0, SIGMAS, gen)
    out["hard_loss"] = out["hard_loss"][0]  # the ball file format: one exact run
    np.savez(bl.RESULTS / "optimize_ball.npz", **out)
    report()


EXACT_ITERS = 2000
EXACT_VARIANTS = ("lr 0.01", "lr 0.01 -> 0.001 (cosine)")


def exact_control():
    """Control for the exact gradient: the same rollout budget as the sampled
    estimators (EXACT_ITERS steps of one rollout), with a constant and a
    cosine-decayed learning rate, from the same starts. Saves partial results
    every 100 steps so it can be stopped early."""
    sim = bl.make_sim()
    u0 = starts(torch.Generator().manual_seed(1))
    n = len(EXACT_VARIANTS)
    u = u0.repeat(n, 1)
    opt = Adam(u.shape)
    lr_min = 0.1 * LR
    rollouts, losses = [], []
    t0 = time.time()
    for it in range(EXACT_ITERS + 1):
        L, g = bl.loss_and_grad(sim, u)
        if it % LOG_EVERY == 0 or it == EXACT_ITERS:
            rollouts.append(it)
            losses.append(L.reshape(n, STARTS))
        if it % 100 == 0 or it == EXACT_ITERS:
            np.savez(bl.RESULTS / "optimize_ball_exact.npz", variants=np.array(EXACT_VARIANTS),
                     rollouts=np.array(rollouts), loss=torch.stack(losses).numpy(), done=it == EXACT_ITERS)
            print(f"iter {it}: {time.time() - t0:.0f} s, median L " +
                  " ".join(f"{x:.4f}" for x in losses[-1].median(dim=1).values.tolist()), flush=True)
        if it == EXACT_ITERS:
            break
        cosine = lr_min + 0.5 * (LR - lr_min) * (1 + math.cos(math.pi * it / EXACT_ITERS))
        opt.lr = torch.tensor([LR] * STARTS + [cosine] * STARTS, dtype=torch.float64)[:, None]
        u = opt.step(u, g)
    report()


def report():
    d = np.load(bl.RESULTS / "optimize_ball.npz")
    configs = [c.split("|") for c in d["configs"]]
    hard_final = d["hard_loss"][-1]
    print(f"\n{'method':>8} {'sigma':>14} {'median L':>9} {'success':>8}   (final; success = L < 0.01; {STARTS} starts)")
    print(f"{'exact':>8} {'-':>14} {np.median(hard_final):9.4f} {np.mean(hard_final < 0.01):8.0%}   ({ITERS} rollouts + adjoints)")
    for k, (m, name) in enumerate(configs):
        final = d["loss"][-1, k]
        print(f"{m:>8} {name:>14} {np.median(final):9.4f} {np.mean(final < 0.01):8.0%}")
    exact = load_exact()
    if exact is not None:
        for k, name in enumerate(exact["variants"]):
            final = exact["loss"][-1, k]
            print(f"{'exact':>8} {name:>14} {np.median(final):9.4f} {np.mean(final < 0.01):8.0%}"
                  f"   ({exact['rollouts'][-1]} rollouts + adjoints{'' if exact['done'] else ', partial'})")
    plot(d, configs)


def load_exact():
    path = bl.RESULTS / "optimize_ball_exact.npz"
    return dict(np.load(path)) if path.exists() else None


def plot(d, configs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    styles = {"first": ("first-order", "#2a78d6", "o"), "zeroth": ("zeroth-order", "#eb6834", "s"),
              "alpha": ("alpha-order", "#1baf7a", "^")}
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(12, 3.9), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, SIGMAS):
        exact = load_exact()
        if exact is not None:
            for k, ls in ((0, "--"), (1, ":")):
                ax.plot(np.maximum(exact["rollouts"], 1), np.median(exact["loss"][:, k], axis=1),
                        color=muted, lw=1.2, ls=ls, label=f"exact, {exact['variants'][k]}")
        else:
            ax.plot(np.maximum(d["hard_rollouts"], 1), np.median(d["hard_loss"], axis=1), color=muted, lw=1.2, ls="--",
                    label="exact")
        for k, (m, n) in enumerate(configs):
            if n != name:
                continue
            label, color, marker = styles[m]
            ax.plot(np.maximum(d["rollouts"], 1), np.median(d["loss"][:, k], axis=1), color=color, lw=2,
                    marker=marker, ms=4, markevery=5, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"sigma = {name}", color=ink, fontsize=11, loc="left")
        ax.set_xlabel("forward rollouts", color=muted)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(colors=muted)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel(f"median deterministic loss ({STARTS} starts)", color=muted)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="outside upper right", ncols=5, fontsize=9)
    fig.suptitle("Ball, terminal-state loss: Adam with each gradient estimator",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "optimize_ball.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'optimize_ball.png'}")


if __name__ == "__main__":
    main()
