"""Optimization in d dimensions on the live simulator.

Adam on u in [0, 1]^d minimizing the deterministic loss, driven by a
smoothed-gradient estimator with sigma annealed geometrically from SIGMA0 to
SIGMA0 * SIGMA_END over the budget (as in bench_optimize).

Scenarios:
  wheel  d torque phases (wheel_ledge, terrain bump amplitude AMP); starts
         stall at the ledge (U_START + a uniform offset per phase)
  box    d / 2 push phases (box_phases); starts push too weakly to reach
         the target (BOX_START + offsets), the optimum lies next to tipping

Estimators (cost = simulations actually used):
  zeroth_order  antithetic Stein mean (M2), N pairs
  linreg        slope of one linear fit to the 2N samples
  gated         two linear experts with a logistic gate (method_c)
  gated_sub     gated with the switch in SUB_K directions (quadratic) from
                adjoint directions accumulated over the steps (SUB_PAIRS
                adjoint pairs per step, decay SUB_DECAY; their cost counted)
  dgs_gh        directional smoothing, Gauss-Hermite with 2K nodes per line
                (Zhang, Tran, Bao 2020), a random orthonormal basis per step
  dgs_c         directional smoothing, method C on each line (K pairs)

  PYTHONPATH=. python opt_highd.py SCEN D AMP METHOD N LR_SCALE SEED0 SEED1 TAG
(N is K per line for dgs_*). Isotropic methods draw directions in orthogonal
blocks (estimators.directions). Writes results/opt_highd/<tag>_<method>_N<N>_
lr<c>_s<seed>.npz with the iterate's loss against the simulations spent.
"""

import sys

import numpy as np
import torch

import ball_landscape as bl
import estimators as est
import method_c as mc

SIGMA0 = 0.05
SIGMA_END = 1 / 30
BUDGET = {"wheel": 3000, "box": 4000}
U_START = {0.0: 0.30, 0.004: 0.38}  # wheel: stalls at the ledge for every start offset
BOX_START = (0.5, 0.5)  # box: slides short of the target, upright
START_OFFSET = 0.05
SUB_K, SUB_PAIRS, SUB_DECAY = 4, 4, 0.9
ADJ = {"wheel": 1.36, "box": 1.86}
OUT = bl.RESULTS / "opt_highd"


def scenario(scen, d, amp):
    """(loss(x), loss_and_grad(x) -> (losses, adjoint gradients), start)."""
    if scen == "wheel":
        import wheel_ledge as wl

        sim = wl.make_sim(amp=amp)
        lg = lambda x: tuple(t.numpy() for t in wl.loss_and_grad(sim, torch.tensor(np.atleast_2d(x)), chunk=512))
        return lambda x: wl.loss(sim, torch.tensor(np.atleast_2d(x)), chunk=512).numpy(), lg, np.full(d, U_START[amp])
    import box_phases as bp
    import box_tip as bt

    sim = bt.make_sim()
    lg = lambda x: tuple(t.numpy() for t in bp.loss_and_grad(sim, torch.tensor(np.atleast_2d(x)), chunk=512))
    return lambda x: bp.loss(sim, torch.tensor(np.atleast_2d(x)), chunk=512).numpy(), lg, np.tile(BOX_START, d // 2)


def gradient(method, f, u, sigma, N, rng, gen, state):
    """Estimate and simulations used. state: per-run memory (gated_sub's
    accumulated adjoint direction matrix)."""
    d = len(u)
    if method == "dgs_gh":
        V = np.linalg.qr(rng.standard_normal((d, d)))[0].T
        g, used, _ = mc.estimate_directional(f, u, sigma, V, np.polynomial.hermite.hermgauss(2 * N), mode="gh")
        return g, used
    if method == "dgs_c":
        V = np.linalg.qr(rng.standard_normal((d, d)))[0].T
        g, used, _ = mc.estimate_directional(f, u, sigma, V, rng.standard_normal(N), mode="c")
        return g, used
    eps = est.directions(N, d, orthogonal=True, generator=gen).numpy()
    if method == "gated_sub":
        e = est.directions(SUB_PAIRS, d, orthogonal=True, generator=gen).numpy()
        _, G = state["lg"](np.concatenate([u + sigma * e, u - sigma * e]))
        Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-300)
        state["C"] = SUB_DECAY * state.get("C", np.zeros((d, d))) + Gn.T @ Gn
        U = np.linalg.eigh(state["C"])[1][:, ::-1][:, :SUB_K]
        g, used, _ = mc.estimate_gated_subspace(f, u, sigma, eps, U)
        return g, used + 2 * SUB_PAIRS * state["adj"]
    if method == "zeroth_order":
        y = f(np.concatenate([u + sigma * eps, u - sigma * eps]))
        return ((y[:N] - y[N:])[:, None] * eps).mean(axis=0) / (2 * sigma), 2 * N
    if method == "linreg":
        g, used, _ = mc.estimate_nd(f, u, sigma, eps, jumps=False)
        return g, used
    g, used, _ = mc.estimate_gated(f, u, sigma, eps)
    return g, used


def run(scen, d, amp, method, N, c, seed):
    f, lg, u0 = scenario(scen, d, amp)
    rng = np.random.default_rng([d, seed, 1])
    gen = torch.Generator().manual_seed(1000 * d + seed)
    state = {"lg": lg, "adj": ADJ[scen]}
    u = np.clip(u0 + START_OFFSET * np.random.default_rng([d, seed]).uniform(-1, 1, d), 0, 1)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2 = 0.9, 0.999
    spent, k = 0, 0
    hist = []
    while spent < BUDGET[scen]:
        sigma = SIGMA0 * SIGMA_END ** (spent / BUDGET[scen])
        hist.append((spent, f(u[None])[0]))  # monitoring only, not counted
        g, used = gradient(method, f, u, sigma, N, rng, gen, state)
        k += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        u = np.clip(u - c * SIGMA0 * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + 1e-12), 0, 1)
        spent += used
    hist.append((spent, f(u[None])[0]))
    return np.array(hist), u


if __name__ == "__main__":
    scen, d, amp, method, N, c, s0, s1, tag = sys.argv[1:]
    d, amp, N, c, s0, s1 = int(d), float(amp), int(N), float(c), int(s0), int(s1)
    OUT.mkdir(exist_ok=True)
    for seed in range(s0, s1):
        hist, u = run(scen, d, amp, method, N, c, seed)
        np.savez(OUT / f"{tag}_{method}_N{N}_lr{c}_s{seed}.npz", hist=hist, u=u)
        print(f"{tag} {method} N={N} lr={c} seed {seed}: final loss {hist[-1, 1]:.4f} best {hist[:, 1].min():.4f}", flush=True)
