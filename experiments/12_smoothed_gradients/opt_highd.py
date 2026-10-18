"""Optimization in d dimensions: the wheel's torque profile, live simulator.

Adam on u in [0, 1]^d (d torque phases) minimizing the deterministic loss
(x(T) - X_TARGET)^2, driven by a smoothed-gradient estimator with sigma
annealed geometrically from SIGMA0 to SIGMA0 / 30 over the budget (as in
bench_optimize). Starts: the wheel stalls at the ledge (u_start per terrain
+ a uniform offset per phase), so the optimizer must find the climb.

Estimators (N antithetic pairs per step; cost = simulations actually used):
  zeroth_order  antithetic Stein average (M2)
  c_nojump      slope of one linear fit to the 2N samples (ablation)
  gated         two linear experts with a logistic gate (method_c)
  method_c      planar jump located by bisection (method_c.estimate_nd)

  PYTHONPATH=. python opt_highd.py run D AMP METHOD N LR_SCALE SEED0 SEED1 TAG
Writes results/opt_highd/<tag>_<method>_N<N>_lr<c>_s<seed>.npz with the
iterate's loss against the simulations spent.
"""

import sys

import numpy as np
import torch

import ball_landscape as bl
import method_c as mc
import wheel_ledge as wl

SIGMA0 = 0.05
SIGMA_END = 1 / 30
BUDGET = 3000
U_START = {0.0: 0.30, 0.004: 0.38}  # stalls at the ledge for every start offset
START_OFFSET = 0.05
OUT = bl.RESULTS / "opt_highd"


def start(d, amp, seed):
    return np.clip(U_START[amp] + START_OFFSET * np.random.default_rng([d, seed]).uniform(-1, 1, d), 0, 1)


def run(d, amp, method, N, c, seed):
    sim = wl.make_sim(amp=amp)
    f = lambda x: wl.loss(sim, torch.tensor(np.atleast_2d(x)), chunk=512).numpy()
    rng = np.random.default_rng([d, seed, 1])
    u = start(d, amp, seed)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2 = 0.9, 0.999
    spent, k = 0, 0
    hist = []
    while spent < BUDGET:
        sigma = SIGMA0 * SIGMA_END ** (spent / BUDGET)
        eps = rng.standard_normal((N, d))
        if method == "zeroth_order":
            y = f(np.concatenate([u[None], u + sigma * eps, u - sigma * eps]))
            L0, yp, ym = y[0], y[1 : N + 1], y[N + 1 :]
            g = ((yp - ym)[:, None] * eps).mean(axis=0) / (2 * sigma)
            used = 2 * N
        else:
            cache = {}

            def fc(x):
                x = np.atleast_2d(x)
                keys = [tuple(np.round(xi, 12)) for xi in x]
                new = [i for i, kk in enumerate(keys) if kk not in cache]
                if new:
                    for i, val in zip(new, f(x[new])):
                        cache[keys[i]] = val
                return np.array([cache[kk] for kk in keys])

            L0 = f(u[None])[0]
            if method == "c_nojump":
                g, used, _ = mc.estimate_nd(fc, u, sigma, eps, jumps=False)
            elif method == "gated":
                g, used, _ = mc.estimate_gated(fc, u, sigma, eps)
            else:
                g, used, _ = mc.estimate_nd(fc, u, sigma, eps, refine=4, slopes="loo")
        hist.append((spent, L0))
        k += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        step = c * SIGMA0 * (m / (1 - b1**k)) / (np.sqrt(v / (1 - b2**k)) + 1e-12)
        u = np.clip(u - step, 0, 1)
        spent += used
    hist.append((spent, f(u[None])[0]))
    return np.array(hist), u


if __name__ == "__main__":
    _, d, amp, method, N, c, s0, s1, tag = sys.argv[1:]
    d, amp, N, c, s0, s1 = int(d), float(amp), int(N), float(c), int(s0), int(s1)
    OUT.mkdir(exist_ok=True)
    for seed in range(s0, s1):
        hist, u = run(d, amp, method, N, c, seed)
        np.savez(OUT / f"{tag}_{method}_N{N}_lr{c}_s{seed}.npz", hist=hist, u=u)
        print(f"{tag} {method} N={N} lr={c} seed {seed}: final loss {hist[-1, 1]:.4f} best {hist[:, 1].min():.4f}", flush=True)
