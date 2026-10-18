"""Wheel and ledge with d torque phases: gradient estimators against a reference.

No dense grid exists in d dimensions, so the reference smoothed gradient is
a large antithetic zeroth-order run (unbiased; its standard error is kept
and reported). Estimators are run live on the simulator: hard derivatives
for first-order, alpha-order and DDCG come from the adjoint on the same 2N
samples; method C uses the same samples plus its own bisection simulations.

Test points: torque profiles near the climbing threshold, u0 + a fixed
random offset per phase (THETA_OFFSET), so the jump lies inside the blur.

  ref:   PYTHONPATH=. python highd_wheel.py ref D K PART NPARTS
  est:   PYTHONPATH=. python highd_wheel.py est D K N R0 R1
  merge: PYTHONPATH=. python highd_wheel.py merge D
Writes results/highd/*.npz and results/highd_wheel_D.npz.
"""

import sys
import time

import numpy as np
import torch

import ball_landscape as bl
import estimators as est
import method_c as mc
import wheel_ledge as wl

SIGMA = 0.05
U0 = (0.30, 0.34, 0.38, 0.45)
THETA_OFFSET = 0.03
REF_PAIRS = 4096  # per part
N_PAIRS = (8, 32, 128)
OUT = bl.RESULTS / "highd"


def theta(d, k):
    rng = np.random.default_rng(1000 + d)
    return np.clip(U0[k] + THETA_OFFSET * rng.uniform(-1, 1, (len(U0), d))[k], 0, 1)


def loss(sim, x):
    return wl.loss(sim, torch.tensor(np.atleast_2d(x)), chunk=512).numpy()


def run_ref(d, k, part, nparts):
    sim, th = wl.make_sim(), theta(d, k)
    eps = np.random.default_rng([d, k, part]).standard_normal((REF_PAIRS, d))
    t0 = time.time()
    Lp, Lm = loss(sim, th + SIGMA * eps), loss(sim, th - SIGMA * eps)
    terms = (Lp - Lm)[:, None] * eps / (2 * SIGMA)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"ref_d{d}_k{k}_p{part}.npz", s1=terms.sum(0), s2=(terms**2).sum(0), n=len(terms))
    print(f"ref d={d} k={k} part {part}/{nparts}: {time.time() - t0:.0f} s", flush=True)


def run_est(d, k, N, r0, r1):
    sim, th = wl.make_sim(), theta(d, k)
    f = lambda x: loss(sim, x)
    rows = []
    for r in range(r0, r1):
        eps = np.random.default_rng([d, k, N, r, 7]).standard_normal((N, d))
        x = np.concatenate([th + SIGMA * eps, th - SIGMA * eps])
        L, g = wl.loss_and_grad(sim, torch.tensor(x), chunk=512)
        Lp, Lm, gp, gm = L[:N][None], L[N:][None], g[:N][None], g[N:][None]
        e = torch.tensor(eps)[None]
        ests = {
            "zeroth_order": est.zeroth_order_from_samples(e, Lp, Lm, SIGMA)[0].numpy(),
            "alpha_order": est.alpha_order_from_samples(e, Lp, Lm, gp, gm, SIGMA, 2.0)[0][0].numpy(),
            "ddcg": est.ddcg_from_samples(e, Lp, Lm, gp, gm, SIGMA)[0][0].numpy(),
        }
        # method C: reuses the 2N sample losses, simulates only its own extra points
        cache = {tuple(np.round(xi, 12)): li for xi, li in zip(x, L.numpy())}
        extra = [0]

        def f_c(z):
            out = []
            for xi in np.atleast_2d(z):
                key = tuple(np.round(xi, 12))
                if key not in cache:
                    cache[key] = f(xi)[0]
                    extra[0] += 1
                out.append(cache[key])
            return np.array(out)

        gc, _, info = mc.estimate_nd(f_c, th, SIGMA, eps)
        gn, _, _ = mc.estimate_nd(f_c, th, SIGMA, eps, jumps=False)
        ests["method_c"], ests["c_nojump"] = gc, gn
        rows.append({"r": r, "extra": extra[0], "jump": info["jump"] is not None, **ests})
        print(f"est d={d} k={k} N={N} r={r}: C extra sims {extra[0]}, jump {info['jump'] is not None}", flush=True)
    OUT.mkdir(exist_ok=True)
    names = ("zeroth_order", "alpha_order", "ddcg", "method_c", "c_nojump")
    np.savez(OUT / f"est_d{d}_k{k}_N{N}_r{r0}-{r1}.npz", extra=[w["extra"] for w in rows],
             jump=[w["jump"] for w in rows], **{n: np.array([w[n] for w in rows]) for n in names})


def merge(d):
    names = ("zeroth_order", "alpha_order", "ddcg", "method_c", "c_nojump")
    adj = 1.36
    cost = {"zeroth_order": lambda N, ex: 2 * N, "alpha_order": lambda N, ex: 2 * N * adj,
            "ddcg": lambda N, ex: 2 * N * adj, "method_c": lambda N, ex: 2 * N + ex, "c_nojump": lambda N, ex: 2 * N}
    refs, ref_se, out = [], [], {}
    for k in range(len(U0)):
        parts = sorted(OUT.glob(f"ref_d{d}_k{k}_p*.npz"))
        s1 = sum(np.load(p)["s1"] for p in parts)
        s2 = sum(np.load(p)["s2"] for p in parts)
        n = sum(int(np.load(p)["n"]) for p in parts)
        mean = s1 / n
        refs.append(mean)
        ref_se.append(np.sqrt(np.maximum(s2 / n - mean**2, 0).sum() / n))
    scale = np.mean([np.linalg.norm(r) for r in refs])
    print(f"d={d}: reference |grad| per theta {[f'{np.linalg.norm(r):.2f}' for r in refs]}, "
          f"standard error {[f'{s:.3f}' for s in ref_se]} (relative to scale {np.mean(ref_se) / scale:.1%})")
    print(f"  {'estimator':13s} {'N':>4s} {'cost':>7s}  err median [q95]")
    for N in N_PAIRS:
        for name in names:
            errs, costs = [], []
            for k in range(len(U0)):
                for p in sorted(OUT.glob(f"est_d{d}_k{k}_N{N}_r*.npz")):
                    r = np.load(p)
                    errs += list(np.linalg.norm(r[name] - refs[k], axis=1) / scale)
                    costs += [cost[name](N, ex) for ex in r["extra"]]
            if errs:
                out[f"{name}_N{N}"] = np.array([np.mean(costs), np.median(errs), np.quantile(errs, 0.95)])
                print(f"  {name:13s} {N:4d} {np.mean(costs):7.1f}  {np.median(errs):.3f} [{np.quantile(errs, 0.95):.3f}]")
    np.savez(bl.RESULTS / f"highd_wheel_{d}.npz", refs=np.array(refs), ref_se=np.array(ref_se), scale=scale, **out)


if __name__ == "__main__":
    cmd, d = sys.argv[1], int(sys.argv[2])
    if cmd == "ref":
        run_ref(d, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "est":
        run_est(d, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        merge(d)
