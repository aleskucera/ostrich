"""Estimators on a pool of antithetic samples: offline repeats at one point.

A pool holds P antithetic pairs at a test point theta (offsets eps (P, d),
losses at theta +- sigma eps, adjoint gradients there). Each repeat draws N
pairs without replacement and applies every estimator to them; the
reference for that repeat is the zeroth-order (antithetic) mean over the
other P - N pairs, independent of the repeat's samples, so
E|g_hat - g_ref|^2 = E|g_hat - g|^2 + se_ref^2 and the reported RMSE has
se_ref^2 subtracted. Errors are relative to |g_ref|. An external reference
(an independent large run) can be given instead.

Estimators (cost in forward-rollout equivalents; ADJ adjoint cost factor):
  zeroth_order  antithetic Stein mean (M2)                       2N
  linreg        slope of a linear least-squares fit              2N
  gated         two linear experts with a logistic gate (ours)   2N
  first_order   mean adjoint gradient (M1)                       2N ADJ
  alpha_order   Suh et al. 2022 (M3), gamma 2                    2N ADJ
  ddcg          Onoda, Parmas et al. 2026                        2N ADJ

  PYTHONPATH=. python pool_eval.py 'results/night/pool_box_d10_F0.7_p*.npz' [ADJ]
"""

import glob
import math
import sys

import numpy as np
import torch

import estimators as est
import method_c as mc

N_PAIRS = (8, 16, 32, 64, 128)
REPEATS = 100
NAMES = ("zeroth_order", "linreg", "gated", "first_order", "alpha_order", "ddcg")


def load(pattern):
    fs = sorted(glob.glob(pattern))
    eps, Lp, Lm, gp, gm = [], [], [], [], []
    for p in fs:
        z = np.load(p)
        n = len(z["eps"])
        eps.append(z["eps"])
        Lp.append(z["L"][:n])
        Lm.append(z["L"][n:])
        if "g" in z:
            gp.append(z["g"][:n])
            gm.append(z["g"][n:])
    sigma = float(np.load(fs[0])["sigma"]) if "sigma" in np.load(fs[0]) else 0.05
    cat = lambda a: np.concatenate(a) if a else None
    return cat(eps), cat(Lp), cat(Lm), cat(gp), cat(gm), sigma


def evaluate(pattern, adj=1.86, seed=0, ref_ext=None):
    """ref_ext: (gradient, standard error) of an independent reference."""
    eps, Lp, Lm, gp, gm, sigma = load(pattern)
    P, d = eps.shape
    terms = (Lp - Lm)[:, None] * eps / (2 * sigma)
    total = terms.sum(axis=0)
    var_terms = terms.var(axis=0).sum()
    names = [n for n in NAMES if gp is not None or n in ("zeroth_order", "linreg", "gated")]
    rng = np.random.default_rng(seed)
    out = {}
    for N in N_PAIRS:
        if N > P // 8:
            break
        sq, rel, scale = {n: [] for n in names}, {n: [] for n in names}, []
        for _ in range(REPEATS):
            i = rng.choice(P, N, replace=False)
            if ref_ext is None:
                ref = (total - terms[i].sum(axis=0)) / (P - N)
                se2 = var_terms / (P - N)
            else:
                ref, se2 = ref_ext[0], ref_ext[1] ** 2
            e = eps[i]
            z = np.concatenate([e, -e]) * sigma
            y = np.concatenate([Lp[i], Lm[i]])
            g = {"zeroth_order": terms[i].mean(axis=0)}
            g["linreg"] = mc._lin(np.hstack([np.ones((2 * N, 1)), z]), y)[0][1:]
            model = mc.fit_gated(z, y, sigma)
            g["gated"] = g["linreg"] if model is None else mc.gated_grad(*model[:3], sigma)
            if gp is not None:
                E, yp, ym = torch.tensor(e)[None], torch.tensor(Lp[i])[None], torch.tensor(Lm[i])[None]
                Gp, Gm = torch.tensor(gp[i])[None], torch.tensor(gm[i])[None]
                g["first_order"] = est.first_order_from_samples(Gp, Gm)[0].numpy()
                g["alpha_order"] = est.alpha_order_from_samples(E, yp, ym, Gp, Gm, sigma, 2.0)[0][0].numpy()
                g["ddcg"] = est.ddcg_from_samples(E, yp, ym, Gp, Gm, sigma)[0][0].numpy()
            sc = np.linalg.norm(ref)
            scale.append(sc)
            for n in names:
                err2 = float(((g[n] - ref) ** 2).sum())
                sq[n].append(err2 - se2)
                rel[n].append(math.sqrt(err2) / sc)
        sc = np.mean(scale)
        for n in names:
            cost = 2 * N * (adj if n in ("first_order", "alpha_order", "ddcg") else 1)
            out[(n, N)] = (cost, np.median(rel[n]), math.sqrt(max(np.mean(sq[n]), 0)) / sc)
    return out, P, d, math.sqrt(var_terms / P) / np.linalg.norm(total / P)


if __name__ == "__main__":
    pattern = sys.argv[1]
    adj = float(sys.argv[2]) if len(sys.argv) > 2 else 1.86
    ref_ext = None
    if len(sys.argv) > 3:  # external reference: files with eps, L (antithetic, sigma as the pool)
        eps_r = np.concatenate([np.load(p)["eps"] for p in sorted(glob.glob(sys.argv[3]))])
        L_r = [np.load(p)["L"] for p in sorted(glob.glob(sys.argv[3]))]
        D_r = np.concatenate([l[: len(l) // 2] - l[len(l) // 2 :] for l in L_r])
        t_r = D_r[:, None] * eps_r / (2 * load(pattern)[5])
        ref_ext = (t_r.mean(axis=0), math.sqrt(t_r.var(axis=0).sum() / len(t_r)))
        print(f"external reference: {len(t_r)} pairs, |g| {np.linalg.norm(ref_ext[0]):.3f}, se {ref_ext[1]:.3f}")
    out, P, d, rse = evaluate(pattern, adj, ref_ext=ref_ext)
    print(f"{pattern}: {P} pairs, d = {d}, reference relative se {rse:.1%} (full pool)")
    for N in N_PAIRS:
        row = [(n, *out[(n, N)]) for n in NAMES if (n, N) in out]
        if row:
            print(f"  N={N:3d}  " + "  ".join(f"{n} {med:.3f}/{rmse:.3f}({c:.0f})" for n, c, med, rmse in row))
