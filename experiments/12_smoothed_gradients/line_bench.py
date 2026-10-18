"""Directional smoothing on the simulator: 1D estimators along random lines.

A directional gradient (Zhang, Tran, Bao 2020) is sum_i D_i v_i over
orthonormal directions v_i, D_i the derivative of the loss smoothed along v_i
alone, so its error is the root sum of per-line errors. This benchmark takes
dense 1D grids of the loss (801 nodes over +- 4 sigma) along random lines
through a d-dimensional test point, the truth D by Stein quadrature of the
grid, and the 1D estimators on samples read from the grid (nearest node):

  gh       Gauss-Hermite quadrature with 2K nodes (the paper's estimator)
  m2       antithetic Stein mean, K pairs
  linreg   slope of a linear fit, K pairs
  c        method C (jump located by bisection on the grid), K pairs

Error: root mean square over lines and repeats of (estimate - D), relative
to the root mean square of D over the lines (the directional gradient's
relative error). Cost: simulations per line (C: as used).

  PYTHONPATH=. python line_bench.py 'results/night/linegrid_wheel_l*.npz'
"""

import glob
import math
import sys

import numpy as np

import method_c as mc

K_PAIRS = (2, 4, 8, 16, 32)
REPEATS = 60


def evaluate(pattern, seed=0):
    lines = [np.load(p) for p in sorted(glob.glob(pattern))]
    sigma = (lines[0]["s"][-1] - lines[0]["s"][0]) / 8
    truth = []
    for z in lines:
        w = np.exp(-0.5 * (z["s"] / sigma) ** 2)
        w /= w.sum()
        truth.append((w * z["L"] * z["s"]).sum() / (w * z["s"] ** 2).sum())
    truth = np.array(truth)
    scale = math.sqrt((truth**2).mean())
    rng = np.random.default_rng(seed)
    out = {}
    for K in K_PAIRS:
        xg, wg = np.polynomial.hermite.hermgauss(2 * K)
        sq = {k: [] for k in ("gh", "m2", "linreg", "c")}
        cost = []
        for z, D in zip(lines, truth):
            s, L = z["s"], z["L"]
            f = lambda x: L[np.clip(np.rint((np.asarray(x) - s[0]) / (s[1] - s[0])).astype(int), 0, len(s) - 1)]
            gh = (f(math.sqrt(2) * sigma * xg) * wg * math.sqrt(2) * xg).sum() / (sigma * math.sqrt(math.pi))
            sq["gh"].append((gh - D) ** 2)
            for _ in range(REPEATS):
                e = rng.standard_normal(K)
                yp, ym = f(sigma * e), f(-sigma * e)
                sq["m2"].append((((yp - ym) * e).mean() / (2 * sigma) - D) ** 2)
                x = np.concatenate([sigma * e, -sigma * e])
                sq["linreg"].append((np.polyfit(x, np.concatenate([yp, ym]), 1)[0] - D) ** 2)
                g, n, _ = mc.estimate_1d(f, 0.0, sigma, e)
                sq["c"].append((g - D) ** 2)
                cost.append(n)
        for k, v in sq.items():
            out[(k, K)] = (2 * K if k != "c" else np.mean(cost), math.sqrt(np.mean(v)) / scale)
    return out, len(lines), scale


if __name__ == "__main__":
    out, n, scale = evaluate(sys.argv[1])
    print(f"{sys.argv[1]}: {n} lines, rms directional derivative {scale:.3f}")
    for K in K_PAIRS:
        print(f"  K={K:2d}  " + "  ".join(f"{k} {out[(k, K)][1]:.3f}({out[(k, K)][0]:.0f})" for k in ("gh", "m2", "linreg", "c")))
