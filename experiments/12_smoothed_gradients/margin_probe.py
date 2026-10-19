"""Margin probe: is there a smooth number whose sign says whether the box tips?

For the d-dim box (box_phases) around a pool point th, runs P antithetic
orthogonal pairs at blur sigma = 0.05 and logs per run the loss, whether the
box tipped (|theta(T)| > 1) and candidate margins:

  e_endpush   energy over the corner barrier at the end of the push,
              e = 1/2 I_c w^2 + m g (y - r), r = hypot(HX, HY), I_c the
              inertia about the corner
  e_max_all   max of e over the whole run
  e_max_after max of e after the push
  tilt_clip   min(max |theta|, 1.5) - atan(HX/HY)
  e_cross     e at the first step with |theta| >= atan(HX/HY), else max e

For each: sign agreement with the tip outcome, held-out (every 4th run) R^2
of a linear and a quadratic fit in z, and held-out tip classification by the
sign of the fitted margin.

  PYTHONPATH=. python margin_probe.py D A P OUT.npz
"""

import math
import sys

import numpy as np
import torch

import box_phases as bp
import box_tip as bt

SIGMA = 0.05
G = 9.81
R_CORNER = math.hypot(bt.HX, bt.HY)
I_CORNER = bt.MASS * (4 * bt.HX**2 + 4 * bt.HY**2) / 12 + bt.MASS * R_CORNER**2
CRIT = math.atan(bt.HX / bt.HY)


def pool_point(d, a):
    th = np.tile([a, 0.5], d // 2) + 0.03 * np.random.default_rng(2000 + d).uniform(-1, 1, d)
    return np.clip(th, 0, 1)


def orth_pairs(d, P, rng):
    """P Gaussian directions, orthogonal within blocks of d."""
    out = []
    while len(out) < P:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        out.extend(Q.T * np.sqrt(rng.chisquare(d, d))[:, None])
    return np.array(out[:P])


def trajectories(sim, u, chunk=512):
    Q, V = [], []
    for u_c in torch.split(u, chunk):
        q, v = bt.initial_state(len(u_c))
        qs, vs, lam = [], [], None
        for f in bp.forces(u_c):
            q, v, info = sim.step(q, v, f, lam0=lam)
            lam = info["z"][:, 3:]
            qs.append(q)
            vs.append(v)
        Q.append(torch.stack(qs))
        V.append(torch.stack(vs))
    return torch.cat(Q, 1).numpy(), torch.cat(V, 1).numpy()


def energy_margin(q, v):
    """Tilt |theta| and energy over the corner barrier per step (T, B), and the
    margin e_cross: e at the first step with tilt >= CRIT, else max e."""
    tilt = np.abs(q[:, :, 2])
    e = 0.5 * I_CORNER * v[:, :, 2] ** 2 + bt.MASS * G * (q[:, :, 1] - R_CORNER)
    crossed = tilt >= CRIT
    first = crossed.argmax(0)
    return tilt, e, np.where(crossed.any(0), e[first, np.arange(q.shape[1])], e.max(0))


def features(Z, quad):
    cols = [np.ones(len(Z)), *Z.T]
    if quad:
        d = Z.shape[1]
        cols += [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, 1)


def heldout(Z, y, tip, quad):
    test = np.arange(len(Z)) % 4 == 0
    X = features(Z, quad)
    A = X[~test]
    c = np.linalg.solve(A.T @ A + 1e-6 * np.eye(X.shape[1]), A.T @ y[~test])
    pred = X[test] @ c
    r2 = 1 - np.mean((y[test] - pred) ** 2) / np.var(y[test])
    return r2, np.mean((pred > 0) == tip[test])


def main():
    d, a, P, out = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    th = pool_point(d, a)
    eps = orth_pairs(d, P, np.random.default_rng(7))
    Z = np.vstack([eps, -eps])
    X = th + SIGMA * Z
    q, v = trajectories(bt.make_sim(), torch.from_numpy(X))
    L = bt.terminal_cost(torch.from_numpy(q[-1])).numpy()
    tip = np.abs(q[-1, :, 2]) > 1.0
    tilt, e, ecross = energy_margin(q, v)
    cands = {
        "e_endpush": e[bt.T_PUSH - 1],
        "e_max_all": e.max(0),
        "e_max_after": e[bt.T_PUSH :].max(0),
        "tilt_clip": np.minimum(tilt.max(0), 1.5) - CRIT,
        "e_cross": ecross,
    }
    print(f"d={d} a={a} runs={len(X)} tipped={tip.mean():.3f}")
    for name, m in cands.items():
        rl, cl = heldout(Z, m, tip, False)
        rq, cq = heldout(Z, m, tip, True)
        print(f"  {name:12s} sign agrees {np.mean((m > 0) == tip):.3f}  heldout R2 lin {rl:.3f} quad {rq:.3f}"
              f"  sign(fit)=tip held-out lin {cl:.3f} quad {cq:.3f}")
    rl, _ = heldout(Z, L, tip, False)
    rq, _ = heldout(Z, L, tip, True)
    print(f"  loss itself  heldout R2 lin {rl:.3f} quad {rq:.3f}")
    np.savez(out, th=th, Z=Z, L=L, tip=tip, e=e, tilt=tilt, **cands)


if __name__ == "__main__":
    main()
