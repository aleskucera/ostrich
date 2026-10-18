"""A d-dimensional toy loss shaped like the wheel at its climb boundary.

Fast stand-in for designing method C in d dimensions (the simulator costs
seconds per sample batch). In sigma units around theta = 0:

  f(z) = a . z + 0.5 z^T H z                    smooth trend
       + 1[m(z) > 0] (DELTA + LAYER exp(-m / W))   jump, steep layer on the
                                                   far side (a wheel that
                                                   barely climbs is late)
       + ROUGH sum_k 1[sin(OMEGA v_k . z + p_k) > 0]  small jumps (bumps)
  m(z) = n . z - c + KAPPA |P z|^2 / sigma        boundary (curved if KAPPA)

grad(z) is the derivative where f is differentiable (what the adjoint gives;
zero for the steps). The true smoothed gradient is a large antithetic
Monte Carlo (its standard error is reported by truth()).
"""

import numpy as np

SIGMA = 0.05
DELTA, LAYER, W = 0.4, 0.2, SIGMA / 30
ROUGH, OMEGA, N_ROUGH = 0.004, 2 * np.pi / (SIGMA / 3), 20


class Toy:
    def __init__(self, d, c=0.0, kappa=0.0, layer=LAYER, rough=ROUGH, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.c, self.kappa, self.layer, self.rough = d, c * SIGMA, kappa, layer, rough
        self.a = rng.standard_normal(d) / np.sqrt(d)
        A = rng.standard_normal((d, d)) / np.sqrt(d)
        self.H = (A + A.T) / SIGMA
        self.n = rng.standard_normal(d)
        self.n /= np.linalg.norm(self.n)
        Q = np.linalg.qr(np.column_stack([self.n, rng.standard_normal((d, d - 1))]))[0]
        self.P = Q[:, 1:].T  # in-plane directions
        self.v = rng.standard_normal((N_ROUGH, d))
        self.ph = rng.uniform(0, 2 * np.pi, N_ROUGH)

    def margin(self, z):
        return z @ self.n - self.c + self.kappa * ((z @ self.P.T) ** 2).sum(-1) / SIGMA

    def __call__(self, z):
        z = np.atleast_2d(z)
        m = self.margin(z)
        up = m > 0
        f = z @ self.a + 0.5 * np.einsum("bi,ij,bj->b", z, self.H, z)
        f += up * (DELTA + self.layer * np.exp(-np.where(up, m, 0) / W))
        f += self.rough * (np.sin(OMEGA * z @ self.v.T + self.ph) > 0).sum(-1)
        return f

    def grad(self, z):
        z = np.atleast_2d(z)
        m = self.margin(z)
        up = m > 0
        dm = self.n + 2 * self.kappa * (z @ self.P.T) @ self.P / SIGMA
        g = self.a + z @ self.H
        g -= (up * self.layer / W * np.exp(-np.where(up, m, 0) / W))[:, None] * dm
        return g

    def truth(self, pairs=2_000_000, seed=123, chunk=200_000):
        rng = np.random.default_rng(seed)
        s1 = np.zeros(self.d)
        s2 = np.zeros(self.d)
        for _ in range(pairs // chunk):
            e = rng.standard_normal((chunk, self.d))
            t = (self(SIGMA * e) - self(-SIGMA * e))[:, None] * e / (2 * SIGMA)
            s1 += t.sum(0)
            s2 += (t**2).sum(0)
        mean = s1 / pairs
        return mean, np.sqrt((s2 / pairs - mean**2).sum() / pairs)
