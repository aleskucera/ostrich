"""Method C in one dimension: locate the big jump, fit each side, integrate.

The gradient of the smoothed loss F(theta) = E[f(theta + sigma eps)] is a
slope term plus, for each jump b of f, (f(b+) - f(b-)) phi_sigma(b - theta).
Instead of discovering the jump term by random sampling (zeroth-order), C
builds a piecewise-smooth surrogate of f from the samples and differentiates
its smoothed version exactly. It uses loss values only (no adjoint, no mode
labels):

  1. Samples x_i = theta +- sigma eps_i (2N forward simulations).
  2. Jump detection: over the splits between neighbouring sorted samples,
     fit a polynomial to each side; keep the best split if it beats one
     polynomial over all samples by the Bayesian information criterion.
  3. Location: bisect between the two neighbours. A midpoint belongs to the
     side whose fitted polynomial predicts its value better (no labels).
     Stops at width < sigma * LOCATE_TOL or after MAX_STEPS simulations.
  4. Sparse side: if a side has fewer than MIN_SIDE samples, simulate
     EXTRA points beyond the jump on that side.
  5. Gradient: Stein's formula F'(theta) = E[f_hat(x) (x - theta)] / sigma^2
     on the surrogate f_hat (left polynomial below b, right above), by
     quadrature. No further simulations.

Bisection points are not used in the side fits: near the jump f can fall
through a steep layer (the corner drop's apex), and points inside it would
pull the fits. Side fits are linear below QUAD_MIN samples, quadratic above.
With no accepted split, f_hat is one polynomial (also the ablation
estimate_1d(..., jumps=False)).
"""

import math

import numpy as np

LOCATE_TOL = 1 / 64
MAX_STEPS = 8
MIN_SIDE = 3
EXTRA = (0.25, 0.75, 1.5)  # in sigma, beyond the jump on the sparse side
QUAD_MIN = 10
KAPPA = 3.0  # a split also needs |jump| > KAPPA x the fits' residual spread


def _fit(x, y):
    """Least-squares polynomial (degree by sample count) and its RSS."""
    deg = 2 if len(x) >= QUAD_MIN else 1 if len(x) >= 2 else 0
    c = np.polyfit(x, y, deg) if deg else np.array([y.mean()])
    return c, float(((np.polyval(c, x) - y) ** 2).sum()), deg + 1


def _bic(rss, n, k):
    return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)


def _prefix_fits(x, y, deg):
    """Least-squares polynomials of degree deg on x[:j] for every j = 1..n,
    from cumulative moments (x is centred and scaled by the caller).
    Returns coefficients (n, deg + 1), highest power first like polyfit, and
    RSS (n,). Rows with fewer than deg + 1 points are NaN."""
    p = deg + 1
    S = np.cumsum(x[:, None] ** np.arange(2 * p - 1), axis=0)
    T = np.cumsum((x[:, None] ** np.arange(p)) * y[:, None], axis=0)
    Y2 = np.cumsum(y**2)
    A = S[:, np.arange(p)[:, None] + np.arange(p)[None, :]]
    ok = np.arange(1, len(x) + 1) >= p
    c = np.full((len(x), p), np.nan)
    c[ok] = np.linalg.solve(A[ok] + 1e-12 * np.eye(p), T[ok][..., None])[..., 0]
    rss = np.full(len(x), np.nan)
    rss[ok] = np.maximum(Y2[ok] - (c[ok] * T[ok]).sum(axis=1), 0.0)
    return c[:, ::-1], rss


def best_split(x, y):
    """Index j such that x[:j] | x[j:] is the best two-piece fit, or None if
    one polynomial is better by BIC or the jump between the two fits at the
    split is within KAPPA residual spreads. x sorted. Same fits as _fit
    (degree by sample count), for all splits at once."""
    n = len(x)
    _, rss0, k0 = _fit(x, y)
    mu, sc = x.mean(), max(x.std(), 1e-12)
    z = (x - mu) / sc
    fwd = {d: _prefix_fits(z, y, d) for d in (0, 1, 2)}
    bwd = {d: _prefix_fits(-z[::-1], y[::-1], d) for d in (0, 1, 2)}
    deg = lambda m: 2 if m >= QUAD_MIN else 1 if m >= 2 else 0
    best, best_bic = None, _bic(rss0, n, k0)
    for j in range(1, n):
        dl, dr = deg(j), deg(n - j)
        cl, rl = fwd[dl][0][j - 1], fwd[dl][1][j - 1]
        cr, rr = bwd[dr][0][n - j - 1], bwd[dr][1][n - j - 1]
        b = _bic(rl + rr, n, dl + dr + 3)
        if b >= best_bic:
            continue
        zs = 0.5 * (z[j - 1] + z[j])
        jump = np.polyval(cr, -zs) - np.polyval(cl, zs)  # bwd fits are in -z
        spread = math.sqrt((rl + rr) / max(n - dl - dr - 2, 1))
        if abs(jump) > KAPPA * spread:
            best, best_bic = j, b
    return best


def smoothed_grad(pieces, theta, sigma, n_quad=2001):
    """Stein's formula on the piecewise polynomial [(lo, hi, coeffs), ...]."""
    x = np.linspace(theta - 8 * sigma, theta + 8 * sigma, n_quad)
    f = np.zeros_like(x)
    for lo, hi, c in pieces:
        m = (x >= lo) & (x < hi)
        f[m] = np.polyval(c, x[m])
    w = np.exp(-0.5 * ((x - theta) / sigma) ** 2)
    w /= w.sum()
    return float((w * f * (x - theta)).sum() / (w * (x - theta) ** 2).sum())


def estimate_1d(f, theta, sigma, eps, jumps=True):
    """f(x array) -> losses (one simulation per point). eps: N standard
    normals (antithetic pairs theta +- sigma eps). Returns (gradient,
    simulations used, info)."""
    x = np.concatenate([theta + sigma * eps, theta - sigma * eps])
    y = f(x)
    order = np.argsort(x)
    x, y = x[order], y[order]
    sims = len(x)
    j = best_split(x, y) if jumps else None
    if j is None:
        c, _, _ = _fit(x, y)
        return smoothed_grad([(-np.inf, np.inf, c)], theta, sigma), sims, {"jump": None}

    a, b = x[j - 1], x[j]
    cl, _, _ = _fit(x[:j], y[:j])
    cr, _, _ = _fit(x[j:], y[j:])
    for _ in range(MAX_STEPS):
        if b - a < LOCATE_TOL * sigma:
            break
        m = 0.5 * (a + b)
        ym = f(np.array([m]))[0]
        sims += 1
        if abs(ym - np.polyval(cl, m)) <= abs(ym - np.polyval(cr, m)):
            a = m
        else:
            b = m
    bj = 0.5 * (a + b)

    xl, yl, xr, yr = x[:j], y[:j], x[j:], y[j:]
    if len(xl) < MIN_SIDE:
        xe = bj - sigma * np.array(EXTRA)
        xl, yl = np.concatenate([xe[::-1], xl]), np.concatenate([f(xe)[::-1], yl])
        sims += len(xe)
    if len(xr) < MIN_SIDE:
        xe = bj + sigma * np.array(EXTRA)
        xr, yr = np.concatenate([xr, xe]), np.concatenate([yr, f(xe)])
        sims += len(xe)
    cl, _, _ = _fit(xl, yl)
    cr, _, _ = _fit(xr, yr)
    g = smoothed_grad([(-np.inf, bj, cl), (bj, np.inf, cr)], theta, sigma)
    return g, sims, {"jump": bj, "delta": float(np.polyval(cr, bj) - np.polyval(cl, bj))}
