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


# --- d dimensions: a flat jump surface, a linear piece on each side ---------


def _lin(X, y):
    """Least squares y ~ X; coefficients and RSS."""
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, float(((X @ c - y) ** 2).sum())


def _side_model(X, y):
    """Linear piece if the side has enough points, else a constant. Returns
    coefficients padded to X's width, RSS and the parameter count."""
    k = X.shape[1]
    if len(y) >= k + 2:
        c, r = _lin(X, y)
        return c, r, k
    c = np.zeros(k)
    c[0] = y.mean()
    return c, float(((y - c[0]) ** 2).sum()), 1


def _prefix_lin(X, y):
    """Least squares y ~ X on the first j rows, for every j, from cumulative
    X^T X, X^T y and y^T y. Returns coefficients (n, k) and RSS (n,); NaN
    where fewer than k + 2 rows (the side then uses a constant)."""
    n, k = X.shape
    A = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
    B = np.cumsum(X * y[:, None], axis=0)
    Y2 = np.cumsum(y**2)
    ok = np.arange(1, n + 1) >= k + 2
    c = np.full((n, k), np.nan)
    c[ok] = np.linalg.solve(A[ok] + 1e-12 * np.eye(k), B[ok][..., None])[..., 0]
    rss = np.full(n, np.nan)
    rss[ok] = np.maximum(Y2[ok] - (c[ok] * B[ok]).sum(axis=1), 0.0)
    return c, rss


def _split_along(t, X, y):
    """Best two-piece split of the samples sorted by t (BIC and the KAPPA
    test at the split, as in 1D). Returns (order, j) or None."""
    n, k = X.shape
    order = np.argsort(t)
    X, y = X[order], y[order]
    _, rss0 = _lin(X, y)
    cf, rf = _prefix_lin(X, y)
    cb, rb = _prefix_lin(X[::-1], y[::-1])
    csum, c2sum = np.cumsum(y), np.cumsum(y**2)
    best, best_bic = None, _bic(rss0, n, k)
    for j in range(2, n - 1):
        m = n - j
        if j >= k + 2:
            cl, rl, kl = cf[j - 1], rf[j - 1], k
        else:
            cl = np.zeros(k)
            cl[0] = csum[j - 1] / j
            rl, kl = max(c2sum[j - 1] - j * cl[0] ** 2, 0.0), 1
        if m >= k + 2:
            cr, rr, kr = cb[m - 1], rb[m - 1], k
        else:
            cr = np.zeros(k)
            cr[0] = (csum[-1] - csum[j - 1]) / m
            rr, kr = max((c2sum[-1] - c2sum[j - 1]) - m * cr[0] ** 2, 0.0), 1
        b = _bic(rl + rr, n, kl + kr + 1)
        if b >= best_bic:
            continue
        xs = 0.5 * (X[j - 1] + X[j])
        spread = math.sqrt((rl + rr) / max(n - kl - kr, 1))
        if abs(xs @ (cr - cl)) > KAPPA * spread:
            best, best_bic = j, b
    return None if best is None else (order, best)


def _normal(z, side, reg=1e-2, iters=25):
    """Unit normal of a linear boundary between the two sides: regularised
    logistic regression on positions z (n, d), Newton steps."""
    Z = np.hstack([np.ones((len(z), 1)), z])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg * np.eye(len(w))
        w -= np.linalg.solve(H, Z.T @ (p - side) + reg * w)
    n = w[1:]
    return n / max(np.linalg.norm(n), 1e-300)


def estimate_nd(f, theta, sigma, eps, jumps=True):
    """d-dimensional method C. f(x (m, d)) -> losses; eps (N, d) standard
    normal directions, antithetic pairs theta +- sigma eps. Returns
    (gradient (d,), simulations used, info).

    Jump surface modelled as a plane with a linear piece on each side:
    grad = P_L b_L + P_R b_R + Delta(p0) n phi_sigma(c), where c is the
    plane's distance from theta along its unit normal n (pointing to the R
    side), P_L = Phi(c / sigma), b the slopes of the pieces and Delta(p0) the
    jump at p0 = theta + c n (the Gaussian's mean on the plane; exact for
    linear pieces). No accepted split: the slope of one linear fit (the
    ablation, also jumps=False)."""
    d = len(theta)
    z = np.concatenate([eps, -eps]) * sigma  # positions relative to theta
    y = f(theta + z)
    sims = len(z)
    X = np.hstack([np.ones((len(z), 1)), z])
    c_all, _ = _lin(X, y)
    if not jumps:
        return c_all[1:], sims, {"jump": None}
    # first direction: zeroth-order estimate from the same pairs
    N = len(eps)
    u = ((y[:N] - y[N:])[:, None] * eps).mean(axis=0)
    u = u / max(np.linalg.norm(u), 1e-300)
    found = _split_along(z @ u, X, y)
    if found is None:
        return c_all[1:], sims, {"jump": None}
    order, j = found
    side = np.zeros(len(z))
    side[order[j:]] = 1.0
    n = _normal(z / sigma, side)
    if (z[side == 1] @ n).mean() < (z[side == 0] @ n).mean():
        n = -n
    cl, _, _ = _side_model(X[side == 0], y[side == 0])
    cr, _, _ = _side_model(X[side == 1], y[side == 1])

    # locate: bisect along up to d segments between opposite-side samples
    # (closest pairs first, crossing points at least sigma/2 apart); d points
    # on a flat surface fix it, fewer constrain the logistic normal
    zl, zr = z[side == 0], z[side == 1]
    dist = ((zl[:, None] - zr[None]) ** 2).sum(axis=-1)
    points = []
    for flat in np.argsort(dist, axis=None):
        if len(points) == d:
            break
        il, ir = np.unravel_index(flat, dist.shape)
        if any(np.linalg.norm(0.5 * (zl[il] + zr[ir]) - q) < 0.5 * sigma for q in points):
            continue
        a, b = zl[il], zr[ir]
        for _ in range(MAX_STEPS):
            if np.linalg.norm(b - a) < LOCATE_TOL * sigma:
                break
            mid = 0.5 * (a + b)
            ym = f(theta + mid[None])[0]
            sims += 1
            xm = np.concatenate([[1.0], mid])
            if abs(ym - xm @ cl) <= abs(ym - xm @ cr):
                a = mid
            else:
                b = mid
        points.append(0.5 * (a + b))
    P = np.array(points)
    if len(P) >= 2:  # n orthogonal to the plane's in-plane directions
        _, sv, Vt = np.linalg.svd(P[1:] - P[0], full_matrices=True)
        span = Vt[: int((sv > 1e-9 * sigma).sum())]
        n = n - span.T @ (span @ n)
        n = n / max(np.linalg.norm(n), 1e-300)
        if (z[side == 1] @ n).mean() < (z[side == 0] @ n).mean():
            n = -n
    c = float((P @ n).mean())

    # sparse side: points beyond the plane along n
    for s, sgn in ((0, -1.0), (1, 1.0)):
        if (side == s).sum() < MIN_SIDE:
            ze = (c + sgn * sigma * np.array(EXTRA))[:, None] * n[None]
            z = np.vstack([z, ze])
            y = np.concatenate([y, f(theta + ze)])
            side = np.concatenate([side, np.full(len(ze), float(s))])
            sims += len(ze)
    X = np.hstack([np.ones((len(z), 1)), z])
    cl, _, _ = _side_model(X[side == 0], y[side == 0])
    cr, _, _ = _side_model(X[side == 1], y[side == 1])
    p0 = np.concatenate([[1.0], c * n])
    delta = float(p0 @ (cr - cl))
    PL = 0.5 * (1 + math.erf(c / (sigma * math.sqrt(2))))
    dens = math.exp(-0.5 * (c / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    g = PL * cl[1:] + (1 - PL) * cr[1:] + delta * n * dens
    return g, sims, {"jump": c, "normal": n, "delta": delta}
