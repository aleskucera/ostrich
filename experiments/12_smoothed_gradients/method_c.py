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


def pieces_1d(f, theta, sigma, eps, jumps=True):
    """The piecewise surrogate of estimate_1d: ([(lo, hi, coeffs), ...],
    simulations used, info)."""
    x = np.concatenate([theta + sigma * eps, theta - sigma * eps])
    y = f(x)
    order = np.argsort(x)
    x, y = x[order], y[order]
    sims = len(x)
    j = best_split(x, y) if jumps else None
    if j is None:
        c, _, _ = _fit(x, y)
        return [(-np.inf, np.inf, c)], sims, {"jump": None}

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
    return [(-np.inf, bj, cl), (bj, np.inf, cr)], sims, {"jump": bj, "delta": float(np.polyval(cr, bj) - np.polyval(cl, bj))}


def smoothed_value(pieces, theta, sigma, n_quad=2001):
    """E[f_hat(theta + sigma eps)] on the piecewise polynomial."""
    x = np.linspace(theta - 8 * sigma, theta + 8 * sigma, n_quad)
    f = np.zeros_like(x)
    for lo, hi, c in pieces:
        m = (x >= lo) & (x < hi)
        f[m] = np.polyval(c, x[m])
    w = np.exp(-0.5 * ((x - theta) / sigma) ** 2)
    return float((w * f).sum() / w.sum())


def estimate_1d(f, theta, sigma, eps, jumps=True):
    """f(x array) -> losses (one simulation per point). eps: N standard
    normals (antithetic pairs theta +- sigma eps). Returns (gradient,
    simulations used, info)."""
    pieces, sims, info = pieces_1d(f, theta, sigma, eps, jumps)
    return smoothed_grad(pieces, theta, sigma), sims, info


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


def _split_along(t, X, y, test=True):
    """Best two-piece split of the samples sorted by t (BIC and the KAPPA
    test at the split, as in 1D). Returns (order, j) or None. test=False:
    the lowest-BIC split even if one piece or a small jump is better."""
    n, k = X.shape
    order = np.argsort(t)
    X, y = X[order], y[order]
    _, rss0 = _lin(X, y)
    cf, rf = _prefix_lin(X, y)
    cb, rb = _prefix_lin(X[::-1], y[::-1])
    csum, c2sum = np.cumsum(y), np.cumsum(y**2)
    best, best_bic = None, _bic(rss0, n, k) if test else np.inf
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
        if not test or abs(xs @ (cr - cl)) > KAPPA * spread:
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


def _oriented_normal(z, side, sigma):
    n = _normal(z / sigma, side)
    return -n if (z[side == 1] @ n).mean() < (z[side == 0] @ n).mean() else n


def _loo(X, y):
    """Least squares, its leave-one-out mean squared error (PRESS / n)."""
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    h = np.einsum("ij,ji->i", X, np.linalg.pinv(X))
    r = (y - X @ c) / np.maximum(1 - h, 1e-9)
    return c, float((r**2).mean())


def _models(X, y, side, slopes):
    """Linear pieces (intercept, slope) of the two sides: _side_model per
    side, or (slopes="loo") a shared slope with a jump offset if that has the
    lower leave-one-out error."""
    cl, _, _ = _side_model(X[side == 0], y[side == 0])
    cr, _, _ = _side_model(X[side == 1], y[side == 1])
    if slopes == "separate":
        return cl, cr
    k = X.shape[1]
    w, e_shared = _loo(np.hstack([X, side[:, None]]), y)
    sep = np.hstack([X * (1 - side)[:, None], X * side[:, None]])
    if min(side.sum(), len(side) - side.sum()) >= k + 2:
        _, e_sep = _loo(sep, y)
        if e_sep < e_shared:
            return cl, cr
    off = np.zeros(k)
    off[0] = w[-1]
    return w[:k], w[:k] + off


def estimate_nd(f, theta, sigma, eps, jumps=True, refine=0, slopes="separate", points=None):
    """d-dimensional method C. f(x (m, d)) -> losses; eps (N, d) standard
    normal directions, antithetic pairs theta +- sigma eps. Returns
    (gradient (d,), simulations used, info).

    Jump surface modelled as a plane with a linear piece on each side:
    grad = P_L b_L + P_R b_R + Delta(p0) n phi_sigma(c), where c is the
    plane's distance from theta along its unit normal n (pointing to the R
    side), P_L = Phi(c / sigma), b the slopes of the pieces and Delta(p0) the
    jump at p0 = theta + c n (the Gaussian's mean on the plane; exact for
    linear pieces). No accepted split: the slope of one linear fit (the
    ablation, also jumps=False).

    refine: rounds of re-splitting along the logistic normal of the current
    sides (the first split is along the noisy zeroth-order direction).
    slopes: "separate" a slope per side, "loo" also try one shared slope
    plus a jump offset and keep the lower leave-one-out error. points: the
    number of boundary points bisected (default d)."""
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
    # refine: untested splits along the logistic normal of the last sides,
    # the jump test on the final direction only
    for _ in range(refine):
        found = _split_along(z @ u, X, y, test=False)
        if found is None:
            break
        side = np.zeros(len(z))
        side[found[0][found[1]:]] = 1.0
        new = _oriented_normal(z, side, sigma)
        if new @ u > 1 - 1e-9:
            break
        u = new
    found = _split_along(z @ u, X, y)
    if found is None:
        return c_all[1:], sims, {"jump": None}
    order, j = found
    side = np.zeros(len(z))
    side[order[j:]] = 1.0
    n = _oriented_normal(z, side, sigma)
    cl, cr = _models(X, y, side, slopes)

    # locate: bisect along up to d segments between opposite-side samples
    # (closest pairs first, segment midpoints at least sigma/2 apart), all
    # segments at once; d points on a flat surface fix it, fewer constrain
    # the logistic normal
    zl, zr = z[side == 0], z[side == 1]
    dist = ((zl[:, None] - zr[None]) ** 2).sum(axis=-1)
    npts, seg = points, []
    for flat in np.argsort(dist, axis=None):
        if len(seg) == min(d if npts is None else npts, d):
            break
        il, ir = np.unravel_index(flat, dist.shape)
        if any(np.linalg.norm(0.5 * (zl[il] + zr[ir] - a - b)) < 0.5 * sigma for a, b in seg):
            continue
        seg.append((zl[il], zr[ir]))
    a = np.array([s_[0] for s_ in seg])
    b = np.array([s_[1] for s_ in seg])
    for _ in range(MAX_STEPS):
        live = np.linalg.norm(b - a, axis=1) >= LOCATE_TOL * sigma
        if not live.any():
            break
        mid = 0.5 * (a[live] + b[live])
        ym = f(theta + mid)
        sims += len(mid)
        xm = np.hstack([np.ones((len(mid), 1)), mid])
        left = np.abs(ym - xm @ cl) <= np.abs(ym - xm @ cr)
        ia = np.flatnonzero(live)
        a[ia[left]] = mid[left]
        b[ia[~left]] = mid[~left]
    points = list(0.5 * (a + b))
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
    cl, cr = _models(X, y, side, slopes)
    p0 = np.concatenate([[1.0], c * n])
    delta = float(p0 @ (cr - cl))
    PL = 0.5 * (1 + math.erf(c / (sigma * math.sqrt(2))))
    dens = math.exp(-0.5 * (c / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    g = PL * cl[1:] + (1 - PL) * cr[1:] + delta * n * dens
    return g, sims, {"jump": c, "normal": n, "delta": delta}


# --- two linear experts with a soft (logistic) gate -------------------------

GATE_REG = 1e-2
EM_ITERS = 30


def _logistic(Z, target, w=None, reg=GATE_REG, iters=25):
    """Logistic regression coefficients (intercept first) for soft targets in
    [0, 1], L2-regularised Newton steps (the intercept unpenalised)."""
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])
    w = np.zeros(Z1.shape[1]) if w is None else w.copy()
    R = reg * np.eye(len(w))
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z1 @ w, -50, 50)))
        H = Z1.T @ (Z1 * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, Z1.T @ (p - target) + R @ w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _wls(X, y, r):
    W = np.sqrt(np.maximum(r, 0))[:, None]
    c, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
    s2 = float((r * (y - X @ c) ** 2).sum() / max(r.sum(), 1e-12))
    return c, max(s2, 1e-12)


def gated_grad(cl, cr, w, sigma, n_quad=4001):
    """Exact gradient at theta of E[f_hat(theta + sigma eps)] for
    f_hat(theta + z) = (1 - p) (cl . [1, z]) + p (cr . [1, z]),
    p = psi(w0 + w . z / sigma), psi the logistic function. Only a = w0 +
    w . eps enters the gate, so the expectations are 1D integrals over
    a ~ N(w0, |w|^2):
      grad = bL + E[psi] (bR - bL) + (w / sigma) (D0 E[psi'] + sigma
             (dB . w) / |w|^2 E[psi' (a - w0)]),
    D0 = cr0 - cl0, dB = bR - bL."""
    w0, wv = w[0], w[1:]
    s = max(np.linalg.norm(wv), 1e-12)
    a = w0 + s * np.linspace(-8, 8, n_quad)
    pdf = np.exp(-0.5 * ((a - w0) / s) ** 2)
    pdf /= pdf.sum()
    psi = 1 / (1 + np.exp(-np.clip(a, -50, 50)))
    dpsi = psi * (1 - psi)
    # a sharp gate: psi' is narrower than the grid; integrate it exactly
    # against the Gaussian instead (psi' -> delta(a), unit mass)
    if s > 50:
        dens0 = math.exp(-0.5 * (w0 / s) ** 2) / (s * math.sqrt(2 * math.pi))
        J0, J1 = dens0, dens0 * (0 - w0)
    else:
        J0, J1 = (pdf * dpsi).sum(), (pdf * dpsi * (a - w0)).sum()
    I0 = (pdf * psi).sum()
    bl, br = cl[1:], cr[1:]
    D0, dB = cr[0] - cl[0], br - bl
    return bl + I0 * dB + (wv / sigma) * (D0 * J0 + sigma * (dB @ wv) / s**2 * J1)


def _otsu(v):
    """Labels v > t for the threshold t that best splits v into two groups
    (largest between-group variance)."""
    o = np.sort(v)
    c = np.cumsum(o)
    k = np.arange(1, len(o))
    between = (c[:-1] / k - (c[-1] - c[:-1]) / (len(o) - k)) ** 2 * k * (len(o) - k)
    j = int(np.argmax(between))
    return (v > 0.5 * (o[j] + o[j + 1])).astype(float)


def _em(X, z, y, r, sigma, shared):
    """EM for two linear experts with a logistic gate from soft labels r.
    Returns (cl, cr, gate w, log-likelihood per sample, r)."""
    n, k = X.shape
    wg = None
    for _ in range(EM_ITERS):
        if shared:
            cs, s2 = _wls(np.hstack([X, r[:, None]]), y, np.ones(n))
            cl, cr = cs[:k], cs[:k].copy()
            cr[0] += cs[-1]
            s2l = s2r = s2
        else:
            cl, s2l = _wls(X, y, 1 - r)
            cr, s2r = _wls(X, y, r)
        wg = _logistic(z / sigma, r, wg)
        a = np.clip(wg[0] + (z / sigma) @ wg[1:], -50, 50)
        ll = -0.5 * (y - X @ cl) ** 2 / s2l - 0.5 * math.log(s2l) - np.logaddexp(0, a)
        lr = -0.5 * (y - X @ cr) ** 2 / s2r - 0.5 * math.log(s2r) - np.logaddexp(0, -a)
        new = 1 / (1 + np.exp(np.clip(ll - lr, -50, 50)))
        done = np.abs(new - r).max() < 1e-6
        r = new
        if done:
            break
    mix = np.logaddexp(ll, lr) - 0.5 * math.log(2 * math.pi)
    return cl, cr, wg, mix, r


def fit_gated(z, y, sigma, shared=True, init_refine=4):
    """Fit the gated two-expert surrogate to samples y at offsets z (n, d)
    from theta (antithetic: z[:n/2] = -z[n/2:]). EM starts from three
    labellings (two groups of the losses; of the residuals of one linear fit;
    the 1D split along the zeroth-order direction refined by the logistic
    normal) and keeps the most likely. Returns (cl, cr, gate w, r) if it
    beats one linear fit by BIC, else None."""
    n, d = z.shape
    N = n // 2
    eps = z[:N] / sigma
    X = np.hstack([np.ones((n, 1)), z])
    c_all, rss0 = _lin(X, y)
    starts = [_otsu(y), _otsu(y - X @ c_all)]
    u = ((y[:N] - y[N:])[:, None] * eps).mean(axis=0)
    u = u / max(np.linalg.norm(u), 1e-300)
    side = None
    for _ in range(init_refine):
        found = _split_along(z @ u, X, y, test=False)
        if found is None:
            break
        side = np.zeros(n)
        side[found[0][found[1]:]] = 1.0
        new = _oriented_normal(z, side, sigma)
        if new @ u > 1 - 1e-9:
            break
        u = new
    if side is not None:
        starts.append(side)
    best = None
    for r0 in starts:
        if min(r0.sum(), n - r0.sum()) < 2:
            continue
        fit = _em(X, z, y, r0.copy(), sigma, shared)
        if best is None or fit[3].sum() > best[3].sum():
            best = fit
    if best is None:
        return None
    cl, cr, wg, mix, r = best
    k_mix = (d + 2 if shared else 2 * (d + 1)) + (d + 1) + (1 if shared else 2)
    bic_mix = -2 * mix.sum() + k_mix * math.log(n)
    bic_one = n * math.log(max(rss0, 1e-300) / n) + n * (1 + math.log(2 * math.pi)) + (d + 2) * math.log(n)
    return (cl, cr, wg, r) if bic_mix < bic_one else None


def gated_predict(model, z, sigma):
    """The surrogate's value at offsets z from theta."""
    cl, cr, w, _ = model
    X = np.hstack([np.ones((len(z), 1)), z])
    p = 1 / (1 + np.exp(-np.clip(w[0] + (z / sigma) @ w[1:], -50, 50)))
    return (1 - p) * (X @ cl) + p * (X @ cr)


def estimate_gated(f, theta, sigma, eps, shared=True):
    """Two linear experts (one per mode) with a logistic gate, fitted by EM to
    the 2N antithetic samples (fit_gated); the gradient of the smoothed
    surrogate in closed form (gated_grad). Covers a sharp plane boundary (the
    gate becomes a step: method C's jump term) and a fuzzy one (the mode
    depends on unresolved small features: P(mode) varies smoothly). No
    accepted mixture: the slope of one linear fit. shared: one slope for both
    experts plus an offset. Returns (gradient, simulations, info)."""
    z = np.concatenate([eps, -eps]) * sigma
    y = f(theta + z)
    model = fit_gated(z, y, sigma, shared)
    if model is None:
        c_all, _ = _lin(np.hstack([np.ones((len(z), 1)), z]), y)
        return c_all[1:], len(z), {"jump": None}
    cl, cr, wg, r = model
    return gated_grad(cl, cr, wg, sigma), len(z), {"jump": True, "gate": wg, "cl": cl, "cr": cr, "r": r,
                                                   "sharpness": float(np.linalg.norm(wg[1:]))}


def cv_reference(eps, y, sigma, folds=2, shared=False):
    """Unbiased reference gradient from many antithetic pairs (eps (N, d),
    y = losses at theta + sigma eps then theta - sigma eps): cross-fitted
    control variate. For each fold, the gated surrogate is fitted on the
    other folds; the fold's pairs give the zeroth-order estimate of the
    residual f - f_hat, plus the surrogate's exact smoothed gradient
    (unbiased for any fixed f_hat). Returns (gradient, standard error,
    standard error of plain zeroth-order on the same pairs)."""
    N, d = eps.shape
    yp, ym = y[:N], y[N:]
    idx = np.arange(N) % folds
    terms = np.zeros((N, d))
    base = np.zeros((folds, d))
    for k in range(folds):
        tr, te = idx != k, idx == k
        z_tr = np.concatenate([eps[tr], -eps[tr]]) * sigma
        model = fit_gated(z_tr, np.concatenate([yp[tr], ym[tr]]), sigma, shared)
        if model is None:
            X = np.hstack([np.ones((len(z_tr), 1)), z_tr])
            c, _ = _lin(X, np.concatenate([yp[tr], ym[tr]]))
            model = (c, c, np.zeros(d + 1), None)
        base[k] = gated_grad(model[0], model[1], model[2], sigma)
        rp = yp[te] - gated_predict(model, sigma * eps[te], sigma)
        rm = ym[te] - gated_predict(model, -sigma * eps[te], sigma)
        terms[te] = (rp - rm)[:, None] * eps[te] / (2 * sigma) + base[k]
    g = terms.mean(axis=0)
    se = math.sqrt(sum(terms[idx == k].var(axis=0).sum() / (idx == k).sum() for k in range(folds)) / folds**2 * folds)
    plain = (yp - ym)[:, None] * eps / (2 * sigma)
    return g, se, math.sqrt(plain.var(axis=0).sum() / N)


# --- directional smoothing: method C along each of d directions -------------


def _bisect_lines(f_lines, a, b, cl, cr, sigma):
    """Lockstep 1D bisection on many lines: f_lines(idx, x) -> losses at
    offsets x on lines idx (one simulation each). Each line keeps a and b on
    opposite sides (a midpoint goes to the side whose polynomial predicts it
    better). Returns the crossings and simulations used."""
    a, b, sims = a.copy(), b.copy(), 0
    for _ in range(MAX_STEPS):
        live = np.flatnonzero(b - a >= LOCATE_TOL * sigma)
        if len(live) == 0:
            break
        m = 0.5 * (a[live] + b[live])
        ym = f_lines(live, m)
        sims += len(live)
        left = np.array([abs(v - np.polyval(cl[i], x)) <= abs(v - np.polyval(cr[i], x)) for i, x, v in zip(live, m, ym)])
        a[live[left]] = m[left]
        b[live[~left]] = m[~left]
    return 0.5 * (a + b), sims


def estimate_directional(f, theta, sigma, V, eps1d, mode="c", jumps=True):
    """Directional Gaussian smoothing (Zhang, Tran, Bao 2020): the gradient
    sum_i D_i v_i over the orthonormal rows v_i of V, D_i the derivative of
    f smoothed along v_i only, x -> E_s[f(theta + (x + sigma s) v_i)].
    mode "c": D_i by 1D method C on the line (antithetic offsets eps1d, the
    jump bisected; all lines in lockstep, one batched simulation call per
    step); "m2": antithetic 1D Stein mean with the same offsets; "gh":
    Gauss-Hermite quadrature with nodes/weights eps1d = (x, w) as in the
    paper. f(x (m, d)) -> losses. Returns (gradient, simulations, info)."""
    d = len(theta)
    if mode == "gh":
        x, w = eps1d
        X = theta + np.sqrt(2) * sigma * (x[None, :, None] * V[:, None, :])  # (d, M, d)
        y = f(X.reshape(-1, d)).reshape(len(V), len(x))
        D = (y * (w * np.sqrt(2) * x)).sum(axis=1) / (sigma * np.sqrt(np.pi))
        return D @ V, y.size, {}
    K = len(eps1d)
    off = sigma * np.concatenate([eps1d, -eps1d])  # (2K,)
    X = theta + off[None, :, None] * V[:, None, :]
    Y = f(X.reshape(-1, d)).reshape(len(V), 2 * K)
    sims = Y.size
    if mode == "m2":
        D = ((Y[:, :K] - Y[:, K:]) * eps1d).mean(axis=1) / (2 * sigma)
        return D @ V, sims, {}
    f_lines = lambda idx, x: f(theta + x[:, None] * V[idx])
    order = np.argsort(off)
    xs = off[order]
    pieces, split = [], []
    for i in range(len(V)):
        ys = Y[i, order]
        j = best_split(xs, ys) if jumps else None
        split.append(j)
        if j is None:
            pieces.append([(-np.inf, np.inf, _fit(xs, ys)[0])])
        else:
            pieces.append(None)
    jl = [i for i, j in enumerate(split) if j is not None]
    if jl:
        a = np.array([xs[split[i] - 1] for i in jl])
        b = np.array([xs[split[i]] for i in jl])
        cl = [_fit(xs[: split[i]], Y[i, order][: split[i]])[0] for i in jl]
        cr = [_fit(xs[split[i] :], Y[i, order][split[i] :])[0] for i in jl]
        bj, n = _bisect_lines(lambda idx, x: f_lines(np.array(jl)[idx], x), a, b, cl, cr, sigma)
        sims += n
        # sparse sides: points beyond the jump, all lines in one call
        extra_idx, extra_x = [], []
        for k, i in enumerate(jl):
            j = split[i]
            if j < MIN_SIDE:
                extra_idx += [k] * len(EXTRA)
                extra_x += list(bj[k] - sigma * np.array(EXTRA))
            if 2 * K - j < MIN_SIDE:
                extra_idx += [k] * len(EXTRA)
                extra_x += list(bj[k] + sigma * np.array(EXTRA))
        ye = f_lines(np.array(jl)[extra_idx], np.array(extra_x)) if extra_idx else np.array([])
        sims += len(extra_idx)
        for k, i in enumerate(jl):
            j = split[i]
            ys = Y[i, order]
            sel = [t for t, kk in enumerate(extra_idx) if kk == k]
            xe, ve = np.array(extra_x)[sel], ye[sel] if len(sel) else np.array([])
            xl, yl = np.concatenate([xs[:j], xe[xe < bj[k]]]), np.concatenate([ys[:j], ve[xe < bj[k]]])
            xr, yr = np.concatenate([xs[j:], xe[xe > bj[k]]]), np.concatenate([ys[j:], ve[xe > bj[k]]])
            ol, orr = np.argsort(xl), np.argsort(xr)
            pieces[i] = [(-np.inf, bj[k], _fit(xl[ol], yl[ol])[0]), (bj[k], np.inf, _fit(xr[orr], yr[orr])[0])]
    D = np.array([smoothed_grad(p, 0.0, sigma) for p in pieces])
    return D @ V, sims, {"lines_with_jump": len(jl) / len(V)}
