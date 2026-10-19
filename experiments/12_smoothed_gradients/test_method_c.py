"""Method C on a toy loss with a known smoothed gradient.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_method_c.py
"""

import math

import numpy as np

import method_c as mc


def test_line_plus_step():
    """f = 0.5 x + 2 * 1[x > 0.03]: F'(theta) = 0.5 + 2 phi_sigma(0.03 - theta).
    C finds the jump and is far closer than zeroth-order with the same samples."""
    sigma, theta, b = 0.05, 0.0, 0.03
    f = lambda x: 0.5 * x + 2.0 * (x > b)
    true = 0.5 + 2.0 * math.exp(-0.5 * ((b - theta) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(0)
    err_c, err_zo = [], []
    for _ in range(200):
        eps = rng.standard_normal(16)
        g, sims, info = mc.estimate_1d(f, theta, sigma, eps)
        assert info["jump"] is not None and abs(info["jump"] - b) < sigma / 32
        assert sims <= 32 + mc.MAX_STEPS + 2 * len(mc.EXTRA)
        err_c.append(abs(g - true))
        err_zo.append(abs(((f(theta + sigma * eps) - f(theta - sigma * eps)) * eps).mean() / (2 * sigma) - true))
    assert np.median(err_c) < 0.02 * true
    assert np.median(err_c) < 0.2 * np.median(err_zo)


def test_smooth_no_jump():
    """On a quadratic no jump is accepted and the estimate is the exact slope."""
    f = lambda x: 3 * x**2 - x
    rng = np.random.default_rng(1)
    g, _, info = mc.estimate_1d(f, 0.2, 0.05, rng.standard_normal(16))
    assert info["jump"] is None
    assert abs(g - (6 * 0.2 - 1)) < 1e-6


def test_plane_jump_2d():
    """f = a . x + 2 * 1[n . x > 0.02] in 2D: grad F = a + 2 n phi_sigma(0.02).
    C recovers it far better than zeroth-order with the same samples."""
    sigma, theta = 0.05, np.zeros(2)
    a, n = np.array([0.5, -0.3]), np.array([0.6, 0.8])
    f = lambda x: x @ a + 2.0 * (x @ n > 0.02)
    true = a + 2.0 * n * math.exp(-0.5 * (0.02 / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(0)
    err_c, err_zo = [], []
    for _ in range(200):
        eps = rng.standard_normal((16, 2))
        g, _, info = mc.estimate_nd(f, theta, sigma, eps)
        err_c.append(np.linalg.norm(g - true))
        y = lambda s: f(theta + s * sigma * eps)
        err_zo.append(np.linalg.norm(((y(1) - y(-1))[:, None] * eps).mean(axis=0) / (2 * sigma) - true))
    assert np.median(err_c) < 0.05 * np.linalg.norm(true)
    assert np.median(err_c) < 0.2 * np.median(err_zo)


def test_gated_grad_closed_form():
    """gated_grad equals the smoothed surrogate's gradient by Monte Carlo
    (sharp and soft gates, separate slopes)."""
    sigma, d = 0.05, 3
    rng = np.random.default_rng(2)
    cl, cr = rng.standard_normal(d + 1), rng.standard_normal(d + 1)
    for scale in (0.5, 3.0, 200.0):
        w = np.concatenate([[0.4], scale * rng.standard_normal(d)])
        e = rng.standard_normal((400_000, d))
        z = sigma * e
        p = 1 / (1 + np.exp(-np.clip(w[0] + e @ w[1:], -50, 50)))
        X = np.hstack([np.ones((len(z), 1)), z])
        f = (1 - p) * (X @ cl) + p * (X @ cr)
        mc_grad = (f[:, None] * e).mean(axis=0) / sigma
        se = np.sqrt(((f[:, None] * e / sigma).var(axis=0)).sum() / len(e))
        assert np.linalg.norm(mc.gated_grad(cl, cr, w, sigma) - mc_grad) < 4 * se


def test_gated_plane_jump():
    """Planar jump with a slope (as test_plane_jump_2d): the gated fit finds
    it from the samples alone and is far closer than zeroth-order."""
    sigma, theta = 0.05, np.zeros(2)
    a, n = np.array([0.5, -0.3]), np.array([0.6, 0.8])
    f = lambda x: x @ a + 2.0 * (x @ n > 0.02)
    true = a + 2.0 * n * math.exp(-0.5 * (0.02 / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    rng = np.random.default_rng(0)
    err_g, err_zo = [], []
    for _ in range(100):
        eps = rng.standard_normal((32, 2))
        g, _, info = mc.estimate_gated(f, theta, sigma, eps)
        assert info["jump"]
        err_g.append(np.linalg.norm(g - true))
        y = lambda s: f(theta + s * sigma * eps)
        err_zo.append(np.linalg.norm(((y(1) - y(-1))[:, None] * eps).mean(axis=0) / (2 * sigma) - true))
    assert np.median(err_g) < 0.1 * np.linalg.norm(true)
    assert np.median(err_g) < 0.3 * np.median(err_zo)


def test_directional_one_line_is_1d():
    """Directional smoothing with one direction is the 1D method."""
    f1 = lambda x: 0.5 * x + 2.0 * (x > 0.03)
    rng = np.random.default_rng(0)
    for _ in range(5):
        e = rng.standard_normal(16)
        g1, s1, _ = mc.estimate_1d(f1, 0.0, 0.05, e)
        gd, sd, _ = mc.estimate_directional(lambda X: f1(X[:, 0]), np.zeros(1), 0.05, np.eye(1), e)
        assert abs(g1 - gd[0]) < 1e-12 and s1 == sd


def test_gated_subspace_curved():
    """d = 10, a jump whose boundary curves inside a 2D subspace span(n, v):
    with that subspace the switch fits the curvature, beating plain gated."""
    sigma, d = 0.05, 10
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    n, v, a = Q[:, 0], Q[:, 1], 0.3 * Q[:, 2]
    f = lambda x: np.atleast_2d(x) @ a + 1.0 * ((np.atleast_2d(x) @ n - 0.01 + 3.0 * (np.atleast_2d(x) @ v) ** 2 / sigma) > 0)
    e = rng.standard_normal((400_000, d))
    true = ((f(sigma * e) - f(-sigma * e))[:, None] * e).mean(axis=0) / (2 * sigma)
    U = np.stack([n, v], axis=1)
    err_g, err_s = [], []
    for _ in range(30):
        eps = rng.standard_normal((40, d))
        err_g.append(np.linalg.norm(mc.estimate_gated(f, np.zeros(d), sigma, eps)[0] - true))
        err_s.append(np.linalg.norm(mc.estimate_gated_subspace(f, np.zeros(d), sigma, eps, U)[0] - true))
    assert np.median(err_s) < 0.7 * np.median(err_g)


def test_margin_grad_closed_form():
    """margin_grad and margin_quad_grad (with zero curvature) equal the
    smoothed surrogate's gradient by Monte Carlo (separate slopes, scatter r)."""
    sigma, d = 0.05, 3
    rng = np.random.default_rng(3)
    cl, cr = rng.standard_normal(d + 1), rng.standard_normal(d + 1)
    c, r = np.array([0.3, 0.8, -0.5, 0.2]), 0.7
    e = rng.standard_normal((400_000, d))
    H = c[0] + e @ c[1:] + r * rng.standard_normal(len(e)) > 0
    X = np.hstack([np.ones((len(e), 1)), sigma * e])
    f = X @ cl + H * (X @ (cr - cl))
    mc_grad = (f[:, None] * e).mean(axis=0) / sigma
    se = np.sqrt(((f[:, None] * e / sigma).var(axis=0)).sum() / len(e))
    assert np.linalg.norm(mc.margin_grad(cl, cr, c, r, sigma) - mc_grad) < 4 * se
    q = np.concatenate([c, np.zeros(d)])
    assert np.linalg.norm(mc.margin_quad_grad(cl, cr, q, r, sigma, M=200_000) - mc_grad) < 6 * se


def test_margin_curved_highd():
    """Curved jump in d = 16 seen through a smooth margin (the toy of the
    margin note): with N = 4d the margin estimate is far closer than the
    gated fit on loss values."""
    sigma, d, kap = 0.05, 16, 0.15
    m = lambda x: 0.3 + x[:, 0] / sigma + 0.5 * kap * (((x[:, 1:] / sigma) ** 2).sum(1) - (d - 1))
    fm = lambda x: (0.5 * x[:, 1] + (m(x) > 0), m(x))
    rng = np.random.default_rng(1)
    Q = rng.chisquare(d - 1, 400_000)
    true = np.zeros(d)
    true[1] = 0.5
    true[0] = np.mean(np.exp(-0.5 * (0.3 + 0.5 * kap * (Q - (d - 1))) ** 2)) / (sigma * math.sqrt(2 * math.pi))
    err = {"margin": [], "quad": [], "gated": []}
    for _ in range(20):
        eps = rng.standard_normal((2 * d, d))
        err["margin"].append(np.linalg.norm(mc.estimate_margin(fm, np.zeros(d), sigma, eps)[0] - true))
        err["quad"].append(np.linalg.norm(mc.estimate_margin(fm, np.zeros(d), sigma, eps, quad=True)[0] - true))
        err["gated"].append(np.linalg.norm(mc.estimate_gated(lambda x: fm(x)[0], np.zeros(d), sigma, eps)[0] - true))
    med = {k: np.median(v) / np.linalg.norm(true) for k, v in err.items()}
    assert med["margin"] < 0.1 and med["quad"] < 0.2 and med["gated"] > 3 * med["margin"], med
