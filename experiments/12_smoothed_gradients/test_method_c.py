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
