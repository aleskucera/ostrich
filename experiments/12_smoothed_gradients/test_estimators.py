"""Estimators on toy losses with known smoothed gradients.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_estimators.py
"""

import math

import torch

import estimators as est

C = torch.tensor([0.7, -1.3], dtype=torch.float64)


def quad(u):
    return (u**2).sum(dim=1) + u @ C


def quad_grad(u):
    return quad(u), 2 * u + C


def step(u):
    return (u[:, 0] > 0).double()


def step_grad(u):
    return step(u), torch.zeros_like(u)


def mean_of(fn, trials=400, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([fn(gen) for _ in range(trials)]).mean(dim=0)


def test_orthogonal_directions():
    gen = torch.Generator().manual_seed(0)
    eps = est.directions(4000, 2, orthogonal=True, generator=gen)
    assert torch.allclose(eps.T @ eps / len(eps), torch.eye(2, dtype=torch.float64), atol=0.06)
    block = eps[:2] / eps[:2].norm(dim=1, keepdim=True)
    assert abs(block[0] @ block[1]) < 1e-12


def test_quadratic_all_agree():
    """Smoothing a quadratic only adds a constant: every estimator targets 2u + C."""
    u = torch.tensor([0.3, -0.2], dtype=torch.float64)
    true = 2 * u + C
    assert torch.allclose(est.hard(quad_grad, u), true)
    assert torch.allclose(est.fd(quad, u), true, atol=1e-8)
    assert torch.allclose(est.first_order(quad_grad, u, 0.1, 8), true)
    # Antithetic ZO on a quadratic: (grad . e) e per sample, unbiased.
    # Per-component sample std ~3, so 2000 x 8 samples give a std error ~0.024.
    zo = mean_of(lambda g: est.zeroth_order(quad, u, 0.1, 8, generator=g), trials=2000)
    assert torch.allclose(zo, true, atol=0.08)


def test_step_function():
    """L = 1[u0 > 0]: exact and first-order gradients are 0; the smoothed
    gradient is phi(u0 / sigma) / sigma along u0. Zeroth-order finds it, and
    the alpha-order bias test switches to zeroth-order."""
    sigma, u = 0.1, torch.tensor([0.05, 0.0], dtype=torch.float64)
    true = torch.tensor([math.exp(-0.5 * (u[0] / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)), 0.0],
                        dtype=torch.float64)
    assert torch.equal(est.hard(step_grad, u), torch.zeros(2, dtype=torch.float64))
    assert torch.equal(est.first_order(step_grad, u, sigma, 16), torch.zeros(2, dtype=torch.float64))
    zo = mean_of(lambda g: est.zeroth_order(step, u, sigma, 16, generator=g))
    assert torch.allclose(zo, true, atol=0.1)
    ao = mean_of(lambda g: est.alpha_order(step_grad, u, sigma, 16, gamma=2.0, generator=g)[0])
    assert torch.allclose(ao, true, atol=0.1)


def test_ddcg_gate():
    """DDCG mostly keeps the first-order mix on a quadratic (alpha > 0) and falls
    back to zeroth-order at a step (alpha = 0), where it is then unbiased."""
    sigma = 0.1
    u = torch.tensor([0.3, -0.2], dtype=torch.float64)
    # The test compares sample variances, so on a quadratic it still fails
    # now and then (sample Var[L] > 1.4x its mean); check the pass rate.
    passed = [est.ddcg(quad_grad, u, sigma, 16, generator=torch.Generator().manual_seed(k))[1] > 0 for k in range(200)]
    assert sum(passed) / len(passed) > 0.7
    u = torch.tensor([0.05, 0.0], dtype=torch.float64)
    _, a = est.ddcg(step_grad, u, sigma, 16, generator=torch.Generator().manual_seed(0))
    assert a == 0
    true = torch.tensor([math.exp(-0.5 * (u[0] / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)), 0.0],
                        dtype=torch.float64)
    dd = mean_of(lambda g: est.ddcg(step_grad, u, sigma, 16, generator=g)[0])
    assert torch.allclose(dd, true, atol=0.1)
