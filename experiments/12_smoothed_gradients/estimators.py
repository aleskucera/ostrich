"""Gradient estimators for the Gaussian-smoothed loss L_sigma(u) = E[L(u + sigma eps)].

  hard         exact gradient of the unsmoothed loss at u
  fd           central finite differences of the unsmoothed loss
  first_order  mean of exact gradients at u +- sigma eps (biased at jumps:
               Suh et al. 2022, Thm 1)
  zeroth_order antithetic pairs (L(u + s e) - L(u - s e)) e / (2 s), unbiased
               for grad L_sigma; directions iid or orthogonal
  alpha_order  Suh et al. 2022: alpha * first + (1 - alpha) * zeroth on shared
               samples, alpha = var_zo / (var_fo + var_zo) from the sample
               variances, and alpha = 0 when |first - zeroth| exceeds `gamma`
               standard errors of the zeroth-order mean (bias test)
  ddcg         Onoda, Parmas et al. 2026 (ICLR, arXiv 2604.18161): the same
               inverse-variance mix, gated by their Eq. 14 test (gradient
               spread vs a quadratic model's prediction from the spread of L);
               if it fails, alpha = 0. Here its zeroth-order part is the
               antithetic estimator above (the paper uses a baseline f(u)).

The *_from_samples functions work on precomputed sample values, so many
points and repeats can share one batched rollout; the per-point wrappers take
`loss(u) -> (B,)` and `loss_grad(u) -> ((B,), (B, d))`. Evaluation counts: one
rollout per loss() point, one rollout plus its adjoint per loss_grad() point.
"""

import math

import torch


def directions(n_pairs, d, orthogonal=True, generator=None):
    """n_pairs standard-normal directions (n_pairs, d). Orthogonal: blocks of d
    mutually orthogonal directions (random rotation), each rescaled to a
    chi_d-distributed norm, so the marginals stay N(0, I)."""
    if not orthogonal:
        return torch.randn(n_pairs, d, generator=generator, dtype=torch.float64)
    blocks = []
    for _ in range(math.ceil(n_pairs / d)):
        Q, R = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))
        Q = Q * torch.sign(torch.diagonal(R))
        norms = torch.randn(d, d, generator=generator, dtype=torch.float64).norm(dim=1)
        blocks.append(Q * norms[:, None])
    return torch.cat(blocks)[:n_pairs]


# --- formulas on samples: eps (..., N, d), L_plus/L_minus (..., N), g_plus/g_minus (..., N, d)


def first_order_from_samples(g_plus, g_minus):
    return 0.5 * (g_plus + g_minus).mean(dim=-2)


def zeroth_order_from_samples(eps, L_plus, L_minus, sigma):
    return ((L_plus - L_minus)[..., None] * eps).mean(dim=-2) / (2 * sigma)


def alpha_order_from_samples(eps, L_plus, L_minus, g_plus, g_minus, sigma, gamma):
    """Returns (estimate, alpha)."""
    n = eps.shape[-2]
    fo_i = 0.5 * (g_plus + g_minus)
    zo_i = (L_plus - L_minus)[..., None] * eps / (2 * sigma)
    fo, zo = fo_i.mean(dim=-2), zo_i.mean(dim=-2)
    var_fo = fo_i.var(dim=-2).sum(dim=-1) / n  # trace of the covariance of the mean
    var_zo = zo_i.var(dim=-2).sum(dim=-1) / n
    alpha = var_zo / (var_fo + var_zo).clamp(min=1e-300)  # both 0 (flat region): alpha = 0
    alpha = torch.where((fo - zo).norm(dim=-1) > gamma * var_zo.sqrt(), 0.0, alpha)
    return alpha[..., None] * fo + (1 - alpha[..., None]) * zo, alpha


def ddcg_from_samples(eps, L_plus, L_minus, g_plus, g_minus, sigma, c=0.3):
    """Returns (estimate, alpha). Eq. 14 on all 2N samples x_i: the test
    v_hat >= 2 (1 - c) Var[L] / sigma^2 - 2 |mean grad|^2 passes for a
    near-quadratic L; a jump inflates Var[L] without showing in the
    gradients, so the test fails and alpha = 0 (Eq. 15)."""
    n = eps.shape[-2]
    fo_i = 0.5 * (g_plus + g_minus)
    zo_i = (L_plus - L_minus)[..., None] * eps / (2 * sigma)
    fo, zo = fo_i.mean(dim=-2), zo_i.mean(dim=-2)
    var_fo = fo_i.var(dim=-2).sum(dim=-1) / n
    var_zo = zo_i.var(dim=-2).sum(dim=-1) / n
    alpha = var_zo / (var_fo + var_zo).clamp(min=1e-300)
    g = torch.cat([g_plus, g_minus], dim=-2)
    L = torch.cat([L_plus, L_minus], dim=-1)
    v_hat = ((g - g.mean(dim=-2, keepdim=True)) ** 2).sum(dim=-1).sum(dim=-1) / (2 * n - 1)
    smooth = v_hat >= 2 * (1 - c) * L.var(dim=-1) / sigma**2 - 2 * (g.mean(dim=-2) ** 2).sum(dim=-1)
    alpha = torch.where(smooth, alpha, 0.0)
    return alpha[..., None] * fo + (1 - alpha[..., None]) * zo, alpha


# --- per-point wrappers


def hard(loss_grad, u):
    return loss_grad(u[None])[1][0]


def fd(loss, u, delta=1e-4):
    d = len(u)
    eye = torch.eye(d, dtype=u.dtype)
    L = loss(torch.cat([u + delta * eye, u - delta * eye]))
    return (L[:d] - L[d:]) / (2 * delta)


def _samples(loss_grad, u, sigma, n_pairs, orthogonal, generator):
    eps = directions(n_pairs, len(u), orthogonal, generator)
    L, g = loss_grad(torch.cat([u + sigma * eps, u - sigma * eps]))
    return eps, L[:n_pairs], L[n_pairs:], g[:n_pairs], g[n_pairs:]


def first_order(loss_grad, u, sigma, n_pairs, orthogonal=True, generator=None):
    _, _, _, g_plus, g_minus = _samples(loss_grad, u, sigma, n_pairs, orthogonal, generator)
    return first_order_from_samples(g_plus, g_minus)


def zeroth_order(loss, u, sigma, n_pairs, orthogonal=True, generator=None):
    eps = directions(n_pairs, len(u), orthogonal, generator)
    L = loss(torch.cat([u + sigma * eps, u - sigma * eps]))
    return zeroth_order_from_samples(eps, L[:n_pairs], L[n_pairs:], sigma)


def alpha_order(loss_grad, u, sigma, n_pairs, gamma=2.0, orthogonal=True, generator=None):
    """Returns (estimate, alpha)."""
    return alpha_order_from_samples(*_samples(loss_grad, u, sigma, n_pairs, orthogonal, generator), sigma, gamma)


def ddcg(loss_grad, u, sigma, n_pairs, c=0.3, orthogonal=True, generator=None):
    """Returns (estimate, alpha)."""
    return ddcg_from_samples(*_samples(loss_grad, u, sigma, n_pairs, orthogonal, generator), sigma, c)
