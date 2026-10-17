"""Gaussian-mixture propagation through contact: stochastic complementarity
with mixture closure.

A state distribution is a Gaussian mixture over s = (q, v) (6,). A contact
step is smooth within a branch (a contact becomes active or not, which face
is hit, ...); branch boundaries are conditions g(s) = 0 the simulator already
computes. One step, per component:

  1. split the component at each boundary, linearised at its mean: the
     probability of a side is a Gaussian CDF and the side's mean and
     covariance are the moments of the truncated Gaussian (closed form);
  2. push each piece through the step, linearised at the piece's mean with
     the simulator's exact step Jacobian (the piece's mean lies inside its
     branch, so the step there follows that branch).

Tassa & Todorov (RSS 2010) keep only the mean of the stochastic
complementarity; keeping the branches as components keeps the modes.
"""

import math

import torch
from torch.func import jacrev

SQRT2 = math.sqrt(2.0)


def norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def norm_pdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def split(mu, Sigma, g):
    """Split N(mu, Sigma) at the boundary g(s) = 0, linearised at mu.

    Returns [(p, mean, cov)] for the sides g < 0 and g >= 0: their
    probabilities and the moments of the Gaussian truncated to each side.
    With u = a.(s - mu) / sd ~ N(0, 1) (a = grad g, sd^2 = a' Sigma a), the
    side g < 0 is u < beta = -g(mu) / sd.
    """
    g0 = g(mu)
    a = jacrev(g)(mu)
    sd = torch.sqrt(a @ Sigma @ a)
    beta = -g0 / sd
    k = Sigma @ a / sd  # the direction in which conditioning moves the mean
    Phi, phi = norm_cdf(beta), norm_pdf(beta)
    sides = []
    for p, m_u, v_u in (
        (Phi, -phi / Phi, 1 - beta * phi / Phi - (phi / Phi) ** 2),
        (1 - Phi, phi / (1 - Phi), 1 + beta * phi / (1 - Phi) - (phi / (1 - Phi)) ** 2),
    ):
        sides.append((p, mu + k * m_u, Sigma - torch.outer(k, k) * (1 - v_u)))
    return sides


def partition(mu, Sigma, boundaries, p_min=1e-9):
    """Split a component into branches, each piece split again (linearised at
    its own mean). `boundaries` is either a list of g's applied to every piece,
    or a function label -> g (or None: no further split), so that the next
    boundary can depend on the branch so far (e.g. which face, then whether
    that face is hit). Returns [(label, p, mean, cov)], label a tuple of
    signs (-1: g < 0, +1: g >= 0)."""
    nxt_g = boundaries if callable(boundaries) else (lambda label: boundaries[len(label)] if len(label) < len(boundaries) else None)
    done, todo = [], [((), torch.tensor(1.0, dtype=mu.dtype), mu, Sigma)]
    while todo:
        label, p, m, S = todo.pop()
        g = nxt_g(label)
        if g is None:
            done.append((label, p, m, S))
            continue
        for sign, (q, m2, S2) in zip((-1, 1), split(m, S, g)):
            if p * q > p_min:
                todo.append((label + (sign,), p * q, m2, S2))
    return sorted(done, key=lambda c: c[0])


def label_samples(s, boundaries):
    """The branch label of each sample s (N, 6), following the same tree as
    `partition` but with the exact boundary functions."""
    nxt_g = boundaries if callable(boundaries) else (lambda label: boundaries[len(label)] if len(label) < len(boundaries) else None)
    labels = []
    for x in s:
        label = ()
        while (g := nxt_g(label)) is not None:
            label += (-1 if g(x) < 0 else 1,)
        labels.append(label)
    return labels


def step_fn(sim):
    """s (6,) -> s' (6,): one simulator step for one state."""

    def f(s):
        q, v, _ = sim.step(s[None, :3], s[None, 3:])
        return torch.cat([q[0], v[0]])

    return f


def propagate(sim, pieces):
    """Push each piece through the step, linearised at its mean."""
    f = step_fn(sim)
    out = []
    for label, p, m, S in pieces:
        J = torch.autograd.functional.jacobian(f, m)
        out.append((label, p, f(m).detach(), J @ S @ J.T))
    return out


def moments(components):
    """Mean and covariance of a mixture [(label, p, mean, cov)]."""
    P = sum(p for _, p, _, _ in components)
    mean = sum(p * m for _, p, m, _ in components) / P
    cov = sum(p * (S + torch.outer(m - mean, m - mean)) for _, p, m, S in components) / P
    return mean, cov
