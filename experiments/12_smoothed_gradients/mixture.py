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
    that face is hit). A node may return a pair (g1, g2): both are split at
    once (split_joint), adding two signs to the label. Returns
    [(label, p, mean, cov)], label a tuple of signs (-1: g < 0, +1: g >= 0)."""
    nxt_g = boundaries if callable(boundaries) else (lambda label: boundaries[len(label)] if len(label) < len(boundaries) else None)
    done, todo = [], [((), torch.tensor(1.0, dtype=mu.dtype), mu, Sigma)]
    while todo:
        label, p, m, S = todo.pop()
        g = nxt_g(label)
        if g is None:
            done.append((label, p, m, S))
            continue
        if isinstance(g, tuple):
            for signs, (q, m2, S2) in split_joint(m, S, *g).items():
                if p * q > p_min:
                    todo.append((label + signs, p * q, m2, S2))
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
            gs = g if isinstance(g, tuple) else (g,)
            label += tuple(-1 if gi(x) < 0 else 1 for gi in gs)
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


def step_jacobians(sim, means, info, f_ext=None):
    """d(q1, v1)/d(q0, v0) (P, 6, 6) at the pieces' means, from the adjoint:
    one batched step_vjp per output row."""
    P = len(means)
    q0, v0 = means[:, :3], means[:, 3:]
    f = torch.zeros(P, 3, dtype=means.dtype) if f_ext is None else f_ext
    rows = []
    for i in range(6):
        g = torch.zeros(P, 6, dtype=means.dtype)
        g[:, i] = 1.0
        g_q, g_v, _ = sim.step_vjp(q0, v0, f, info, g[:, :3], g[:, 3:])
        rows.append(torch.cat([g_q, g_v], dim=1))
    return torch.stack(rows, dim=1)


def propagate_batched(sim, pieces):
    """As `propagate`, all pieces through one batched step."""
    if not pieces:
        return []
    means = torch.stack([m for _, _, m, _ in pieces])
    q1, v1, info = sim.step(means[:, :3], means[:, 3:])
    J = step_jacobians(sim, means, info)
    out_means = torch.cat([q1, v1], dim=1)
    return [(label, p, out_means[i], J[i] @ S @ J[i].T) for i, (label, p, _, S) in enumerate(pieces)]


def reduce_by_history(components, k_per_history, jitter=1e-12):
    """Reduce within groups of equal contact history only (the component's
    label holds its history): components that went through different
    contact sequences, i.e. different modes, are never merged."""
    groups = {}
    for c in components:
        groups.setdefault(c[0], []).append(c)
    out = []
    for history, group in groups.items():
        out += [(history,) + c[1:] for c in reduce(group, k_per_history, jitter)]
    return out


def reduce(components, k_max, jitter=1e-12):
    """Merge components [(label, p, mean, cov)] down to k_max by moment matching,
    each time the pair with the smallest Runnalls cost
    B = 0.5 [(wi + wj) logdet S_ij - wi logdet S_i - wj logdet S_j].
    All pair costs are computed in one batch; after a merge only the merged
    component's costs are recomputed."""
    if len(components) <= k_max:
        return [((), p, m, S) for _, p, m, S in components]
    w = torch.stack([p for _, p, _, _ in components])
    m = torch.stack([mu for _, _, mu, _ in components])
    S = torch.stack([cov for _, _, _, cov in components])
    eye = torch.eye(m.shape[1], dtype=m.dtype) * jitter
    logdet = lambda C: torch.linalg.slogdet(C + eye)[1]

    def merged(i, j):
        """Moments of merging components i (index or tensor) with j."""
        wi, wj = w[i], w[j]
        ws = wi + wj
        ms = (wi[..., None] * m[i] + wj[..., None] * m[j]) / ws[..., None]
        di, dj = m[i] - ms, m[j] - ms
        Ss = (wi[..., None, None] * (S[i] + di[..., :, None] * di[..., None, :])
              + wj[..., None, None] * (S[j] + dj[..., :, None] * dj[..., None, :])) / ws[..., None, None]
        return ws, ms, Ss

    alive = torch.ones(len(w), dtype=torch.bool)
    ld = logdet(S)
    I, J = torch.triu_indices(len(w), len(w), offset=1)
    ws, _, Ss = merged(I, J)
    cost = torch.full((len(w), len(w)), float("inf"), dtype=m.dtype)
    cost[I, J] = 0.5 * (ws * logdet(Ss) - w[I] * ld[I] - w[J] * ld[J])
    n = len(w)
    while n > k_max:
        flat = torch.argmin(cost)
        i, j = divmod(flat.item(), cost.shape[1])
        wi, mi, Si = merged(torch.tensor(i), torch.tensor(j))
        w[i], m[i], S[i], ld[i] = wi, mi, Si, logdet(Si)
        alive[j] = False
        cost[j, :] = float("inf")
        cost[:, j] = float("inf")
        others = torch.nonzero(alive).flatten()
        others = others[others != i]
        if len(others):
            ii = torch.full_like(others, i)
            wo, _, So = merged(ii, others)
            c = 0.5 * (wo * logdet(So) - w[i] * ld[i] - w[others] * ld[others])
            lo, hi = torch.minimum(ii, others), torch.maximum(ii, others)
            cost[lo, hi] = c
        n -= 1
    idx = torch.nonzero(alive).flatten()
    return [((), w[k], m[k], S[k]) for k in idx]


def _orthant_moments(c1, c2, rho, n=600):
    """P, E[z], E[z z'] of (z1, z2) ~ N(0, [[1, rho], [rho, 1]]) truncated to
    z1 < c1, z2 < c2, by quadrature over z1 with z2 | z1 ~ N(rho z1, 1 - rho^2)
    in closed form (E[Y 1{Y < c}] = m Phi(t) - s phi(t), E[Y^2 1{Y < c}] =
    (m^2 + s^2) Phi(t) - s (m + c) phi(t), t = (c - m) / s)."""
    lo = torch.minimum(torch.as_tensor(-10.0, dtype=c1.dtype), c1 - 1.0)
    x = lo + (c1.clamp(max=10.0) - lo) * torch.linspace(0, 1, n, dtype=c1.dtype)
    wq = torch.full_like(x, 1.0)
    wq[0] = wq[-1] = 0.5
    wq = wq * (x[1] - x[0])  # trapezoid weights
    s = torch.sqrt((1 - rho**2).clamp(min=1e-12))
    m = rho * x
    t = (c2 - m) / s
    Phi, phi = norm_cdf(t), norm_pdf(t)
    base = norm_pdf(x) * wq
    P = (base * Phi).sum()
    E1 = (base * x * Phi).sum() / P
    E11 = (base * x**2 * Phi).sum() / P
    EY = m * Phi - s * phi
    EYY = (m**2 + s**2) * Phi - s * (m + c2) * phi
    E2 = (base * EY).sum() / P
    E22 = (base * EYY).sum() / P
    E12 = (base * x * EY).sum() / P
    mean = torch.stack([E1, E2])
    second = torch.stack([torch.stack([E11, E12]), torch.stack([E12, E22])])
    return P, mean, second - torch.outer(mean, mean)


def split_joint(mu, Sigma, g1, g2):
    """Split N(mu, Sigma) by two boundaries at once, both linearised at mu.
    Returns {(s1, s2): (p, mean, cov)} for the four sign regions (-1: g < 0),
    from the bivariate normal of (g1, g2): no dependence on the order of the
    boundaries. Falls back to splitting in turn if they are (nearly) parallel."""
    gs = (g1, g2)
    g0 = torch.stack([g(mu) for g in gs])
    A = torch.stack([jacrev(g)(mu) for g in gs], dim=1)  # (n, 2)
    C = A.T @ Sigma @ A
    sd = torch.sqrt(torch.diagonal(C))
    rho = C[0, 1] / (sd[0] * sd[1])
    if rho.abs() > 1 - 1e-6:
        out = {}
        for s1, (p1, m1, S1) in zip((-1, 1), split(mu, Sigma, g1)):
            for s2, (p2, m2, S2) in zip((-1, 1), split(m1, S1, g2)):
                out[(s1, s2)] = (p1 * p2, m2, S2)
        return out
    K = Sigma @ A @ torch.linalg.inv(C)  # regression of s on (g1, g2) - g0
    b = -g0 / sd  # g_i < 0  <=>  z_i < b_i, z = (g - g0) / sd standardised
    out = {}
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            sign = torch.tensor([-s1, -s2], dtype=mu.dtype)  # flip so each region is a lower orthant
            P, m_z, C_z = _orthant_moments(sign[0] * b[0], sign[1] * b[1], sign[0] * sign[1] * rho)
            D = torch.diag(sd * sign)  # (g - g0) = D z'
            m_g = D @ m_z
            C_g = D @ C_z @ D
            out[(s1, s2)] = (P, mu + K @ m_g, Sigma - K @ C @ K.T + K @ C_g @ K.T)
    return out


def propagate_unscented(sim, pieces, jitter=1e-14):
    """As `propagate_batched`, but each piece's output mean and covariance
    come from its 2n cubature points m +- sqrt(n lambda_i) v_i (principal
    axes, equal weights 1/2n) pushed through the real step, not from a
    linearisation. All outputs are simulated states, so a piece cannot be
    sent to an unphysical region by an extrapolated Jacobian."""
    if not pieces:
        return []
    n = 6
    S = torch.stack([S for _, _, _, S in pieces])
    lam, V = torch.linalg.eigh(S)
    offsets = (V * torch.sqrt(n * lam.clamp(min=0))[:, None, :]).transpose(1, 2)  # (P, n, n), rows are the axes
    means = torch.stack([m for _, _, m, _ in pieces])
    pts = torch.cat([means[:, None, :] + offsets, means[:, None, :] - offsets], dim=1)
    flat = pts.reshape(-1, n)
    q1, v1, _ = sim.step(flat[:, :3], flat[:, 3:])
    out = torch.cat([q1, v1], dim=1).reshape(len(pieces), 2 * n, n)
    m_out = out.mean(dim=1)
    d = out - m_out[:, None, :]
    S_out = d.transpose(1, 2) @ d / (2 * n) + jitter * torch.eye(n, dtype=S.dtype)
    return [(label, p, m_out[i], S_out[i]) for i, (label, p, _, _) in enumerate(pieces)]
