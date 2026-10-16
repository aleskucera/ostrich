"""Adjoint (Sim2D.step_vjp / rollout_vjp) against autodiff through the unrolled
Newton iterations and against central finite differences.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_adjoint.py
"""

import pytest
import torch

import sim2d

EXACT = {}
OSTRICH = {"friction": "frozen", "contact_compliance": 1e-6}


def fd_grad(loss, x0, eps=1e-6):
    g = torch.zeros(x0.numel())
    for j in range(x0.numel()):
        d = torch.zeros(x0.numel())
        d[j] = eps
        d = d.reshape(x0.shape)
        g[j] = (loss(x0 + d) - loss(x0 - d)) / (2 * eps)
    return g.reshape(x0.shape)


def ad_grad(loss, x0):
    x = x0.clone().requires_grad_(True)
    return torch.autograd.grad(loss(x), x)[0]


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


# --- single step ------------------------------------------------------------

STEP_CASES = {
    "sliding_box": (sim2d.box(17.0, 0.2, 0.05), [0.0, 0.050003, 0.0], [1.5, 0.0, 0.0], 0.0),
    "resting_box": (sim2d.box(17.0, 0.2, 0.05), [0.0, 0.050003, 0.0], [0.0, 0.0, 0.0], 0.0),
    "rolling_ball": (sim2d.ball(1.0, 0.05), [0.0, 0.050005, 0.0], [1.0, 0.0, -20.0], 0.0),
    "impact_e08": (sim2d.ball(1.0, 0.05), [0.0, 0.062, 0.0], [0.3, -2.0, 0.0], 0.8),
}


def step_jacobians(sim, q, v):
    """d(q1, v1) / d(q0, v0) (6 x 6) from the adjoint, unrolled AD and FD."""
    _, _, info = sim.step(q, v)
    rows = []
    for i in range(6):
        g = torch.zeros(1, 6)
        g[0, i] = 1.0
        g_q, g_v, _ = sim.step_vjp(q, v, torch.zeros(1, 3), info, g[:, :3], g[:, 3:])
        rows.append(torch.cat([g_q, g_v], 1)[0])
    J_adj = torch.stack(rows)

    def f(x):
        q1, v1, _ = sim.step(x[:, :3], x[:, 3:])
        return torch.cat([q1, v1], 1)[0]

    x0 = torch.cat([q, v], 1)
    J_ad = torch.autograd.functional.jacobian(f, x0)[:, 0, :]
    J_fd = torch.stack([fd_grad(lambda x: f(x)[i], x0)[0] for i in range(6)])
    return J_adj, J_ad, J_fd


@pytest.mark.parametrize("case", STEP_CASES)
def test_step_jacobian(case):
    body, q0, v0, e = STEP_CASES[case]
    sim = sim2d.Sim2D(body, sim2d.SimConfig(restitution=e, tol=1e-13))
    J_adj, J_ad, J_fd = step_jacobians(sim, torch.tensor([q0]), torch.tensor([v0]))
    assert rel_err(J_adj, J_ad) < 1e-7
    # FD truncation error near stick/roll kinks is O(eps): ~1e-4 at eps 1e-6.
    assert rel_err(J_adj, J_fd) < 1e-3


# --- trajectories -----------------------------------------------------------


def test_bouncing_ball_trajectory():
    """Frictionless ball, e = 0.8, two bounces; loss = height mid-flight after
    the second bounce, so it depends on both impacts' timing and targets."""
    sim = sim2d.Sim2D(sim2d.ball(1.0, 0.05), sim2d.SimConfig(mu=0.0, restitution=0.8, tol=1e-13))
    T, q0, v0 = 70, torch.tensor([[0.0, 0.3, 0.0]]), torch.tensor([[0.5, 0.0, 0.0]])
    forces = torch.zeros(T, 1, 3)

    def loss_v(v):
        return sim2d.rollout(sim, q0, v, forces)[1][0, 1]

    def loss_q(q):
        return sim2d.rollout(sim, q, v0, forces)[1][0, 1]

    tape, q, v = sim2d.rollout(sim, q0, v0, forces)
    vy = torch.stack([t[1][0, 1] for t in tape] + [v[0, 1]])
    assert int(((vy[1:] > 0) & (vy[:-1] < 0)).sum()) == 2
    assert v[0, 1] > 0.1 and q[0, 1] - 0.05 > 0.01  # airborne, rising

    g_q, g_v, _ = sim2d.rollout_vjp(sim, tape, torch.tensor([[0.0, 1.0, 0.0]]), torch.zeros(1, 3))
    for g_adj, loss, x0 in ((g_v, loss_v, v0), (g_q, loss_q, q0)):
        assert rel_err(g_adj, ad_grad(loss, x0)) < 1e-7
        assert rel_err(g_adj, fd_grad(loss, x0)) < 1e-5


def pushed_box():
    """Box pushed forward and down for 20 steps, then sliding to rest."""
    sim = sim2d.Sim2D(sim2d.box(17.0, 0.2, 0.05), sim2d.SimConfig(tol=1e-13))
    q0, T = torch.tensor([[0.0, 0.050003, 0.0]]), 40

    def forces(fxy):
        push = torch.cat([fxy.reshape(1, 1, 2).expand(20, 1, 2), torch.zeros(20, 1, 1)], dim=2)
        return torch.cat([push, torch.zeros(T - 20, 1, 3)])

    def loss(fxy):
        return sim2d.rollout(sim, q0, torch.zeros(1, 3), forces(fxy))[1][0, 0]

    f0 = torch.tensor([150.0, -60.0])
    tape, _, v = sim2d.rollout(sim, q0, torch.zeros(1, 3), forces(f0))
    assert abs(v[0, 0]) < 1e-6  # came to rest

    def adjoint(**mode):
        g = sim2d.rollout_vjp(sim, tape, torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1, 3), **mode)[2]
        return g[:20, 0, :2].sum(dim=0)

    return loss, f0, adjoint


def test_pushed_box_trajectory():
    loss, f0, adjoint = pushed_box()
    g = adjoint(**EXACT)
    assert rel_err(g, ad_grad(loss, f0)) < 1e-7
    assert rel_err(g, fd_grad(loss, f0)) < 1e-6


def test_frozen_friction_adjoint_misses_normal_coupling():
    """Characterises Ostrich's adjoint (friction w held fixed): sliding
    friction acts like a viscous damper whose magnitude ignores lambda_n, so
    pushing down (fy < 0, more friction) has no effect on the gradient and the
    push gradient is ~5x too small."""
    loss, f0, adjoint = pushed_box()
    g_exact, g_ostrich = adjoint(**EXACT), adjoint(**OSTRICH)
    assert g_exact[1] > 1e-3
    assert abs(g_ostrich[1]) < 1e-8
    assert rel_err(g_ostrich, g_exact) > 0.5
