"""Box scenario: a settled box pushed through its centre of mass.

The 17 kg box (0.4 x 0.1 m, mu = 0.5) gets a constant push for the first
T_PUSH steps, with magnitude F in [0, 300] N and direction a in [-60, +30]
deg from horizontal (negative pushes down, raising the normal force and
with it the friction). Loss: (x(T) - X_TARGET)^2 at T = 1 s. Below the
friction limit F cos a < mu (m g - F sin a) the box does not move: the
loss is flat there and the exact gradient is zero (the failure mode
smoothing should fix). Parameters are normalised to u in [0, 1]^2.
"""

import math
from functools import lru_cache

import torch

import sim2d

MASS, HX, HY = 17.0, 0.2, 0.05
FORCE = (0.0, 300.0)
ANGLE = (math.radians(-60.0), math.radians(30.0))
T_PUSH, T_STEPS = 30, 100
X_TARGET = 0.5


# Friction regularisers: Ostrich's (slip floor 1e-4 m, which lets a stuck box
# creep and gives the stuck region a small gradient) or tight (1e-8 m, close
# to exact Coulomb stick: the stuck region is flat).
FRICTION = {
    "ostrich": {},
    "tight": {"friction_slip_eps_sq": 1e-16, "friction_denom_eps": 1e-10},
}


def make_sim(friction="ostrich"):
    return sim2d.Sim2D(sim2d.box(MASS, HX, HY), sim2d.SimConfig(**FRICTION[friction]))


@lru_cache
def settled_state():
    """The box after settling on the ground from 0.5 mm above it."""
    sim = make_sim()
    q, v = torch.tensor([[0.0, HY + 0.0005, 0.0]]), torch.zeros(1, 3)
    for _ in range(30):
        q, v, _ = sim.step(q, v)
    return q[0], v[0]


def push(u):
    """u (B, 2) -> magnitude F, angle a."""
    F = FORCE[0] + (FORCE[1] - FORCE[0]) * u[:, 0]
    a = ANGLE[0] + (ANGLE[1] - ANGLE[0]) * u[:, 1]
    return F, a


def forces(u):
    """Per-step external forces (T_STEPS, B, 3): the push, then nothing."""
    F, a = push(u)
    f = torch.zeros(T_STEPS, len(u), 3, dtype=torch.float64)
    f[:T_PUSH, :, 0] = F * torch.cos(a)
    f[:T_PUSH, :, 1] = F * torch.sin(a)
    return f


def initial_state(n):
    q, v = settled_state()
    return q.expand(n, 3).clone(), v.expand(n, 3).clone()


def loss(sim, u, chunk=4096):
    out = []
    for u_c in torch.split(u, chunk):
        q, v = initial_state(len(u_c))
        for f in forces(u_c):
            q, v, _ = sim.step(q, v, f)
        out.append((q[:, 0] - X_TARGET) ** 2)
    return torch.cat(out)


def loss_and_grad(sim, u, **adjoint):
    """L(u) and dL/du through the adjoint (adjoint kwargs as Sim2D.step_vjp)."""
    q0, v0 = initial_state(len(u))
    tape, q, _ = sim2d.rollout(sim, q0, v0, forces(u))
    L = (q[:, 0] - X_TARGET) ** 2
    g_qT = torch.zeros(len(u), 3, dtype=torch.float64)
    g_qT[:, 0] = 2 * (q[:, 0] - X_TARGET)
    _, _, g_f = sim2d.rollout_vjp(sim, tape, g_qT, torch.zeros_like(g_qT), **adjoint)
    gx, gy = g_f[:T_PUSH, :, 0].sum(dim=0), g_f[:T_PUSH, :, 1].sum(dim=0)
    F, a = push(u)
    c, s = torch.cos(a), torch.sin(a)
    g_u = torch.stack(
        [(gx * c + gy * s) * (FORCE[1] - FORCE[0]), F * (-gx * s + gy * c) * (ANGLE[1] - ANGLE[0])],
        dim=1,
    )
    return L, g_u
