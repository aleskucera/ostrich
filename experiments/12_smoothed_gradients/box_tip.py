"""Box tip: a tall box pushed toward a target; it stays, slides, or tips over.

A settled 0.2 x 0.4 m box (5 kg, mu = 0.4) gets a horizontal push for the
first T_PUSH steps, magnitude F in [0, 50] N, applied at height h in
[0.05, 0.38] m above the ground (as an equivalent force and torque about the
centre). Loss at T: (x(T) - X_TARGET)^2 + W_TILT theta(T)^2, the box should
arrive upright.

Three outcomes:
  stays  F below the friction limit mu m g = 20 N (and too low to tip): the
         loss is flat, the exact gradient zero (the box scenario's failure)
  slides the box moves upright: the loss is smooth in (F, h)
  tips   a high or strong push rotates it past its corner: it falls on its
         side, theta(T) ~ -pi/2, a large jump in the loss
The tipped region lies next to the good sliding pushes (high pushes slide
the box further per newton), so descent walks toward the jump.

Parameters u in [0, 1]^2 map to (F, h).
"""

import math
from functools import lru_cache

import torch

import sim2d

MASS, HX, HY = 5.0, 0.1, 0.2
MU = 0.4
FORCE = (0.0, 50.0)
HEIGHT = (0.05, 0.38)
T_PUSH, T_STEPS = 40, 150
X_TARGET = 0.5
W_TILT = 1.0  # m^2 per rad^2


class BoxSim(sim2d.Restarts, sim2d.Sim2D):
    pass


def make_sim():
    return BoxSim(sim2d.box(MASS, HX, HY), sim2d.SimConfig(mu=MU, max_iters=40))


@lru_cache
def settled_state():
    """The box after settling on the ground from 0.5 mm above it."""
    sim = make_sim()
    q, v = torch.tensor([[0.0, HY + 0.0005, 0.0]]), torch.zeros(1, 3)
    for _ in range(30):
        q, v, _ = sim.step(q, v)
    return q[0], v[0]


def push(u):
    """u (B, 2) -> magnitude F, height h above the ground."""
    return FORCE[0] + (FORCE[1] - FORCE[0]) * u[:, 0], HEIGHT[0] + (HEIGHT[1] - HEIGHT[0]) * u[:, 1]


def forces(u):
    """Per-step external forces (T_STEPS, B, 3): horizontal push F at height
    h, i.e. force F and torque -F (h - HY) about the centre (box upright at
    the start; the push point is taken fixed in the world frame)."""
    F, h = push(u)
    f = torch.zeros(T_STEPS, len(u), 3, dtype=torch.float64)
    f[:T_PUSH, :, 0] = F
    f[:T_PUSH, :, 2] = -F * (h - HY)
    return f


def initial_state(n):
    q, v = settled_state()
    return q.expand(n, 3).clone(), v.expand(n, 3).clone()


def terminal_cost(q):
    return (q[:, 0] - X_TARGET) ** 2 + W_TILT * q[:, 2] ** 2


def run(sim, u, record=False):
    q, v = initial_state(len(u))
    path, lam = [q], None
    for f in forces(u):
        q, v, info = sim.step(q, v, f, lam0=lam)
        lam = info["z"][:, 3:]
        if record:
            path.append(q)
    return (q, v, torch.stack(path)) if record else (q, v)


def loss(sim, u, chunk=2048):
    return torch.cat([terminal_cost(run(sim, u_c)[0]) for u_c in torch.split(u, chunk)])


def loss_and_grad(sim, u, chunk=2048, **adjoint):
    """L(u) and dL/du (B, 2) through the adjoint."""
    Ls, gs = [], []
    for u_c in torch.split(u, chunk):
        q0, v0 = initial_state(len(u_c))
        tape, q, _ = sim2d.rollout(sim, q0, v0, forces(u_c), warm_start=True)
        g_q = torch.zeros_like(q)
        g_q[:, 0] = 2 * (q[:, 0] - X_TARGET)
        g_q[:, 2] = 2 * W_TILT * q[:, 2]
        _, _, g_f = sim2d.rollout_vjp(sim, tape, g_q, torch.zeros_like(q), **adjoint)
        gx, gw = g_f[:T_PUSH, :, 0].sum(dim=0), g_f[:T_PUSH, :, 2].sum(dim=0)
        F, h = push(u_c)
        g_F = gx - (h - HY) * gw
        g_h = -F * gw
        Ls.append(terminal_cost(q))
        gs.append(torch.stack([g_F * (FORCE[1] - FORCE[0]), g_h * (HEIGHT[1] - HEIGHT[0])], dim=1))
    return torch.cat(Ls), torch.cat(gs)
