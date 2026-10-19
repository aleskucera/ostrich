"""Apex drop in d dimensions: steer a falling ball onto the right side of a ridge.

The corner_drop wedge (apex at (0, H)) with the ball released at rest just
left of the apex. While it falls (the first T_PUSH steps) a horizontal force
pushes it, d phases with their own force F_k = F_MAX (2 u_k - 1). Where the
ball first touches the wedge decides the side it rolls down: a fork of the
"which side of an edge" kind. Loss: final distance to a target on the right
plus a force cost, so the best push only just gets the ball over the apex:

  L = (x(T) - X_TARGET)^2 + W_F mean_k (F_k / F_MAX)^2
"""

import torch

import corner_drop as cd
import sim2d

X0 = -0.02
F_MAX = 2.0
T_PUSH = 24  # about the fall time to the apex
W_F = 0.2
X_TARGET = 1.0
MU = 0.3


def make_sim():
    """Inelastic, with friction: the ball rolls down the side it lands on
    (corner_drop's bouncy frictionless wedge adds small jumps from the
    number of bounces)."""
    return cd.WedgeSim(sim2d.SimConfig(restitution=0.0, mu=MU))


def forces(u):
    """u (B, d) -> per-step external forces (T_STEPS, B, 3)."""
    B, d = u.shape
    phase = torch.arange(T_PUSH) * d // T_PUSH
    f = torch.zeros(cd.T_STEPS, B, 3, dtype=u.dtype)
    f[:T_PUSH, :, 0] = (F_MAX * (2 * u - 1)).T[phase]
    return f


def initial_state(n):
    q = torch.tensor([[X0, cd.Y0, 0.0]], dtype=torch.float64).expand(n, 3).clone()
    return q, torch.zeros(n, 3, dtype=torch.float64)


def cost(q, u):
    return (q[:, 0] - X_TARGET) ** 2 + W_F * ((2 * u - 1) ** 2).mean(dim=1)


def loss(sim, u, chunk=512):
    out = []
    for u_c in torch.split(u, chunk):
        q, v = initial_state(len(u_c))
        for f in forces(u_c):
            q, v, _ = sim.step(q, v, f)
        out.append(cost(q, u_c))
    return torch.cat(out)
