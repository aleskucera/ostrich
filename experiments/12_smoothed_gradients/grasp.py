"""Grasp and lift: a force-controlled two-finger gripper lifts a block.

A 6 x 12 cm block (0.3 kg) rests on the table. Two vertical finger pads
(4 cm tall, centred on the block's mid-height at the start) press on its
side faces with a commanded squeeze force F each and move up at speed v.
The pads hold the block by friction (mu = 0.5): it rises with them while
2 mu F covers its weight and acceleration, slips down relative to the pads
otherwise, and drops onto the table once it has slipped so far that the
pads no longer overlap its faces (the jump).

Parameters u in [0, 1]^(2k): k phases over the run, each a squeeze
F = F_MAX u and a lift speed v = V_MAX u. Loss: height error to the target
plus a squeeze cost (use as little force as possible, the classic grasp
trade-off; the optimum sits next to dropping):

  L = ((y(T) - Y_TARGET) / Y_SCALE)^2 + W_F mean_k (F_k / F_MAX)^2

Contacts (sim2d with moving geometry): the two bottom corners on the table,
and one contact per pad at the block's side-face midpoint with a prescribed
normal force F (zero velocity across the pad, the pad's velocity along it).
"""

import torch

import sim2d

MASS, HX, HY = 0.3, 0.03, 0.06
MU = 0.5
PAD_H = 0.02  # half height of a pad
F_MAX, V_MAX = 8.0, 0.4
T_STEPS = 120
Y_TARGET, Y_SCALE, W_F = HY + 0.15, 0.05, 0.5


class GraspSim(sim2d.Restarts, sim2d.Sim2D):
    """geom per world and step: (F, lift speed, pad centre height)."""

    def __init__(self):
        pts = torch.tensor([[-HX, -HY], [HX, -HY], [-HX, 0.0], [HX, 0.0]])
        super().__init__(sim2d.Body(MASS, MASS * (4 * HX**2 + 4 * HY**2) / 12, pts, torch.zeros(4)),
                         sim2d.SimConfig(mu=MU, max_iters=40))
        self.n = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])

    def contact_frame(self, q_prev, geom=None):
        s = torch.zeros(4, 2)
        valid = torch.ones(4, dtype=torch.bool)
        if geom is not None:
            # pads: anchored at the face point (gap 0; their normal force is prescribed),
            # valid while the pad overlaps the block's face
            p = q_prev[:2] + sim2d._rotate(q_prev[2], self.body.points[2:])
            s = torch.cat([s[:2], p])
            on = (geom[2] - q_prev[1]).abs() < HY + PAD_H
            valid = torch.cat([valid[:2], on.expand(2)])
        return self.n, s, valid

    def surface_velocity(self, q_prev, geom):
        return torch.cat([torch.zeros(2, 2), torch.stack([torch.zeros(2), geom[1].expand(2)], dim=1)])

    def prescribed_normal(self, q_prev, geom):
        return torch.tensor([False, False, True, True]), torch.cat([torch.zeros(2), geom[0].expand(2)])


def make_sim():
    return GraspSim()


def schedule(u):
    """u (B, 2k) -> geom per step (T, B, 3): F, lift speed, pad centre height
    at the start of the step."""
    B, d = u.shape
    k = d // 2
    phase = torch.arange(T_STEPS) * k // T_STEPS
    F = (F_MAX * u[:, 0::2]).T[phase]
    v = (V_MAX * u[:, 1::2]).T[phase]
    y = HY + torch.cumsum(torch.cat([torch.zeros(1, B, dtype=u.dtype), v[:-1]]), dim=0) * sim2d.SimConfig.dt
    return torch.stack([F, v, y], dim=2)


def initial_state(n):
    q = torch.tensor([[0.0, HY, 0.0]], dtype=torch.float64).expand(n, 3).clone()
    return q, torch.zeros(n, 3, dtype=torch.float64)


def cost(q, u):
    F = u[:, 0::2]
    return ((q[:, 1] - Y_TARGET) / Y_SCALE) ** 2 + W_F * (F**2).mean(dim=1)


def run(sim, u, record=False):
    q, v = initial_state(len(u))
    path, lam = [q], None
    for g in schedule(u):
        q, v, info = sim.step(q, v, lam0=lam, geom=g)
        lam = info["z"][:, 3:]
        if record:
            path.append(q)
    return (q, v, torch.stack(path)) if record else (q, v)


def loss(sim, u, chunk=512):
    return torch.cat([cost(run(sim, u_c)[0], u_c) for u_c in torch.split(u, chunk)])
