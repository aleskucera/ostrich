"""Box tip with a push profile: k phases, each with its own force and height.

The push of box_tip (first T_PUSH steps) split into k equal phases; phase j
has magnitude F_j and height h_j, so u in [0, 1]^(2k) maps to
(F_1, h_1, ..., F_k, h_k) with box_tip's ranges. k = 1 is box_tip. Same
loss: (x(T) - X_TARGET)^2 + W_TILT theta(T)^2. Tipping is a single event
(one corner pivot), unlike the wheel's repeated contacts at the ledge.
"""

import torch

import box_tip as bt
import sim2d


def forces(u):
    """u (B, 2k) -> per-step external forces (T_STEPS, B, 3)."""
    B, d = u.shape
    k = d // 2
    phase = torch.arange(bt.T_PUSH) * k // bt.T_PUSH
    F = bt.FORCE[0] + (bt.FORCE[1] - bt.FORCE[0]) * u[:, 0::2]
    h = bt.HEIGHT[0] + (bt.HEIGHT[1] - bt.HEIGHT[0]) * u[:, 1::2]
    f = torch.zeros(bt.T_STEPS, B, 3, dtype=torch.float64)
    f[: bt.T_PUSH, :, 0] = F.T[phase]
    f[: bt.T_PUSH, :, 2] = -(F * (h - bt.HY)).T[phase]
    return f


def run(sim, u):
    q, v = bt.initial_state(len(u))
    lam = None
    for f in forces(u):
        q, v, info = sim.step(q, v, f, lam0=lam)
        lam = info["z"][:, 3:]
    return q, v


def loss(sim, u, chunk=2048):
    return torch.cat([bt.terminal_cost(run(sim, u_c)[0]) for u_c in torch.split(u, chunk)])


def loss_and_grad(sim, u, chunk=2048, **adjoint):
    """L(u) and dL/du (B, 2k) through the adjoint (as box_tip.loss_and_grad,
    the force and torque gradients summed per phase)."""
    Ls, gs = [], []
    for u_c in torch.split(u, chunk):
        B, d = u_c.shape
        k = d // 2
        q0, v0 = bt.initial_state(B)
        tape, q, _ = sim2d.rollout(sim, q0, v0, forces(u_c), warm_start=True)
        g_q = torch.zeros_like(q)
        g_q[:, 0] = 2 * (q[:, 0] - bt.X_TARGET)
        g_q[:, 2] = 2 * bt.W_TILT * q[:, 2]
        _, _, g_f = sim2d.rollout_vjp(sim, tape, g_q, torch.zeros_like(q), **adjoint)
        phase = torch.arange(bt.T_PUSH) * k // bt.T_PUSH
        gx = torch.zeros(B, k, dtype=torch.float64).index_add_(1, phase, g_f[: bt.T_PUSH, :, 0].T)
        gw = torch.zeros(B, k, dtype=torch.float64).index_add_(1, phase, g_f[: bt.T_PUSH, :, 2].T)
        F = bt.FORCE[0] + (bt.FORCE[1] - bt.FORCE[0]) * u_c[:, 0::2]
        h = bt.HEIGHT[0] + (bt.HEIGHT[1] - bt.HEIGHT[0]) * u_c[:, 1::2]
        g = torch.zeros(B, d, dtype=torch.float64)
        g[:, 0::2] = (gx - (h - bt.HY) * gw) * (bt.FORCE[1] - bt.FORCE[0])
        g[:, 1::2] = -F * gw * (bt.HEIGHT[1] - bt.HEIGHT[0])
        Ls.append(bt.terminal_cost(q))
        gs.append(g)
    return torch.cat(Ls), torch.cat(gs)
