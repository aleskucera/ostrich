"""Wheel and ledge: a driven wheel over bumpy terrain to a ledge it climbs or not.

A disc (the wheel of a planar robot) starts at rest on flat ground and is
driven by an axle torque. It crosses a stretch of small bumps (a polyline of
short segments, like a terrain mesh) and reaches a ledge of height 0.6 R.
Depending on the torque it climbs the ledge and drives on, or stalls against
its edge. Loss: (x(T) - X_TARGET)^2, target beyond the ledge.

Two kinds of discontinuity, as on a robot on a mesh:
  macro  climb / stall at the ledge: a large jump in x(T), and below it a
         flat region (a stalled wheel ends at the same place for any torque)
  micro  the bumps: contacts switch between segments and vertices, the wheel
         hops off bump tops, giving many small jumps and kinks in x(T)

Torque profile: d piecewise-constant phases over the horizon, parameters
u in [0, 1]^d mapped to [0, TAU_MAX] N m (forward drive).

Contact candidates: every terrain segment (applies while the wheel centre
projects onto it) and every convex vertex (applies in the cone between its
two segments, normal from the vertex to the centre), as in corner_drop. Only
the K nearest applicable features enter a step's contact set.
"""

import math

import numpy as np
import torch

import sim2d

R, MASS = 0.1, 2.0
INERTIA = 0.5 * MASS * R**2
MU = 0.8
TAU_MAX = 0.6
T_STEPS = 250  # 2.5 s
X_TARGET = 2.0
K = 4  # contact candidates per step

LEDGE_X, LEDGE_H = 1.4, 0.6 * R
BUMPS = (0.3, 1.1, 0.04, 0.004)  # start, end, segment length, height amplitude


def terrain(seed=0):
    """Polyline vertices (N, 2) from left to right; solid below."""
    x0, x1, seg, amp = BUMPS
    rng = np.random.default_rng(seed)
    xs = np.arange(x0, x1 + 1e-9, seg)
    ys = amp * rng.uniform(-1, 1, len(xs))
    ys[0] = ys[-1] = 0.0
    pts = [(-1.0, 0.0), *zip(xs, ys), (LEDGE_X, 0.0), (LEDGE_X, LEDGE_H), (8.0, LEDGE_H)]
    return torch.tensor(pts, dtype=torch.float64)


class WheelSim(sim2d.Restarts, sim2d.Sim2D):
    def __init__(self, cfg, verts=None):
        body = sim2d.Body(MASS, INERTIA, torch.zeros(K, 2), torch.full((K,), R))
        super().__init__(body, cfg)
        v = terrain() if verts is None else verts
        a, b = v[:-1], v[1:]
        self.seg_a = a
        self.seg_len = (b - a).norm(dim=1)
        self.seg_dir = (b - a) / self.seg_len[:, None]
        self.seg_n = torch.stack([-self.seg_dir[:, 1], self.seg_dir[:, 0]], dim=1)  # outward (up-left of the direction)
        d_in, d_out = self.seg_dir[:-1], self.seg_dir[1:]
        convex = (d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]) < 0  # turns clockwise
        self.vert = v[1:-1][convex]
        self.vert_in = torch.nonzero(convex)[:, 0]  # index of the segment ending at the vertex
        self.verts = v

    def contact_frame(self, q_prev):
        c = q_prev[:2]
        t = ((c - self.seg_a) * self.seg_dir).sum(dim=1)
        on = (t >= 0) & (t <= self.seg_len)
        d_seg = ((c - self.seg_a) * self.seg_n).sum(dim=1)
        rel = c - self.vert
        d_vert = rel.norm(dim=1)
        in_cone = (t[self.vert_in] > self.seg_len[self.vert_in]) & (t[self.vert_in + 1] < 0)
        n_vert = rel / d_vert.clamp(min=1e-12)[:, None]
        n = torch.cat([self.seg_n, n_vert])
        s = torch.cat([self.seg_a, self.vert])
        valid = torch.cat([on, in_cone])
        dist = torch.where(valid, torch.cat([d_seg.abs(), d_vert]), torch.inf)
        idx = torch.sort(torch.topk(-dist, K).indices).values  # stable slot order (warm start)
        return n[idx], s[idx], valid[idx]


def make_sim(**cfg):
    return WheelSim(sim2d.SimConfig(**{"mu": MU, "max_iters": 40, **cfg}))


def torques(u):
    """u (B, d) -> per-step external forces (T_STEPS, B, 3): phase j of the
    horizon gets torque TAU_MAX u_j (negative z: clockwise, drives to +x)."""
    B, d = u.shape
    phase = torch.arange(T_STEPS) * d // T_STEPS
    f = torch.zeros(T_STEPS, B, 3, dtype=torch.float64)
    f[:, :, 2] = -TAU_MAX * u.T[phase]
    return f


def initial_state(n):
    q = torch.tensor([0.0, R, 0.0]).expand(n, 3).clone()
    return q, torch.zeros(n, 3)


def run(sim, u, record=False):
    """Final q, v (B, 3) (and the q path (T+1, B, 3))."""
    q, v = initial_state(len(u))
    path, lam = [q], None
    for f in torques(u):
        q, v, info = sim.step(q, v, f, lam0=lam)
        lam = info["z"][:, 3:]
        if record:
            path.append(q)
    return (q, v, torch.stack(path)) if record else (q, v)


def loss(sim, u, chunk=2048):
    out = []
    for u_c in torch.split(u, chunk):
        q, _ = run(sim, u_c)
        out.append((q[:, 0] - X_TARGET) ** 2)
    return torch.cat(out)


def loss_and_grad(sim, u, chunk=2048, **adjoint):
    """L(u) and dL/du (B, d) through the adjoint."""
    Ls, gs = [], []
    for u_c in torch.split(u, chunk):
        q0, v0 = initial_state(len(u_c))
        f = torques(u_c)
        tape, q, _ = sim2d.rollout(sim, q0, v0, f, warm_start=True)
        g_q = torch.zeros_like(q)
        g_q[:, 0] = 2 * (q[:, 0] - X_TARGET)
        _, _, g_f = sim2d.rollout_vjp(sim, tape, g_q, torch.zeros_like(q), **adjoint)
        d = u_c.shape[1]
        phase = torch.arange(T_STEPS) * d // T_STEPS
        g_u = torch.zeros(len(u_c), d, dtype=torch.float64).index_add_(1, phase, -TAU_MAX * g_f[:, :, 2].T)
        Ls.append((q[:, 0] - X_TARGET) ** 2)
        gs.append(g_u)
    return torch.cat(Ls), torch.cat(gs)
