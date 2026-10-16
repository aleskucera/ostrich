"""Corner drop: a ball dropped onto the apex of a wedge, a bimodal outcome.

A symmetric wedge (ridge) on the ground plane: apex at (0, H), faces down to
(+-W, 0). A ball dropped from height Y0 at horizontal position x0 lands on
the left or the right face and slides down that side, so the outcome x(T)
splits into two modes at x0 = 0. With noise on the drop position,
x0 ~ N(theta, sigma), the mode weights are smooth functions of theta.

Contact candidates for the ball (all with thickness r at its centre): the
ground, the left face, the right face and the apex vertex. A face applies
while the centre projects onto it; the apex covers the cone between them,
with its normal from the apex to the centre. As in Ostrich, the normal and
anchor of each contact are fixed at the start of the step.
"""

import math

import torch

import sim2d

R, MASS = 0.05, 1.0
W, H = 0.25, 0.12  # wedge half-width and height
Y0 = 0.45  # drop height of the ball centre
E = 0.5  # restitution
T_STEPS = 150  # 1.5 s
X_TARGET = 0.5


class WedgeSim(sim2d.Sim2D):
    """A ball against the ground plane and a wedge (ridge) at x = 0."""

    def __init__(self, cfg):
        body = sim2d.Body(MASS, 0.4 * MASS * R**2, torch.zeros(4, 2), torch.full((4,), R))
        super().__init__(body, cfg)
        apex = torch.tensor([0.0, H])
        self.apex = apex
        self.dir_left = torch.tensor([-W, -H]) / math.hypot(W, H)  # along the faces, from the apex
        self.dir_right = torch.tensor([W, -H]) / math.hypot(W, H)
        self.n_left = torch.tensor([-H, W]) / math.hypot(W, H)  # outward normals
        self.n_right = torch.tensor([H, W]) / math.hypot(W, H)
        self.face_len = math.hypot(W, H)

    def contact_frame(self, q_prev):
        c = q_prev[:2]
        rel = c - self.apex
        s_left, s_right = rel @ self.dir_left, rel @ self.dir_right
        on_left = (s_left >= 0) & (s_left <= self.face_len)
        on_right = (s_right >= 0) & (s_right <= self.face_len)
        on_apex = (s_left < 0) & (s_right < 0)
        n_apex = rel / rel.norm().clamp(min=1e-12)
        ground = torch.tensor([0.0, 1.0])
        n = torch.stack([ground, self.n_left, self.n_right, n_apex])
        s = torch.stack([torch.zeros(2), self.apex, self.apex, self.apex])
        valid = torch.stack([torch.tensor(True), on_left, on_right, on_apex])
        return n, s, valid


def make_sim(restitution=E, **cfg):
    return WedgeSim(sim2d.SimConfig(restitution=restitution, mu=0.0, **cfg))


def drop(sim, x0, steps=T_STEPS, record=False):
    """Drop balls from (x0, Y0) at rest; returns final q, v (and the q path)."""
    q = torch.stack([x0, torch.full_like(x0, Y0), torch.zeros_like(x0)], dim=1)
    v = torch.zeros_like(q)
    path = [q]
    for _ in range(steps):
        q, v, _ = sim.step(q, v)
        if record:
            path.append(q)
    return (q, v, torch.stack(path)) if record else (q, v)


def surface_distance(sim, c):
    """Signed distance from points c (..., 2) to the ground-and-wedge surface
    (negative inside the ground or the wedge)."""
    rel = c - sim.apex
    feet = [sim.apex + (rel @ d).clamp(0.0, sim.face_len)[..., None] * d for d in (sim.dir_left, sim.dir_right)]
    to_faces = torch.stack([(c - f).norm(dim=-1) for f in feet]).min(dim=0).values
    inside = ((rel @ sim.n_left) < 0) & ((rel @ sim.n_right) < 0) & (c[..., 1] > 0)
    wedge = torch.where(inside, -to_faces, to_faces)
    return torch.minimum(c[..., 1], wedge)
