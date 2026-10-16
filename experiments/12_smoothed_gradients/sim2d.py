"""Small 2D contact simulator that mirrors Ostrich's step residual.

One body (ball or box) per world on the ground plane y = 0, batched over
worlds, torch float64 on CPU. State per world: q = (x, y, theta),
v = (vx, vy, omega).

Only the residual mirrors Ostrich:
  dynamics   constraints/dynamics_constraint.py  (2D: no gyroscopic term)
  normal     constraints/contact_constraint.py::compute_contact_core
  friction   constraints/friction_constraint.py::compute_friction_model
             (isotropic branch, 1 tangent direction)
The solve is plain dense Newton, K = jacrev(residual), torch.linalg.solve,
with per-world backtracking on ||R||^2. The friction weight w is a function of
the iterate (Ostrich lags it by one Newton iteration and under-relaxes it;
at a converged fixed point both coincide), so here the root of R(z, w(z)) is
solved directly and jacrev differentiates through w. Picard on w stalls
around 1e-4 on a sliding box, and plain Newton fails at the stopping step.
Ostrich's Schur complement, PCR, linesearch, backtracking, warm start and
compliance damper only change the path to the root, so they are left out.

Known deviations: Ostrich's friction preconditioner r averages the effective
masses of two tangent directions; in 2D there is only one. A 3D box has four
corners, the 2D one two (each carrying double load), so effective masses and
hence the eps^2 smoothing of the normal FB differ. Since the applied normal
force is dphi/dd * lambda_n but the friction budget is mu * lambda_n, Ostrich's
sliding box decelerates at 1.0017 mu g, this one at 1.0013 mu g.

Known limitation: on violent impacts with sliding (random box states with
|v_y| up to 3.5 m/s, |omega| up to 5 rad/s) ~2-4 % of steps do not converge.
Newton overshoots outside the friction cone into a region where the slip is
below the slip floor, w ~ 1e-5, and the Jacobian is nearly singular: a
stationary point of ||R||^2 that is not a root. Projecting onto the cone or
falling back to the frozen-w direction made it worse. Check
info["residual_world"].
"""

from dataclasses import dataclass

import torch
from torch.func import jacrev
from torch.func import vmap

torch.set_default_dtype(torch.float64)

NORMAL = torch.tensor([0.0, 1.0])
TANGENT = torch.tensor([1.0, 0.0])


@dataclass
class Body:
    mass: float
    inertia: float
    points: torch.Tensor  # (C, 2) contact points in the body frame
    thickness: torch.Tensor  # (C,) offset of each point along the normal


def ball(mass: float, radius: float, inertia: float | None = None) -> Body:
    """Ostrich sphere-plane contact: the centre, with thickness = radius."""
    if inertia is None:
        inertia = 0.4 * mass * radius**2
    return Body(mass, inertia, torch.zeros(1, 2), torch.tensor([radius]))


def box(mass: float, hx: float, hy: float, inertia: float | None = None) -> Body:
    """Ostrich box-plane contact: the corners, with thickness 0."""
    if inertia is None:
        inertia = mass * (4 * hx**2 + 4 * hy**2) / 12
    corners = torch.tensor([[-hx, -hy], [hx, -hy], [-hx, hy], [hx, hy]])
    return Body(mass, inertia, corners, torch.zeros(4))


@dataclass
class SimConfig:
    dt: float = 0.01
    mu: float = 0.5
    gravity: float = 9.81
    # builder.rigid_gap. A contact is detected if its gap at the start of the
    # step, or the gap the unconstrained step would reach, is below it (the
    # latter is not in Ostrich: without it a fast fall tunnels through the
    # margin and is pushed out next step, a spurious bounce).
    margin: float = 0.01
    # Restitution: an impacting contact (v_n^free < -restitution_v_thr and
    # d_free < 0) targets the end-of-step gap d* = d_k - e h v_n^- instead of
    # 0, so the step's displacement gives the outgoing normal velocity
    # e |v_n^-| (v_n^- at the start of the step). e = 0 is Ostrich (target 0);
    # note the jump at e -> 0+, where the ball stops at d_k for one step.
    restitution: float = 0.0
    restitution_v_thr: float = 0.2
    fb_eps_sq: float = 1e-8  # compliance.contact_fb_smooth_eps_sq
    load_gate: float = 1e-6  # relaxation.friction_load_gate
    # Regularisers hard-coded in compute_friction_model: under the slip norm
    # (a 1e-4 m displacement floor) and in the denominator of w (times dt).
    friction_slip_eps_sq: float = 1e-8
    friction_denom_eps: float = 1e-6
    max_iters: int = 200
    tol: float = 1e-10  # max-norm of the residual at the friction fixed point


def _rotate(theta, p):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], dim=1)


def integrate(q, v, h):
    """Ostrich's pose update: explicit quaternion step q += h/2 * w (x) q,
    then normalisation, which in 2D advances theta by 2 atan(h w / 2)."""
    return torch.stack([q[..., 0] + h * v[..., 0], q[..., 1] + h * v[..., 1],
                        q[..., 2] + 2.0 * torch.atan(0.5 * h * v[..., 2])], dim=-1)


def _cross2(a, b):
    return a[:, 0] * b[1] - a[:, 1] * b[0]


class Sim2D:
    def __init__(self, body: Body, cfg: SimConfig = SimConfig()):
        self.body = body
        self.cfg = cfg
        self.C = body.points.shape[0]
        self.m_diag = torch.tensor([body.mass, body.mass, body.inertia])
        self.f_gravity = torch.tensor([0.0, -body.mass * cfg.gravity, 0.0])

    # --- per-world functions (vmapped over worlds) -------------------------

    def gap(self, q):
        p = _rotate(q[2], self.body.points)
        return q[1] + p[:, 1] - self.body.thickness

    def jacobians(self, q_prev):
        """Normal and tangent Jacobians (C, 3), at the previous pose."""
        lever = _rotate(q_prev[2], self.body.points) - self.body.thickness[:, None] * NORMAL
        J_n = torch.cat([NORMAL.expand(self.C, 2), _cross2(lever, NORMAL)[:, None]], dim=1)
        J_t = torch.cat([TANGENT.expand(self.C, 2), _cross2(lever, TANGENT)[:, None]], dim=1)
        return J_n, J_t

    def effective_mass(self, J):
        return (J**2 / self.m_diag).sum(dim=1)

    def split(self, z):
        return z[:3], z[3 : 3 + self.C], z[3 + self.C :]

    def free_gap(self, q_prev, v_prev, f_ext):
        """Start-of-step gap, unconstrained-step gap d_free and normal velocity
        of the unconstrained step v_free = v_prev + h M^-1 (f_ext + gravity)."""
        h = self.cfg.dt
        v_free = v_prev + h * (f_ext + self.f_gravity) / self.m_diag
        J_n, _ = self.jacobians(q_prev)
        v_n = J_n @ v_free
        d0 = self.gap(q_prev)
        return d0, d0 + h * v_n, v_n

    def gap_target(self, q_prev, v_prev, f_ext):
        """Restitution target d* = d_k - e h v_n^- for impacting contacts."""
        if self.cfg.restitution == 0.0:
            return torch.zeros(self.C)
        d0, d_free, v_n_free = self.free_gap(q_prev, v_prev, f_ext)
        v_n_prev = self.jacobians(q_prev)[0] @ v_prev
        impact = (v_n_free < -self.cfg.restitution_v_thr) & (d_free < 0.0)
        return torch.where(impact, d0 - self.cfg.restitution * self.cfg.dt * v_n_prev, 0.0)

    def residual(self, z, q_prev, v_prev, f_ext, active, friction_on):
        h = self.cfg.dt
        v, lam_n, lam_t = self.split(z)
        w = self.friction_weight(z, q_prev)
        J_n, J_t = self.jacobians(q_prev)

        # Normal: scaled FB on (gap at end of step, lambda_n), alpha = 1,
        # beta = h^2 * effective mass, residual divided by h.
        d = self.gap(integrate(q_prev, v, h)) - self.gap_target(q_prev, v_prev, f_ext)
        beta = h**2 * self.effective_mass(J_n)
        norm = torch.sqrt(d**2 + (beta * lam_n) ** 2 + self.cfg.fb_eps_sq)
        phi = d + beta * lam_n - norm
        dphi_dd = 1.0 - d / norm  # Ostrich applies the normal force along dphi/dd * J_n

        force_n = (torch.where(active, dphi_dd * lam_n, 0.0)[:, None] * J_n).sum(dim=0)
        force_t = (torch.where(friction_on, lam_t, 0.0)[:, None] * J_t).sum(dim=0)
        r_dyn = self.m_diag * (v - v_prev) - h * (f_ext + self.f_gravity) - h * (force_n + force_t)

        # Rows of undetected contacts / unloaded friction pin their multiplier to 0.
        r_n = torch.where(active, phi / h, lam_n)
        r_t = torch.where(friction_on, J_t @ v + w * lam_t, lam_t)
        return torch.cat([r_dyn, r_n, r_t])

    def friction_weight(self, z, q_prev):
        """w of compute_friction_model (isotropic branch), from the iterate z."""
        h, mu = self.cfg.dt, self.cfg.mu
        v, lam_n, lam_t = self.split(z)
        _, J_t = self.jacobians(q_prev)
        r = self.effective_mass(J_t)

        d_t_norm = torch.sqrt((J_t @ v * h) ** 2 + self.cfg.friction_slip_eps_sq)
        raw = torch.abs(lam_t * h)
        limit = mu * lam_n * h
        clamped = torch.minimum(raw, limit)
        gap = limit - clamped
        phi = d_t_norm + r * gap - torch.sqrt(d_t_norm**2 + (r * gap) ** 2)
        w = r * (d_t_norm - phi) / (r * clamped + phi + self.cfg.friction_denom_eps * h)
        return torch.clamp(w, 0.0, 1e5)

    # --- batched step -----------------------------------------------------

    def step(self, q_prev, v_prev, f_ext=None):
        """One step for all worlds. q_prev, v_prev, f_ext: (B, 3).

        Returns q, v (B, 3) and an info dict with lam_n, lam_t (B, C), the
        number of Newton iterations and the final residual max-norm.
        """
        cfg, C = self.cfg, self.C
        B = q_prev.shape[0]
        if f_ext is None:
            f_ext = torch.zeros(B, 3)

        d0, d_free, _ = vmap(self.free_gap)(q_prev, v_prev, f_ext)
        active = torch.minimum(d0, d_free) < cfg.margin
        residual = vmap(self.residual)
        jacobian = vmap(jacrev(self.residual))

        z = torch.cat([v_prev, torch.zeros(B, 2 * C)], dim=1)
        for it in range(cfg.max_iters + 1):
            friction_on = active & (cfg.mu * z[:, 3 : 3 + C] > cfg.load_gate)
            args = (q_prev, v_prev, f_ext, active, friction_on)
            r = residual(z, *args)
            res_world = r.abs().amax(dim=1)
            res = res_world.amax()
            if it > 0 and res < cfg.tol or it == cfg.max_iters:
                break
            dz = torch.linalg.solve(jacobian(z, *args), r)
            merit = (r**2).sum(dim=1)
            alpha = torch.ones(B)
            for _ in range(30):
                trial = (residual(z - alpha[:, None] * dz, *args) ** 2).sum(dim=1)
                reject = trial > (1.0 - 1e-4 * alpha) * merit
                if not reject.any():
                    break
                alpha = torch.where(reject, 0.5 * alpha, alpha)
            z = z - alpha[:, None] * dz

        v = z[:, :3]
        info = {
            "lam_n": z[:, 3 : 3 + C],
            "lam_t": z[:, 3 + C :],
            "active": active,
            "iters": it,
            "residual": res.item(),
            "residual_world": res_world,
        }
        return integrate(q_prev, v, cfg.dt), v, info
