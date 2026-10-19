"""Small 2D contact simulator that mirrors Ostrich's step residual.

One body (ball or box) per world on the ground plane y = 0, batched over
worlds, torch float64 on CPU. State per world: q = (x, y, theta),
v = (vx, vy, omega).

Only the residual mirrors Ostrich:
  dynamics   constraints/dynamics_constraint.py  (2D: no gyroscopic term)
  normal     constraints/contact_constraint.py::compute_contact_core
  friction   constraints/friction_constraint.py::compute_friction_model
             (isotropic branch, 1 tangent direction)
The solve is plain dense Newton (K = residual_jacobian, written by hand and
checked against jacrev; torch.linalg.solve),
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
from torch.func import vjp
from torch.func import vmap

torch.set_default_dtype(torch.float64)

NORMAL = torch.tensor([0.0, 1.0])


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
    """z-component of a x b, row-wise for (C, 2) arrays."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


class Sim2D:
    def __init__(self, body: Body, cfg: SimConfig = SimConfig()):
        self.body = body
        self.cfg = cfg
        self.C = body.points.shape[0]
        self.m_diag = torch.tensor([body.mass, body.mass, body.inertia])
        self.f_gravity = torch.tensor([0.0, -body.mass * cfg.gravity, 0.0])

    # --- per-world functions (vmapped over worlds) -------------------------

    def contact_frame(self, q_prev, **geom):
        """Contact candidates at the start of the step, as Ostrich's collision
        detection fixes them: per contact a normal n (C, 2) pointing from the
        static geometry to the body, an anchor point s (C, 2) on the geometry,
        and whether the candidate applies at this pose (C,). Default: every
        body point against the ground plane y = 0."""
        return NORMAL.expand(self.C, 2), torch.zeros(self.C, 2), torch.ones(self.C, dtype=torch.bool)

    # Moving geometry (scenes that pass step(..., geom=g), g (B, G) per world
    # and step; static scenes never see it): each hook gets geom=g[world].
    def surface_velocity(self, q_prev, **geom):
        """Velocity (C, 2) of the geometry under each contact (kinematic
        surfaces); enters the gap and the friction slip as a relative motion."""
        return torch.zeros(self.C, 2)

    def prescribed_normal(self, q_prev, **geom):
        """Contacts whose normal force is commanded, not solved for (a
        force-controlled finger pad): mask (C,) and force (C,). Their normal
        row is lam_n = force while valid, and the force acts along J_n."""
        return torch.zeros(self.C, dtype=torch.bool), torch.zeros(self.C)

    def gap(self, q, frame):
        """Signed distance along each fixed normal: n . (p(q) - s) - thickness."""
        n, s, _ = frame
        p = q[:2] + _rotate(q[2], self.body.points)
        return (n * (p - s)).sum(dim=1) - self.body.thickness

    def jacobians(self, q_prev, **geom):
        """Normal and tangent Jacobians (C, 3), at the previous pose."""
        n, _, _ = self.contact_frame(q_prev, **geom)
        t = torch.stack([n[:, 1], -n[:, 0]], dim=1)
        lever = _rotate(q_prev[2], self.body.points) - self.body.thickness[:, None] * n
        J_n = torch.cat([n, _cross2(lever, n)[:, None]], dim=1)
        J_t = torch.cat([t, _cross2(lever, t)[:, None]], dim=1)
        return J_n, J_t

    def effective_mass(self, J):
        return (J**2 / self.m_diag).sum(dim=1)

    def split(self, z):
        return z[:3], z[3 : 3 + self.C], z[3 + self.C :]

    def free_gap(self, q_prev, v_prev, f_ext, **geom):
        """Start-of-step gap, unconstrained-step gap d_free and normal velocity
        of the unconstrained step v_free = v_prev + h M^-1 (f_ext + gravity),
        relative to the geometry (moving geometry: the frame's anchors are at
        the end of the step, so the start gap adds back h u_n)."""
        h = self.cfg.dt
        v_free = v_prev + h * (f_ext + self.f_gravity) / self.m_diag
        J_n, _ = self.jacobians(q_prev, **geom)
        frame = self.contact_frame(q_prev, **geom)
        v_n = J_n @ v_free
        d_end = self.gap(q_prev, frame)
        if not geom:
            return d_end, d_end + h * v_n, v_n
        u_n = (frame[0] * self.surface_velocity(q_prev, **geom)).sum(dim=1)
        return d_end + h * u_n, d_end + h * v_n, v_n - u_n

    def gap_target(self, q_prev, v_prev, f_ext, **geom):
        """Restitution target d* = d_k - e h v_n^- for impacting contacts."""
        if self.cfg.restitution == 0.0:
            return torch.zeros(self.C)
        d0, d_free, v_n_free = self.free_gap(q_prev, v_prev, f_ext, **geom)
        v_n_prev = self.jacobians(q_prev, **geom)[0] @ v_prev
        if geom:
            v_n_prev = v_n_prev - (self.contact_frame(q_prev, **geom)[0] * self.surface_velocity(q_prev, **geom)).sum(dim=1)
        impact = (v_n_free < -self.cfg.restitution_v_thr) & (d_free < 0.0)
        return torch.where(impact, d0 - self.cfg.restitution * self.cfg.dt * v_n_prev, 0.0)

    def residual(self, z, q_prev, v_prev, f_ext, active, friction_on, z_w=None, **geom):
        """Step residual R(z) for unknowns z = (v, lam_n, lam_t). The friction
        weight w is computed from z_w (default z); the adjoint passes a fixed
        z_w to hold w frozen, as Ostrich's adjoint does."""
        h = self.cfg.dt
        v, lam_n, lam_t = self.split(z)
        w = self.friction_weight(z if z_w is None else z_w, q_prev, **geom)
        J_n, J_t = self.jacobians(q_prev, **geom)

        # Normal: scaled FB on (gap at end of step, lambda_n), alpha = 1,
        # beta = h^2 * effective mass, residual divided by h.
        d = self.gap(integrate(q_prev, v, h), self.contact_frame(q_prev, **geom)) - self.gap_target(q_prev, v_prev, f_ext, **geom)
        beta = h**2 * self.effective_mass(J_n)
        norm = torch.sqrt(d**2 + (beta * lam_n) ** 2 + self.cfg.fb_eps_sq)
        phi = d + beta * lam_n - norm
        dphi_dd = 1.0 - d / norm  # Ostrich applies the normal force along dphi/dd * J_n
        if geom:
            pres, lam_p = self.prescribed_normal(q_prev, **geom)
            dphi_dd = torch.where(pres, 1.0, dphi_dd)
            phi = torch.where(pres, h * (lam_n - lam_p), phi)

        force_n = (torch.where(active, dphi_dd * lam_n, 0.0)[:, None] * J_n).sum(dim=0)
        force_t = (torch.where(friction_on, lam_t, 0.0)[:, None] * J_t).sum(dim=0)
        r_dyn = self.m_diag * (v - v_prev) - h * (f_ext + self.f_gravity) - h * (force_n + force_t)

        # Rows of undetected contacts / unloaded friction pin their multiplier to 0.
        r_n = torch.where(active, phi / h, lam_n)
        v_t = J_t @ v
        if geom:
            n = self.contact_frame(q_prev, **geom)[0]
            v_t = v_t - (torch.stack([n[:, 1], -n[:, 0]], dim=1) * self.surface_velocity(q_prev, **geom)).sum(dim=1)
        r_t = torch.where(friction_on, v_t + w * lam_t, lam_t)
        return torch.cat([r_dyn, r_n, r_t])

    def friction_weight(self, z, q_prev, **geom):
        """w of compute_friction_model (isotropic branch), from the iterate z."""
        return self.friction_weight_partials(z, q_prev, **geom)[0]

    def friction_weight_partials(self, z, q_prev, **geom):
        """w and its partial derivatives dw/dv_t, dw/dlam_n, dw/dlam_t (each (C,)).
        The derivatives follow torch's conventions (abs'(0) = 0, clamp passes
        the gradient on [0, 1e5]), so they match autodiff of w."""
        h, mu = self.cfg.dt, self.cfg.mu
        v, lam_n, lam_t = self.split(z)
        _, J_t = self.jacobians(q_prev, **geom)
        r = self.effective_mass(J_t)

        v_t = J_t @ v
        if geom:
            n = self.contact_frame(q_prev, **geom)[0]
            v_t = v_t - (torch.stack([n[:, 1], -n[:, 0]], dim=1) * self.surface_velocity(q_prev, **geom)).sum(dim=1)
        d_t_norm = torch.sqrt((v_t * h) ** 2 + self.cfg.friction_slip_eps_sq)
        raw = torch.abs(lam_t * h)
        limit = mu * lam_n * h
        below = raw < limit
        clamped = torch.where(below, raw, limit)  # min(raw, limit)
        gap = limit - clamped
        S = torch.sqrt(d_t_norm**2 + (r * gap) ** 2)
        phi = d_t_norm + r * gap - S
        N = d_t_norm - phi
        D = r * clamped + phi + self.cfg.friction_denom_eps * h
        w_raw = r * N / D
        w = torch.clamp(w_raw, 0.0, 1e5)

        # Chain rule: w_raw(d_t_norm, gap, clamped), with gap and clamped
        # functions of raw = |lam_t h| and limit = mu lam_n h.
        phi_d = 1.0 - d_t_norm / S
        phi_g = r * (1.0 - r * gap / S)
        dw = lambda dN, dD: r * (dN * D - N * dD) / D**2
        dw_dnorm = dw(1.0 - phi_d, phi_d)
        dw_dgap = dw(-phi_g, phi_g)
        dw_dclamped = dw(0.0, r)
        b = below.to(z.dtype)
        dw_draw = b * (dw_dclamped - dw_dgap)
        dw_dlimit = (1.0 - b) * dw_dclamped + b * dw_dgap
        inside = ((w_raw >= 0.0) & (w_raw <= 1e5)).to(z.dtype)
        dw_dvt = inside * dw_dnorm * v_t * h**2 / d_t_norm
        dw_dlam_n = inside * dw_dlimit * mu * h
        dw_dlam_t = inside * dw_draw * h * torch.sign(lam_t)
        return w, dw_dvt, dw_dlam_n, dw_dlam_t

    def residual_jacobian(self, z, q_prev, v_prev, f_ext, active, friction_on, **geom):
        """dR/dz of `residual` (exact: includes dw/dz), written out by hand;
        equal to jacrev(self.residual) and several times cheaper."""
        h, C = self.cfg.dt, self.C
        v, lam_n, lam_t = self.split(z)
        J_n, J_t = self.jacobians(q_prev, **geom)
        a, o = active.to(z.dtype), friction_on.to(z.dtype)

        # Gap at the end of the step and its derivative g = dd/dv (C, 3).
        q1 = integrate(q_prev, v, h)
        p = self.body.points
        frame = self.contact_frame(q_prev, **geom)
        n = frame[0]
        d = self.gap(q1, frame) - self.gap_target(q_prev, v_prev, f_ext, **geom)
        dtheta_dw = h / (1.0 + (0.5 * h * v[2]) ** 2)
        c1, s1 = torch.cos(q1[2]), torch.sin(q1[2])
        dp_dtheta = torch.stack([-s1 * p[:, 0] - c1 * p[:, 1], c1 * p[:, 0] - s1 * p[:, 1]], dim=1)
        g = torch.stack([n[:, 0] * h, n[:, 1] * h, (n * dp_dtheta).sum(dim=1) * dtheta_dw], dim=1)

        # Scaled FB phi(d, lam_n) and its derivatives.
        beta = h**2 * self.effective_mass(J_n)
        norm = torch.sqrt(d**2 + (beta * lam_n) ** 2 + self.cfg.fb_eps_sq)
        phi_d = 1.0 - d / norm
        phi_l = beta * (1.0 - beta * lam_n / norm)
        phi_dd = -((beta * lam_n) ** 2 + self.cfg.fb_eps_sq) / norm**3
        phi_dl = d * beta**2 * lam_n / norm**3
        if geom:  # commanded normal force: row h (lam_n - force), force along J_n
            pres = self.prescribed_normal(q_prev, **geom)[0]
            phi_d, phi_dd, phi_dl = torch.where(pres, 1.0, phi_d), torch.where(pres, 0.0, phi_dd), torch.where(pres, 0.0, phi_dl)
            g = torch.where(pres[:, None], 0.0, g)
            phi_l = torch.where(pres, h, phi_l)

        w, dw_dvt, dw_dlam_n, dw_dlam_t = self.friction_weight_partials(z, q_prev, **geom)

        # Blocks assembled with cat (in-place writes are not allowed under vmap).
        # Dynamics rows: M (v - v_prev) - h (f + g) - h sum(phi_d lam_n J_n + lam_t J_t)
        dyn = torch.cat([
            torch.diag(self.m_diag) - h * torch.einsum("i,ij,ik->jk", a * lam_n * phi_dd, J_n, g),
            -h * (J_n * (a * (phi_d + lam_n * phi_dl))[:, None]).T,
            -h * (J_t * o[:, None]).T,
        ], dim=1)
        # Normal rows: phi / h (active) or lam_n
        normal = torch.cat([
            (a * phi_d / h)[:, None] * g,
            torch.diag(a * phi_l / h + (1.0 - a)),
            torch.zeros(C, C, dtype=z.dtype),
        ], dim=1)
        # Friction rows: v_t + w lam_t (friction on) or lam_t
        friction = torch.cat([
            (o * (1.0 + lam_t * dw_dvt))[:, None] * J_t,
            torch.diag(o * lam_t * dw_dlam_n),
            torch.diag(o * (w + lam_t * dw_dlam_t) + (1.0 - o)),
        ], dim=1)
        K = torch.cat([dyn, normal, friction], dim=0)
        return K

    # --- batched step -----------------------------------------------------

    def step(self, q_prev, v_prev, f_ext=None, lam0=None, geom=None):
        """One step for all worlds. q_prev, v_prev, f_ext: (B, 3).

        lam0 (B, 2C): initial contact forces for Newton (warm start, e.g. the
        previous step's info["z"][:, 3:]); default zero. It changes only the
        path to the root, not the root.

        geom (B, G): per-world state of moving geometry for this step, handed
        to the hooks contact_frame / surface_velocity / prescribed_normal as
        geom=geom[world] (forward only: step_vjp ignores it).

        Returns q, v (B, 3) and an info dict with lam_n, lam_t (B, C), the
        number of Newton iterations and the final residual max-norm.
        """
        cfg, C = self.cfg, self.C
        B = q_prev.shape[0]
        if f_ext is None:
            f_ext = torch.zeros(B, 3)

        if geom is None:
            d0, d_free, _ = vmap(self.free_gap)(q_prev, v_prev, f_ext)
            valid = vmap(lambda q: self.contact_frame(q)[2])(q_prev)
            active = (torch.minimum(d0, d_free) < cfg.margin) & valid
            residual = vmap(self.residual)
            jacobian = vmap(self.residual_jacobian)
        else:
            d0, d_free, _ = vmap(lambda q, v, f, g: self.free_gap(q, v, f, geom=g))(q_prev, v_prev, f_ext, geom)
            valid = vmap(lambda q, g: self.contact_frame(q, geom=g)[2])(q_prev, geom)
            pres = vmap(lambda q, g: self.prescribed_normal(q, geom=g)[0])(q_prev, geom)
            active = ((torch.minimum(d0, d_free) < cfg.margin) | pres) & valid
            residual = lambda *a, _r=vmap(lambda z, q, v, f, ac, o, g: self.residual(z, q, v, f, ac, o, None, geom=g)): _r(*a, geom)
            jacobian = lambda *a, _j=vmap(lambda z, q, v, f, ac, o, g: self.residual_jacobian(z, q, v, f, ac, o, geom=g)): _j(*a, geom)

        z = torch.cat([v_prev, torch.zeros(B, 2 * C) if lam0 is None else lam0], dim=1)
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
            # Converged worlds take no step: their merit is at rounding level,
            # so the decrease test would fail on noise and halve 30 times.
            done = (res_world < cfg.tol) & (it > 0)
            alpha = torch.where(done, 0.0, 1.0)
            for _ in range(30):
                trial = (residual(z - alpha[:, None] * dz, *args) ** 2).sum(dim=1)
                reject = (trial > (1.0 - 1e-4 * alpha) * merit) & ~done
                if not reject.any():
                    break
                alpha = torch.where(reject, 0.5 * alpha, alpha)
            z = z - alpha[:, None] * dz

        v = z[:, :3]
        info = {
            "lam_n": z[:, 3 : 3 + C],
            "lam_t": z[:, 3 + C :],
            "active": active,
            "z": z,
            "friction_on": friction_on,
            "iters": it,
            "residual": res.item(),
            "residual_world": res_world,
        }
        return integrate(q_prev, v, cfg.dt), v, info

    # --- adjoint ------------------------------------------------------------

    def step_vjp(self, q_prev, v_prev, f_ext, info, g_q, g_v, friction="exact", contact_compliance=0.0):
        """Pull loss gradients (g_q, g_v) on a step's outputs back to its inputs.

        Implicit function theorem at the converged root z of R(z, p) = 0,
        p = (q_prev, v_prev, f_ext): with K = dR/dz, mu = K^-T g_z and
        dL/dp = -mu^T dR/dp, plus the explicit dependence of q = integrate(q_prev, v).
        Contact sets (active, friction_on) are those of the forward step.

        friction="exact": K contains dw/dz, the exact derivative of the
          friction fixed point. "frozen": w held fixed in z (Ostrich's
          adjoint, friction_linearization="true"); w's dependence on p is kept.
        contact_compliance: added to the normal rows' dR/dlambda_n, as
          AdjointConfig.contact_compliance (Ostrich default 1e-6; 0 = exact).

        Returns gradients on q_prev, v_prev, f_ext, each (B, 3).
        """
        assert friction in ("exact", "frozen")
        h, C = self.cfg.dt, self.C
        active = info["active"]

        def world(z, q_prev, v_prev, f_ext, active, friction_on, g_q, g_v):
            z_w = None if friction == "exact" else z  # a separate argument: constant for jacrev
            K = jacrev(self.residual)(z, q_prev, v_prev, f_ext, active, friction_on, z_w)
            K[3 : 3 + C, 3 : 3 + C] += torch.diag(torch.where(active, contact_compliance / h, 0.0))
            g_qprev_explicit, g_v_from_q = vjp(lambda q, v: integrate(q, v, h), q_prev, z[:3])[1](g_q)
            g_z = torch.cat([g_v + g_v_from_q, torch.zeros(2 * C)])
            mu = torch.linalg.solve(K.T, g_z)
            _, pull = vjp(lambda q, v, f: self.residual(z, q, v, f, active, friction_on, z_w), q_prev, v_prev, f_ext)
            g_qprev, g_vprev, g_f = pull(-mu)
            return g_qprev + g_qprev_explicit, g_vprev, g_f

        if f_ext is None:
            f_ext = torch.zeros_like(q_prev)
        return vmap(world)(info["z"], q_prev, v_prev, f_ext, active, info["friction_on"], g_q, g_v)


class Restarts:
    """Mixin for Sim2D: a world whose Newton solve has not converged after
    max_iters is solved again from zero contact forces, then from each
    candidate carrying an equal share of the weight, with no friction and
    with sliding friction mu * share of either sign; the result with the
    smaller residual is kept. Newton can stall in one start and converge from
    another (wedged contacts, tumbling impacts); only the start differs, the
    root is the same. Use as class S(Restarts, Sim2D), with a low max_iters."""

    def step(self, q_prev, v_prev, f_ext=None, lam0=None, geom=None):
        q, v, info = super().step(q_prev, v_prev, f_ext, lam0, geom)
        C = self.C
        share = torch.full((C,), self.body.mass * self.cfg.gravity * self.cfg.dt / C)
        starts = [None] + [torch.cat([share, sign * self.cfg.mu * share]) for sign in (0.0, -1.0, 1.0)]
        for start in starts:
            bad = torch.nonzero(info["residual_world"] > self.cfg.tol)[:, 0]
            if len(bad) == 0:
                break
            l0 = None if start is None else start.expand(len(bad), 2 * C)
            f = None if f_ext is None else f_ext[bad]
            q_b, v_b, info_b = super().step(q_prev[bad], v_prev[bad], f, l0, None if geom is None else geom[bad])
            fixed = info_b["residual_world"] < info["residual_world"][bad]
            sel = bad[fixed]
            q, v = q.clone(), v.clone()
            q[sel], v[sel] = q_b[fixed], v_b[fixed]
            for key in ("lam_n", "lam_t", "active", "z", "friction_on", "residual_world"):
                info[key] = info[key].clone()
                info[key][sel] = info_b[key][fixed]
        info["residual"] = info["residual_world"].max().item()
        return q, v, info


def rollout(sim, q0, v0, forces, warm_start=False):
    """Steps all worlds through forces (T, B, 3). Returns the list of
    (q_prev, v_prev, f_ext, info) per step and the final (q, v). With
    warm_start, each Newton solve starts from the previous step's contact
    forces (3x fewer iterations on a resting or sliding box; same roots)."""
    q, v, tape, lam = q0, v0, [], None
    for f in forces:
        q_next, v_next, info = sim.step(q, v, f, lam0=lam)
        tape.append((q, v, f, info))
        q, v = q_next, v_next
        if warm_start:
            lam = info["z"][:, 3:]
    return tape, q, v


def rollout_vjp(sim, tape, g_q, g_v, **adjoint):
    """Backpropagates loss gradients on the final (q, v) through a rollout
    with Sim2D.step_vjp. Returns gradients on q0, v0 and the forces (T, B, 3)."""
    g_forces = []
    for q_prev, v_prev, f, info in reversed(tape):
        g_q, g_v, g_f = sim.step_vjp(q_prev, v_prev, f, info, g_q, g_v, **adjoint)
        g_forces.append(g_f)
    return g_q, g_v, torch.stack(g_forces[::-1])
