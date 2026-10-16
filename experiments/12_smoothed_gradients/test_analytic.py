"""sim2d against analytic solutions.

Each scenario is checked against the exact solution of the discrete
time-stepper (implicit Euler + Coulomb friction + inelastic contact), which
isolates the contact formulation from the O(h) time discretisation, and
against the continuous-time solution within O(h).

Two configs: EXACT shrinks the regularisers that sim2d copies from Ostrich,
OSTRICH keeps Ostrich's values. Their two effects:
  * fb_eps_sq leaves a gap d = eps^2 / (2 beta lambda) > 0 at a loaded contact,
    so dphi/dd = 1 - d / (d + beta lambda) < 1 and lambda exceeds the applied
    load N: 2 beta^2 lambda^3 = N (eps^2 + 2 beta^2 lambda^2). Friction scales
    with lambda. This is predicted exactly (`contact_lambda`) and built into
    the references (0.14 % for the 17 kg box, 0.51 % for the 1 kg ball).
  * the friction slip floor and denominator eps pull the friction force inside
    the cone at low slip speed (stop step off by ~1e-4 m/s, then a short
    creep). Not modelled; covered by the OSTRICH tolerances.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_analytic.py
"""

import numpy as np
import pytest
import torch

import sim2d

G, MU, H = 9.81, 0.5, 0.01
EXACT = sim2d.SimConfig(fb_eps_sq=1e-16, friction_slip_eps_sq=1e-16, friction_denom_eps=1e-10)
OSTRICH = sim2d.SimConfig()
CONFIGS = {"exact": EXACT, "ostrich": OSTRICH}

# Per config: (rtol on forces, atol on positions, atol on velocities)
TOL = {"exact": (1e-8, 1e-9, 1e-8), "ostrich": (1e-4, 2e-5, 2e-4)}


def contact_lambda(sim, q, load):
    """lambda_n of a contact carrying the applied normal load `load`, for
    each contact of the body at pose q (the eps^2 effect above)."""
    beta = H**2 * sim.effective_mass(sim.jacobians(torch.tensor(q))[0]).numpy()
    lam = []
    loads = np.broadcast_to(load, beta.shape) if np.ndim(load) == 0 else load
    for b, n in zip(beta, loads):
        roots = np.roots([2 * b**2, -2 * b**2 * n, 0.0, -n * sim.cfg.fb_eps_sq])
        lam.append(max(x.real for x in roots if abs(x.imag) < 1e-9))
    return np.array(lam)


def run(sim, q0, v0, n_steps):
    q, v = torch.tensor([q0]), torch.tensor([v0])
    qs, vs, infos = [], [], []
    for _ in range(n_steps):
        q, v, info = sim.step(q, v)
        qs.append(q[0].numpy().copy())
        vs.append(v[0].numpy().copy())
        infos.append(info)
    return np.array(qs), np.array(vs), infos


# --- resting box ------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIGS)
def test_resting_box(name):
    rtol, atol_q, atol_v = TOL[name]
    m, hx, hy = 17.0, 0.2, 0.05
    sim = sim2d.Sim2D(sim2d.box(m, hx, hy), CONFIGS[name])
    qs, vs, infos = run(sim, [0.0, hy + 0.0005, 0.0], [0.0, 0.0, 0.0], 50)

    lam_n = infos[-1]["lam_n"][0, :2].numpy()
    lam_pred = contact_lambda(sim, qs[-1], 0.5 * m * G)[:2]
    assert lam_n == pytest.approx(lam_pred, rel=1e-9)
    # resting gap d = eps^2 / (2 beta lambda)
    beta = H**2 * sim.effective_mass(sim.jacobians(torch.tensor(qs[-1]))[0])[0].item()
    assert qs[-1][1] - hy == pytest.approx(sim.cfg.fb_eps_sq / (2 * beta * lam_pred[0]), abs=1e-12)
    assert np.abs(vs[-1]).max() < atol_v


# --- box sliding to a stop ----------------------------------------------------


def discrete_slide(x0, v0, mu, n_steps):
    """Implicit Euler with Coulomb friction: v -= mu g h until v = 0."""
    x, v, xs, vs = x0, v0, [], []
    for _ in range(n_steps):
        v = max(v - mu * G * H, 0.0)
        x += H * v
        xs.append(x)
        vs.append(v)
    return np.array(xs), np.array(vs)


@pytest.mark.parametrize("name", CONFIGS)
def test_sliding_box(name):
    rtol, atol_q, atol_v = TOL[name]
    m, hx, hy, v0 = 17.0, 0.2, 0.05, 2.0
    sim = sim2d.Sim2D(sim2d.box(m, hx, hy), CONFIGS[name])
    q_rest, _, _ = run(sim, [0.0, hy + 0.0005, 0.0], [0.0, 0.0, 0.0], 20)
    qs, vs, infos = run(sim, q_rest[-1].tolist(), [v0, 0.0, 0.0], 60)

    # Applied loads balance the friction moment: N_front - N_back =
    # F_friction hy / hx, with F_friction = mu * sum(lambda) (fixed point).
    mu_eff = MU
    for _ in range(20):
        dn = 0.5 * mu_eff * m * G * hy / hx
        loads = np.array([0.5 * m * G - dn, 0.5 * m * G + dn])  # back, front
        lam_pred = contact_lambda(sim, q_rest[-1], loads)[:2]
        mu_eff = MU * lam_pred.sum() / (m * G)

    xs, vs_exact = discrete_slide(q_rest[-1][0], v0, mu_eff, 60)
    assert np.abs(vs[:, 0] - vs_exact).max() < atol_v
    assert np.abs(qs[:, 0] - xs).max() < atol_q
    assert np.abs(qs[:, 2]).max() < atol_q  # moment balance: no pitching

    # While sliding: normal loads as predicted, friction on the cone. Skip two
    # steps after the kick: with eps^2 the front and back corners rest at
    # different gaps, so the box first pitches by ~2e-6 rad (load split
    # alternates by ~2e-4 meanwhile).
    for info in infos[2:30]:
        lam_n, lam_t = info["lam_n"][0, :2].numpy(), info["lam_t"][0, :2].numpy()
        assert lam_n == pytest.approx(lam_pred, rel=rtol)
        assert lam_t == pytest.approx(-MU * lam_n, rel=rtol)

    # Continuous time: stopping distance v0^2 / (2 mu g), within O(h v0).
    x_stop = v0**2 / (2 * MU * G)
    assert abs(qs[-1][0] - q_rest[-1][0] - x_stop) < H * v0


# --- ball sliding into rolling ----------------------------------------------


def discrete_roll(x0, v0, m, r, inertia, mu, n_steps):
    """Kinetic friction until the step in which slip would reverse; that step
    sticks with impulse P = (v + w r) / (1/m + r^2/I), then pure rolling."""
    x, v, w, rolling = x0, v0, 0.0, False
    xs, vs, ws = [], [], []
    for _ in range(n_steps):
        if not rolling:
            dv, dw = mu * G * H, mu * m * G * r * H / inertia
            if (v - dv) + (w - dw) * r > 0.0:
                v, w = v - dv, w - dw
            else:
                p = (v + w * r) / (1.0 / m + r**2 / inertia)
                v, w, rolling = v - p / m, w - p * r / inertia, True
        x += H * v
        xs.append(x)
        vs.append(v)
        ws.append(w)
    return np.array(xs), np.array(vs), np.array(ws)


@pytest.mark.parametrize("name", CONFIGS)
def test_ball_slide_to_roll(name):
    rtol, atol_q, atol_v = TOL[name]
    m, r, v0 = 1.0, 0.05, 1.0
    body = sim2d.ball(m, r)
    sim = sim2d.Sim2D(body, CONFIGS[name])
    q_rest, _, _ = run(sim, [0.0, r + 0.0005, 0.0], [0.0, 0.0, 0.0], 10)
    qs, vs, _ = run(sim, q_rest[-1].tolist(), [v0, 0.0, 0.0], 40)

    mu_eff = MU * contact_lambda(sim, q_rest[-1], m * G)[0] / (m * G)
    xs, vs_exact, ws_exact = discrete_roll(q_rest[-1][0], v0, m, r, body.inertia, mu_eff, 40)
    assert np.abs(vs[:, 0] - vs_exact).max() < atol_v
    assert np.abs(vs[:, 2] - ws_exact).max() < atol_v / r
    assert np.abs(qs[:, 0] - xs).max() < atol_q

    # Continuous time: rolling from t* = 2 v0 / (7 mu g) at v = 5/7 v0
    # (angular momentum about the contact point is conserved, so exact).
    t_roll = 2 * v0 / (7 * MU * G)
    T = 40 * H
    x_end = v0 * t_roll - 0.5 * MU * G * t_roll**2 + 5 / 7 * v0 * (T - t_roll)
    assert vs[-1][0] == pytest.approx(5 / 7 * v0, abs=atol_v)
    assert abs(qs[-1][0] - q_rest[-1][0] - x_end) < H * v0


# --- free fall and inelastic landing ----------------------------------------


def discrete_drop(y0, m, r, margin, eps_sq, n_steps):
    """Free flight; a contact detected at the start of a step (gap < margin)
    stops the ball on the surface at the end of that step (position-level,
    inelastic). An undetected contact lets the step penetrate; the next step
    then pushes the ball back to the surface. "Surface" is r + d with the
    eps^2 gap d = eps^2 / (2 beta lambda) of that step's contact impulse."""
    beta = H**2 / m
    y, v, ys, vs = y0, 0.0, [], []
    for _ in range(n_steps):
        v_free = v - G * H
        if y - r < margin and y + H * v_free < r:
            v = (r - y) / H
            for _ in range(5):
                lam = m * (v - v_free) / H
                v = (r + eps_sq / (2 * beta * lam) - y) / H
        else:
            v = v_free
        y += H * v
        ys.append(y)
        vs.append(v)
    return np.array(ys), np.array(vs)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("y0", [0.3, 0.3123])
def test_drop(name, y0):
    _, atol_q, atol_v = TOL[name]
    r = 0.05
    sim = sim2d.Sim2D(sim2d.ball(1.0, r), CONFIGS[name])
    qs, vs, _ = run(sim, [0.0, y0, 0.0], [0.0, 0.0, 0.0], 40)
    ys, vs_exact = discrete_drop(y0, 1.0, r, sim.cfg.margin, sim.cfg.fb_eps_sq, 40)
    assert np.abs(qs[:, 1] - ys).max() < atol_q
    assert np.abs(vs[:, 1] - vs_exact).max() < atol_v

    # Continuous time: free flight until y = r; the discrete parabola lags it
    # by O(h) in time.
    t_land = np.sqrt(2 * (y0 - r) / G)
    k_land = int(np.argmax(np.isclose(ys, r, atol=1e-6) | (ys < r)))
    assert abs((k_land + 1) * H - t_land) < 2 * H
