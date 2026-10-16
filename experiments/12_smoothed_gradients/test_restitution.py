"""Restitution: shifted gap target d* = d_k - e h v_n^- (see SimConfig).

(a) Bouncing ball against Newton's impact law: outgoing velocity, energy for
    e = 1, apex heights e^(2n) h0 converging as O(h), bounce times.
(b) Active set: restitution only changes how hard an impacting contact pushes,
    it never activates a contact the inelastic solver leaves unloaded.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_restitution.py
"""

import numpy as np
import torch
from torch.func import vmap

import sim2d

G, R = 9.81, 0.05


def drop(h0s, e, dt=0.01, duration=1.6, fb_eps_sq=1e-8):
    """Balls dropped from heights h0s (one world each). Returns y, vy of shape
    (steps, worlds). The exactness tests shrink fb_eps_sq: it leaves a gap
    eps^2 / (2 beta lambda) at the end of an impact (~1e-5 m/s here)."""
    cfg = sim2d.SimConfig(dt=dt, restitution=e, fb_eps_sq=fb_eps_sq)
    sim = sim2d.Sim2D(sim2d.ball(1.0, R), cfg)
    n = len(h0s)
    q = torch.stack([torch.zeros(n), torch.as_tensor(h0s), torch.zeros(n)], dim=1)
    v = torch.zeros(n, 3)
    ys, vys = [], []
    for _ in range(int(round(duration / dt))):
        q, v, info = sim.step(q, v)
        assert info["residual"] < sim.cfg.tol
        ys.append(q[:, 1].clone())
        vys.append(v[:, 1].clone())
    return torch.stack(ys).numpy(), torch.stack(vys).numpy()


def apexes(y):
    return [y[k] for k in range(1, len(y) - 1) if y[k] >= y[k - 1] and y[k] > y[k + 1] and y[k] - R > 1e-3]


# Drop heights sweep the impact's phase within the step.
H0S = np.linspace(0.30, 0.32, 25)


def test_outgoing_velocity():
    """The impact step leaves with exactly e |v_n^-| (v_n^- at its start)."""
    e = 0.8
    _, vys = drop(H0S, e, duration=0.4, fb_eps_sq=1e-16)
    for i in range(len(H0S)):
        k = int(np.argmax(vys[:, i] > 0))
        assert abs(vys[k, i] + e * vys[k - 1, i]) < 1e-9


def test_energy_conserved_for_e1():
    """e = 1: every apex returns to h0, whatever the impact phase."""
    ys, _ = drop(H0S, 1.0, fb_eps_sq=1e-16)
    for i, h0 in enumerate(H0S):
        peaks = apexes(ys[:, i])
        assert len(peaks) >= 3
        assert np.allclose(peaks[:3], h0, rtol=1e-6)


def test_apex_heights_converge():
    """e = 0.8: apex n -> R + e^(2n) (h0 - R), error O(h)."""
    e = 0.8
    errors = []
    for dt in (0.01, 0.005):
        ys, _ = drop(H0S, e, dt=dt)
        err = [(apexes(ys[:, i])[0] - R) / (e**2 * (h0 - R)) - 1 for i, h0 in enumerate(H0S)]
        errors.append(np.mean(err))
    print(f"\n  mean apex-1 error: h=0.01 {errors[0]:+.4f}, h=0.005 {errors[1]:+.4f}")
    assert abs(errors[0]) < 0.06
    assert 1.6 < errors[0] / errors[1] < 2.6  # first order in h


def test_bounce_times():
    """First impact at sqrt(2 (h0 - R) / g), then flights of 2 e v / g."""
    e, dt = 0.8, 0.01
    ys, vys = drop(H0S, e, dt=dt)
    for i, h0 in enumerate(H0S):
        impacts = [k for k in range(1, len(vys)) if vys[k, i] > 0 and vys[k - 1, i] <= 0]
        v_impact = np.sqrt(2 * G * (h0 - R))
        t1 = np.sqrt(2 * (h0 - R) / G)
        assert abs((impacts[0] + 0.5) * dt - t1) < dt  # impact inside step k
        for n in (1, 2):
            flight = 2 * e**n * v_impact / G
            assert abs((impacts[n] - impacts[n - 1]) * dt - flight) < 3 * dt


def random_box_states(n, seed=0):
    """Tilted, spinning, mostly falling box states near the ground."""
    gen = torch.Generator().manual_seed(seed)
    body = sim2d.box(17.0, 0.2, 0.05)
    theta = (torch.rand(n, generator=gen) - 0.5) * 0.6
    lowest = torch.stack([sim2d._rotate(t, body.points)[:, 1] for t in theta]).min(dim=1).values
    y = -lowest + torch.rand(n, generator=gen) * 0.04 - 0.002
    q = torch.stack([torch.zeros(n), y, theta], dim=1)
    v = torch.stack(
        [
            (torch.rand(n, generator=gen) - 0.5) * 2,
            -torch.rand(n, generator=gen) * 4 + 0.5,
            (torch.rand(n, generator=gen) - 0.5) * 10,
        ],
        dim=1,
    )
    return body, q, v


def test_active_set():
    """Contacts loaded with e = 0.8 are exactly those loaded with e = 0, and
    steps without an impact are bit-identical. Compared on worlds where both
    solves converged (a few violent friction impacts do not, see sim2d)."""
    body, q, v = random_box_states(400)
    out = {}
    for e in (0.0, 0.8):
        q1, v1, info = sim2d.Sim2D(body, sim2d.SimConfig(restitution=e)).step(q, v)
        out[e] = (info["lam_n"], q1, v1, info["residual_world"] < 1e-8)
    ok = out[0.0][3] & out[0.8][3]
    target = vmap(sim2d.Sim2D(body, sim2d.SimConfig(restitution=0.8)).gap_target)(q, v, torch.zeros(len(q), 3))
    impact = (target != 0).any(dim=1)
    assert ok.sum() > 0.9 * len(q)
    assert (ok & impact).sum() > 100

    loaded = 1e-4 * body.mass * G
    on0, on8 = out[0.0][0] > loaded, out[0.8][0] > loaded
    assert torch.equal(on0[ok], on8[ok])
    assert not ((target != 0) & ~on0)[ok].any()
    assert torch.equal(out[0.0][1][~impact], out[0.8][1][~impact])
    assert torch.equal(out[0.0][2][~impact], out[0.8][2][~impact])
