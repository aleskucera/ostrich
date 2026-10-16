"""sim2d against Ostrich (CPU, tight solver settings) and against analytic facts.

Run: .venv/bin/python -m pytest experiments/12_smoothed_gradients/test_sim2d.py -s
"""

import numpy as np
import pytest
import torch
import warp as wp

import ostrich_ref as ref
import sim2d

HX, HY, RADIUS = 0.2, 0.05, 0.05


def run_sim2d(body, n_steps, y0, kick=None, kick_at=0, cfg=sim2d.SimConfig()):
    sim = sim2d.Sim2D(body, cfg)
    q = torch.tensor([[0.0, y0, 0.0]])
    v = torch.zeros(1, 3)
    qs, vs, infos = [q[0].numpy().copy()], [v[0].numpy().copy()], []
    for k in range(n_steps):
        if kick is not None and k == kick_at:
            v = torch.tensor([kick])
            vs[-1] = v[0].numpy().copy()
        q, v, info = sim.step(q, v)
        qs.append(q[0].numpy().copy())
        vs.append(v[0].numpy().copy())
        infos.append(info)
    return np.array(qs), np.array(vs), infos


def compare(shape, n_steps, kick=None, kick_at=0):
    with wp.ScopedDevice("cpu"):
        model = ref.build(shape, size=(HX, 0.2, HY), radius=RADIUS)
        mass, inertia = ref.mass_and_inertia(model)
        q_o, v_o = ref.run(model, n_steps, kick=kick, kick_at=kick_at)
    if shape == "box":
        body = sim2d.box(mass, HX, HY, inertia=inertia)
    else:
        body = sim2d.ball(mass, RADIUS, inertia=inertia)
    q_m, v_m, infos = run_sim2d(body, n_steps, HY + 0.0005, kick, kick_at)

    dq = q_o - q_m
    dq[:, 2] = np.angle(np.exp(1j * dq[:, 2]))  # wrap: the ball turns past pi
    dq, dv = np.abs(dq).max(axis=0), np.abs(v_o - v_m).max(axis=0)
    print(f"\n  {shape}: max|dq| = {dq}, max|dv| = {dv}, "
          f"sim2d iters <= {max(i['iters'] for i in infos)}")
    assert max(i["residual"] for i in infos) < sim2d.SimConfig.tol
    return dq, dv


@pytest.mark.parametrize(
    "shape, n_steps, kick, kick_at, tol_q, tol_v",
    [
        ("box", 100, None, 0, 1e-5, 2e-4),  # rest
        ("box", 80, [2.0, 0.0, 0.0], 30, 5e-4, 2e-3),  # slide to a stop
        # theta: float32 quaternion accumulates ~1e-6 relative over ~13 rad of rolling
        ("sphere", 80, [1.0, 0.0, 0.0], 30, 1e-4, 2e-3),  # slide into rolling
    ],
    ids=["rest_box", "slide_box", "roll_ball"],
)
def test_matches_ostrich(shape, n_steps, kick, kick_at, tol_q, tol_v):
    dq, dv = compare(shape, n_steps, kick, kick_at)
    assert dq.max() < tol_q
    assert dv.max() < tol_v


def test_sliding_friction_on_the_cone():
    """While sliding, the friction force sits on the Coulomb cone."""
    body = sim2d.box(17.0, HX, HY)
    _, vs, infos = run_sim2d(body, 60, HY + 0.0005, kick=[2.0, 0.0, 0.0], kick_at=10)
    for v, info in zip(vs[11:30], infos[10:29]):
        assert v[0] > 0.5
        lam_n, lam_t = info["lam_n"][0], info["lam_t"][0]
        assert torch.allclose(lam_t.abs(), 0.5 * lam_n, rtol=1e-3, atol=1e-6)
    assert abs(vs[-1][0]) < 1e-5  # stopped: 2 m/s / (mu g) = 0.41 s


def test_ball_rolls_at_five_sevenths():
    """A sliding ball ends rolling at v = 5/7 v0, omega = -v / r."""
    _, vs, _ = run_sim2d(sim2d.ball(1.0, RADIUS), 100, RADIUS + 0.0005, kick=[1.0, 0.0, 0.0], kick_at=10)
    assert vs[-1][0] == pytest.approx(5 / 7, rel=1e-4)
    assert vs[-1][2] == pytest.approx(-vs[-1][0] / RADIUS, rel=1e-4)


def test_batched_worlds_are_independent():
    """A batch of worlds gives the same result as each world alone."""
    sim = sim2d.Sim2D(sim2d.box(17.0, HX, HY))
    q = torch.tensor([[0.0, HY + 0.0005, 0.0], [0.0, HY + 0.003, 0.02], [1.0, 0.3, 0.5]])
    v = torch.tensor([[2.0, 0.0, 0.0], [0.3, -0.5, 1.0], [0.0, 0.0, 0.0]])
    q_b, v_b, _ = sim.step(q, v)
    for i in range(3):
        q_i, v_i, _ = sim.step(q[i : i + 1], v[i : i + 1])
        assert torch.allclose(q_b[i], q_i[0], atol=1e-12)
        assert torch.allclose(v_b[i], v_i[0], atol=1e-12)


@pytest.mark.parametrize("shape, e, mu", [("box", 0.0, 0.5), ("box", 0.8, 0.5), ("ball", 0.8, 0.5), ("ball", 0.8, 0.0)])
def test_residual_jacobian_matches_jacrev(shape, e, mu):
    """The hand-written dR/dz equals autodiff of the residual, on random
    iterates (inside and outside the friction cone) near the ground."""
    from torch.func import jacrev, vmap

    gen = torch.Generator().manual_seed(0)
    body = sim2d.box(17.0, HX, HY) if shape == "box" else sim2d.ball(1.0, 0.05)
    sim = sim2d.Sim2D(body, sim2d.SimConfig(restitution=e, mu=mu))
    B, C = 200, sim.C
    rand = lambda *size: torch.rand(*size, generator=gen, dtype=torch.float64)
    theta = (rand(B) - 0.5) * 0.6
    lowest = torch.stack([sim2d._rotate(t, body.points)[:, 1] - body.thickness for t in theta]).min(dim=1).values
    q = torch.stack([torch.zeros(B), -lowest + rand(B) * 0.03 - 0.002, theta], dim=1)
    v = torch.stack([(rand(B) - 0.5) * 4, -rand(B) * 3 + 0.5, (rand(B) - 0.5) * 8], dim=1)
    f = torch.zeros(B, 3)
    d0, d_free, _ = vmap(sim.free_gap)(q, v, f)
    active = torch.minimum(d0, d_free) < sim.cfg.margin
    z = torch.cat([v + 0.1 * (rand(B, 3) - 0.5), rand(B, C) * 200, (rand(B, C) - 0.5) * 300], dim=1)
    on = active & (sim.cfg.mu * z[:, 3 : 3 + C] > sim.cfg.load_gate)
    args = (z, q, v, f, active, on)
    K_hand = vmap(sim.residual_jacobian)(*args)
    K_auto = vmap(jacrev(sim.residual))(*args)
    assert active.sum() > B / 2
    assert ((K_hand - K_auto).abs().amax(dim=(1, 2)) / K_auto.abs().amax(dim=(1, 2))).max() < 1e-12
