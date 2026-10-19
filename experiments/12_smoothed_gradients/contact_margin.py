"""Generic margins from the contact solver, no scene knowledge.

Per run and contact c the simulator gives, every step, the gap g_c >= 0 and
the normal force lam_c >= 0 (complementary: at most one is nonzero). One
signed number per contact and step:

  s_c = g_c - lam_c dt^2 / m      (force as the displacement it makes in a step)

positive while the contact is open, negative while it carries load,
continuous when a contact is made or lost. Its friction counterpart, from
the friction reserve and the slip speed (also complementary),

  f_c = (mu lam_c - |lam_t,c|) dt / m - |v_slip,c|

positive while the contact sticks, negative while it slips. Candidate
margins per contact: min, max, final value and time sum of s_c and f_c.
Mode label: the set of contacts carrying load at the end of the run
(contact signature).

  PYTHONPATH=. python contact_margin.py SCENE D A P OUT.npz   (SCENE box: the
  box_phases pool point of margin_probe; grasp: the grasp point; reports
  each candidate against the outcome: tipped / block dropped)
"""

import sys

import numpy as np
import torch
from torch.func import vmap

import box_phases as bp
import box_tip as bt
import margin_probe as mp


def contact_trajectories(sim, q, v, steps):
    """Run all worlds through steps (a list of sim.step keyword dicts, e.g.
    f_ext or geom per step) and record per step and contact (T, B, C): gap
    at the end of the step, lam_n, lam_t and the slip speed relative to the
    geometry. Returns those and the final q (B, 3), as numpy."""
    rec = {"gap": [], "lam_n": [], "lam_t": [], "slip": []}
    lam = None
    for kw in steps:
        g = kw.get("geom")
        if g is None:
            frame = vmap(sim.contact_frame)(q)
            J_t = vmap(sim.jacobians)(q)[1]
            u_t = torch.zeros_like(frame[1][..., 0])
        else:
            frame = vmap(lambda a, b: sim.contact_frame(a, geom=b))(q, g)
            J_t = vmap(lambda a, b: sim.jacobians(a, geom=b))(q, g)[1]
            n = frame[0]
            u_t = (torch.stack([n[..., 1], -n[..., 0]], dim=-1) * vmap(lambda a, b: sim.surface_velocity(a, geom=b))(q, g)).sum(-1)
        q, v, info = sim.step(q, v, lam0=lam, **kw)
        lam = info["z"][:, 3:]
        rec["gap"].append(vmap(sim.gap)(q, frame))
        rec["lam_n"].append(info["lam_n"])
        rec["lam_t"].append(info["lam_t"])
        rec["slip"].append((torch.einsum("bcj,bj->bc", J_t, v) - u_t).abs())
    out = {k: torch.stack(r).numpy() for k, r in rec.items()}
    out["q"] = q.numpy()
    return out


def candidates(tr, mass, dt, mu):
    """Candidate margins {name: (B,)} and contact signatures (B,) as ints."""
    s = tr["gap"] - tr["lam_n"] * dt**2 / mass
    f = (mu * tr["lam_n"] - np.abs(tr["lam_t"])) * dt / mass - tr["slip"]
    loaded = tr["lam_n"][-1] > 1e-3 * mass * 9.81
    sig = (loaded * (1 << np.arange(loaded.shape[1]))).sum(1)
    cand = {}
    for c in range(s.shape[2]):
        for name, x in (("s", s[:, :, c]), ("f", f[:, :, c])):
            cand[f"min_{name}{c}"] = x.min(0)
            cand[f"max_{name}{c}"] = x.max(0)
            cand[f"end_{name}{c}"] = x[-1]
            cand[f"sum_{name}{c}"] = x.sum(0)
    return cand, sig


def best_sign(m, label):
    """Agreement of sign(m) with label, either orientation; returns (acc, orientation)."""
    a = np.mean((m > 0) == label)
    return (a, 1) if a >= 0.5 else (1 - a, -1)


def grasp_point(d, a):
    """Grasp pool point: squeeze a (units of F_MAX) and the lift speed of the
    2-phase optimum in every phase, with the pool's small offsets."""
    import grasp as gr

    base = np.tile([a, 0.14 / gr.V_MAX], d // 2)
    return np.clip(base + 0.03 * np.random.default_rng(2000 + d).uniform(-1, 1, d), 0, 1)


def scene_runs(scene, X, chunk=512):
    """Contact trajectories for the parameters X (B, d), losses and outcome."""
    parts = []
    for x in torch.split(torch.from_numpy(X), chunk):
        if scene == "box":
            sim = bt.make_sim()
            q0, v0 = bt.initial_state(len(x))
            tr = contact_trajectories(sim, q0, v0, [{"f_ext": f} for f in bp.forces(x)])
            tr["L"] = bt.terminal_cost(torch.from_numpy(tr["q"])).numpy()
            tr["out"] = np.abs(tr["q"][:, 2]) > 1.0
            consts = (bt.MASS, sim.cfg.dt, bt.MU)
        else:
            import grasp as gr

            sim = gr.make_sim()
            q0, v0 = gr.initial_state(len(x))
            tr = contact_trajectories(sim, q0, v0, [{"geom": g} for g in gr.schedule(x)])
            tr["L"] = gr.cost(torch.from_numpy(tr["q"]), x).numpy()
            tr["out"] = tr["q"][:, 1] < gr.HY + 0.01
            consts = (gr.MASS, sim.cfg.dt, gr.MU)
        parts.append(tr)
    return {k: np.concatenate([p[k] for p in parts], axis=0 if parts[0][k].ndim == 1 or k == "q" else 1) for k in parts[0]}, consts


def main():
    scene, d, a, P, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    th = mp.pool_point(d, a) if scene == "box" else grasp_point(d, a)
    eps = mp.orth_pairs(d, P, np.random.default_rng(7))
    Z = np.vstack([eps, -eps])
    tr, consts = scene_runs(scene, th + mp.SIGMA * Z)
    tip = tr["out"]
    cand, sig = candidates(tr, *consts)
    print(f"{scene} d={d} a={a} runs={len(Z)} outcome rate={tip.mean():.3f}")
    vals, counts = np.unique(sig, return_counts=True)
    for v, n in zip(vals, counts):
        print(f"  signature {v:04b}: {n} runs, tipped {tip[sig == v].mean():.3f}")
    rows = []
    for name, m in cand.items():
        acc, o = best_sign(m, tip)
        rq, _ = mp.heldout(Z, o * m, tip, True)
        rows.append((acc, rq, name))
    for acc, rq, name in sorted(rows, reverse=True)[:8]:
        print(f"  {name:8s} sign agrees {acc:.3f}  heldout quad R2 {rq:.3f}")
    np.savez(out, th=th, Z=Z, L=tr["L"], tip=tip, sig=sig, **cand)


if __name__ == "__main__":
    main()
