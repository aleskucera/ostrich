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
margins per contact: min, max, final value and time sum of s_c and f_c,
and the first-touch geometry: contact point and normal (world x, y) at the
first step the contact carries load (if never, at its smallest gap).
Contacts of one body pair (sim.pairs, a list of candidate index lists;
default one pair per candidate) are merged: per step the smallest s, the f
of the most loaded candidate, and the earliest touch. Mode labels: the set
of pairs carrying load at the end (sig), or ever during the run, per
candidate (sig_ever).

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
    rec = {"gap": [], "lam_n": [], "lam_t": [], "slip": [], "nx": [], "ny": [], "px": [], "py": []}
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
        n = frame[0]
        p = q[:, None, :2] + torch.stack([torch.cos(q[:, 2:3]) * sim.body.points[:, 0] - torch.sin(q[:, 2:3]) * sim.body.points[:, 1],
                                          torch.sin(q[:, 2:3]) * sim.body.points[:, 0] + torch.cos(q[:, 2:3]) * sim.body.points[:, 1]], dim=-1)
        p = p - sim.body.thickness[None, :, None] * n
        rec["nx"].append(n[..., 0]); rec["ny"].append(n[..., 1])
        rec["px"].append(p[..., 0]); rec["py"].append(p[..., 1])
    out = {k: torch.stack(r).numpy() for k, r in rec.items()}
    out["q"] = q.numpy()
    return out


def candidates(tr, mass, dt, mu, pairs=None):
    """Candidate margins {name: (B,)} and signatures (final set of loaded
    pairs, set of candidates ever loaded), as ints."""
    T, B, C = tr["gap"].shape
    pairs = [[c] for c in range(C)] if pairs is None else pairs
    thr = 1e-3 * mass * 9.81
    s_c = tr["gap"] - tr["lam_n"] * dt**2 / mass
    f_c = (mu * tr["lam_n"] - np.abs(tr["lam_t"])) * dt / mass - tr["slip"]
    loaded_c = tr["lam_n"] > thr
    sig_ever = (loaded_c.any(0) * (1 << np.arange(C))).sum(1)
    cand, final = {}, []
    b = np.arange(B)
    for k, members in enumerate(pairs):
        s = s_c[:, :, members].min(2)
        top = tr["lam_n"][:, :, members].argmax(2)
        f = np.take_along_axis(f_c[:, :, members], top[..., None], 2)[..., 0]
        final.append(loaded_c[-1][:, members].any(1))
        for name, x in (("s", s), ("f", f)):
            cand[f"min_{name}{k}"] = x.min(0)
            cand[f"max_{name}{k}"] = x.max(0)
            cand[f"end_{name}{k}"] = x[-1]
            cand[f"sum_{name}{k}"] = x.sum(0)
        # first touch: earliest loaded step of any member, else the smallest gap
        load = loaded_c[:, :, members]
        any_load = load.any(2)
        t_touch = np.where(any_load.any(0), any_load.argmax(0), s.argmin(0))
        c_touch = np.where(any_load.any(0), load[t_touch, b].argmax(1), tr["gap"][t_touch, b][:, members].argmin(1))
        c_idx = np.array(members)[c_touch]
        for key in ("px", "py", "nx", "ny"):
            cand[f"first_{key}{k}"] = tr[key][t_touch, b, c_idx]
    sig = (np.stack(final, 1) * (1 << np.arange(len(pairs)))).sum(1)
    return cand, sig, sig_ever


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
        elif scene == "grasp":
            import grasp as gr

            sim = gr.make_sim()
            q0, v0 = gr.initial_state(len(x))
            tr = contact_trajectories(sim, q0, v0, [{"geom": g} for g in gr.schedule(x)])
            tr["L"] = gr.cost(torch.from_numpy(tr["q"]), x).numpy()
            tr["out"] = tr["q"][:, 1] < gr.HY + 0.01
            consts = (gr.MASS, sim.cfg.dt, gr.MU)
        if scene == "apex":
            import apex_drop as ad

            sim = ad.make_sim()
            q0, v0 = ad.initial_state(len(x))
            tr = contact_trajectories(sim, q0, v0, [{"f_ext": f} for f in ad.forces(x)])
            tr["L"] = ad.cost(torch.from_numpy(tr["q"]), x).numpy()
            tr["out"] = tr["q"][:, 0] > 0
            consts = (1.0, sim.cfg.dt, ad.MU, [[0], [1, 2, 3]])  # pairs: ball-ground, ball-wedge
        parts.append(tr)
    return {k: np.concatenate([p[k] for p in parts], axis=0 if parts[0][k].ndim == 1 or k == "q" else 1) for k in parts[0]}, consts


def main():
    scene, d, a, P, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    if scene == "box":
        th = mp.pool_point(d, a)
    elif scene == "grasp":
        th = grasp_point(d, a)
    else:  # apex: constant push a in every phase, with the pool's offsets
        th = np.clip(a + 0.03 * np.random.default_rng(2000 + d).uniform(-1, 1, d), 0, 1)
    eps = mp.orth_pairs(d, P, np.random.default_rng(7))
    Z = np.vstack([eps, -eps])
    tr, consts = scene_runs(scene, th + mp.SIGMA * Z)
    tip = tr["out"]
    cand, sig, sig_ever = candidates(tr, *consts)
    print(f"{scene} d={d} a={a} runs={len(Z)} outcome rate={tip.mean():.3f}")
    vals, counts = np.unique(sig, return_counts=True)
    for v, n in zip(vals, counts):
        print(f"  signature {v:04b}: {n} runs, outcome {tip[sig == v].mean():.3f}")
    vals, counts = np.unique(sig_ever, return_counts=True)
    for v, n in zip(vals, counts):
        print(f"  ever-loaded {v:04b}: {n} runs, outcome {tip[sig_ever == v].mean():.3f}")
    rows = []
    for name, m in cand.items():
        acc, o = best_sign(m, tip)
        rq, _ = mp.heldout(Z, o * m, tip, True)
        rows.append((acc, rq, name))
    for acc, rq, name in sorted(rows, reverse=True)[:8]:
        print(f"  {name:8s} sign agrees {acc:.3f}  heldout quad R2 {rq:.3f}")
    np.savez(out, th=th, Z=Z, L=tr["L"], tip=tip, sig=sig, sig_ever=sig_ever, **cand)


if __name__ == "__main__":
    main()
