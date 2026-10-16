"""Step 2 of PLAN.md: oracle ceiling for the FB preconditioning factor r.

Sweeps global multipliers on the contact and friction r
(`compliance.contact_fb_r_scale`, `compliance.friction_fb_r_scale`) and
measures how fast Newton converges with each.

Two things keep the comparison fair:

* **Paired steps.** A reference trajectory is driven with k = (1, 1). At every
  step, the engine state is snapshotted and each (k_n, k_f) solves *the same*
  step from that snapshot. Without this, trajectories diverge and per-step
  "best k" would compare different problems.
* **One ruler.** The engine's own residual depends on r (e.g. a separating
  contact with lambda > 0 contributes ~r*lambda). After every NR iteration the
  residual is therefore re-evaluated with k = (1, 1) at the current iterate,
  and all metrics use that reference residual.

The engine is not deterministic run to run (float32 GPU reductions), so the
same step solved twice gives slightly different histories. Each step therefore
runs (1, 1) twice from the same snapshot (the grid entry and the reference
advance, stored as "1,1#2"). Picking the better of those two per step is a
pure-noise "oracle": a per-step oracle over the grid has to beat it before
it means anything.

Usage:
    python dev/fb_study/oracle_sweep.py --json sweep.json
"""
import argparse
import dataclasses
import json
import pathlib

import numpy as np
import newton
import warp as wp

from ostrich.core.residual import compute_residual

from profile_nr_iters import BACKTRACK_MIN_ITER, iters_to_tol, make_engine
from relax_bench import SCENES, build_model

K_N = [0.01, 0.1, 0.3, 3.0, 10.0, 100.0]
# 0.03 = dt: the paper's friction choice r = h*w.
K_F = [0.01, 0.03, 0.1, 0.3, 3.0, 10.0, 100.0]
GRID = [(1.0, 1.0)] + [(k, 1.0) for k in K_N] + [(1.0, k) for k in K_F]
TOLS = (1e-3, 1e-4)
BUDGETS = (6, 8, 16)
NOISE = "1,1#2"


def gather_arrays(obj, seen=None, depth=0):
    """All distinct wp.arrays reachable from `obj` through attributes."""
    seen = {} if seen is None else seen
    if depth > 6 or id(obj) in seen:
        return seen
    if isinstance(obj, wp.array):
        if obj.size > 0 and obj.is_contiguous:
            seen[id(obj)] = obj
        return seen
    if isinstance(obj, (list, tuple)):
        for v in obj:
            gather_arrays(v, seen, depth + 1)
        return seen
    if isinstance(obj, dict):
        for v in obj.values():
            gather_arrays(v, seen, depth + 1)
        return seen
    mod = type(obj).__module__ or ""
    if hasattr(obj, "__dict__") and mod.startswith(("ostrich", "newton", "warp.optim")):
        seen[id(obj)] = None  # mark visited
        for v in vars(obj).values():
            gather_arrays(v, seen, depth + 1)
    return seen


class Snapshot:
    def __init__(self, *roots):
        found = {}
        for r in roots:
            gather_arrays(r, found)
        self.arrays = [a for a in found.values() if a is not None]
        self.saved = [wp.clone(a) for a in self.arrays]

    def take(self):
        for src, dst in zip(self.arrays, self.saved):
            wp.copy(dst, src)

    def restore(self):
        for src, dst in zip(self.saved, self.arrays):
            wp.copy(dst, src)


def with_scales(config, k_n, k_f):
    comp = dataclasses.replace(
        config.compliance, contact_fb_r_scale=k_n, friction_fb_r_scale=k_f
    )
    return dataclasses.replace(config, compliance=comp)


class RefResidualProbe:
    """After every NR iteration, records the residual norm under k = (1, 1)."""

    def __init__(self, engine):
        self.engine = engine
        self.ref_config = with_scales(engine.config, 1.0, 1.0)
        self._orig = engine._check_convergence
        self.norms: list[float] = []
        engine._check_convergence = self._wrapped

    def _norm(self, config):
        e = self.engine
        compute_residual(e.ostrich_model, e.ostrich_contacts, e.data, config, e.dims)
        e.data.tiled_sq_norm.compute(e.data.res.full, e.data.res_norm_sq)
        return float(np.sqrt(e.data.res_norm_sq.numpy()[0]))

    def _wrapped(self):
        self.norms.append(self._norm(self.ref_config))
        self._norm(self.engine.config)  # restore the engine's own residual
        return self._orig()

    def drain(self):
        out, self.norms = self.norms, []
        return out


def set_config(engine, config):
    # Same trick as dev/relax_bench.py: kernels read config at launch time.
    object.__setattr__(engine, "config", config)


def run(scene, cap, settle_steps, steps, dt):
    cfg = SCENES[scene]
    model = build_model(cfg["mu"], cfg["ground_mu"])
    engine = make_engine(model, settle_steps + steps, cap)
    base_config = engine.config
    probe = RefResidualProbe(engine)

    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    targets = np.zeros(9, dtype=np.float32)
    targets[6:9] = cfg["drive"]
    target_arr = wp.array(targets, dtype=wp.float32, device=model.device)

    snap = None
    hist = {f"{kn:g},{kf:g}": [] for kn, kf in GRID}
    phase = []
    for t in range(settle_steps + steps):
        settling = t < settle_steps
        state_in.clear_forces()
        contacts = model.collide(state_in)
        if settling:
            control.joint_target_vel.zero_()
        else:
            wp.copy(control.joint_target_vel, target_arr)

        if snap is None:
            snap = Snapshot(engine, state_in, state_out, contacts)
        snap.take()
        for kn, kf in GRID:
            snap.restore()
            set_config(engine, with_scales(base_config, kn, kf))
            engine.step(state_in=state_in, state_out=state_out,
                        control=control, contacts=contacts, dt=dt)
            hist[f"{kn:g},{kf:g}"].append(probe.drain())

        # Advance the reference trajectory with k = (1, 1).
        snap.restore()
        set_config(engine, base_config)
        engine.step(state_in=state_in, state_out=state_out,
                    control=control, contacts=contacts, dt=dt)
        hist.setdefault(NOISE, []).append(probe.drain())
        phase.append("settle" if settling else "drive")

        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(state_in, name), getattr(state_out, name))
    return {"phase": phase, "hist": hist}


def step_metrics(h, cap):
    """Per-step scalar metrics from one reference-residual history."""
    out = {}
    for tol in TOLS:
        n = iters_to_tol(h, tol, BACKTRACK_MIN_ITER)
        out[f"to_{tol:g}"] = cap + 1 if n is None else n  # unreached -> penalty
    for b in BUDGETS:
        out[f"best_at_{b}"] = min(h[:b])  # backtracking keeps the best iterate
    return out


def summarize(rec, cap, phase):
    idx = [i for i, p in enumerate(rec["phase"]) if p == phase]
    keys = [k for k in rec["hist"] if k != NOISE]
    m = {k: [step_metrics(rec["hist"][k][i], cap) for i in idx] for k in rec["hist"]}
    out = {"steps": len(idx), "per_config": {}, "oracle": {}}
    for metric in m[keys[0]][0]:
        table = np.array([[m[k][s][metric] for k in keys] for s in range(len(idx))])
        noise = np.array([m[NOISE][s][metric] for s in range(len(idx))])
        log = metric.startswith("best_at")
        agg = (lambda x: float(np.exp(np.mean(np.log(x))))) if log else (lambda x: float(np.mean(x)))
        per = {k: agg(table[:, j]) for j, k in enumerate(keys)}
        best_fixed = min(per, key=per.get)
        oracle = agg(table.min(axis=1))
        argbest = [keys[j] for j in table.argmin(axis=1)]
        out["oracle"][metric] = {
            "agg": "geomean" if log else "mean",
            "k=1": per["1,1"],
            "k=1_rerun": agg(noise),
            "noise_oracle": agg(np.minimum(table[:, keys.index("1,1")], noise)),
            "best_fixed": [best_fixed, per[best_fixed]],
            "per_step_oracle": oracle,
            "frac_steps_best_is_k1": float(np.mean(table.min(axis=1) >= table[:, keys.index("1,1")])),
            "argbest_counts": {k: argbest.count(k) for k in keys if argbest.count(k)},
        }
        for k in keys:
            out["per_config"].setdefault(k, {})[metric] = per[k]
        if not log:
            for k in keys:
                out["per_config"][k][f"{metric}_unreached"] = int(
                    np.sum(table[:, keys.index(k)] > cap)
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="cruise,skid")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=3e-2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    for scene in args.scenes.split(","):
        steps = args.steps or SCENES[scene]["steps"]
        rec = run(scene, args.cap, args.settle_steps, steps, args.dt)
        summ = {ph: summarize(rec, args.cap, ph) for ph in ("settle", "drive")}
        results[scene] = {"summary": summ, "raw": rec}
        for ph, s in summ.items():
            print(f"\n=== {scene} / {ph} ({s['steps']} steps, cap {args.cap})")
            print(f"  {'k_n,k_f':>10} " + " ".join(f"{m:>12}" for m in s["oracle"]))
            for k, row in s["per_config"].items():
                print(f"  {k:>10} " + " ".join(f"{row[m]:>12.4g}" for m in s["oracle"]))
            for metric, o in s["oracle"].items():
                print(f"  {metric}: k=1 {o['k=1']:.4g} (rerun {o['k=1_rerun']:.4g}, "
                      f"noise oracle {o['noise_oracle']:.4g}) | best fixed {o['best_fixed'][0]} "
                      f"{o['best_fixed'][1]:.4g} | per-step oracle {o['per_step_oracle']:.4g} "
                      f"| k=1 already best on {o['frac_steps_best_is_k1']:.0%}")
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
