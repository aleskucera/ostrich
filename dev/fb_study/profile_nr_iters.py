"""Step 1 of PLAN.md: profile Newton-Raphson iteration counts.

Drives the Ostrich engine directly (eager mode, one world) on the Helhest
scenes from `dev/relax_bench.py`, with the production engine config from
`examples/conf/engine/ostrich.yaml`. Records, per step:

  * NR iterations actually run, and whether the step hit `nr.max_iters`,
  * the residual norm after every NR iteration (so iterations-to-tolerance
    can be read off for any atol afterwards),
  * PCR iterations per NR iteration, and the contact count.

Two protocols:
  production  - nr.max_iters=16 (as shipped): what the engine pays today.
  converge    - nr.max_iters=64: how many iterations convergence really needs.

Wall-clock is NOT measured: eager mode syncs every NR iteration.

Usage:
    python dev/fb_study/profile_nr_iters.py --json out.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import newton
import warp as wp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))

from ostrich import ComplianceConfig
from ostrich import ContactsConfig
from ostrich import LinearSolverConfig
from ostrich import LoggingConfig
from ostrich import NewtonRaphsonConfig
from ostrich import OstrichEngineConfig
from ostrich import WarmStartConfig
from ostrich.collision import ContactReductionConfig

from relax_bench import SCENES, _PCRProbe, build_model

PROTOCOLS = {"production": 16, "converge": 64}
NR_ATOL = 1e-5
BACKTRACK_MIN_ITER = 4


class _ResidualProbe:
    """Records sqrt(res_norm_sq) after every NR iteration of a step."""

    def __init__(self, engine):
        self._engine = engine
        self._orig = engine._check_convergence
        self.norms: list[float] = []
        engine._check_convergence = self._wrapped

    def _wrapped(self):
        self.norms.append(float(np.sqrt(self._engine.data.res_norm_sq.numpy()[0])))
        return self._orig()

    def drain(self) -> list[float]:
        out, self.norms = self.norms, []
        return out


def make_engine(model, steps: int, nr_max_iters: int):
    # Mirrors examples/conf/engine/ostrich.yaml.
    config = OstrichEngineConfig(
        differentiable=False,
        compliance=ComplianceConfig(joint=6e-10, contact=1e-10, friction=1e-8),
        nr=NewtonRaphsonConfig(
            max_iters=nr_max_iters, backtrack_min_iter=BACKTRACK_MIN_ITER, atol=NR_ATOL
        ),
        linear=LinearSolverConfig(max_iters=26, tol=1e-3, atol=1e-3, regularization=1e-6),
        contacts=ContactsConfig(
            max_per_world=256,
            reduction=ContactReductionConfig(policy="cluster", max_per_pair=8),
        ),
        warm_start=WarmStartConfig(enabled=True),
    )
    return config.create_engine(model=model, sim_steps=steps, logging_config=LoggingConfig())


def run(scene: str, nr_max_iters: int, settle_steps: int, steps: int, dt: float) -> dict:
    cfg = SCENES[scene]
    model = build_model(cfg["mu"], cfg["ground_mu"])
    engine = make_engine(model, settle_steps + steps, nr_max_iters)
    pcr = _PCRProbe(engine.cr_solver)
    res = _ResidualProbe(engine)

    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    targets = np.zeros(9, dtype=np.float32)
    targets[6:9] = cfg["drive"]
    target_arr = wp.array(targets, dtype=wp.float32, device=model.device)

    rec = {"phase": [], "nr_iters": [], "res_hist": [], "pcr_iters": [], "n_contacts": []}
    for k in range(settle_steps + steps):
        # Settle (drop + impact, zero drive) then drive: different regimes,
        # reported separately.
        settling = k < settle_steps
        state_in.clear_forces()
        contacts = model.collide(state_in)
        if settling:
            control.joint_target_vel.zero_()
        else:
            wp.copy(control.joint_target_vel, target_arr)

        engine.step(state_in=state_in, state_out=state_out,
                    control=control, contacts=contacts, dt=dt)

        rec["phase"].append("settle" if settling else "drive")
        rec["nr_iters"].append(int(engine.data.iter_count.numpy()[0]))
        rec["res_hist"].append(res.drain())
        rec["pcr_iters"].append(pcr.drain())
        rec["n_contacts"].append(int(engine.ostrich_contacts.contact_count.numpy()[0]))

        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(state_in, name), getattr(state_out, name))

    return rec


def iters_to_tol(hist: list[float], atol: float, min_iter: int) -> int | None:
    """NR iterations the engine's stopping rule would need at `atol`."""
    for i, r in enumerate(hist, start=1):
        if i >= min_iter and r <= atol:
            return i
    return None


def summarize(rec: dict, nr_max_iters: int, phase: str) -> dict:
    idx = [i for i, p in enumerate(rec["phase"]) if p == phase]
    nr = np.array([rec["nr_iters"][i] for i in idx])
    pcr_per_nr = np.concatenate([rec["pcr_iters"][i] for i in idx])
    final_res = np.array([rec["res_hist"][i][-1] for i in idx])
    out = {
        "steps": len(idx),
        "nr_mean": float(nr.mean()),
        "nr_p50": float(np.percentile(nr, 50)),
        "nr_p90": float(np.percentile(nr, 90)),
        "nr_max": int(nr.max()),
        "frac_at_cap": float(np.mean(nr >= nr_max_iters)),
        "frac_at_min": float(np.mean(nr <= BACKTRACK_MIN_ITER)),
        "pcr_per_nr_mean": float(pcr_per_nr.mean()),
        "pcr_per_step_mean": float(np.mean([sum(rec["pcr_iters"][i]) for i in idx])),
        "final_res_p50": float(np.median(final_res)),
        "final_res_max": float(final_res.max()),
        "contacts_mean": float(np.mean([rec["n_contacts"][i] for i in idx])),
        "nr_hist": {int(v): int(c) for v, c in zip(*np.unique(nr, return_counts=True))},
    }
    # Iterations needed at looser tolerances (unreached -> counted as None).
    for atol in (1e-3, 1e-4, 1e-5):
        need = [iters_to_tol(rec["res_hist"][i], atol, BACKTRACK_MIN_ITER) for i in idx]
        reached = [n for n in need if n is not None]
        out[f"to_{atol:g}_mean"] = float(np.mean(reached)) if reached else None
        out[f"to_{atol:g}_unreached"] = len(need) - len(reached)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="cruise,skid")
    ap.add_argument("--protocols", default="production,converge")
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--steps", type=int, default=None, help="drive steps (default: per scene)")
    ap.add_argument("--dt", type=float, default=3e-2)  # examples/conf/simulation/helhest.yaml
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results = {}
    for scene in args.scenes.split(","):
        steps = args.steps or SCENES[scene]["steps"]
        for proto in args.protocols.split(","):
            cap = PROTOCOLS[proto]
            rec = run(scene, cap, args.settle_steps, steps, args.dt)
            summ = {ph: summarize(rec, cap, ph) for ph in ("settle", "drive")}
            results[f"{scene}/{proto}"] = {"summary": summ, "raw": rec}
            for ph, s in summ.items():
                print(f"\n== {scene} / {proto} (cap {cap}) / {ph}")
                for key, val in s.items():
                    print(f"  {key:>20}: {val}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
