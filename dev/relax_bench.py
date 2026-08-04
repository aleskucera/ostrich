"""Forward-only benchmark for the structural relaxation rungs.

Phase 0 of RELAXATION_PLAN.md. Drives the Ostrich engine directly — no
simulator, no hydra, no rendering, no backward pass — so that a rung's cost
is measured against identical initial states and identical control sequences.

Primary metric is NR iterations per step and PCR iterations per NR step, per
RELAXATION_BRANCH.md §8. Those are engine-internal and noise-free; wall-clock
here runs in eager mode (no CUDA graph) and is NOT a valid cost baseline.

Usage:
    python dev/relax_bench.py                     # cruise, cone vs bilateral
    python dev/relax_bench.py --scene skid
    python dev/relax_bench.py --steps 200 --json out.json
"""
import argparse
import json
import sys
import pathlib

import numpy as np
import newton
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ostrich import ComplianceConfig
from ostrich import ContactsConfig
from ostrich import LinearSolverConfig
from ostrich import LoggingConfig
from ostrich import NewtonRaphsonConfig
from ostrich import OstrichEngineConfig
from ostrich import RelaxationConfig
from ostrich import WarmStartConfig
from ostrich.collision import ContactReductionConfig
from ostrich.core.model_builder import OstrichModelBuilder

from examples.helhest.common import create_helhest_model


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

SCENES = {
    # S1 — cruise. Flat hard ground, gentle straight drive, friction nowhere
    # near saturation. This is the equivalence scene: every rung must agree.
    "cruise": dict(drive=(2.0, 2.0, 2.0), mu=0.9, ground_mu=0.9, steps=120),
    # S2 — skid turn. Hard differential drive that saturates the cone, so the
    # bilateral rung must diverge and must be the optimistic one.
    "skid": dict(drive=(8.0, -8.0, 0.0), mu=0.35, ground_mu=0.35, steps=120),
}


def build_model(mu: float, ground_mu: float) -> newton.Model:
    builder = OstrichModelBuilder()
    builder.rigid_gap = 1.0
    create_helhest_model(
        builder,
        xform=wp.transform((0.0, 0.0, 0.6), wp.quat_identity()),
        control_mode="velocity",
        k_p=150.0,
        k_d=0.0,
        is_visible=False,
        friction_left_right=mu,
        friction_rear=mu * 0.5,
    )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=ground_mu))
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


# ---------------------------------------------------------------------------
# PCR probe
# ---------------------------------------------------------------------------


class _PCRProbe:
    """Records the PCR iteration count of every linear solve.

    `cr_solver.iter_count` is zeroed at the top of each `solve()`, so reading
    it once per engine.step would only ever report the *last* NR iteration.
    Wrapping `solve` is the only way to get the per-NR-step series §8 asks for.

    This syncs the device once per NR iteration. The engine's eager path
    already syncs per iteration for its convergence check, so the added cost is
    small — but it is another reason not to read wall-clock off this harness.
    """

    def __init__(self, solver):
        self._solver = solver
        self._orig = solver.solve
        self.counts: list[int] = []
        solver.solve = self._wrapped

    def _wrapped(self, *args, **kwargs):
        out = self._orig(*args, **kwargs)
        self.counts.append(int(self._solver.iter_count.numpy()[0]))
        return out

    def drain(self) -> list[int]:
        out, self.counts = self.counts, []
        return out


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------


def saturation_ratio(lam_n: np.ndarray, lam_f: np.ndarray, mu: float,
                     load_floor: float = 1e-3) -> np.ndarray:
    """||lambda_f|| / (mu * lambda_n), guarded against a vanishing normal load.

    Unguarded this diverges exactly where it means least: a near-airborne
    contact carries lambda_n -> 0 and reports enormous saturation while
    transmitting no force. Contacts under the floor return NaN ("no verdict")
    rather than a large number, so they cannot be mistaken for violations.
    """
    f_norm = np.linalg.norm(lam_f.reshape(-1, 2), axis=1)
    budget = mu * lam_n
    out = np.full(f_norm.shape, np.nan)
    live = budget > load_floor
    out[live] = f_norm[live] / budget[live]
    return out


def gyro_ratio(engine) -> float:
    """||w x I w|| * dt / ||M * du||, the inertia axis's free certificate."""
    d = engine.data
    vel = d.body_vel.numpy()[0]
    vel_prev = d.body_vel_prev.numpy()[0]
    pose = d.body_pose.numpy()[0]
    mass = engine.ostrich_model.body_mass.numpy()[0]
    inertia = engine.ostrich_model.body_inertia.numpy()[0]

    gyro_mag, mom_mag = 0.0, 0.0
    for b in range(len(mass)):
        q = pose[b, 3:7]
        R = _quat_to_mat(q)
        I_w = R @ inertia[b] @ R.T
        w = vel[b, 3:6]
        gyro_mag += np.linalg.norm(np.cross(w, I_w @ w)) * float(d.dt)
        dv = vel[b] - vel_prev[b]
        mom_mag += np.linalg.norm(np.concatenate([mass[b] * dv[0:3], I_w @ dv[3:6]]))
    return gyro_mag / max(mom_mag, 1e-12)


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(scene: str, relaxation: RelaxationConfig, steps: int, dt: float,
        nr_max_iters: int, backtrack_min_iter: int, nr_atol: float,
        friction_compliance: float = 1e-8, settle_steps: int = 40,
        max_per_pair: int = 8, mu_override: float | None = None) -> dict:
    cfg = SCENES[scene]
    mu = cfg["mu"] if mu_override is None else mu_override
    model = build_model(mu, cfg["ground_mu"] if mu_override is None else mu_override)

    # Mirrors examples/conf/engine/ostrich.yaml — the configuration the
    # project actually runs. The dataclass defaults differ substantially
    # (linear.max_iters, all three compliances, contact reduction), so
    # benchmarking against them would not describe this engine's baseline.
    engine_config = OstrichEngineConfig(
        differentiable=False,
        relaxation=relaxation,
        compliance=ComplianceConfig(
            joint=6e-10, contact=1e-10, friction=friction_compliance
        ),
        nr=NewtonRaphsonConfig(
            max_iters=nr_max_iters,
            backtrack_min_iter=backtrack_min_iter,
            atol=nr_atol,
        ),
        linear=LinearSolverConfig(
            max_iters=26, tol=1e-3, atol=1e-3, regularization=1e-6
        ),
        contacts=ContactsConfig(
            max_per_world=256,
            reduction=ContactReductionConfig(policy="cluster", max_per_pair=max_per_pair),
        ),
        warm_start=WarmStartConfig(enabled=True),
    )
    engine = engine_config.create_engine(
        model=model, sim_steps=steps, logging_config=LoggingConfig()
    )
    probe = _PCRProbe(engine.cr_solver)

    state_in, state_out = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    targets = np.zeros(9, dtype=np.float32)
    targets[6:9] = cfg["drive"]
    target_arr = wp.array(targets, dtype=wp.float32, device=model.device)

    rec = {
        "nr_iters": [], "pcr_iters": [], "res_norm": [], "n_contacts": [],
        "pose": [], "saturation_max": [], "lam_n_min": [], "gyro_cert": [],
        "wheel_spin": [],
    }

    # Settle phase: drop-and-impact from the spawn height is a different
    # regime from cruising, and it would otherwise dominate every statistic.
    # Run it under the FULL formulation for every rung, so all rungs start
    # from a bit-identical settled state (§8: identical initial states).
    settle_relax = engine.config.relaxation
    for _ in range(settle_steps):
        state_in.clear_forces()
        contacts = model.collide(state_in)
        control.joint_target_vel.zero_()
        object.__setattr__(engine.config, "relaxation", RelaxationConfig())
        engine.step(state_in=state_in, state_out=state_out,
                    control=control, contacts=contacts, dt=dt)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)
    object.__setattr__(engine.config, "relaxation", settle_relax)
    rec["settled_z"] = float(state_in.body_q.numpy()[0][2])

    for _ in range(steps):
        state_in.clear_forces()
        contacts = model.collide(state_in)
        wp.copy(control.joint_target_vel, target_arr)

        probe.drain()
        engine.step(state_in=state_in, state_out=state_out,
                    control=control, contacts=contacts, dt=dt)

        rec["nr_iters"].append(int(engine.data.iter_count.numpy()[0]))
        # engine-side count: post-reduction, which is what the solver sees.
        rec["n_contacts"].append(int(engine.ostrich_contacts.contact_count.numpy()[0]))
        # Wheel spin. Bodies are [chassis, left, right, rear]; a bilateral
        # no-slip rung that over-constrains the contact patch drives these
        # to zero (the wheel is welded, not rolling).
        w_bodies = engine.data.body_vel.numpy()[0][1:, 3:6]
        rec["wheel_spin"].append(float(np.abs(w_bodies).max()))
        rec["pcr_iters"].append(probe.drain())
        rec["res_norm"].append(float(np.sqrt(engine.data.res_norm_sq.numpy()[0])))
        # newton State arrays are flat over all bodies (not per-world);
        # engine.data arrays are (world, body). Don't index a world here.
        rec["pose"].append(state_out.body_q.numpy().tolist())
        rec["gyro_cert"].append(gyro_ratio(engine))

        n_contacts = int(contacts.rigid_contact_count.numpy()[0])
        lam_n = engine.data.constr_force.n.numpy()[0][:n_contacts]
        lam_f = engine.data.constr_force.f.numpy()[0][: 2 * n_contacts]
        if n_contacts:
            sat = saturation_ratio(lam_n, lam_f, mu)
            rec["saturation_max"].append(
                float(np.nanmax(sat)) if not np.all(np.isnan(sat)) else float("nan")
            )
            rec["lam_n_min"].append(float(lam_n.min()))
        else:
            rec["saturation_max"].append(float("nan"))
            rec["lam_n_min"].append(float("nan"))

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)

    return rec


def summarize(name: str, rec: dict) -> dict:
    nr = np.array(rec["nr_iters"])
    pcr_flat = np.array([c for step in rec["pcr_iters"] for c in step])
    chassis = np.array([p[0][:3] for p in rec["pose"]])
    sat = np.array(rec["saturation_max"])
    nc = np.array(rec["n_contacts"])
    return {
        "rung": name,
        "n_contacts_mean": float(nc.mean()),
        "wheel_spin_mean": float(np.array(rec["wheel_spin"]).mean()),
        "settled_z": rec["settled_z"],
        "nr_per_step_mean": float(nr.mean()),
        "nr_per_step_max": int(nr.max()),
        "nr_total": int(nr.sum()),
        "pcr_per_nr_mean": float(pcr_flat.mean()) if pcr_flat.size else 0.0,
        "pcr_total": int(pcr_flat.sum()),
        "final_res_norm": float(rec["res_norm"][-1]),
        "final_xyz": chassis[-1].tolist(),
        "saturation_max": float(np.nanmax(sat)) if not np.all(np.isnan(sat)) else float("nan"),
        "lam_n_min": float(np.nanmin(np.array(rec["lam_n_min"]))),
        "gyro_cert_max": float(np.nanmax(np.array(rec["gyro_cert"]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="cruise", choices=sorted(SCENES))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dt", type=float, default=3e-2)
    ap.add_argument("--protocol", default="converge", choices=("converge", "budget"),
                    help="converge: iterations-to-tolerance (loose atol, ample budget). "
                         "budget: residual-at-fixed-budget (tight atol, capped iters).")
    ap.add_argument("--nr-max-iters", type=int, default=None)
    ap.add_argument("--nr-atol", type=float, default=None)
    ap.add_argument("--backtrack-min-iter", type=int, default=4)
    ap.add_argument("--friction-compliance", type=float, default=1e-8)
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--max-per-pair", type=int, default=8)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    steps = args.steps or SCENES[args.scene]["steps"]

    # The two protocols measure different things and must not be conflated
    # (RELAXATION_PLAN.md, review point 2). "converge" lets each rung run to a
    # loose tolerance and reports iterations-to-convergence; "budget" fixes the
    # iteration count across rungs and reports residual-at-budget, which is the
    # protocol the attribution claim needs.
    if args.protocol == "converge":
        nr_max_iters = args.nr_max_iters or 64
        nr_atol = args.nr_atol if args.nr_atol is not None else 1e-3
    else:
        nr_max_iters = args.nr_max_iters or 16
        nr_atol = args.nr_atol if args.nr_atol is not None else 1e-5

    # The 2x2 attribution control (RELAXATION_PLAN.md Phase 2): "mu50" keeps
    # the cone and its active set but makes it never bind. If mu50 is as fast
    # as the bilateral rungs, the cost is the cone BINDING (stick-slip mode
    # switching). If mu50 stays fast while bilateral is slow, the cone was
    # never the driver and removing it bought nothing.
    rungs = {
        "full": (RelaxationConfig(), None),
        "mu50": (RelaxationConfig(), 50.0),
        "bilateral": (RelaxationConfig(friction="bilateral"), None),
        "bilat_patch": (RelaxationConfig(friction="bilateral_patch"), None),
        "no_gyro": (RelaxationConfig(gyro=False), None),
    }

    results, raw = [], {}
    for name, (relax, mu_ovr) in rungs.items():
        rec = run(args.scene, relax, steps, args.dt,
                  nr_max_iters, args.backtrack_min_iter, nr_atol,
                  args.friction_compliance, args.settle_steps, args.max_per_pair,
                  mu_ovr)
        raw[name] = rec
        results.append(summarize(name, rec))

    hdr = f"{'rung':<12}{'NR/step':>9}{'NRmax':>7}{'PCR/NR':>9}{'PCRtot':>9}{'res':>11}{'sat':>11}{'min λn':>10}{'#con':>7}{'|ω|wheel':>10}"
    print(f"\nscene={args.scene} steps={steps} dt={args.dt} "
          f"compliance.friction={args.friction_compliance:g} settle={args.settle_steps} "
          f"max_per_pair={args.max_per_pair} "
          f"protocol={args.protocol} nr.max_iters={nr_max_iters} nr.atol={nr_atol:g} "
          f"backtrack_min_iter={args.backtrack_min_iter}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['rung']:<12}{r['nr_per_step_mean']:>9.2f}{r['nr_per_step_max']:>7d}"
              f"{r['pcr_per_nr_mean']:>9.2f}{r['pcr_total']:>9d}"
              f"{r['final_res_norm']:>11.2e}{r['saturation_max']:>11.3g}"
              f"{r['lam_n_min']:>10.3f}{r['n_contacts_mean']:>7.1f}"
              f"{r['wheel_spin_mean']:>10.3f}")

    base = np.array(results[0]["final_xyz"])
    print("\nfinal chassis xyz, and displacement from the full rung:")
    for r in results:
        d = np.linalg.norm(np.array(r["final_xyz"]) - base)
        print(f"  {r['rung']:<12}{np.array2string(np.array(r['final_xyz']), precision=5)}"
              f"   |Δ| = {d:.3e}")
    print(f"\ngyro certificate (max over steps): {results[0]['gyro_cert_max']:.3e}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"scene": args.scene, "steps": steps, "summary": results,
                        "raw": raw}, indent=2)
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
