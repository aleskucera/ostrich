"""Phase A diagnostic: helhest drop test for the dt-asymmetry investigation.

Drops the helhest robot from z=0.5 onto a flat ground (no obstacle, no
controller) for 0.5s, with HDF5 logging on every step. Across multiple
dt's we measure impact-event metrics to disambiguate which mechanism
produces the observed "stiff at small dt, compliant at large dt" feel:

  1. FB smoothing collapse        (smoothing radius ~ h^2)
  2. Cold-start Newton overshoot  (~ pen / h^2)
  3. Impulse-time-window effect   (peak F = m*v / dt — structural)
  4. Newton non-convergence       (residual carried across many steps)

Usage:
    python experiments/2_dt_stability/diagnose_helhest_drop.py --dt 0.001 --out results/helhest_drop_dt0.001.h5
    python experiments/2_dt_stability/diagnose_helhest_drop.py --analyze results/helhest_drop_dt*.h5
"""
import argparse
import os
import pathlib
import sys

import warp

warp.config.quiet = True

import newton
import numpy as np
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.logging.hdf5_reader import HDF5Reader

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# Use sweep_axion's tuning (close to user's working config) so we observe the
# regime they care about — not our test-tuned defaults.
SIM_DURATION = 0.5
START_HEIGHT = 0.5
MAX_NEWTON_ITERS = 16
NEWTON_ATOL = 1e-5
CONTACT_COMPLIANCE = 1e-8        # /dt^2 scaling applied internally
FRICTION_COMPLIANCE = 2e-2
JOINT_COMPLIANCE = 6e-8
REGULARIZATION = 1e-6
DT_LIST = [0.05, 0.02, 0.005, 0.002, 0.001, 0.0005]

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def build_model() -> newton.Model:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from examples.helhest.common import create_helhest_model

    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=0.0)
    )

    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(0.0, 0.0, START_HEIGHT), wp.quat_identity()),
        control_mode="velocity",
        k_p=0.0,    # disable PD so it's a pure drop, no torques
        k_d=0.0,
        friction_left_right=0.0,
        friction_rear=0.0,
    )
    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def run(dt: float, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    num_steps = int(round(SIM_DURATION / dt))

    config = AxionEngineConfig(
        max_newton_iters=MAX_NEWTON_ITERS,
        newton_atol=NEWTON_ATOL,
        contact_compliance=CONTACT_COMPLIANCE,
        friction_compliance=FRICTION_COMPLIANCE,
        joint_compliance=JOINT_COMPLIANCE,
        regularization=REGULARIZATION,
        max_contacts_per_world=16,
    )
    logging_config = LoggingConfig(
        enable_hdf5_logging=True,
        hdf5_log_file=str(out_path),
        max_simulation_steps=num_steps,
    )

    engine = AxionEngine(model=model, sim_steps=num_steps, config=config, logging_config=logging_config)
    s_in, s_out = model.state(), model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)

    print(f"helhest drop dt={dt}, {num_steps} steps")
    for _ in range(num_steps):
        c = model.collide(s_in)
        engine.step(s_in, s_out, ctrl, c, dt)
        wp.copy(s_in.body_q, s_out.body_q)
        wp.copy(s_in.body_qd, s_out.body_qd)

    engine.save_logs()
    print(f"Wrote {out_path}")


def analyze(paths: list[pathlib.Path]) -> None:
    """Tabulate impact-event metrics per run."""
    rows = []
    for path in paths:
        with HDF5Reader(str(path)) as r:
            bp = r.get_dataset("body_pose")[:, 0, 0]   # chassis (body 0)
            bv = r.get_dataset("body_vel")[:, 0, 0]
            cf = r.get_dataset("_constr_force")[:, 0]
            cm = r.get_dataset("_constr_active_mask")[:, 0]
            ic = r.get_dataset("iter_count")[:, 0]
            rn = r.get_dataset("res_norm_sq")[:, 0]
            # logger's "dt" attr is captured at init (always 0); parse from filename
            dt = float(path.name.split("dt")[-1].split(".h5")[0])
            n_j = int(r.get_attribute("/dims", "N_j"))
            n_ctrl = int(r.get_attribute("/dims", "N_ctrl"))
            n_n = int(r.get_attribute("/dims", "N_n"))
            n_off = n_j + n_ctrl                # start of contact-normal slots

        z = bp[:, 2]
        vz = bv[:, 2]   # body_vel layout (linear, angular) → vz is index 2

        # Impact = first step where any contact carries non-trivial force.
        # active_mask alone is not enough since rigid_gap=0.1 m generates
        # constraint pairs in the broadphase well before actual touching.
        FN_IMPACT_THRESH = 1.0   # N
        max_fn_per_step = cf[:, n_off:n_off + n_n].max(axis=1)
        contact_steps = np.where(max_fn_per_step > FN_IMPACT_THRESH)[0]
        if len(contact_steps) == 0:
            rows.append((path.name, dt, "NO IMPACT"))
            continue

        first_contact = int(contact_steps[0])

        # Pre-impact velocity = velocity one step before first contact
        # (or the velocity at first contact, taken before contact resolution)
        v_pre = vz[max(0, first_contact - 1)] if first_contact > 0 else vz[0]
        z_at_contact = z[first_contact]

        # Peak f_n in the first ~5 steps after contact
        impact_window = slice(first_contact, min(first_contact + 5, len(z)))
        f_n_vals = cf[impact_window, n_off:n_off + n_n].max(axis=1)
        peak_fn = float(f_n_vals.max())
        peak_step = first_contact + int(np.argmax(f_n_vals))

        # Velocity 5 steps after impact (post-bounce)
        post_idx = min(first_contact + 5, len(z) - 1)
        v_post = vz[post_idx]

        # Energy ratio (KE only, since PE roughly preserved at contact event)
        # Use velocity^2 ratio for a unitless number
        energy_ratio = (v_post / v_pre) ** 2 if abs(v_pre) > 1e-6 else float("nan")

        # Settle: when does z stop changing more than 1mm over 50 steps?
        settle_idx = None
        for i in range(first_contact, len(z) - 50):
            window = z[i:i + 50]
            if window.max() - window.min() < 1e-3:
                settle_idx = i
                break
        settle_t = (settle_idx - first_contact) * dt if settle_idx is not None else float("nan")

        # Final z (last 50 steps mean)
        z_final = float(z[-50:].mean())

        # Per-step iter and res info during impact
        iter_at_impact = float(ic[impact_window].mean())

        rows.append({
            "name": path.name, "dt": dt,
            "first_contact_step": first_contact,
            "z_at_contact": z_at_contact,
            "v_pre": float(v_pre),
            "v_post": float(v_post),
            "peak_fn": peak_fn,
            "energy_ratio": float(energy_ratio),
            "settle_t": float(settle_t),
            "z_final": z_final,
            "iter_at_impact": iter_at_impact,
        })

    print(f"\n{'='*100}")
    print(f"{'name':<35} {'dt':>8} {'v_pre':>8} {'v_post':>8} {'peak_fn':>10} {'KE_ratio':>9} {'settle_t':>9} {'z_final':>9} {'iter@imp':>9}")
    print('-' * 100)
    for r in sorted([x for x in rows if isinstance(x, dict)], key=lambda x: -x["dt"]):
        print(
            f"{r['name']:<35} {r['dt']:>8.4f} {r['v_pre']:>8.3f} {r['v_post']:>8.3f} "
            f"{r['peak_fn']:>10.2f} {r['energy_ratio']:>9.3f} {r['settle_t']:>9.3f} "
            f"{r['z_final']:>9.4f} {r['iter_at_impact']:>9.2f}"
        )
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--sweep", action="store_true", help="Run all DT_LIST in one go")
    p.add_argument("--analyze", nargs="+", type=pathlib.Path, default=None)
    args = p.parse_args()

    if args.analyze:
        analyze(args.analyze)
    elif args.sweep:
        for dt in DT_LIST:
            out = RESULTS_DIR / f"helhest_drop_dt{dt:g}.h5"
            run(dt, out)
    elif args.dt is not None:
        out = args.out or RESULTS_DIR / f"helhest_drop_dt{args.dt:g}.h5"
        run(args.dt, out)
    else:
        p.error("specify --dt, --sweep, or --analyze")


if __name__ == "__main__":
    main()
