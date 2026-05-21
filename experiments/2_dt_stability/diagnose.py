"""Phase 1 diagnostic for the dt-dependence problem.

Two test scenes for two failure modes:

1. Normal contact: a 1 kg sphere resting on a frictionless ground plane.
   Symptom at small dt is overshoot of lambda_n leading to the body
   hovering and oscillating above the floor instead of settling on it.

2. Friction sticking: a 1 kg sphere resting on a frictional ground
   plane (mu>0) with a horizontal force applied each step that is
   below the static friction limit. The body should stay at rest;
   the symptom at small dt is slipping or x-jitter from overshoot
   in lambda_f at the sticking boundary.

Usage:
    # contact-normal test
    python experiments/2_dt_stability/diagnose.py --dt 0.05  --out results/diag_dt0.05.h5
    python experiments/2_dt_stability/diagnose.py --dt 0.001 --out results/diag_dt0.001.h5

    # friction test (mu>0, applied tangential force fx<mu*M*g)
    python experiments/2_dt_stability/diagnose.py --dt 0.05  --mu 0.5 --fx 2.0
    python experiments/2_dt_stability/diagnose.py --dt 0.001 --mu 0.5 --fx 2.0

    # plot
    python experiments/2_dt_stability/diagnose.py --plot results/*.h5
"""
import argparse
import pathlib

import warp

warp.config.quiet = True
import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.logging.hdf5_reader import HDF5Reader

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Defaults match AxionEngineConfig defaults — the no-flag run exercises
# the "shipped" behavior. Override --compliance to A/B against alternatives.
SIM_DURATION = 0.3
MAX_NEWTON_ITERS = 16
NEWTON_ATOL = 1e-3
CONTACT_COMPLIANCE = 1e-4   # interpreted as dt-adaptive: effective e = compliance/dt^2
FRICTION_COMPLIANCE = 1e-6


@wp.kernel
def apply_horizontal_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1),
    fx: float,
):
    body_idx = wp.tid()
    if body_idx == 0:
        # axion spatial vector layout: (f_x, f_y, f_z, tau_x, tau_y, tau_z)
        body_f[body_idx] = body_f[body_idx] + wp.spatial_vector(
            fx, 0.0, 0.0, 0.0, 0.0, 0.0
        )


def build_model(mu: float = 0.0) -> newton.Model:
    """1 kg sphere of radius 0.1 m on a ground plane with friction mu."""
    builder = AxionModelBuilder()

    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu, restitution=0.0)
    )

    radius = 0.1
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()),
        label="ball",
    )
    target_mass = 1.0
    volume = (4.0 / 3.0) * 3.141592653589793 * radius**3
    density = target_mass / volume

    builder.add_shape_sphere(
        body=body,
        radius=radius,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=density, mu=mu, restitution=0.0
        ),
    )

    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def run(
    dt: float,
    out_path: pathlib.Path,
    contact_compliance: float = CONTACT_COMPLIANCE,
    friction_compliance: float = FRICTION_COMPLIANCE,
    mu: float = 0.0,
    fx: float = 0.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(mu=mu)
    num_steps = int(round(SIM_DURATION / dt))

    config = AxionEngineConfig(
        max_newton_iters=MAX_NEWTON_ITERS,
        newton_atol=NEWTON_ATOL,
        contact_compliance=contact_compliance,
        friction_compliance=friction_compliance,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig(
        enable_hdf5_logging=True,
        hdf5_log_file=str(out_path),
        max_simulation_steps=num_steps,
    )

    engine = AxionEngine(
        model=model, sim_steps=num_steps, config=config, logging_config=logging_config
    )

    state_in = model.state()
    state_out = model.state()
    control = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    print(
        f"Running diagnostic at dt={dt}, {num_steps} steps "
        f"(max_newton_iters={MAX_NEWTON_ITERS}, newton_atol={NEWTON_ATOL}, "
        f"contact_compliance={contact_compliance}, friction_compliance={friction_compliance}, "
        f"mu={mu}, fx={fx})"
    )

    for step in range(num_steps):
        if fx != 0.0:
            state_in.body_f.zero_()
            wp.launch(
                kernel=apply_horizontal_force_kernel,
                dim=1,
                inputs=[state_in.body_f, fx],
                device=engine.device,
            )

        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    engine.save_logs()
    print(f"Wrote {out_path}")


def plot(paths: list[pathlib.Path], friction_mode: bool = False) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if friction_mode:
        # 3 panels: x position, x velocity, friction force
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for path in paths:
            with HDF5Reader(str(path)) as r:
                bp = r.get_dataset("body_pose")[:, 0, 0]
                bv = r.get_dataset("body_vel")[:, 0, 0]
                cf = r.get_dataset("_constr_force")[:, 0]
                dt = r.get_attribute("/", "dt") if "dt" in r.list_attributes("/") else None
            x = bp[:, 0]
            vx = bv[:, 0]   # body_vel layout is (linear_xyz, angular_xyz)
            # friction force tangential basis is two scalars per contact —
            # plot the magnitude of the first contact's tangential pair
            f_t1 = cf[:, 1] if cf.shape[1] > 1 else np.zeros_like(x)
            f_t2 = cf[:, 2] if cf.shape[1] > 2 else np.zeros_like(x)
            f_t = np.sqrt(f_t1**2 + f_t2**2)
            label = f"{path.name}" + (f"  (dt={dt:g})" if dt else "")
            axes[0].plot(x, label=label)
            axes[1].plot(vx, label=label)
            axes[2].plot(f_t, label=label)
        axes[0].set_ylabel("ball x [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].set_ylabel("ball v_x [m/s]")
        axes[1].axhline(0, color="k", ls="--", alpha=0.4)
        axes[1].grid(True, alpha=0.3)
        axes[2].set_ylabel("|friction force| [N]")
        axes[2].set_xlabel("step")
        axes[2].grid(True, alpha=0.3)
    else:
        # 2 panels: iter_count, res_norm_sq
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        for path in paths:
            with HDF5Reader(str(path)) as r:
                ic = r.get_dataset("iter_count")
                rn = r.get_dataset("res_norm_sq")
                dt = r.get_attribute("/", "dt") if "dt" in r.list_attributes("/") else None
            ic = np.asarray(ic)[:, 0] if ic.ndim > 1 else np.asarray(ic)
            rn = np.asarray(rn)[:, 0] if rn.ndim > 1 else np.asarray(rn)
            label = f"{path.name}" + (f"  (dt={dt:g})" if dt else "")
            axes[0].plot(ic, label=label)
            axes[1].semilogy(np.maximum(rn, 1e-20), label=label)
        axes[0].set_ylabel("iter_count")
        axes[0].axhline(MAX_NEWTON_ITERS, color="k", ls="--", alpha=0.4,
                        label=f"max_newton_iters={MAX_NEWTON_ITERS}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("res_norm_sq (log)")
        axes[1].set_xlabel("step")
        axes[1].axhline(NEWTON_ATOL**2, color="k", ls="--", alpha=0.4,
                        label=f"newton_atol²={NEWTON_ATOL**2:g}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    out = RESULTS_DIR / ("diagnose_friction_plot.png" if friction_mode else "diagnose_plot.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dt", type=float, default=None, help="Run a diagnostic at this dt")
    parser.add_argument("--compliance", type=float, default=CONTACT_COMPLIANCE,
                        help=f"contact_compliance (default {CONTACT_COMPLIANCE})")
    parser.add_argument("--friction-compliance", type=float, default=FRICTION_COMPLIANCE,
                        help=f"friction_compliance (default {FRICTION_COMPLIANCE})")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="ground friction coefficient (default 0; >0 enables friction test)")
    parser.add_argument("--fx", type=float, default=0.0,
                        help="horizontal force in N applied each step (default 0)")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="HDF5 output path")
    parser.add_argument("--plot", nargs="+", type=pathlib.Path, default=None,
                        help="Plot one or more HDF5 logs")
    parser.add_argument("--plot-friction", nargs="+", type=pathlib.Path, default=None,
                        help="Friction-mode plot (x, v_x, f_t per step)")
    args = parser.parse_args()

    if args.plot or args.plot_friction:
        plot(args.plot or args.plot_friction, friction_mode=bool(args.plot_friction))
    elif args.dt is not None:
        if args.out is not None:
            out = args.out
        else:
            tag = f"dt{args.dt:g}"
            if args.mu > 0:
                tag += f"_mu{args.mu:g}_fx{args.fx:g}"
            if args.compliance != CONTACT_COMPLIANCE:
                tag += f"_e{args.compliance:g}"
            if args.friction_compliance != FRICTION_COMPLIANCE:
                tag += f"_ef{args.friction_compliance:g}"
            out = RESULTS_DIR / f"diag_{tag}.h5"
        run(
            args.dt, out,
            contact_compliance=args.compliance,
            friction_compliance=args.friction_compliance,
            mu=args.mu, fx=args.fx,
        )
    else:
        parser.error("specify either --dt or --plot")


if __name__ == "__main__":
    main()
