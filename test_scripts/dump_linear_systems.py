"""Dump the per-NR-iter linear systems PCR solves, from the Helhest
obstacle-benchmark scene, for offline learned-preconditioner experiments.

Strategy
--------
We drive `OstrichEngine.step()` directly in a plain Python loop (no
simulator / hydra / viewer and crucially no surrounding `wp.capture`).
With `device.is_capturing == False` and the profiler disabled,
`base_engine._solve()` takes the eager while-loop NR path
(`base_engine.py:487`), so every NR iteration is a real Python call and
the inner PCR solve is invoked eagerly once per iter.

We monkeypatch `engine.cr_solver.solve` to snapshot, for every PCR call:
  * the matrix-free system inputs from `A.data` (a SystemLinearData):
    J_values, constr_body_idx, constr_active_mask, C_values,
    body_inv_mass, world_inv_inertia, plus A's regularization;
  * the right-hand side `b`;
  * the engine's own PCR solution `x` after the solve (reference).

That is exactly the system each PCR call faces, regardless of call site
(NR loop *and* the two warm-start passes — the latter have C zeroed,
flagged via `c_all_zero`).

Output: a single compressed .npz with stacked arrays (N_c is fixed for a
given model — contact slots are padded and gated by the active mask).

Run:
    python test_scripts/dump_linear_systems.py
    python test_scripts/dump_linear_systems.py --steps 25 --out data/baselines/helhest_systems.npz
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import warp as wp

# Repo root on path so `examples.helhest.common` imports cleanly.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import newton  # noqa: E402
from ostrich import OstrichEngineConfig  # noqa: E402
from ostrich import LoggingConfig  # noqa: E402
from ostrich.core.model_builder import OstrichModelBuilder  # noqa: E402

from examples.helhest.common import create_helhest_model  # noqa: E402

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


def build_helhest_obstacle_model(friction: float = 0.5):
    """Mirror HelhestObstacleBenchmark.build_model (obstacle_benchmark.py)."""
    builder = OstrichModelBuilder()
    builder.rigid_gap = 1.0

    create_helhest_model(
        builder,
        xform=wp.transform((-1.5, 0.0, 0.6), wp.quat_identity()),
        control_mode="velocity",
        k_p=150.0,
        k_d=0.0,
        friction_left_right=friction,
        friction_rear=friction * 0.5,
    )

    FRICTION, RESTITUTION = 0.4, 0.0
    for x, hz in [(2.0, 0.10), (5.0, 0.25), (8.0, 0.40), (11.0, 0.65)]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=hz,
            cfg=newton.ModelBuilder.ShapeConfig(mu=FRICTION, restitution=RESTITUTION),
        )

    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(
            ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=FRICTION, restitution=RESTITUTION
        )
    )

    return builder.finalize_replicated(num_worlds=1, gravity=-9.81)


def build_helhest_surface_model(friction: float = 0.5):
    """Mirror HelhestSurfaceBenchmark.build_model (surface_drive_benchmark.py):
    Helhest driving over the surface.obj triangle-mesh terrain."""
    import openmesh

    builder = OstrichModelBuilder()
    builder.rigid_gap = 0.5

    create_helhest_model(
        builder,
        xform=wp.transform((-1.5, 0.0, 1.7), wp.quat_identity()),
        control_mode="velocity",
        k_p=150.0,
        k_d=0.0,
        friction_left_right=friction,
        friction_rear=friction * 0.5,
    )

    assets = REPO_ROOT / "examples/assets/surface.obj"
    sm = openmesh.read_trimesh(str(assets))
    idx = np.array(sm.face_vertex_indices(), dtype=np.int32).flatten()
    pts = np.array(sm.points()) * np.array([6.0, 6.0, 4.0]) + np.array([0.0, 0.0, 0.05])
    surface_mesh = newton.Mesh(pts, idx)

    globals_builder = newton.ModelBuilder()
    globals_builder.add_shape_mesh(
        body=-1,
        mesh=surface_mesh,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0, has_shape_collision=True,
            mu=0.5, ke=150.0, kd=150.0, kf=500.0,
        ),
    )

    return builder.finalize_replicated(
        num_worlds=1, gravity=-9.81, global_builder=globals_builder
    )


class SystemCapture:
    """Wraps `cr_solver.solve` to snapshot every PCR system."""

    def __init__(self, engine, max_systems: int):
        self.engine = engine
        self.max_systems = max_systems
        self.records = []
        self.step = 0           # set by the driver loop before engine.step
        self.iter_in_step = 0   # reset per step; ++ per captured PCR solve
        self._orig_solve = engine.cr_solver.solve
        engine.cr_solver.solve = self._wrapped_solve

    def _wrapped_solve(self, A, b, x, preconditioner, iters, tol, atol, log=False):
        d = A.data  # SystemLinearData
        if len(self.records) < self.max_systems:
            C = d.C_values.numpy()
            rec = {
                "J_values": d.J_values.numpy(),                  # (W,Nc,2,6)
                "constr_body_idx": d.constraint_body_idx.numpy(),  # (W,Nc,2)
                "constr_active_mask": d.constraint_active_mask.numpy(),  # (W,Nc)
                "C_values": C,                                    # (W,Nc)
                "body_inv_mass": d.body_inv_mass.numpy(),         # (W,Nb)
                "world_inv_inertia": d.world_inv_inertia.numpy(),  # (W,Nb,3,3)
                "b": b.numpy(),                                   # (W,Nc)
                "regularization": float(getattr(A, "regularization", 1e-6)),
                "c_all_zero": bool(np.all(C == 0.0)),
                "step_idx": int(self.step),
                "iter_in_step": int(self.iter_in_step),
            }
        else:
            rec = None
        self.iter_in_step += 1

        ret = self._orig_solve(A, b, x, preconditioner, iters, tol, atol, log=log)

        if rec is not None:
            rec["x_engine"] = x.numpy()  # engine PCR solution (reference)
            self.records.append(rec)
        return ret

    def stacked(self):
        keys_arr = [
            "J_values", "constr_body_idx", "constr_active_mask", "C_values",
            "body_inv_mass", "world_inv_inertia", "b", "x_engine",
        ]
        out = {k: np.stack([r[k] for r in self.records]) for k in keys_arr}
        out["regularization"] = np.array([r["regularization"] for r in self.records])
        out["c_all_zero"] = np.array([r["c_all_zero"] for r in self.records])
        out["step_idx"] = np.array([r["step_idx"] for r in self.records])
        out["iter_in_step"] = np.array([r["iter_in_step"] for r in self.records])
        return out


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--max-systems", type=int, default=600)
    ap.add_argument("--drive-velocity", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--scene", choices=["obstacle", "surface"], default="obstacle")
    ap.add_argument("--out", type=str, default=None,
                    help="defaults to data/baselines/helhest_<scene>_systems.npz")
    args = ap.parse_args()
    if args.out is None:
        nm = "systems" if args.scene == "obstacle" else "surface_systems"
        args.out = f"data/baselines/helhest_{nm}.npz"

    with wp.ScopedDevice(args.device):
        model = (build_helhest_surface_model() if args.scene == "surface"
                 else build_helhest_obstacle_model())

        cfg = OstrichEngineConfig()
        engine = cfg.create_engine(
            model=model, sim_steps=args.steps, logging_config=LoggingConfig(),
            differentiable_simulation=False,
        )

        current_state = model.state()
        next_state = model.state()
        control = model.control()
        contacts = model.collide(current_state)
        newton.eval_fk(model, model.joint_q, model.joint_qd, current_state)

        # Velocity-mode wheel drive: last 3 joint DOFs are the wheels
        # (6 free-base DOFs precede them). Matches obstacle_benchmark.
        n_dof = control.joint_target_vel.shape[0]
        targets = np.zeros(n_dof, dtype=np.float32)
        targets[-3:] = args.drive_velocity
        joint_target = wp.array(targets, dtype=wp.float32, device=model.device)

        cap = SystemCapture(engine, max_systems=args.max_systems)

        print(f"dims: N_w={engine.dims.N_w} N_b={engine.dims.N_b} "
              f"N_c={engine.dims.N_c} (N_j={engine.dims.N_j} "
              f"N_ctrl={engine.dims.N_ctrl} N_n={engine.dims.N_n} N_f={engine.dims.N_f})")

        for s in range(args.steps):
            cap.step = s
            cap.iter_in_step = 0
            current_state.clear_forces()
            contacts = model.collide(current_state)
            wp.copy(control.joint_target_vel, joint_target)
            engine.step(current_state, next_state, control, contacts, args.dt)
            current_state, next_state = next_state, current_state
            if (s + 1) % 5 == 0:
                print(f"  step {s + 1}/{args.steps}  systems={len(cap.records)}")
            if len(cap.records) >= args.max_systems:
                print("  reached --max-systems cap, stopping early")
                break

        data = cap.stacked()
        meta = dict(
            N_w=engine.dims.N_w, N_b=engine.dims.N_b, N_c=engine.dims.N_c,
            N_j=engine.dims.N_j, N_ctrl=engine.dims.N_ctrl,
            N_n=engine.dims.N_n, N_f=engine.dims.N_f,
            offset_j=engine.dims.offset_j, offset_ctrl=engine.dims.offset_ctrl,
            offset_n=engine.dims.offset_n, offset_f=engine.dims.offset_f,
        )
        data["meta"] = np.array([meta], dtype=object)

        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)

        n = len(cap.records)
        n_warm = int(data["c_all_zero"].sum())
        print(f"\nsaved {n} systems ({n_warm} warm-start C=0, {n - n_warm} NR) "
              f"-> {out_path}")
        print(f"J_values shape {data['J_values'].shape}, "
              f"b shape {data['b'].shape}")


if __name__ == "__main__":
    main()
