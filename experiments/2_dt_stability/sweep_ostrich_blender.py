"""Run Ostrich over a fixed list of dt values and dump trajectories for Blender.

Each dt produces one "iteration" of the resulting npz. The Blender importer for
this experiment (experiments/2_dt_stability/import_to_blender.py) stacks them
on the timeline so you can watch the same obstacle traversal at progressively
coarser timesteps.

Usage:
    python experiments/2_dt_stability/sweep_ostrich_blender.py \
        --export experiments/2_dt_stability/results/ostrich_dt.npz
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import override

import warp

warp.config.quiet = True

import newton  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from ostrich import OstrichEngineConfig
from ostrich.core.engine_config import ComplianceConfig
from ostrich.core.engine_config import ContactsConfig
from ostrich.core.engine_config import LinearSolverConfig
from ostrich.core.engine_config import LinesearchConfig
from ostrich.core.engine_config import NewtonRaphsonConfig  # noqa: E402
from ostrich import LoggingConfig  # noqa: E402
from ostrich import RenderingConfig  # noqa: E402
from ostrich import SimulationConfig  # noqa: E402
from ostrich import FrameRecorder  # noqa: E402

os.environ["PYOPENGL_PLATFORM"] = "glx"

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_ostrich import (  # noqa: E402
    CONTACT_COMPLIANCE,
    FRICTION_COMPLIANCE,
    HelhestObstacleSim,
    K_P,
    MU,
    OBSTACLE_HEIGHT,
    OBSTACLE_MU,
    OBSTACLE_X,
    RAMP_TIME,
    WHEEL_VEL,
)

# Shorter duration than the headless dt sweep (8 s) — for Blender video the
# robot only needs to clear the obstacle and continue briefly.
DURATION = 4.0

# Shared with sweep_mujoco_blender.py and sweep_semi_implicit_blender.py so
# the three runs line up frame-by-frame in compare_to_blender.py. The list
# spans semi-implicit's boundary (~0.65 ms) and MuJoCo's (~1.5 ms); ostrich
# stays stable across the whole range.
DT_LIST = [5e-4, 1e-3, 2e-2, 1.3e-1]
DEFAULT_FPS = 30


# ---------------------------------------------------------------------------
# Subclass with full per-step body pose capture
# ---------------------------------------------------------------------------


class HelhestObstacleBlenderSim(HelhestObstacleSim):
    """HelhestObstacleSim that records every body's pose at every sim step."""

    def simulate_and_capture(self, recorder: FrameRecorder) -> tuple[np.ndarray, bool, str]:
        """Run all sim steps, capturing body_q onto the recorder's fps grid.

        Returns (poses [T, num_bodies, 7], is_stable, note). The recorder
        interpolates as it goes, so the poses come out on the render grid
        with no resampling pass afterwards.
        Stability matches sweep_ostrich's simulate_and_check predicate.
        """
        body_q = self.current_state.body_q.numpy()
        if body_q.ndim == 3:
            body_q = body_q[0]
        recorder.start(body_q)
        x_final = float(body_q[0, 0])

        z_min = float(body_q[0, 2])
        z_max = z_min
        has_nan = False
        total_steps = self.clock.total_sim_steps
        dt = self.clock.dt

        # Bound applies to *every* body, not just chassis — otherwise a wheel
        # can fly past ~20 km before the chassis-only check fires, leaving
        # extreme keyframes in the npz that bloat Blender's BVH/shadow bounds
        # and tank EEVEE render time across the whole timeline.
        DIVERGE_BOUND = 50.0

        for step in range(total_steps):
            self._single_physics_step(0)
            wp.synchronize()
            body_q = self.current_state.body_q.numpy()
            if body_q.ndim == 3:
                body_q = body_q[0]
            if (not np.all(np.isfinite(body_q))) or np.any(
                np.abs(body_q[:, :3]) > DIVERGE_BOUND
            ):
                has_nan = True
                break
            recorder.record(body_q, (step + 1) * dt)
            x_final = float(body_q[0, 0])
            z = float(body_q[0, 2])
            z_min = min(z_min, z)
            z_max = max(z_max, z)

        poses = recorder.finish()

        is_stable = (
            not has_nan and z_min > 0.05 and z_max < 2.0 and x_final > self._obstacle_x + 1.0
        )
        if has_nan:
            note = "diverged"
        elif z_max >= 2.0 or z_min <= 0.05:
            note = "chassis out of bounds"
        elif x_final <= self._obstacle_x + 1.0:
            note = f"stalled at x={x_final:.2f}"
        else:
            note = "stable"
        return poses, is_stable, note


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def _build_configs(dt: float):
    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=dt,
        num_worlds=1,
        use_cuda_graph=True,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    # The old flat OstrichEngineConfig also carried contact_fb_alpha=0.5 and
    # contact/friction fb_beta. Those knobs no longer exist on the config:
    # friction's are gone entirely and contact's alpha is a module-import
    # constant, so reproducing the original 0.5 needs OSTRICH_CONTACT_FB_ALPHA=0.5
    # in the environment (it defaults to 1.0).
    engine_config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(
            max_iters=24,
            backtrack_min_iter=18,
            atol=1e-8,
        ),
        linear=LinearSolverConfig(
            max_iters=16,
            tol=1e-5,
            atol=1e-5,
            regularization=1e-6,
        ),
        compliance=ComplianceConfig(
            joint=6e-10,
            contact=CONTACT_COMPLIANCE,
            friction=FRICTION_COMPLIANCE,
        ),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=16),
    )
    logging_config = LoggingConfig()
    return sim_config, render_config, engine_config, logging_config


# ---------------------------------------------------------------------------
# Shape extraction (same logic as spline_surface_fast's exporter)
# ---------------------------------------------------------------------------


def extract_shapes(model) -> list[dict]:
    """Per-shape descriptors compatible with the Blender importer."""
    shape_body = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_thickness = model.shape_margin.numpy()
    shape_is_solid = model.shape_is_solid.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_source = model.shape_source

    visible_mask = int(newton.ShapeFlags.VISIBLE)
    mesh_types = {int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)}
    shapes: list[dict] = []
    for s in range(len(shape_body)):
        if not (shape_flags[s] & visible_mask):
            continue
        gt = int(shape_type[s])
        entry = {
            "body_idx": int(shape_body[s]),
            "geo_type": gt,
            "geo_scale": np.array(shape_scale[s], dtype=np.float32),
            "geo_thickness": float(shape_thickness[s]),
            "geo_is_solid": bool(shape_is_solid[s]),
            "local_xform": shape_transform[s].astype(np.float32),
        }
        if gt in mesh_types and shape_source[s] is not None:
            mesh = shape_source[s]
            entry["mesh_verts"] = np.asarray(mesh.vertices, dtype=np.float32)
            entry["mesh_faces"] = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        shapes.append(entry)
    return shapes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=pathlib.Path, required=True, help="Output .npz")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help=f"Target fps (default {DEFAULT_FPS})"
    )
    args = parser.parse_args()

    pose_iters: list[np.ndarray] = []
    iter_labels: list[str] = []
    iter_stable: list[bool] = []
    shapes: list[dict] | None = None

    num_render_bodies = 4  # chassis + 3 wheels in helhest

    for dt in DT_LIST:
        print(f"  simulating dt={dt}s ({int(DURATION/dt)} steps)...", end=" ", flush=True)
        sim_config, render_config, engine_config, logging_config = _build_configs(dt)
        sim = HelhestObstacleBlenderSim(
            sim_config,
            render_config,
            engine_config,
            logging_config,
            wheel_vel=WHEEL_VEL,
            obstacle_x=OBSTACLE_X,
            obstacle_height=OBSTACLE_HEIGHT,
            initial_yaw=0.0,
        )
        if shapes is None:
            shapes = extract_shapes(sim.model)
            print(f"\nExtracted {len(shapes)} shape descriptors.")
            print(f"  simulating dt={dt}s ({int(DURATION/dt)} steps)...", end=" ", flush=True)

        # Capture straight onto the render grid, trimmed to the helhest
        # robot bodies (chassis + 3 wheels).
        recorder = FrameRecorder(args.fps, DURATION, num_render_bodies)
        poses, stable, note = sim.simulate_and_capture(recorder)
        pose_iters.append(poses)
        iter_stable.append(stable)
        suffix = "" if stable else f"  ({note})"
        iter_labels.append(f"dt = {dt*1000:.1f} ms{suffix}")
        print("STABLE" if stable else f"UNSTABLE ({note})")

    body_pose_iters = np.stack(pose_iters, axis=0).astype(np.float32)
    iter_dt_values = np.array(DT_LIST, dtype=np.float32)
    iter_indices = np.arange(len(DT_LIST), dtype=np.int32)
    iter_stable_arr = np.array(iter_stable, dtype=bool)

    args.export.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.export,
        dt=np.float32(1.0 / args.fps),
        fps=np.float32(args.fps),
        body_pose_iters=body_pose_iters,
        iter_indices=iter_indices,
        iter_dt_values=iter_dt_values,
        iter_labels=np.array(iter_labels, dtype=object),
        iter_stable=iter_stable_arr,
        shapes=np.array(shapes, dtype=object),
    )
    print(f"\nSaved {args.export}  ({body_pose_iters.shape}, {len(shapes)} shapes)")


if __name__ == "__main__":
    main()
