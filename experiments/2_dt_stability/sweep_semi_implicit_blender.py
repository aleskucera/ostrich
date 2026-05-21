"""Run Semi-Implicit over a fixed list of dt values and dump trajectories for Blender.

Mirrors sweep_axion_blender.py / sweep_mujoco_blender.py — same DT_LIST so the
three sweeps line up frame-by-frame in compare_to_blender.py.

Usage:
    python experiments/2_dt_stability/sweep_semi_implicit_blender.py \
        --export experiments/2_dt_stability/results/semi_implicit_dt.npz
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
from axion import ExecutionConfig  # noqa: E402
from axion import LoggingConfig  # noqa: E402
from axion import RenderingConfig  # noqa: E402
from axion import SemiImplicitEngineConfig  # noqa: E402
from axion import SimulationConfig  # noqa: E402

os.environ["PYOPENGL_PLATFORM"] = "glx"

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_semi_implicit import (  # noqa: E402
    HelhestObstacleSim,
    K_D,
    K_P,
    KD_CONTACT,
    KE,
    KF,
    MU,
    OBSTACLE_HEIGHT,
    OBSTACLE_X,
    RAMP_TIME,
    WHEEL_VEL,
)

# Shorter duration than the headless dt sweep (8 s).
DURATION = 4.0

# Shared with sweep_axion_blender.py and sweep_mujoco_blender.py so the three
# runs line up frame-by-frame in compare_to_blender.py.
# Semi-implicit's nominal max stable dt is ~0.65 ms; only the 0.5 ms entry
# should stay stable, everything above blows up.
DT_LIST = [5e-4, 1e-3, 2e-2, 1.3e-1]
DEFAULT_FPS = 30


# ---------------------------------------------------------------------------
# Subclass with full per-step body pose capture
# ---------------------------------------------------------------------------

class HelhestObstacleBlenderSim(HelhestObstacleSim):
    """HelhestObstacleSim that records every body's pose at every sim step."""

    def simulate_and_capture(self) -> tuple[np.ndarray, np.ndarray, bool, str]:
        body_q = self.current_state.body_q.numpy()
        if body_q.ndim == 3:
            body_q = body_q[0]
        raw_poses: list[np.ndarray] = [body_q.astype(np.float32).copy()]
        raw_times: list[float] = [0.0]

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
            raw_poses.append(body_q.astype(np.float32).copy())
            raw_times.append((step + 1) * dt)
            z = float(body_q[0, 2])
            z_min = min(z_min, z)
            z_max = max(z_max, z)

        raw_poses_arr = np.stack(raw_poses, axis=0).astype(np.float32)
        raw_times_arr = np.asarray(raw_times, dtype=np.float32)

        x_final = float(raw_poses_arr[-1, 0, 0])
        is_stable = (
            not has_nan
            and z_min > 0.05
            and z_max < 2.0
            and x_final > self._obstacle_x + 1.0
        )
        if has_nan:
            note = "diverged"
        elif z_max >= 2.0 or z_min <= 0.05:
            note = "chassis out of bounds"
        elif x_final <= self._obstacle_x + 1.0:
            note = f"stalled at x={x_final:.2f}"
        else:
            note = "stable"
        return raw_poses_arr, raw_times_arr, is_stable, note


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def _build_configs(dt: float):
    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=dt,
        num_worlds=1,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=1,
    )
    # Stiffer joint attachments (10× the defaults) — same values
    # tune_semi_implicit.py settled on. Helps the wheel revolutes hold
    # against the obstacle reaction without exploding at moderate dt.
    engine_config = SemiImplicitEngineConfig(
        angular_damping=0.05,
        friction_smoothing=0.1,
        joint_attach_ke=1.0e5,
        joint_attach_kd=1.0e3,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )
    return sim_config, render_config, exec_config, engine_config, logging_config


# ---------------------------------------------------------------------------
# Shape extraction (same logic as the axion sweep blender exporter)
# ---------------------------------------------------------------------------

def extract_shapes(model) -> list[dict]:
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
# Resampling onto a common fps grid
# ---------------------------------------------------------------------------

def _resample_poses(
    raw_poses: np.ndarray, raw_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    target_T = len(target_times)
    num_bodies = raw_poses.shape[1]
    out = np.zeros((target_T, num_bodies, 7), dtype=np.float32)
    for b in range(num_bodies):
        for d in range(3):
            out[:, b, d] = np.interp(target_times, raw_times, raw_poses[:, b, d])
        for k, t_target in enumerate(target_times):
            idx = int(np.searchsorted(raw_times, t_target, side="right")) - 1
            idx = max(0, min(idx, len(raw_times) - 1))
            if idx + 1 >= len(raw_times):
                out[k, b, 3:7] = raw_poses[idx, b, 3:7]
                continue
            t0 = float(raw_times[idx])
            t1 = float(raw_times[idx + 1])
            alpha = 0.0 if t1 == t0 else (float(t_target) - t0) / (t1 - t0)
            q0 = raw_poses[idx, b, 3:7].astype(np.float32)
            q1 = raw_poses[idx + 1, b, 3:7].astype(np.float32)
            if float(np.dot(q0, q1)) < 0.0:
                q1 = -q1
            qm = (1.0 - alpha) * q0 + alpha * q1
            n = float(np.linalg.norm(qm))
            out[k, b, 3:7] = qm / max(n, 1e-9)
    return out


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

    target_T = int(round(DURATION * args.fps))
    target_times = np.linspace(0.0, DURATION, target_T, dtype=np.float32)

    pose_iters: list[np.ndarray] = []
    iter_labels: list[str] = []
    iter_stable: list[bool] = []
    shapes: list[dict] | None = None

    num_render_bodies = 4  # chassis + 3 wheels in helhest

    for dt in DT_LIST:
        print(f"  simulating dt={dt}s ({int(DURATION/dt)} steps)...", end=" ", flush=True)
        sim_config, render_config, exec_config, engine_config, logging_config = _build_configs(dt)
        sim = HelhestObstacleBlenderSim(
            sim_config,
            render_config,
            exec_config,
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

        raw_poses, raw_times, stable, note = sim.simulate_and_capture()
        raw_poses = raw_poses[:, :num_render_bodies]
        resampled = _resample_poses(raw_poses, raw_times, target_times)
        pose_iters.append(resampled)
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
