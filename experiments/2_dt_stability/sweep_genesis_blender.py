"""Run Genesis over a fixed list of dt values and dump trajectories for Blender.

Mirrors sweep_mujoco_blender.py / sweep_axion_blender.py / sweep_semi_implicit_blender.py
— same DT_LIST so the four sweeps line up frame-by-frame in compare_to_blender.py.

Genesis loads the same Helhest+obstacle MJCF used by the MuJoCo sweep. Wheel
joints are velocity-controlled via Genesis's PD controller. Forward sim only —
no autograd — so collision and constraint solving stay enabled (which the
gradient-mode genesis scripts in examples/comparison/genesis/ disable).

Usage:
    python experiments/2_dt_stability/sweep_genesis_blender.py \
        --export experiments/2_dt_stability/results/genesis_dt.npz
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import tempfile

import genesis as gs
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_mujoco import (  # noqa: E402
    HELHEST_OBSTACLE_XML,
    KV,
    MU,
    OBSTACLE_HEIGHT,
    OBSTACLE_MU,
    OBSTACLE_X,
    RAMP_TIME,
    WHEEL_VEL,
)
from sweep_mujoco_blender import (  # noqa: E402
    RENDER_BODIES,
    _resample_poses,
    extract_shapes,
)

DURATION = 4.0
DT_LIST = [5e-4, 1e-3, 2e-2, 1.3e-1]
DEFAULT_FPS = 30

WHEEL_JOINT_NAMES = ("left_wheel_j", "right_wheel_j", "rear_wheel_j")

# Genesis must be initialized once per process. GPU is the default in the
# genesis comparison scripts in examples/comparison/genesis/.
gs.init(backend=gs.gpu, logging_level="warning")


def _build_xml(dt: float) -> str:
    # Pass the MJCF default — Genesis's constraint solver auto-bumps timeconst
    # to >= 2*dt internally (warns "timeconst is changed from 0.005 to ..."),
    # so we don't need to pre-bump here.
    return HELHEST_OBSTACLE_XML.format(
        dt=dt,
        kv=KV,
        mu=MU,
        obstacle_mu=OBSTACLE_MU,
        obstacle_x=OBSTACLE_X,
        obstacle_height=OBSTACLE_HEIGHT,
        solref_timeconst=0.005,
        chassis_qw=1.0,
        chassis_qz=0.0,
    )


def _write_xml_tmpfile(xml: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="helhest_obstacle_genesis_")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


def simulate_dt(dt: float, target_times: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """Run Genesis at given dt, capture per-step poses, resample to target_times.

    Returns (resampled_poses [target_T, B, 7], is_stable, note). Stability matches
    the predicate used by the other sweeps (no NaN, chassis z in (0.05, 2.0),
    chassis x_final past obstacle_x + 1).
    """
    xml = _build_xml(dt)
    xml_path = _write_xml_tmpfile(xml)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
            requires_grad=False,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=False,
            enable_joint_limit=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path))
    scene.build()

    wheel_dof_idx = [
        robot.get_joint(name).dofs_idx_local[0] for name in WHEEL_JOINT_NAMES
    ]
    link_idx = [robot.get_link(name).idx_local for name in RENDER_BODIES]

    def _snapshot() -> np.ndarray:
        pos = robot.get_links_pos(links_idx_local=link_idx).detach().cpu().numpy()
        quat_wxyz = robot.get_links_quat(links_idx_local=link_idx).detach().cpu().numpy()
        if pos.ndim == 3:  # (n_envs, n_links, 3) when batched — we run a single env
            pos = pos[0]
            quat_wxyz = quat_wxyz[0]
        out = np.zeros((len(link_idx), 7), dtype=np.float32)
        out[:, 0:3] = pos.astype(np.float32)
        # Genesis quaternions are wxyz; the Blender importer expects xyzw.
        out[:, 3] = quat_wxyz[:, 1]
        out[:, 4] = quat_wxyz[:, 2]
        out[:, 5] = quat_wxyz[:, 3]
        out[:, 6] = quat_wxyz[:, 0]
        return out

    raw_poses: list[np.ndarray] = [_snapshot()]
    raw_times: list[float] = [0.0]

    has_nan = False
    z_min = float(raw_poses[0][0, 2])
    z_max = float(raw_poses[0][0, 2])
    total_steps = int(round(DURATION / dt))
    for step in range(total_steps):
        t = (step + 1) * dt
        ramp = min(t / RAMP_TIME, 1.0)
        wv = WHEEL_VEL * ramp
        robot.control_dofs_velocity([wv, wv, wv], dofs_idx_local=wheel_dof_idx)
        scene.step()
        snap = _snapshot()
        # 50 m bound applies to *every* body, not just chassis — keeps
        # diverged trajectories from baking far-away keyframes into the npz,
        # which would bloat Blender's BVH/shadow bounds and tank EEVEE perf.
        if not np.all(np.isfinite(snap)) or np.any(np.abs(snap[:, :3]) > 50.0):
            has_nan = True
            break
        raw_poses.append(snap)
        raw_times.append(t)
        z_min = min(z_min, float(snap[0, 2]))
        z_max = max(z_max, float(snap[0, 2]))

    raw_poses_arr = np.stack(raw_poses, axis=0).astype(np.float32)
    raw_times_arr = np.asarray(raw_times, dtype=np.float32)
    x_final = float(raw_poses_arr[-1, 0, 0])
    is_stable = (
        not has_nan and z_min > 0.05 and z_max < 2.0 and x_final > OBSTACLE_X + 1.0
    )
    if has_nan:
        note = "diverged"
    elif z_max >= 2.0 or z_min <= 0.05:
        note = "chassis out of bounds"
    elif x_final <= OBSTACLE_X + 1.0:
        note = f"stalled at x={x_final:.2f}"
    else:
        note = "stable"

    resampled = _resample_poses(raw_poses_arr, raw_times_arr, target_times)
    scene.destroy()
    try:
        os.unlink(xml_path)
    except OSError:
        pass
    return resampled, is_stable, note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=pathlib.Path, required=True, help="Output .npz")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help=f"Target fps (default {DEFAULT_FPS})"
    )
    args = parser.parse_args()

    target_T = int(round(DURATION * args.fps))
    target_times = np.linspace(0.0, DURATION, target_T, dtype=np.float32)

    # Reuse the MuJoCo-side shape extractor (Genesis loads the same MJCF, so
    # the visualization geometry is identical to the MuJoCo sweep).
    setup_xml = _build_xml(DT_LIST[0])
    mj_model = mujoco.MjModel.from_xml_string(setup_xml)
    shapes = extract_shapes(mj_model)
    print(f"Extracted {len(shapes)} shape descriptors.")

    pose_iters: list[np.ndarray] = []
    iter_labels: list[str] = []
    iter_stable: list[bool] = []
    for dt in DT_LIST:
        print(f"  simulating dt={dt}s ({int(DURATION/dt)} steps)...", end=" ", flush=True)
        poses, stable, note = simulate_dt(dt, target_times)
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
