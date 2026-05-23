"""Shared utilities for the box sim-to-real benchmark.

This experiment is the box-obstacle counterpart of ``experiments/1_sim_to_real``:
instead of flat-ground turn/accel maneuvers it drives each physics engine with
the recorded wheel commands of a real helhest_junior run *over the box* and
scores how well the engine reproduces the measured trajectory — including the
climb (Z).

Ground truth is produced by ``prepare_gt.py`` from the synced rosbag runs
(~/rosbags_experiment/synced/run_*.h5) into a JSON here under ``data/``. Every
engine consumes the SAME GT JSON so the comparison is apples-to-apples:
  - the wheel commands are stored already remapped to sim DOF order [L,R,rear]
    and sign-flipped so "forward" is positive on every wheel (see replay_real);
  - the real trajectory is the total-station prism point, aligned to start at
    the origin with initial heading +X, z relative to start.

The metric mirrors the replay tooling: track the prism point in sim, correct
the wheel-vs-pose stream-zeroing offset by cross-correlating forward-x, then
take the combined 3D L2 over the overlapping valid window.
"""
import json
import pathlib

import numpy as np

# Reuse the validated geometry/alignment helpers from the replay tool.
from examples.helhest_junior.replay_real import (
    PRISM_OFFSET,
    best_time_shift,
    prism_track,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def load_gt(path) -> dict:
    """Load a box GT JSON written by prepare_gt.py."""
    with open(path) as f:
        gt = json.load(f)
    # Convert lists to arrays for convenience.
    gt["control"]["t"] = np.asarray(gt["control"]["t"], dtype=np.float64)
    gt["control"]["lrr"] = np.asarray(gt["control"]["lrr"], dtype=np.float32)  # [T,3] L,R,rear
    for k in ("t", "x", "y", "z"):
        gt["real"][k] = np.asarray(gt["real"][k], dtype=np.float64)
    return gt


def resample_setpoints(gt: dict, dt: float, duration: float) -> np.ndarray:
    """Resample the GT wheel commands [L,R,rear] (sim order+sign) onto a dt grid."""
    T = int(round(duration / dt))
    tg = np.arange(T) * dt
    src_t = gt["control"]["t"]
    src = gt["control"]["lrr"]
    out = np.zeros((T, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(tg, src_t, src[:, c])
    return out


def score(sim_pose: np.ndarray, sim_dt: float, gt: dict, prism_offset=PRISM_OFFSET):
    """Combined 3D L2 error (m) of an engine's chassis trajectory vs the real
    prism trajectory, prism-tracked and time-aligned.

    sim_pose: [N,7] chassis pose (px,py,pz, qx,qy,qz,qw) in the engine's world.
    Returns dict with combined/xy/z RMSE (metres), the shift, and aligned arrays
    for plotting.
    """
    sim = prism_track(sim_pose, np.asarray(prism_offset, dtype=np.float32))
    sim = sim - sim[0]  # relative to start, like the real trajectory
    st = np.arange(sim.shape[0]) * sim_dt

    rt = gt["real"]["t"]
    rx, ry, rz = gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]
    real_xy = np.column_stack([rx, ry])  # already valid-only (no NaN) in the GT

    shift = best_time_shift(sim, st, real_xy, rt)
    sta = st - shift

    m = (rt >= 0) & (rt <= min(rt.max(), sta.max()))
    rtt = rt[m]
    sx = np.interp(rtt, sta, sim[:, 0])
    sy = np.interp(rtt, sta, sim[:, 1])
    sz = np.interp(rtt, sta, sim[:, 2])
    dx, dy, dz = sx - rx[m], sy - ry[m], sz - rz[m]

    combined = float(np.sqrt(np.mean(dx**2 + dy**2 + dz**2)))
    xy = float(np.sqrt(np.mean(dx**2 + dy**2)))
    z = float(np.sqrt(np.mean(dz**2)))
    return {
        "combined": combined,
        "xy": xy,
        "z": z,
        "shift": float(shift),
        "sim_rel": sim,
        "sim_t_aligned": sta,
    }
