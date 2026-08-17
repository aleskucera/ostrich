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
    for k in ("t", "x", "y", "z", "yaw_rel"):
        if k in gt["real"]:  # yaw_rel was added later; tolerate old JSONs
            gt["real"][k] = np.asarray(gt["real"][k], dtype=np.float64)
    return gt


# Characteristic length used to fold yaw error into a position-equivalent
# combined error. A point at distance L from the rotation centre is displaced
# ~|Δp + L·Δyaw|; using L ≈ chassis half-length gives a metric that penalises
# missing yaw response (e.g. a sim with locked chassis rotation cannot hide
# behind low position L2). 0.5 m matches roughly half the junior wheelbase.
YAW_LEVER_ARM = 0.5  # m

# Settled-baseline window for the z reference (seconds). The chassis spawns
# slightly high and drops onto its wheels in the first ~0.2 s; the robot reaches
# the box near-face only after travelling ~1 m (≳2.5 s at the recorded speeds).
# [0.3, 1.0] s is safely after the settle and before any box contact, so its
# median z is the flat-ground resting height used as z=0 for every engine.
Z_SETTLE_LO = 0.3  # s
Z_SETTLE_HI = 1.0  # s


def _yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about world +Z) from quaternions [N, 4] in xyzw order."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


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


def llc_transform(setpoints: np.ndarray, deficit: float, omega0: float) -> np.ndarray:
    """Speed-dependent low-level-controller tracking model applied to wheel
    commands: real motor regulators track poorly at low wheel speeds and well
    at high speeds, so the executed speed is
        cmd' = cmd * (1 - deficit * exp(-|cmd| / omega0)).
    deficit in [0,1) is the zero-speed fractional tracking loss; omega0 [rad/s]
    the speed scale over which tracking recovers. (deficit=0 -> identity.)
    Identified from the longitudinal commanded-distance ratios (slow runs
    under-execute more than fast runs), disentangling actuator tracking from
    tire friction, which acts laterally."""
    if deficit <= 0.0:
        return setpoints
    eff = 1.0 - deficit * np.exp(-np.abs(setpoints) / omega0)
    return (setpoints * eff).astype(setpoints.dtype)


def score(sim_pose: np.ndarray, sim_dt: float, gt: dict, prism_offset=PRISM_OFFSET,
          yaw_lever_arm: float = YAW_LEVER_ARM):
    """Position + yaw error of an engine's chassis trajectory vs the real one,
    prism-tracked, time-aligned, and prism-vs-prism.

    Returns RMSE values (metres / radians) plus a combined score that folds
    yaw into a position-equivalent error via the lever arm:
        combined_with_yaw = sqrt(<|Δp|²> + (L · Δyaw_rmse)²)
    The yaw term penalises sims whose chassis won't rotate (e.g. over-cranked
    torsional friction), which pure position L2 misses entirely.
    """
    sim = prism_track(sim_pose, np.asarray(prism_offset, dtype=np.float32))
    st = np.arange(sim.shape[0]) * sim_dt
    # Zero x,y against the spawn pose (no transient there), but zero z against
    # the *settled* pre-box baseline rather than sim[0]. The chassis spawns
    # slightly above its resting height and drops onto its wheels over the
    # first ~0.2 s; engines that settle more (e.g. MuJoCo's softer contact)
    # would otherwise get their whole z curve shifted down by that settle,
    # because sim[0,2] is read before the drop completes. The real data is
    # already settled at t=0, so its baseline has no such drop. We take the
    # median z over a settled window [Z_SETTLE_LO, Z_SETTLE_HI] s — after the
    # contact settle, before the robot reaches the box — as the z=0 reference.
    sim = sim - sim[0]  # x,y relative to start (z fixed up next)
    lo = int(round(Z_SETTLE_LO / sim_dt))
    hi = min(int(round(Z_SETTLE_HI / sim_dt)), sim.shape[0])
    if hi > lo:
        # sim[:,2] was just shifted by sim[0,2]; rebase z on the settled median.
        z_baseline_rel = float(np.median(sim[lo:hi, 2]))
        sim[:, 2] -= z_baseline_rel

    rt = gt["real"]["t"]
    rx, ry, rz = gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]
    real_xy = np.column_stack([rx, ry])

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

    # Yaw RMSE (relative to each track's start, so absolute-frame offsets
    # don't pollute the comparison).
    yaw_rmse_rad = float("nan")
    combined_with_yaw = combined
    sim_yaw_interp = None
    if "yaw_rel" in gt["real"]:
        sim_yaw = _yaw_from_quat_xyzw(sim_pose[:, 3:7])
        sim_yaw_rel = sim_yaw - sim_yaw[0]
        sim_yaw_interp = np.interp(rtt, sta, sim_yaw_rel)
        dyaw = sim_yaw_interp - gt["real"]["yaw_rel"][m]
        yaw_rmse_rad = float(np.sqrt(np.mean(dyaw**2)))
        combined_with_yaw = float(np.sqrt(combined**2 + (yaw_lever_arm * yaw_rmse_rad) ** 2))

    return {
        "combined": combined,                         # position only
        "combined_with_yaw": combined_with_yaw,       # the honest metric
        "xy": xy,
        "z": z,
        "yaw_rmse_rad": yaw_rmse_rad,
        "yaw_rmse_deg": float(np.degrees(yaw_rmse_rad)),
        "shift": float(shift),
        "sim_rel": sim,
        "sim_t_aligned": sta,
        # sim yaw sampled on the real timestamps rtt (for yaw-trace overlays)
        "sim_yaw_rel_on_real_t": sim_yaw_interp,
        "real_t_used": rtt,
    }
