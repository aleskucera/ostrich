"""Shared GT loading, command preparation and segmented window scoring.

Replays are open-loop: an engine receives the recorded wheel-speed commands of a
real drive and integrates for the whole bag. Over multi-minute runs every engine
diverges from the odometry, so a single whole-run alignment mostly measures
compounding luck. The headline metric therefore cuts the run into WINDOW_S-second
windows, re-anchors the sim to the real pose at each window start (SE(2) + z
offset), and scores position+yaw RMSE per window; aggregates are the mean/median
across windows.

Unlike the older total-station experiment, wheel commands and pose share one
clock (header stamps from one recorder), so there is no cross-correlation
time-shift search here.

The measured actuator response (data/actuator_id.json: first-order lag
tau ~ 0.10 s, gain ~ 1.0) is identical hardware for every engine, so the bridge
pre-filters the commanded setpoints with that lag once, host-side, and every
engine receives the same inputs. `lag_tau=0` gives the unfiltered ablation.
"""

import json
import pathlib

import numpy as np

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

GT_BAGS = ["fast_experiment0", "fast_experiment1", "calibrate"]  # tuning/eval set
HOLDOUT_BAG = "motors0"

WINDOW_S = 3.0  # was 15.0 until 2026-08-12; 3 s measures short-horizon (MPC-scale) fidelity
YAW_LEVER_ARM = 0.5  # m; same rationale as 1_sim_to_real_box/common_box.py
ACTUATOR_TAU = 0.10  # s; median of data/actuator_id.json, all wheels ~equal


def load_gt(bag: str) -> dict:
    with open(DATA_DIR / f"gt_{bag}.json") as f:
        gt = json.load(f)
    gt["control"]["t"] = np.asarray(gt["control"]["t"])
    gt["control"]["lrr"] = np.asarray(gt["control"]["lrr"])
    gt["real"]["t"] = np.asarray(gt["real"]["t"])
    gt["real"]["pos"] = np.asarray(gt["real"]["pos"])
    gt["real"]["quat_xyzw"] = np.asarray(gt["real"]["quat_xyzw"])
    if "measured" in gt:
        gt["measured"]["t"] = np.asarray(gt["measured"]["t"])
        gt["measured"]["lrr"] = np.asarray(gt["measured"]["lrr"])
    return gt


def prepare_commands(gt: dict, dt: float, lag_tau: float = ACTUATOR_TAU,
                     duration: float | None = None) -> np.ndarray:
    """Resample commands [T,3] (sim order L,R,rear) onto the dt grid, then apply
    the measured first-order actuator lag so every engine gets identical inputs."""
    duration = duration if duration is not None else float(gt["control"]["t"][-1])
    T = int(round(duration / dt))
    tg = np.arange(T) * dt
    src_t, src = gt["control"]["t"], gt["control"]["lrr"]
    out = np.empty((T, 3), dtype=np.float64)
    for c in range(3):
        out[:, c] = np.interp(tg, src_t, src[:, c])
    if lag_tau > 0:
        blend = dt / (dt + lag_tau)
        state = out[0].copy()
        for i in range(T):
            state += blend * (out[i] - state)
            out[i] = state
    return out


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def window_score(sim_pose: np.ndarray, sim_dt: float, gt: dict,
                 window_s: float = WINDOW_S,
                 yaw_lever_arm: float = YAW_LEVER_ARM) -> dict:
    """Segmented re-anchored scoring of a full-bag sim trajectory.

    sim_pose: [T,7] chassis (x,y,z, qx,qy,qz,qw), one row per step, t=0 at bag t=0.
    Returns per-window records plus aggregates. Windows where the real robot moved
    less than 0.3 m still count (holding still badly is an error too) but carry
    their path length so analysis can filter.
    """
    st = np.arange(sim_pose.shape[0]) * sim_dt
    sim_yaw = np.unwrap(yaw_from_quat_xyzw(sim_pose[:, 3:7]))

    rt = gt["real"]["t"]
    rp = gt["real"]["pos"]
    ryaw = np.unwrap(yaw_from_quat_xyzw(gt["real"]["quat_xyzw"]))

    t_end = min(st[-1], rt[-1])
    windows = []
    w0 = 0.0
    while w0 + window_s <= t_end:
        m = (rt >= w0) & (rt <= w0 + window_s)
        rtt = rt[m]
        if len(rtt) < 10:
            w0 += window_s
            continue
        # Sim state interpolated onto the real timestamps of this window.
        sx = np.interp(rtt, st, sim_pose[:, 0])
        sy = np.interp(rtt, st, sim_pose[:, 1])
        sz = np.interp(rtt, st, sim_pose[:, 2])
        syaw = np.interp(rtt, st, sim_yaw)

        # Re-anchor: SE(2) transform matching sim to real at the window start,
        # plus a z offset. (Full SE(3) would also absorb terrain-slope error the
        # engines cannot know about; yaw+translation is the honest middle.)
        dyaw0 = ryaw[m][0] - syaw[0]
        c, s = np.cos(dyaw0), np.sin(dyaw0)
        px, py = sx - sx[0], sy - sy[0]
        ax = rp[m][0, 0] + c * px - s * py
        ay = rp[m][0, 1] + s * px + c * py
        az = rp[m][0, 2] + (sz - sz[0])
        ayaw = syaw - syaw[0] + ryaw[m][0]

        dx, dy, dz = ax - rp[m][:, 0], ay - rp[m][:, 1], az - rp[m][:, 2]
        dyaw = np.angle(np.exp(1j * (ayaw - ryaw[m])))
        pos_rmse = float(np.sqrt(np.mean(dx * dx + dy * dy + dz * dz)))
        yaw_rmse = float(np.sqrt(np.mean(dyaw * dyaw)))
        seg = np.linalg.norm(np.diff(rp[m][:, :2], axis=0), axis=1).sum()
        windows.append({
            "t0": float(w0),
            "pos_rmse": pos_rmse,
            "xy_rmse": float(np.sqrt(np.mean(dx * dx + dy * dy))),
            "z_rmse": float(np.sqrt(np.mean(dz * dz))),
            "yaw_rmse_rad": yaw_rmse,
            "combined": float(np.sqrt(pos_rmse ** 2 + (yaw_lever_arm * yaw_rmse) ** 2)),
            "real_path_len": float(seg),
            "end_err": float(np.hypot(dx[-1], dy[-1])),
        })
        w0 += window_s

    comb = np.asarray([w["combined"] for w in windows])
    return {
        "windows": windows,
        "n_windows": len(windows),
        "combined_mean": float(comb.mean()) if len(comb) else float("nan"),
        "combined_median": float(np.median(comb)) if len(comb) else float("nan"),
        "pos_rmse_mean": float(np.mean([w["pos_rmse"] for w in windows])) if windows else float("nan"),
        "yaw_rmse_deg_mean": float(np.degrees(np.mean([w["yaw_rmse_rad"] for w in windows]))) if windows else float("nan"),
    }


def score_result(res: dict, gt: dict, **kw) -> dict:
    """window_score() over a bridge result dict (adds stability passthrough)."""
    pose = np.asarray(res["pose"])
    out = window_score(pose, res["dt"], gt, **kw)
    out["stable"] = bool(res.get("stable", True))
    out["diverged_at_s"] = res.get("diverged_at_s")
    out["wall_clock_s"] = res.get("wall_clock_s")
    return out
