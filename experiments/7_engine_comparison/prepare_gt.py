"""Convert helhest_stack ROS 2 mcap bags into ground-truth JSONs for the engine comparison.

This experiment replays the recorded wheel setpoints of real helhest_junior drives
open-loop in each physics engine (ostrich / AGX / Chrono) and scores the simulated
chassis trajectory against the odin lidar-inertial odometry.

This script has NO ostrich dependencies on purpose: it needs the ``rosbags`` library,
which lives in the helhest_stack venv, so run it with that interpreter:

    ~/projects/helhest_stack/.venv/bin/python prepare_gt.py            # all default bags
    ~/projects/helhest_stack/.venv/bin/python prepare_gt.py fast_experiment1

Conventions (verified against the bags, 2026-08-12):
  - joint topics carry names [left_wheel_j, rear_wheel_j, right_wheel_j]; all three
    are forward-positive with tracking gain ~1.0 — no sign flips (unlike the older
    total-station bags, whose raw left/right axes were mirrored).
  - commands are stored in SIM DOF order [left, right, rear] ("lrr"), matching
    examples/helhest_junior/common.py (dof 6=left, 7=right, 8=rear).
  - header stamps are used everywhere; bag receive times arrive in bursts and made
    finite-difference velocities garbage during exploration.
  - /odin1/odometry is the pose of odin1_base_link in odom_odin; the static
    transform to base_link is a pure translation (-0.1, 0, -0.1), applied here so
    the stored pose is base_link.
"""

import json
import pathlib
import sys

import numpy as np
from rosbags.highlevel import AnyReader

BAGS_DIR = pathlib.Path.home() / "projects" / "helhest_stack" / "bags"
DATA_DIR = pathlib.Path(__file__).parent / "data"

DEFAULT_BAGS = ["fast_experiment0", "fast_experiment1", "calibrate", "motors0"]

# base_link expressed in odin1_base_link, from /tf_static (identity rotation).
ODIN_TO_BASE = np.array([-0.1, 0.0, -0.1])

# bag joint order is by name; sim order is [left, right, rear].
SIM_ORDER = ["left_wheel_j", "right_wheel_j", "rear_wheel_j"]


def _stamp(msg, fallback_ns: int) -> float:
    """Header stamp in seconds, falling back to the bag receive time if unset."""
    s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return s if s > 0 else fallback_ns * 1e-9


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (xyzw). q: [N,4], v: [3] -> [N,3]."""
    u, w = q[:, :3], q[:, 3:4]
    uv = np.cross(u, np.broadcast_to(v, u.shape))
    return v + 2.0 * (w * uv + np.cross(u, uv))


def extract(bag: str) -> dict:
    joint_cols = None
    t_sp, v_sp = [], []
    t_js, v_js = [], []
    t_od, p_od, q_od = [], [], []

    with AnyReader([BAGS_DIR / bag]) as reader:
        conns = [c for c in reader.connections
                 if c.topic in ("/joint_setpoints", "/joint_states", "/odin1/odometry")]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/odin1/odometry":
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                t_od.append(_stamp(msg, ts))
                p_od.append((p.x, p.y, p.z))
                q_od.append((q.x, q.y, q.z, q.w))
            else:
                if joint_cols is None:
                    names = list(msg.name)
                    joint_cols = [names.index(n) for n in SIM_ORDER]
                vel = np.asarray(msg.velocity, dtype=np.float64)[joint_cols]
                if conn.topic == "/joint_setpoints":
                    t_sp.append(_stamp(msg, ts))
                    v_sp.append(vel)
                else:
                    t_js.append(_stamp(msg, ts))
                    v_js.append(vel)

    t_sp = np.asarray(t_sp)
    t_js = np.asarray(t_js)
    t_od = np.asarray(t_od)
    p_od = np.asarray(p_od)
    q_od = np.asarray(q_od)

    # One common clock zero for every stream: the earliest sample in the bag.
    t0 = min(t_sp[0], t_js[0], t_od[0])

    # odin1_base_link -> base_link (pure translation in the odin body frame).
    p_base = p_od + _quat_rotate(q_od, ODIN_TO_BASE)

    # Streams must be time-sorted for np.interp consumers; header stamps arrive
    # slightly out of order when messages are batched.
    for t_arr, others in ((t_sp, [v_sp]), (t_js, [v_js])):
        order = np.argsort(t_arr, kind="stable")
        t_arr[:] = t_arr[order]
        others[0][:] = [others[0][i] for i in order]
    od_order = np.argsort(t_od, kind="stable")
    t_od, p_base, q_od = t_od[od_order], p_base[od_order], q_od[od_order]

    return {
        "bag": bag,
        "control": {  # commanded wheel speeds, sim order [left, right, rear], rad/s
            "t": (t_sp - t0).tolist(),
            "lrr": np.stack(v_sp).tolist(),
        },
        "measured": {  # measured wheel speeds, same order — diagnostics / --drive measured
            "t": (t_js - t0).tolist(),
            "lrr": np.stack(v_js).tolist(),
        },
        "real": {  # base_link pose in odom_odin, quaternions xyzw
            "t": (t_od - t0).tolist(),
            "pos": p_base.tolist(),
            "quat_xyzw": q_od.tolist(),
        },
        "meta": {
            "t0_unix": t0,
            "odin_to_base_link": ODIN_TO_BASE.tolist(),
            "surface": "outdoor_flat",
            "pose_source": "/odin1/odometry (lidar-inertial), 14.5 Hz",
            "duration_s": float(max(t_sp[-1], t_od[-1]) - t0),
        },
    }


def main() -> None:
    bags = sys.argv[1:] or DEFAULT_BAGS
    DATA_DIR.mkdir(exist_ok=True)
    for bag in bags:
        gt = extract(bag)
        out = DATA_DIR / f"gt_{bag}.json"
        with open(out, "w") as f:
            json.dump(gt, f)
        n_sp = len(gt["control"]["t"])
        n_od = len(gt["real"]["t"])
        print(f"{bag}: {gt['meta']['duration_s']:.1f}s, {n_sp} setpoints, "
              f"{n_od} poses -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
