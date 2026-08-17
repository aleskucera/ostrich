"""Convert campaign-2 mcap bags (Odin localization) into box GT JSONs.

Campaign 2 (2026-08-17, bags/ostrich*) replaces the total-station prism with
the Odin lidar-inertial unit: full 6-DoF pose at ~14 Hz with no occlusion
gaps, plus /joint_setpoints (~100 Hz) for the wheel commands. This converter
emits the SAME GT JSON schema as prepare_gt.py, so the engine sweeps and the
pre-registered quality gate run unchanged.

Conventions (verified against the bags and helhest_stack docs, 2026-08-17):
  - /joint_setpoints names are [left_wheel_j, rear_wheel_j, right_wheel_j],
    velocities ALL-POSITIVE-forward (post the 2026-07-27 LLC fix; the old
    [-L,-rear,+R] echo convention is gone). GT "lrr" order is [left, right,
    rear] -> reorder [0, 2, 1], no sign flips.
  - Trajectory: map -> odom_odin -> odin1_base_link (dynamic) composed with
    the static odin1_base_link -> base_link mount, i.e. the robot base frame.
    prism_offset is therefore (0,0,0): the sim tracks the chassis origin.
  - Alignment matches align_real_to_sim: start at origin, initial heading +X,
    z relative to start; yaw_rel relative to first sample.
  - Box: half-extents from campaign 1 (same physical box); center x/y
    estimated from the climb interval (z above half the box height) along the
    aligned trajectory.

Run with an interpreter that has `rosbags` (e.g. helhest_stack's venv):
    ~/projects/helhest_stack/.venv/bin/python \
        experiments/1_sim_to_real_box/prepare_gt_mcap.py --bags-dir \
        ~/projects/helhest_stack/bags --runs ostrich0 ostrich1 ostrich2 ostrich3
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))

DATA_DIR = pathlib.Path(__file__).parent / "data"
DT_RECORD = 0.01
BOX_HALF_EXTENTS = [0.37, 0.575, 0.06]  # campaign-1 box (same physical box)


def q_rot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def q_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def yaw_of(q):
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def build_gt(bag_dir: pathlib.Path) -> dict:
    from rosbags.highlevel import AnyReader

    with AnyReader([bag_dir]) as reader:
        # --- static odin -> base mount ---
        mount_t, mount_q = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections if c.topic == "/tf_static"]):
            m = reader.deserialize(raw, conn.msgtype)
            for tr in m.transforms:
                if (tr.header.frame_id == "odin1_base_link"
                        and tr.child_frame_id == "base_link"):
                    t, r = tr.transform.translation, tr.transform.rotation
                    mount_t = np.array([t.x, t.y, t.z])
                    mount_q = np.array([r.x, r.y, r.z, r.w])

        # --- dynamic tf chain ---
        mo, ob = {}, {}
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections if c.topic == "/tf"]):
            m = reader.deserialize(raw, conn.msgtype)
            for tr in m.transforms:
                t, r = tr.transform.translation, tr.transform.rotation
                rec = (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
                if tr.child_frame_id == "odom_odin":
                    mo[ts] = rec
                elif tr.child_frame_id == "odin1_base_link":
                    ob[ts] = rec
        mo_t = np.array(sorted(mo))
        T, P, Y = [], [], []
        for ts in sorted(ob):
            i = min(np.searchsorted(mo_t, ts), len(mo_t) - 1)
            p1, q1 = mo[mo_t[i]]
            p2, q2 = ob[ts]
            # map -> odin1_base_link, then apply the static mount
            p_odin = p1 + q_rot(q1, p2)
            q_odin = q_mul(q1, q2)
            p_base = p_odin + q_rot(q_odin, mount_t)
            q_base = q_mul(q_odin, mount_q)
            T.append(ts / 1e9)
            P.append(p_base)
            Y.append(yaw_of(q_base))
        T = np.array(T)
        P = np.array(P)
        Y = np.unwrap(np.array(Y))

        # --- wheel setpoints ---
        st, sv = [], []
        names = None
        for conn, ts, raw in reader.messages(
                connections=[c for c in reader.connections if c.topic == "/joint_setpoints"]):
            m = reader.deserialize(raw, conn.msgtype)
            if names is None:
                names = list(m.name)
            st.append(ts / 1e9)
            sv.append(list(m.velocity))
        st = np.array(st)
        sv = np.array(sv)
        assert names == ["left_wheel_j", "rear_wheel_j", "right_wheel_j"], names
        sv_lrr = sv[:, [0, 2, 1]]  # -> [left, right, rear], all-positive-forward

    # --- common time base: start at first setpoint or first pose ---
    t0 = max(T[0], st[0])
    T -= t0
    st -= t0
    keep = T >= 0.0
    T, P, Y = T[keep], P[keep], Y[keep]

    # --- align: start at origin, initial heading +X, z relative ---
    yaw0 = Y[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    R = np.array([[c, -s], [s, c]])
    xy = (R @ (P[:, :2] - P[0, :2]).T).T
    z = P[:, 2] - P[0, 2]
    yaw_rel = Y - yaw0

    # --- command grid ---
    duration = float(min(T[-1], st[-1]))
    t_grid = np.arange(0.0, duration, DT_RECORD)
    lrr = np.zeros((len(t_grid), 3), dtype=np.float64)
    for d in range(3):
        lrr[:, d] = np.interp(t_grid, st, sv_lrr[:, d])

    # --- box center from the climb interval ---
    hz = BOX_HALF_EXTENTS[2]
    on_box = z > hz  # above half the box height
    if on_box.any():
        box_x = float(np.mean(xy[on_box, 0]))
        box_y = float(np.mean(xy[on_box, 1]))
    else:
        box_x, box_y = float("nan"), float("nan")

    return {
        "source": "real_robot_box_campaign2_odin",
        "run_id": bag_dir.name,
        "robot": "helhest_junior",
        "dt_record": DT_RECORD,
        "duration_s": duration,
        "box": {"center": [box_x, box_y, hz], "half_extents": BOX_HALF_EXTENTS},
        # base_link expressed in the sim chassis (front-axle) frame: the tf
        # base_link rides ~0.23 m AHEAD of the front axle (Odin nose mast).
        # Measured by registering sim-vs-real climb onset against the
        # lidar-fitted pallet near face over ostrich0-3 (dx = +0.217..+0.245,
        # +-0.015). The sim must track this point AND shift the box by it.
        "prism_offset": [0.23, 0.0, 0.0],
        "control": {"t": t_grid.tolist(), "lrr": lrr.tolist()},
        "real": {
            "t": T.tolist(),
            "x": xy[:, 0].tolist(),
            "y": xy[:, 1].tolist(),
            "z": z.tolist(),
            "yaw_rel": yaw_rel.tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags-dir", type=pathlib.Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    # prepare_gt imports the sim replay stack at module level; stub it so the
    # gate is importable from a plain rosbags environment.
    import types
    for name in ("examples", "examples.helhest_junior"):
        sys.modules.setdefault(name, types.ModuleType(name))
    stub = types.ModuleType("examples.helhest_junior.replay_real")
    for a in ("BOX_CENTER", "BOX_HALF_EXTENTS", "PRISM_OFFSET", "SYNCED_DIR",
              "align_real_to_sim", "load_setpoints"):
        setattr(stub, a, None)
    sys.modules["examples.helhest_junior.replay_real"] = stub
    from prepare_gt import quality_check  # noqa: E402 (pre-registered gate)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        gt = build_gt(args.bags_dir / run)
        gt["quality"] = quality_check(gt)
        out = DATA_DIR / f"{run}.json"
        with open(out, "w") as f:
            json.dump(gt, f)
        verdict = "CLEAN" if gt["quality"]["clean"] else "REJECTED"
        fails = [k for k, c in gt["quality"]["checks"].items() if not c["pass"]]
        print(f"{run}: dur={gt['duration_s']:.1f}s "
              f"box_x={gt['box']['center'][0]:.2f} -> {out.name} "
              f"[{verdict}{': ' + ','.join(fails) if fails else ''}]")


if __name__ == "__main__":
    main()
