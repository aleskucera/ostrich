"""Extract ground truth trajectory and wheel velocities from Helhest rosbags.

Reads NUC rosbags recorded on the real Helhest robot and extracts:
  1. Ground truth xy trajectory from /local_odom
  2. Actual per-wheel angular velocities from /joint_states
  3. Commanded twist from /cmd_vel

Output JSON can be used directly as ground truth for simulator parameter sweeps.

Requirements (optional, only needed to run this script):
    uv pip install mcap rosbags

Usage:
    python examples/comparison_accuracy/helhest/extract_rosbag.py
    python examples/comparison_accuracy/helhest/extract_rosbag.py --bag-dir data/rosbags/nuc
    python examples/comparison_accuracy/helhest/extract_rosbag.py --bag helhest_2026_04_10-14_46_18
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print(
        "ERROR: rosbags not installed. Install with:\n"
        "  uv pip install mcap rosbags\n"
        "This dependency is optional and only needed for rosbag extraction.",
        file=sys.stderr,
    )
    sys.exit(1)


ROSBAG_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "rosbags" / "nuc"
OUTPUT_DIR = pathlib.Path(__file__).parent

# Wheel geometry from URDF (embedded in rosbags and matching HelhestConfig)
WHEEL_RADIUS = 0.36
TRACK_WIDTH = 0.72  # left_wheel y=+0.36, right_wheel y=-0.36


def quat_to_yaw(qw, qx, qy, qz):
    """Extract yaw from quaternion (ROS convention: x,y,z,w in message but we pass w first)."""
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def extract_bag(bag_path: pathlib.Path) -> dict | None:
    """Extract trajectory, wheel velocities, and cmd_vel from a single rosbag.

    Returns None if the bag lacks /joint_states or /local_odom.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(str(bag_path)) as reader:
        topic_names = {conn.topic for conn in reader.connections}

        has_joints = "/joint_states" in topic_names
        has_local_odom = "/local_odom" in topic_names
        has_cmd_vel = "/cmd_vel" in topic_names

        if not has_local_odom:
            print(f"  SKIP {bag_path.name}: no /local_odom")
            return None

        cmd_vel_msgs = []
        local_odom_msgs = []
        joint_state_msgs = []

        for conn, timestamp, rawdata in reader.messages():
            t = timestamp / 1e9

            if conn.topic == "/cmd_vel":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                cmd_vel_msgs.append(
                    (t, msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)
                )

            elif conn.topic == "/local_odom":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                v = msg.twist.twist
                local_odom_msgs.append((t, p.x, p.y, p.z, o.w, o.x, o.y, o.z, v.linear.x, v.angular.z))

            elif conn.topic == "/joint_states":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                names = list(msg.name)
                vels = [float(v) for v in msg.velocity]
                joint_state_msgs.append((t, names, vels))

    # --- Determine active driving window ---
    if cmd_vel_msgs:
        cmd_start = cmd_vel_msgs[0][0]
        cmd_end = cmd_vel_msgs[-1][0]
        cmd_duration = cmd_end - cmd_start
        unique_cmds = set()
        for _, vx, vy, wz in cmd_vel_msgs:
            unique_cmds.add((round(vx, 4), round(vy, 4), round(wz, 4)))
    elif joint_state_msgs:
        # No cmd_vel — fall back to joint_states window
        print(f"  NOTE {bag_path.name}: no /cmd_vel, using joint_states window")
        cmd_start = joint_state_msgs[0][0]
        cmd_end = joint_state_msgs[-1][0]
        cmd_duration = cmd_end - cmd_start
        unique_cmds = set()
    else:
        print(f"  SKIP {bag_path.name}: no /cmd_vel and no /joint_states")
        return None

    # --- Extract local_odom trajectory during active driving ---
    # Include 0.5s before cmd_start (to capture initial pose) and 0.5s after
    active_odom = [
        m for m in local_odom_msgs if cmd_start - 0.5 <= m[0] <= cmd_end + 0.5
    ]

    if len(active_odom) < 5:
        print(f"  SKIP {bag_path.name}: too few local_odom msgs during driving ({len(active_odom)})")
        return None

    # Compute trajectory relative to starting pose
    t0 = active_odom[0][0]
    x0, y0 = active_odom[0][1], active_odom[0][2]
    yaw0 = quat_to_yaw(active_odom[0][4], active_odom[0][5], active_odom[0][6], active_odom[0][7])

    # Rotate trajectory so initial heading is along +x
    cos_y0 = math.cos(-yaw0)
    sin_y0 = math.sin(-yaw0)

    trajectory = []
    for m in active_odom:
        t_rel = m[0] - t0
        dx, dy = m[1] - x0, m[2] - y0
        x_rot = cos_y0 * dx - sin_y0 * dy
        y_rot = sin_y0 * dx + cos_y0 * dy
        yaw = quat_to_yaw(m[4], m[5], m[6], m[7]) - yaw0
        trajectory.append({
            "t": round(t_rel, 4),
            "x": round(x_rot, 5),
            "y": round(y_rot, 5),
            "yaw": round(yaw, 5),
            "vx": round(m[8], 4),
            "wz": round(m[9], 4),
        })

    # --- Extract wheel velocities during steady-state driving ---
    wheel_data = None
    if has_joints and joint_state_msgs:
        # Skip first 2s to avoid transient acceleration
        steady_start = cmd_start + 2.0
        steady_end = cmd_end - 0.5
        if steady_end <= steady_start:
            steady_start = cmd_start + 0.5
            steady_end = cmd_end

        steady_joints = [
            m for m in joint_state_msgs if steady_start <= m[0] <= steady_end
        ]

        if steady_joints:
            # Determine joint name ordering
            names = steady_joints[0][1]
            left_idx = names.index("left_wheel_j")
            right_idx = names.index("right_wheel_j")
            rear_idx = names.index("rear_wheel_j")

            left_vels = [m[2][left_idx] for m in steady_joints]
            right_vels = [m[2][right_idx] for m in steady_joints]
            rear_vels = [m[2][rear_idx] for m in steady_joints]

            wheel_data = {
                "left_wheel_vel": {
                    "mean": round(float(np.mean(left_vels)), 4),
                    "std": round(float(np.std(left_vels)), 4),
                },
                "right_wheel_vel": {
                    "mean": round(float(np.mean(right_vels)), 4),
                    "std": round(float(np.std(right_vels)), 4),
                },
                "rear_wheel_vel": {
                    "mean": round(float(np.mean(rear_vels)), 4),
                    "std": round(float(np.std(rear_vels)), 4),
                },
                "num_samples": len(steady_joints),
                "steady_state_window": [
                    round(steady_start - t0, 2),
                    round(steady_end - t0, 2),
                ],
            }

            # Also store the full wheel velocity timeseries (during active driving)
            all_joints = [
                m for m in joint_state_msgs if cmd_start - 0.5 <= m[0] <= cmd_end + 0.5
            ]
            wheel_timeseries = []
            for m in all_joints:
                wheel_timeseries.append({
                    "t": round(m[0] - t0, 4),
                    "left": round(m[2][left_idx], 4),
                    "right": round(m[2][right_idx], 4),
                    "rear": round(m[2][rear_idx], 4),
                })
            wheel_data["timeseries"] = wheel_timeseries

    # --- Summary ---
    traj_end = trajectory[-1]
    dist = math.sqrt(traj_end["x"] ** 2 + traj_end["y"] ** 2)

    result = {
        "source": "real_robot",
        "bag_name": bag_path.name,
        "cmd_vel": {
            "commands": [list(c) for c in sorted(unique_cmds)],
            "duration_s": round(cmd_duration, 2),
            "num_messages": len(cmd_vel_msgs),
        },
        "trajectory": {
            "frame": "local_odom (rotated to initial heading along +x)",
            "duration_s": round(trajectory[-1]["t"], 2),
            "displacement_m": round(dist, 3),
            "num_points": len(trajectory),
            "rate_hz": round(len(trajectory) / trajectory[-1]["t"], 1) if trajectory[-1]["t"] > 0 else 0,
            "points": trajectory,
        },
    }

    if wheel_data is not None:
        result["wheel_velocities"] = wheel_data
        wl = wheel_data["left_wheel_vel"]["mean"]
        wr = wheel_data["right_wheel_vel"]["mean"]
        result["target_ctrl_rad_s"] = [
            round(wl, 4),
            round(wr, 4),
            round(wheel_data["rear_wheel_vel"]["mean"], 4),
        ]

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bag-dir",
        type=pathlib.Path,
        default=ROSBAG_DIR,
        help=f"Directory containing NUC rosbags (default: {ROSBAG_DIR})",
    )
    parser.add_argument(
        "--bag",
        type=str,
        default=None,
        help="Process only this bag name (e.g. helhest_2026_04_10-14_46_18)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help=f"Output directory for JSON files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not args.bag_dir.is_dir():
        print(f"ERROR: bag directory not found: {args.bag_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.bag:
        bag_dirs = [args.bag_dir / args.bag]
    else:
        bag_dirs = sorted(d for d in args.bag_dir.iterdir() if d.is_dir())

    for bag_dir in bag_dirs:
        if not bag_dir.is_dir():
            print(f"SKIP: not a directory: {bag_dir}")
            continue

        print(f"Processing {bag_dir.name}...")
        result = extract_bag(bag_dir)

        if result is None:
            continue

        out_path = args.output_dir / f"{bag_dir.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> {out_path}")

        # Print summary
        cmd = result["cmd_vel"]["commands"]
        traj = result["trajectory"]
        print(f"  cmd_vel: {cmd} for {result['cmd_vel']['duration_s']}s")
        print(f"  trajectory: {traj['num_points']} pts, {traj['duration_s']}s, {traj['displacement_m']}m")
        if "wheel_velocities" in result:
            wv = result["wheel_velocities"]
            print(
                f"  wheels (steady-state): "
                f"L={wv['left_wheel_vel']['mean']:.3f} "
                f"R={wv['right_wheel_vel']['mean']:.3f} "
                f"Rear={wv['rear_wheel_vel']['mean']:.3f} rad/s"
            )
            print(f"  target_ctrl: {result['target_ctrl_rad_s']}")
        print()


if __name__ == "__main__":
    main()
