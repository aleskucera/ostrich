"""Convert synced rosbag box runs into box GT JSONs for the benchmark.

Reuses the replay tool's loaders so the wheel-command remap/sign and the
prism/heading alignment are identical to the validated single-engine replay.

Usage:
    # one run
    python experiments/1_sim_to_real_box/prepare_gt.py --run 18_04_51
    # all synced runs
    python experiments/1_sim_to_real_box/prepare_gt.py --all
"""
import argparse
import json
import pathlib

import numpy as np

from examples.helhest_junior.replay_real import (
    BOX_CENTER,
    BOX_HALF_EXTENTS,
    PRISM_OFFSET,
    SYNCED_DIR,
    align_real_to_sim,
    load_setpoints,
)

DATA_DIR = pathlib.Path(__file__).parent / "data"
DT_RECORD = 0.01     # fine grid the commands are stored on; engines resample
DURATION = 12.0


def build_gt(h5_path: pathlib.Path) -> dict:
    # Wheel commands already in sim order [L,R,rear] with the forward-sign flip.
    setpoints, real, run_id, t_grid = load_setpoints(h5_path, "setpoint", DT_RECORD, DURATION)
    real_aligned, real_t = align_real_to_sim(real)  # prism point: start at 0, heading +X, z rel

    valid = ~np.isnan(real_aligned[:, 0])
    ra, rt = real_aligned[valid], real_t[valid]
    # Yaw delta from start (radians). Frame-independent — only relative
    # rotation matters when comparing to sim chassis yaw.
    raw_yaw = real["yaw"][valid]
    yaw_rel = (raw_yaw - raw_yaw[0]).astype(np.float64)

    return {
        "source": "real_robot_box",
        "run_id": str(run_id),
        "robot": "helhest_junior",
        "dt_record": DT_RECORD,
        "duration_s": DURATION,
        "box": {"center": list(BOX_CENTER), "half_extents": list(BOX_HALF_EXTENTS)},
        "prism_offset": [float(v) for v in PRISM_OFFSET],
        "control": {  # wheel commands, sim DOF order [left,right,rear], forward=+
            "t": t_grid.tolist(),
            "lrr": setpoints.tolist(),
        },
        "real": {  # total-station prism point, aligned, valid samples only
            "t": rt.tolist(),
            "x": ra[:, 0].tolist(),
            "y": ra[:, 1].tolist(),
            "z": ra[:, 2].tolist(),
            "yaw_rel": yaw_rel.tolist(),  # rad, relative to first valid sample
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run id suffix, e.g. 18_04_51")
    ap.add_argument("--all", action="store_true", help="convert every synced run")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        runs = sorted(SYNCED_DIR.glob("run_*.h5"))
    elif args.run:
        runs = [SYNCED_DIR / f"run_2026_05_20-{args.run}.h5"]
    else:
        ap.error("pass --run <id> or --all")

    for h5_path in runs:
        gt = build_gt(h5_path)
        out = DATA_DIR / f"{h5_path.stem}.json"
        with open(out, "w") as f:
            json.dump(gt, f)
        print(f"{gt['run_id']}: {len(gt['control']['t'])} cmd samples, "
              f"{len(gt['real']['t'])} valid real points -> {out.name}")


if __name__ == "__main__":
    main()
