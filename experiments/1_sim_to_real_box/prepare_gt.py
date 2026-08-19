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

# --- Pre-registered clean-run criteria (fixed 2026-08-15, BEFORE the second
# data-collection campaign; see COLLECTION_PROTOCOL.md). A run is "clean" iff
# every check passes; the verdict is stamped into the GT JSON so run selection
# is code, not prose. Reviewers asked how the two 2026-05-20 runs were chosen;
# these thresholds formalize the criteria that were previously applied by
# inspection (tracking dropouts and yaw folds).
# Calibrated against campaign 1 (2026-05-20): good crossers show 0.75-0.79 s
# total-station occlusion during the climb and ~0.94 coverage; the
# historically-excluded runs sit at 0.81-0.88 coverage / >=0.79 s gaps.
QC_MIN_COVERAGE = 0.90      # valid prism samples / expected samples
QC_MAX_GAP_S = 0.85         # max total-station dropout gap
QC_MAX_YAW_STEP = 0.5       # rad between consecutive samples (fold detector)
QC_MIN_X_PROGRESS = None    # set at runtime: box far edge + 0.3 m (crossed it)
QC_MIN_Z_PEAK = 0.06        # m relative climb peak (actually climbed the box)
QC_MIN_MEAN_CMD = 0.5       # rad/s mean |wheel cmd| (robot actually driven)


def quality_check(gt: dict) -> dict:
    rt = np.asarray(gt["real"]["t"])
    x = np.asarray(gt["real"]["x"])
    z = np.asarray(gt["real"]["z"])
    yaw = np.asarray(gt["real"]["yaw_rel"])
    cmds = np.asarray(gt["control"]["lrr"])
    box = gt["box"]

    span = rt[-1] - rt[0] if len(rt) > 1 else 0.0
    expected = span / np.median(np.diff(rt)) if len(rt) > 2 else 1.0
    coverage = len(rt) / max(expected, 1.0)
    max_gap = float(np.max(np.diff(rt))) if len(rt) > 1 else float("inf")
    max_yaw_step = float(np.max(np.abs(np.diff(yaw)))) if len(yaw) > 1 else 0.0
    x_target = box["center"][0] + box["half_extents"][0] + 0.3
    checks = {
        "coverage": (float(min(coverage, 1.0)), coverage >= QC_MIN_COVERAGE),
        "max_gap_s": (max_gap, max_gap <= QC_MAX_GAP_S),
        "max_yaw_step_rad": (max_yaw_step, max_yaw_step <= QC_MAX_YAW_STEP),
        "x_progress_m": (float(np.max(x)), float(np.max(x)) >= x_target),
        "z_peak_m": (float(np.max(z)), float(np.max(z)) >= QC_MIN_Z_PEAK),
        "mean_abs_cmd": (float(np.mean(np.abs(cmds))),
                         float(np.mean(np.abs(cmds))) >= QC_MIN_MEAN_CMD),
    }
    return {"clean": bool(all(ok for _, ok in checks.values())),
            "checks": {k: {"value": float(v), "pass": bool(ok)}
                       for k, (v, ok) in checks.items()}}


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
        gt["quality"] = quality_check(gt)
        out = DATA_DIR / f"{h5_path.stem}.json"
        with open(out, "w") as f:
            json.dump(gt, f)
        verdict = "CLEAN" if gt["quality"]["clean"] else "REJECTED"
        fails = [k for k, c in gt["quality"]["checks"].items() if not c["pass"]]
        print(f"{gt['run_id']}: {len(gt['control']['t'])} cmd samples, "
              f"{len(gt['real']['t'])} valid real points -> {out.name}  "
              f"[{verdict}{': ' + ','.join(fails) if fails else ''}]")


if __name__ == "__main__":
    main()
