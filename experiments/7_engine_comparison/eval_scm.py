"""Chrono SCM deformable-terrain evaluation: GT bags + holdout + turn gains.

    .venv/bin/python experiments/7_engine_comparison/eval_scm.py

Runs the SCM (Bekker/Janosi) terrain variant of the Chrono runner (via the
source build) on the three GT bags, the motors0 holdout, and the four
turn-radius pairs; writes results/scm_chrono.json.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common
from engines import bridge

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--phi", type=float, default=None, help="Mohr friction angle override [deg]")
_args = _ap.parse_args()

DT = 2e-3
PARAMS = {"terrain": "scm"}
if _args.phi is not None:
    PARAMS["scm_phi"] = _args.phi
TURN_PAIRS = [(1.0, 3.0), (1.5, 3.5), (2.0, 4.0), (0.5, 3.5)]
HALF_TRACK, WHEEL_RADIUS = 0.365, 0.35

jobs, gts = [], {}
for bag in common.GT_BAGS + [common.HOLDOUT_BAG]:
    gt = common.load_gt(bag)
    gts[bag] = gt
    jobs.append(bridge.make_job(f"bag_{bag}", common.prepare_commands(gt, DT), DT,
                                params=PARAMS))
for k, (wl, wr) in enumerate(TURN_PAIRS):
    cmds = np.tile([wl, wr, 0.5 * (wl + wr)], (int(8.0 / DT), 1))
    jobs.append(bridge.make_job(f"turn{k}", cmds, DT, params=PARAMS))

results = {r["id"]: r for r in bridge.run_chrono_scm(jobs, procs=4)}

out = {"engine": "chrono_scm", "params": PARAMS, "dt": DT, "bags": {}, "turn": []}
for bag, gt in gts.items():
    s = common.score_result(results[f"bag_{bag}"], gt)
    out["bags"][bag] = {k: s[k] for k in ("combined_mean", "combined_median",
                                          "yaw_rmse_deg_mean", "n_windows", "stable")}
    print(f"{bag}: combined {s['combined_mean']:.3f} m, yaw {s['yaw_rmse_deg_mean']:.1f} deg,"
          f" stable={s['stable']}")
for k, (wl, wr) in enumerate(TURN_PAIRS):
    r = results[f"turn{k}"]
    pose = np.asarray(r["pose"])
    t = np.arange(pose.shape[0]) * DT
    yaw = np.unwrap(common.yaw_from_quat_xyzw(pose[:, 3:7]))
    m = t > 2.0
    wz = (yaw[m][-1] - yaw[m][0]) / (t[m][-1] - t[m][0])
    ideal = WHEEL_RADIUS * (wr - wl) / (2 * HALF_TRACK)
    alpha = ideal / wz if abs(wz) > 1e-4 else None
    out["turn"].append({"pair": [wl, wr], "alpha": alpha})
    print(f"turn ({wl},{wr}): alpha={alpha:.2f}" if alpha else f"turn ({wl},{wr}): no yaw")

gt_errs = [out["bags"][b]["combined_mean"] for b in common.GT_BAGS]
out["error_gt_mean"] = float(np.mean(gt_errs))
suffix = f"_phi{_args.phi:g}" if _args.phi is not None else ""
with open(common.RESULTS_DIR / f"scm_chrono{suffix}.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"GT-mean {out['error_gt_mean']:.3f} m -> results/scm_chrono.json")
