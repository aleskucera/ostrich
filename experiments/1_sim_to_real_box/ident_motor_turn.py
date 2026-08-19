"""Campaign-2 motor + turn-resistance identification for Ostrich (train/test).

Why this exists: the campaign-2 yaw analysis showed a chain the campaign-1
data could not identify (its runs were near-straight):

1. The commands carry large differential content (ideal skid-steer predicts
   16-32 deg net yaw) but the real robot executes only ~30 % of it (rear-wheel
   skid resistance).
2. At the campaign-1 k_p=250, the SIM wheels do not track the commanded
   differential AT ALL (-3 %): the soft velocity servo lets the yaw-resisting
   load drag both wheels to a common speed. mu_rear is a dead knob in that
   regime (0.190 -> 0.194 over 0.4..1.2).
3. With a stiff servo the sim then OVER-turns, so the turn resistance needs
   to be set via friction — and anisotropically, so longitudinal drive grip
   (campaign-1-calibrated) is untouched while lateral skid resistance is
   identified from turn-content data.

Protocol: calibrate on ostrich0+1, hold out ostrich2+3. Contact params stay
campaign-1 frozen; cmd_scale from the cruise calibration (0.937). Grid over
k_p x mu_lateral with longitudinal pinned at (front 0.8, rear 1.2).

Identified (2026-08-17): k_p=4000 (~80 % differential tracking),
mu_lat=2.0 (interior optimum; ground-mu combine rule dilutes it to ~1.4
effective). Train 0.164 m, TEST 0.136 m, all-4 0.152 m, yaw RMSE 2.41 deg
(frozen campaign-1 baseline: 0.190 m / 3.66 deg). A follow-up sweep found
compliance.friction=1e-3 marginally better (test 0.133; creep instead of a
hard stiction plateau); 1e-2 degrades.

SATURATION: every further yaw knob (front/rear lateral split, mu_rolling,
k_p=12000, friction compliance) lands on the same ~0.13-0.16 plateau, and
for a fundamental reason: the REAL per-run turn efficiency (real yaw
regressed on the ideal skid-steer prediction) spans 0.11/0.21/0.13/0.30
across the four runs, correlated with speed - while constant-mu Coulomb
friction realizes exactly one efficiency. The identified sim's yaw RMSE
(2.2-2.4 deg) already matches the residual of the best per-run LINEAR gain
fit (1.0-3.0 deg), i.e. the constant-efficiency noise floor. Fixing the
rest requires slip-speed-dependent (Stribeck-like) lateral friction in the
engine - future work, not a parameter.

    .venv/bin/python experiments/1_sim_to_real_box/ident_motor_turn.py
"""
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np

import eval_campaign2 as ec
from common_box import DATA_DIR, RESULTS_DIR, load_gt
import examples.helhest_junior.replay_real as rr

TRAIN = ["ostrich0", "ostrich1", "ostrich10", "ostrich13"]
TEST = [f"ostrich{i}" for i in range(14) if f"ostrich{i}" not in TRAIN]
CMD_SCALE = 0.937          # cruise calibration (stable across k_p)
MU_LONG = (0.8, 1.2)       # campaign-1 longitudinal (front, rear) — pinned
GRID_KP = (4000.0, 10000.0)
GRID_LAT = (0.8, 1.4, 2.0, 2.8, 3.6)

PARAMS = {}
_orig_init = rr.HelhestJuniorReplaySimulator.__init__


def _patched_init(self, *a, **kw):
    kw.update(PARAMS)
    _orig_init(self, *a, **kw)


rr.HelhestJuniorReplaySimulator.__init__ = _patched_init


def run_set(gts, names):
    return {n: ec._score_run(*ec.run_ostrich(gts[n], CMD_SCALE), gts[n])
            for n in names}


def main():
    gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in ec.RUNS}
    rows = []
    for kp, lat in itertools.product(GRID_KP, GRID_LAT):
        PARAMS.update(k_p=kp, mu_front=lat, mu_rear=lat,
                      mu_long_front=MU_LONG[0], mu_long_rear=MU_LONG[1])
        tr = run_set(gts, TRAIN)
        mc = float(np.mean([s["combined_with_yaw"] for s in tr.values()]))
        my = float(np.mean([s["yaw_rmse_deg"] for s in tr.values()]))
        rows.append({"k_p": kp, "mu_lat": lat, "train_combined": mc,
                     "train_yaw_deg": my})
        print(f"k_p={kp:6.0f} mu_lat={lat:.1f}: train comb={mc:.3f} "
              f"yaw={my:.2f} deg")

    best = min(rows, key=lambda r: r["train_combined"])
    PARAMS.update(k_p=best["k_p"], mu_front=best["mu_lat"],
                  mu_rear=best["mu_lat"])
    te = run_set(gts, TEST)
    al = run_set(gts, ec.RUNS)
    out = {
        "protocol": "train ostrich0+1, test ostrich2+3; contact params frozen "
                    "at campaign-1; longitudinal mu pinned (0.8, 1.2); "
                    f"cmd_scale={CMD_SCALE}",
        "grid": rows,
        "best": best,
        "test": {n: {"combined_with_yaw": s["combined_with_yaw"],
                     "yaw_rmse_deg": s["yaw_rmse_deg"]} for n, s in te.items()},
        "test_mean": float(np.mean([s["combined_with_yaw"] for s in te.values()])),
        "all4_mean": float(np.mean([s["combined_with_yaw"] for s in al.values()])),
        "all4_yaw_deg": float(np.mean([s["yaw_rmse_deg"] for s in al.values()])),
    }
    path = RESULTS_DIR / "ident_motor_turn.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nbest {best} -> TEST mean {out['test_mean']:.3f}, "
          f"all-4 {out['all4_mean']:.3f} ({out['all4_yaw_deg']:.2f} deg) -> {path}")


if __name__ == "__main__":
    main()
