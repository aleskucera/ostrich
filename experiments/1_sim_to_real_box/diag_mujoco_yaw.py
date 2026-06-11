"""Diagnostic: does MuJoCo's tuned config suppress the real turn?

Runs a focused MuJoCo grid over torsional friction (tor) + condim + mu and
scores every config under TWO metrics:

  - current      : combined_with_yaw, lever arm L=0.5 m (what the sweep used)
  - yaw-honest   : same form but L=2.0 m, so missing heading is penalised hard

For each config we also report the raw chassis yaw response (maxabs, deg) and
lateral excursion, plus the real run's yaw, so we can see directly whether the
"best" config is the one that flattens the turn.

CPU-only (MuJoCo); safe to run on the laptop. ~24 configs x 2 runs.
"""
import itertools
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from common_box import load_gt, score, DATA_DIR
from sweep_mujoco import simulate, BASE_PARAMS

RUNS = ["run_2026_05_20-18_10_33", "run_2026_05_20-18_04_51"]
DT = 0.002
KV = 1000
SOLREF0 = 0.005
L_CURRENT = 0.5
L_HONEST = 2.0

# Grid — fine torsional sweep (the transition 0->0.5 is a cliff; resolve it)
# x both wheel geometries (cylinder vs capsule).
CONDIMS = [6]
MUS = [1.2, 1.5]
TORS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
WHEEL_GEOMS = ["cylinder", "capsule"]


def raw_yaw_maxabs_deg(pose):
    q = pose[:, 3:7]
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yr = yaw - yaw[0]
    return float(np.degrees(np.abs(yr).max())), float(np.abs(pose[:, 1]).max())


def lateral_excursion(pose, dt, t_skip=0.5):
    """Lateral (y) peak-to-peak after the initial settle transient."""
    i0 = int(t_skip / dt)
    sy = pose[i0:, 1]
    return float(sy.max() - sy.min())


def main():
    gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in RUNS}
    box = next(iter(gts.values()))["box"]
    box_geom = dict(box_x=box["center"][0], box_y=box["center"][1], box_z=box["center"][2],
                    box_hx=box["half_extents"][0], box_hy=box["half_extents"][1],
                    box_hz=box["half_extents"][2])

    real_exc = {}
    print("Real per-run targets:")
    for r, gt in gts.items():
        yr = np.asarray(gt["real"]["yaw_rel"])
        ry = np.asarray(gt["real"]["y"])
        ry = ry[np.isfinite(ry)]
        real_exc[r] = float(ry.max() - ry.min())
        print(f"  {r}: yaw maxabs {np.degrees(np.abs(yr).max()):.2f} deg, "
              f"lateral excursion {real_exc[r]*1000:.0f} mm")
    real_exc_avg = float(np.mean(list(real_exc.values())))
    print(f"  AVG real lateral excursion: {real_exc_avg*1000:.0f} mm")
    print()

    rows = []
    for geom, mu, tor in itertools.product(WHEEL_GEOMS, MUS, TORS):
        params = {**BASE_PARAMS, **box_geom, "dt": DT, "kv": KV,
                  "ground_friction": mu, "box_friction": mu,
                  "front_friction": mu, "rear_friction": mu,
                  "ground_torsional": tor, "front_torsional": tor, "rear_torsional": tor,
                  "solref0": SOLREF0, "condim": CONDIMS[0], "integrator": "implicitfast",
                  "wheel_geom": geom}
        per_run = {}
        for r, gt in gts.items():
            pose = simulate(params, gt)
            s = score(pose, DT, gt)
            yaw_max, y_max = raw_yaw_maxabs_deg(pose)
            exc = lateral_excursion(pose, DT)
            pos = s["combined"]
            yrad = s["yaw_rmse_rad"]
            per_run[r] = dict(
                pos=pos, yaw_rmse_deg=s["yaw_rmse_deg"], yaw_max_deg=yaw_max,
                exc=exc, exc_ratio=exc / max(real_exc[r], 1e-6),
                cur=float(np.sqrt(pos**2 + (L_CURRENT * yrad) ** 2)),
                honest=float(np.sqrt(pos**2 + (L_HONEST * yrad) ** 2)),
            )
        agg = {k: float(np.mean([per_run[r][k] for r in RUNS]))
               for k in ("pos", "yaw_rmse_deg", "yaw_max_deg", "exc", "cur", "honest")}
        rows.append(dict(geom=geom, mu=mu, tor=tor, **agg))

    # Table — grouped by geom
    for geom in WHEEL_GEOMS:
        print(f"### wheel_geom = {geom}  (real avg excursion {real_exc_avg*1000:.0f} mm) ###")
        hdr = (f"{'mu':>4} {'tor':>5} | {'pos':>6} {'yawRMSE':>8} {'yawMax':>7} "
               f"{'exc_mm':>7} {'%real':>6} | {'cur(L.5)':>9} {'honest(L2)':>11}")
        print(hdr)
        print("-" * len(hdr))
        for row in [r for r in rows if r["geom"] == geom]:
            print(f"{row['mu']:>4} {row['tor']:>5} | "
                  f"{row['pos']:>6.3f} {row['yaw_rmse_deg']:>8.2f} "
                  f"{row['yaw_max_deg']:>7.2f} {row['exc']*1000:>7.1f} "
                  f"{row['exc']/real_exc_avg*100:>5.0f}% | "
                  f"{row['cur']:>9.4f} {row['honest']:>11.4f}")
        print()

    best_cur = min(rows, key=lambda r: r["cur"])
    best_honest = min(rows, key=lambda r: r["honest"])
    # Config whose lateral excursion best matches the real drift.
    best_exc = min(rows, key=lambda r: abs(r["exc"] - real_exc_avg))
    print(f"BEST under CURRENT (L=0.5): geom={best_cur['geom']} mu={best_cur['mu']} "
          f"tor={best_cur['tor']} -> cur={best_cur['cur']:.4f} "
          f"exc={best_cur['exc']*1000:.0f}mm yawMax={best_cur['yaw_max_deg']:.2f}deg")
    print(f"BEST under HONEST  (L=2.0): geom={best_honest['geom']} mu={best_honest['mu']} "
          f"tor={best_honest['tor']} -> honest={best_honest['honest']:.4f} "
          f"exc={best_honest['exc']*1000:.0f}mm yawMax={best_honest['yaw_max_deg']:.2f}deg")
    print(f"BEST drift MATCH         : geom={best_exc['geom']} mu={best_exc['mu']} "
          f"tor={best_exc['tor']} -> exc={best_exc['exc']*1000:.0f}mm "
          f"(real {real_exc_avg*1000:.0f}mm)  pos={best_exc['pos']:.3f} "
          f"yawRMSE={best_exc['yaw_rmse_deg']:.2f}deg")


if __name__ == "__main__":
    main()
