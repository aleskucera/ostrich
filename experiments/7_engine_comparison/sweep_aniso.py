"""Anisotropic wheel-friction sweep (ostrich mu_perp / AGX oriented friction).

    .venv/bin/python experiments/7_engine_comparison/sweep_aniso.py ostrich
    .venv/bin/python experiments/7_engine_comparison/sweep_aniso.py agx

Rationale: isotropic Coulomb forces lateral force = longitudinal force at equal
normal load, which pins the skid-steer turn gain far from the real alpha ~= 2
(isotropic-tuned ostrich/AGX sit at ~4.4). Anisotropy separates the two: keep
longitudinal traction high (real forward gain ~0.9-1.0) while LOWERING lateral
resistance to let the robot yaw. Each config scores the three GT bags AND the
four turn-gain pairs, so we see error and alpha move together.

Ostrich caveat: the contact combines wheel and ground mu by AVERAGING per axis,
so effective lateral mu = (mu_lat + mu_ground)/2 — grid values are wheel-side.

Results -> results/sweep_aniso_<engine>.json
"""

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

TURN_PAIRS = [(1.0, 3.0), (1.5, 3.5), (2.0, 4.0), (0.5, 3.5)]
HALF_TRACK, WHEEL_RADIUS = 0.365, 0.35
ENGINE_DT = {"ostrich": 0.02, "agx": 0.01}

# ostrich: mu_front/mu_rear are LATERAL when mu_long_* is set. Longitudinal held
# near the isotropic-best traction values; lateral swept downward to free yaw.
GRIDS = {
    # mu_rolling pinned to the isotropic-best 0.3 — torsional friction resists
    # yaw directly and the 0.7 default masked the lateral sweep entirely.
    # ground_mu 0.2: the contact AVERAGES wheel and ground mu per axis, so with
    # the default 0.8 ground the effective lateral could never drop below ~0.4
    # (v1 grid result: best 4.73 m, alpha 5-7.7). Low ground mu + boosted
    # wheel-side longitudinal reaches the AGX-winning effective (0.9 / 0.2).
    "ostrich": [
        {"mu_front": lat_f, "mu_rear": lat_r,
         "mu_long_front": 1.6, "mu_long_rear": 0.6, "mu_rolling": 0.3,
         "ground_mu": 0.2}
        for lat_f, lat_r in itertools.product((0.1, 0.2, 0.4), (0.05, 0.2))
    ],
    "agx": [
        {"mu_front": long_f, "mu_rear": long_r,
         "mu_lat_front": lat_f, "mu_lat_rear": lat_r,
         "oriented_friction": True}
        for (long_f, long_r), (lat_f, lat_r) in itertools.product(
            [(0.7, 0.4)], itertools.product((0.15, 0.3, 0.5), (0.1, 0.2, 0.4)))
    ],
}


def turn_alpha(res):
    pose = np.asarray(res["pose"])
    dt = res["dt"]
    t = np.arange(pose.shape[0]) * dt
    yaw = np.unwrap(common.yaw_from_quat_xyzw(pose[:, 3:7]))
    m = t > 2.0
    wz = (yaw[m][-1] - yaw[m][0]) / (t[m][-1] - t[m][0])
    return None if abs(wz) < 1e-4 else float(wz)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine", choices=("ostrich", "agx"))
    args = ap.parse_args()
    engine = args.engine
    dt = ENGINE_DT[engine]

    gts = {bag: common.load_gt(bag) for bag in common.GT_BAGS}
    rows = []
    for ci, params in enumerate(GRIDS[engine]):
        gmu = params.pop("ground_mu", 0.8)
        jobs = [bridge.make_job(f"c{ci}_{bag}", common.prepare_commands(gt, dt), dt,
                                params=params, ground_mu=gmu)
                for bag, gt in gts.items()]
        for k, (wl, wr) in enumerate(TURN_PAIRS):
            cmds = np.tile([wl, wr, 0.5 * (wl + wr)], (int(8.0 / dt), 1))
            jobs.append(bridge.make_job(f"c{ci}_turn{k}", cmds, dt, params=params,
                                        ground_mu=gmu))
        params = {**params, "ground_mu": gmu}
        if engine == "agx":
            results = bridge.run_agx(jobs, timeout=7200.0)
        else:
            from engines.ostrich_runner import run_ostrich
            results = run_ostrich(jobs)
        results = {r["id"]: r for r in results}

        per_bag = {bag: common.score_result(results[f"c{ci}_{bag}"], gt)
                   for bag, gt in gts.items()}
        stable = all(s["stable"] for s in per_bag.values())
        error = (float(np.mean([s["combined_mean"] for s in per_bag.values()]))
                 if stable else float("inf"))
        alphas = []
        for k, (wl, wr) in enumerate(TURN_PAIRS):
            wz = turn_alpha(results[f"c{ci}_turn{k}"])
            ideal = WHEEL_RADIUS * (wr - wl) / (2 * HALF_TRACK)
            alphas.append(None if wz is None else float(ideal / wz))
        rows.append({"params": params, "error": error, "stable": stable,
                     "alphas": alphas,
                     "per_bag": {b: {"combined_mean": s["combined_mean"],
                                     "yaw_rmse_deg_mean": s["yaw_rmse_deg_mean"]}
                                 for b, s in per_bag.items()}})
        amean = np.mean([a for a in alphas if a]) if any(alphas) else float("nan")
        print(f"  {params}: err={error:.3f} m, alpha_mean={amean:.2f}"
              f"{'' if stable else ' UNSTABLE'}", flush=True)

    ok = [r for r in rows if r["stable"]]
    best = min(ok, key=lambda r: r["error"]) if ok else None
    out = {"engine": engine, "grid": rows, "best": best,
           "note": "alphas ordered as pairs " + str(TURN_PAIRS)}
    save = common.RESULTS_DIR / f"sweep_aniso_{engine}.json"
    with open(save, "w") as f:
        json.dump(out, f, indent=1)
    if best:
        print(f"\nbest: {best['params']} err={best['error']:.3f} "
              f"alphas={[round(a,2) if a else None for a in best['alphas']]}")
    print(f"wrote {save}")


if __name__ == "__main__":
    main()
