"""Axis D: behavioral scenarios without ground truth — cross-engine spread.

    .venv/bin/python experiments/7_engine_comparison/run_scenarios.py chrono agx ostrich

Three scenarios, all synthetic jobs through the same replay runners (each run
TWICE per engine to expose nondeterminism), at the engine's tuned best params
(sweep results) with defaults as a second config:

  step16      16 cm full-width step 2.5 m ahead; drive 3 rad/s for 10 s. The
              real robot has climbed 16 cm, so "cleared" is the anchor. Metrics:
              cleared (x > 4 m and upright and back at ride height), time to
              clear, max |pitch|.
  turn_radius fixed wheel-speed differentials for 8 s on flat ground; realized
              yaw rate from NET unwrapped yaw over the steady window (never the
              mean of instantaneous wz — contact jitter integrates to nothing).
              Metric: turn gain alpha = ideal differential-drive wz / realized
              wz. Real-robot context from the calibrate bag: alpha ~= 2.
  rock_field  3x3 grid of loose 0.32 m cubes (density 400, mu 0.5) starting
              1.5 m ahead, 0.9 m spacing; drive 3 rad/s for 10 s. Metrics:
              traversal success (x > 5 m, upright), lateral RMS, wall clock
              (contact-rich stress case).

Results -> results/scenarios.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

ENGINE_DT = {"chrono": 0.005, "agx": 0.01, "ostrich": 0.05}
HALF_TRACK = 0.365
WHEEL_RADIUS = 0.35

TURN_PAIRS = [(1.0, 3.0), (1.5, 3.5), (2.0, 4.0), (0.5, 3.5)]


def scenario_jobs(engine, params, rep):
    dt = params.get("dt", ENGINE_DT[engine])
    run_params = {k: v for k, v in params.items() if k != "dt"}
    jobs = []

    T = int(10.0 / dt)
    cmds = np.full((T, 3), 3.0)
    cmds[: int(0.5 / dt)] = 0.0
    step_scene = [{"type": "static_box", "half_extents": [0.35, 4.0, 0.08],
                   "pos": [2.85, 0.0, 0.08], "mu": 0.8}]
    jobs.append(bridge.make_job(f"step16_r{rep}", cmds, dt, params=run_params,
                                scene=step_scene))

    for k, (wl, wr) in enumerate(TURN_PAIRS):
        T = int(8.0 / dt)
        cmds = np.tile([wl, wr, 0.5 * (wl + wr)], (T, 1))
        jobs.append(bridge.make_job(f"turn{k}_r{rep}", cmds, dt, params=run_params))

    T = int(10.0 / dt)
    cmds = np.full((T, 3), 3.0)
    cmds[: int(0.5 / dt)] = 0.0
    rocks = [{"type": "dynamic_box", "half_extents": [0.16, 0.16, 0.16],
              "pos": [1.5 + i * 0.9, (j - 1) * 0.9, 0.16], "mu": 0.5,
              "density": 400.0}
             for i in range(3) for j in range(3)]
    jobs.append(bridge.make_job(f"rocks_r{rep}", cmds, dt, params=run_params,
                                scene=rocks))
    return jobs


def pitch_from_quat_xyzw(q):
    return np.arcsin(np.clip(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]), -1, 1))


def analyze(job_id, res):
    pose = np.asarray(res["pose"])
    dt = res["dt"]
    t = np.arange(pose.shape[0]) * dt
    yaw = np.unwrap(common.yaw_from_quat_xyzw(pose[:, 3:7]))
    pitch = pitch_from_quat_xyzw(pose[:, 3:7])
    upright = bool(np.abs(pitch).max() < np.radians(60)) and res["stable"]

    if job_id.startswith("step16"):
        # Cleared = rear wheel fully past the step's far edge (x=3.2), upright,
        # and back at flat-ground ride height.
        cleared_mask = pose[:, 0] > 4.5
        cleared = (bool(cleared_mask.any()) and upright
                   and abs(pose[-1, 2] - 0.35) < 0.05)
        return {"scenario": "step16", "cleared": cleared,
                "time_to_clear_s": float(t[cleared_mask.argmax()]) if cleared_mask.any() else None,
                "max_pitch_deg": float(np.degrees(np.abs(pitch).max())),
                "climbed_top": bool((np.abs(pose[:, 2] - 0.51) < 0.05).any()),
                "x_final": float(pose[-1, 0]), "wall_clock_s": res["wall_clock_s"]}

    if job_id.startswith("turn"):
        k = int(job_id[4])
        wl, wr = TURN_PAIRS[k]
        m = t > 2.0  # steady window
        wz = (yaw[m][-1] - yaw[m][0]) / (t[m][-1] - t[m][0])
        ideal = WHEEL_RADIUS * (wr - wl) / (2 * HALF_TRACK)
        seg = np.linalg.norm(np.diff(pose[m][:, :2], axis=0), axis=1).sum()
        v = seg / (t[m][-1] - t[m][0])
        return {"scenario": "turn_radius", "pair": [wl, wr],
                "alpha": float(ideal / wz) if abs(wz) > 1e-4 else None,
                "radius_m": float(v / abs(wz)) if abs(wz) > 1e-4 else None,
                "wz_deg_s": float(np.degrees(wz))}

    if job_id.startswith("rocks"):
        return {"scenario": "rock_field",
                "success": bool(pose[-1, 0] > 5.0) and upright,
                "x_final": float(pose[-1, 0]),
                "lateral_rms": float(np.sqrt(np.mean(pose[:, 1] ** 2))),
                "wall_clock_s": res["wall_clock_s"], "stable": res["stable"]}
    raise ValueError(job_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engines", nargs="+", choices=("chrono", "agx", "ostrich"))
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()

    out_path = common.RESULTS_DIR / "scenarios.json"
    rows = json.load(open(out_path))["rows"] if out_path.exists() else []
    rows = [r for r in rows if r["engine"] not in args.engines]

    for engine in args.engines:
        sweep_path = common.RESULTS_DIR / f"sweep_{engine}.json"
        bp = json.load(open(sweep_path))["best"]["params"] if sweep_path.exists() else None
        configs = [("defaults", {})] + ([("best", bp)] if bp else [])
        for label, params in configs:
            jobs = [j for rep in (0, 1) for j in scenario_jobs(engine, params or {}, rep)]
            print(f"=== {engine} {label} ({len(jobs)} jobs) ===", flush=True)
            if engine == "chrono":
                results = bridge.run_chrono(jobs, timeout=7200.0, procs=args.procs)
            elif engine == "agx":
                results = bridge.run_agx(jobs, timeout=7200.0)
            else:
                from engines.ostrich_runner import run_ostrich
                results = run_ostrich(jobs)
            for job, res in zip(jobs, results):
                a = analyze(job["id"], res)
                a.update({"engine": engine, "config": label, "job": job["id"]})
                rows.append(a)
                print(f"    {job['id']}: {a}", flush=True)
            # Save after every engine/config batch so a stop loses nothing.
            common.RESULTS_DIR.mkdir(exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"rows": rows}, f, indent=1,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
