"""Axis C: accuracy + real-time factor vs timestep, per engine.

    .venv/bin/python experiments/7_engine_comparison/run_speed_dt.py chrono agx ostrich

Replays fast_experiment1 (125 s of aggressive driving) at a per-engine dt grid,
at BOTH the engine's defaults and its tuned best (read from results/
sweep_<engine>.json when present). Records the segmented window error, stability,
and the real-time factor RTF = simulated seconds / wall-clock seconds of the
stepping loop. CPU engines run threads=1 and engine-default threads as separate
rows; ostrich runs on GPU with CUDA graphs (its native mode — different hardware,
flagged in the output; graph capture time is included in the wall clock).

The headline per engine is the RTF at the largest USABLE dt: stable AND window
error within 1.5x of that engine's best error at its accuracy-optimal dt.

Results -> results/speed_dt.json
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

BAG = "fast_experiment1"

DT_GRID = {
    "chrono": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
    "agx": [2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1],
    "ostrich": [5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
}
THREAD_ROWS = {"chrono": [1], "agx": [1, 0], "ostrich": [0]}  # 0 = engine default / GPU


def best_params(engine):
    path = common.RESULTS_DIR / f"sweep_{engine}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["best"]["params"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engines", nargs="+", choices=("chrono", "agx", "ostrich"))
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()

    gt = common.load_gt(BAG)
    out_path = common.RESULTS_DIR / "speed_dt.json"
    all_rows = json.load(open(out_path))["rows"] if out_path.exists() else []
    all_rows = [r for r in all_rows if r["engine"] not in args.engines]

    for engine in args.engines:
        bp = best_params(engine)
        configs = [("defaults", {})] + ([("best", bp)] if bp else [])
        for label, params in configs:
            for threads in THREAD_ROWS[engine]:
                jobs = []
                for dt in DT_GRID[engine]:
                    p = {k: v for k, v in (params or {}).items() if k != "dt"}
                    if engine != "ostrich":
                        p["threads"] = threads
                    cmds = common.prepare_commands(gt, dt)
                    jobs.append(bridge.make_job(f"dt_{dt:g}", cmds, dt, params=p))
                print(f"=== {engine} {label} threads={threads} "
                      f"({len(jobs)} dts) ===", flush=True)
                t0 = time.perf_counter()
                if engine == "chrono":
                    results = bridge.run_chrono(jobs, timeout=7200.0, procs=args.procs)
                elif engine == "agx":
                    results = bridge.run_agx(jobs, timeout=7200.0)
                else:
                    from engines.ostrich_runner import run_ostrich
                    results = run_ostrich(jobs)
                print(f"    batch took {time.perf_counter()-t0:.0f}s", flush=True)
                for job, res in zip(jobs, results):
                    s = common.score_result(res, gt)
                    sim_s = res["n_steps"] * res["dt"]
                    rtf = sim_s / res["wall_clock_s"] if res["wall_clock_s"] else 0.0
                    row = {
                        "engine": engine, "config": label, "threads": threads,
                        "dt": res["dt"], "stable": s["stable"],
                        "combined_mean": s["combined_mean"],
                        "rtf": rtf, "wall_clock_s": res["wall_clock_s"],
                        "hardware": "gpu+cudagraph" if engine == "ostrich" else "cpu",
                    }
                    all_rows.append(row)
                    print(f"    dt={res['dt']:g}: {'ok ' if s['stable'] else 'DIV'} "
                          f"err={s['combined_mean']:.3f} m rtf={rtf:.1f}x", flush=True)

    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"bag": BAG, "rows": all_rows}, f, indent=1)
    print(f"wrote {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
