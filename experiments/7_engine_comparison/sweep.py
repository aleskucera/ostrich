"""Staged solver/friction sweep for one engine (axes A: accuracy, B: sensitivity).

    .venv/bin/python experiments/7_engine_comparison/sweep.py chrono
    .venv/bin/python experiments/7_engine_comparison/sweep.py agx --holdout
    .venv/bin/python experiments/7_engine_comparison/sweep.py ostrich

Protocol: every engine replays the three GT bags (fast_experiment0/1, calibrate)
open-loop and is scored with the segmented 15 s window metric (common.py); the
per-config error is the mean of the three bags' window-mean combined errors.
Sweeps are STAGED, not full cross-products: stage 1 = friction/material grid at
the engine's baseline solver, stage 2 = solver settings at stage-1 best, stage 3
= dt / formulation variants at the running best. The pinned "defaults" row (100%
stock solver, file-default materials) is always evaluated and reported alongside
the best, per the defaults-vs-tuned protocol. --holdout evaluates defaults+best
on the untouched motors0 bag.

Results -> results/sweep_<engine>.json:
  {engine, defaults, best, grid: [{stage, params, error, per_bag, wall_clock_s}]}
"""

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

ENGINE_DT = {"chrono": 0.005, "agx": 0.01, "ostrich": 0.05}


# --- staged grids -----------------------------------------------------------
MU_GRID = [
    {"mu_front": mf, "mu_rear": mr}
    for mf, mr in itertools.product((0.5, 0.7, 1.0), (0.2, 0.4, 0.7, 1.0))
]

STAGES = {
    "chrono": [
        ("friction", MU_GRID),
        ("solver", [
            {"solver": s, "iterations": it}
            for s, it in itertools.product(("psor", "bb", "apgd"), (50, 500))
        ]),
        ("variants", [
            {"dt": 0.002}, {"dt": 0.01},
            {"wheel_shape": "cylinder"},
            {"system": "smc", "dt": 0.001},
            {"rolling_resistance": 0.05}, {"rolling_resistance": 0.09},
        ]),
    ],
    "agx": [
        ("friction", MU_GRID + [{"rolling_resistance": r} for r in (0.35, 0.7)]),
        ("solver", [
            {"friction_solve_type": "direct"},
            {"friction_solve_type": "direct", "exact_cone_projection": True},
            {"friction_solve_type": "direct_and_iterative"},
            {"friction_solve_type": "direct", "joint_solve_type": "direct_and_iterative"},
            {"num_resting_iterations": 64},
            {"num_resting_iterations": 256, "num_friction_iterations": 64},
        ]),
        ("variants", [
            {"dt": 0.005}, {"dt": 0.02},
            {"youngs_modulus": 1e7}, {"youngs_modulus": 1e8},
        ]),
    ],
    "ostrich": [
        ("friction", [
            {"mu_front": mf, "mu_rear": mr}
            for mf, mr in itertools.product((0.6, 0.8, 1.0), (0.3, 0.5, 0.8, 1.0))
        ]),
        ("solver", [
            {"compliance_contact": c, "mu_rolling": mr}
            for c, mr in itertools.product((1e-7, 1e-6, 1e-5), (0.3, 0.7))
        ]),
        ("variants", [{"dt": 0.02}, {"dt": 0.1}]),
    ],
}


def run_engine(engine, jobs, procs):
    if engine == "chrono":
        return bridge.run_chrono(jobs, timeout=7200.0, procs=procs)
    if engine == "agx":
        return bridge.run_agx(jobs, timeout=7200.0)
    from engines.ostrich_runner import run_ostrich
    return run_ostrich(jobs)


def eval_config(engine, params, gts, procs, tag):
    """One config = replay every GT bag; error = mean of per-bag window means."""
    dt = params.get("dt", ENGINE_DT[engine])
    run_params = {k: v for k, v in params.items() if k != "dt"}
    jobs = [bridge.make_job(f"{tag}_{bag}", common.prepare_commands(gt, dt), dt,
                            params=run_params)
            for bag, gt in gts.items()]
    t0 = time.perf_counter()
    results = run_engine(engine, jobs, procs)
    wall = time.perf_counter() - t0
    per_bag = {}
    for job, res in zip(jobs, results):
        bag = job["id"][len(tag) + 1:]
        s = common.score_result(res, gts[bag])
        per_bag[bag] = {"combined_mean": s["combined_mean"],
                        "combined_median": s["combined_median"],
                        "yaw_rmse_deg_mean": s["yaw_rmse_deg_mean"],
                        "n_windows": s["n_windows"],
                        "stable": s["stable"]}
    stable = all(b["stable"] for b in per_bag.values())
    error = float(np.mean([b["combined_mean"] for b in per_bag.values()])) \
        if stable else float("inf")
    return {"params": params, "error": error, "stable": stable,
            "per_bag": per_bag, "wall_clock_s": wall}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine", choices=("chrono", "agx", "ostrich"))
    ap.add_argument("--procs", type=int, default=4, help="parallel chrono processes")
    ap.add_argument("--holdout", action="store_true",
                    help="also evaluate defaults+best on the held-out motors0 bag")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    engine = args.engine

    gts = {bag: common.load_gt(bag) for bag in common.GT_BAGS}

    # Crash-resumable: every completed config is appended to a JSONL partial;
    # a restart loads it and skips configs whose params already have a row.
    # (The in-process ostrich runner has segfaulted after many sim rebuilds.)
    partial = common.RESULTS_DIR / f"sweep_{engine}.partial.jsonl"
    grid = []
    if partial.exists():
        with open(partial) as f:
            grid = [json.loads(line) for line in f if line.strip()]
        print(f"(resume: {len(grid)} configs loaded from {partial.name})", flush=True)

    def record(stage, row):
        row["stage"] = stage
        grid.append(row)
        with open(partial, "a") as f:
            f.write(json.dumps(row) + "\n")
        flag = "" if row["stable"] else " UNSTABLE"
        print(f"  [{stage}] {row['params']}: {row['error']:.3f} m "
              f"({row['wall_clock_s']:.0f}s){flag}", flush=True)

    defaults_row = next((r for r in grid if r["stage"] == "defaults"), None)
    if defaults_row is None:
        print(f"=== {engine}: defaults ===", flush=True)
        defaults_row = eval_config(engine, {}, gts, args.procs, "defaults")
        record("defaults", defaults_row)

    best_params: dict = {}
    best_error = defaults_row["error"]
    for stage_name, stage_grid in STAGES[engine]:
        print(f"=== {engine}: stage {stage_name} ({len(stage_grid)} configs) ===",
              flush=True)
        for i, delta in enumerate(stage_grid):
            params = {**best_params, **delta}
            if any(r["params"] == params for r in grid):
                continue
            row = eval_config(engine, params, gts, args.procs, f"{stage_name}{i}")
            record(stage_name, row)
        stage_best = min((r for r in grid), key=lambda r: r["error"])
        best_params = dict(stage_best["params"])
        best_error = stage_best["error"]
        print(f"  -> best after {stage_name}: {best_params} ({best_error:.3f} m)",
              flush=True)

    out = {
        "engine": engine,
        "defaults": defaults_row,
        "best": {"params": best_params, "error": best_error},
        "grid": grid,
        "gt_bags": common.GT_BAGS,
        "window_s": common.WINDOW_S,
    }

    if args.holdout:
        hold = {common.HOLDOUT_BAG: common.load_gt(common.HOLDOUT_BAG)}
        print(f"=== {engine}: holdout {common.HOLDOUT_BAG} ===", flush=True)
        out["holdout"] = {
            "defaults": eval_config(engine, {}, hold, args.procs, "hold_def"),
            "best": eval_config(engine, best_params, hold, args.procs, "hold_best"),
        }
        print(f"  defaults: {out['holdout']['defaults']['error']:.3f} m, "
              f"best: {out['holdout']['best']['error']:.3f} m", flush=True)

    save = args.save or (common.RESULTS_DIR / f"sweep_{engine}.json")
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(save, "w") as f:
        json.dump(out, f, indent=1)
    partial.unlink(missing_ok=True)
    print(f"\n{engine}: defaults {defaults_row['error']:.3f} m -> "
          f"best {best_error:.3f} m {best_params}\nwrote {save}")


if __name__ == "__main__":
    main()
