"""Final roster evaluation: every headline config, scored at the current window.

    .venv/bin/python experiments/7_engine_comparison/final_eval.py

Re-simulates defaults, isotropic-tuned best, and the best terrain/friction
variant for each engine on the three GT bags + the motors0 holdout, and scores
them all with common.WINDOW_S (now 3 s). Config *selection* still comes from
the sweeps (which ran at 15 s windows); this pass makes the headline numbers
comparable under the new metric without re-running the grids.

Results -> results/final_eval.json (consumed by plot_results.py when present).
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

ENGINE_DT = {"chrono": 0.005, "agx": 0.01, "ostrich": 0.05}


def sweep_best(engine):
    p = common.RESULTS_DIR / f"sweep_{engine}.json"
    return dict(json.load(open(p))["best"]["params"]) if p.exists() else {}


def aniso_best(engine):
    p = common.RESULTS_DIR / f"sweep_aniso_{engine}.json"
    return dict(json.load(open(p))["best"]["params"]) if p.exists() else None


def roster():
    rows = []
    for engine in ("ostrich", "agx", "chrono"):
        rows.append((f"{engine} defaults", engine, {}))
        rows.append((f"{engine} tuned", engine, sweep_best(engine)))
        if engine == "chrono":
            rows.append(("chrono SCM (phi=20)", "chrono_scm",
                         {"terrain": "scm", "scm_phi": 20.0}))
        else:
            ab = aniso_best(engine)
            if ab:
                rows.append((f"{engine} anisotropic", engine, ab))
    return rows


def run(kind, jobs):
    if kind == "chrono":
        return bridge.run_chrono(jobs, timeout=7200.0, procs=4)
    if kind == "chrono_scm":
        return bridge.run_chrono_scm(jobs, timeout=7200.0, procs=4)
    if kind == "agx":
        return bridge.run_agx(jobs, timeout=7200.0)
    from engines.ostrich_runner import run_ostrich
    return run_ostrich(jobs)


def main():
    bags = {b: common.load_gt(b) for b in common.GT_BAGS + [common.HOLDOUT_BAG]}
    out_rows = []
    for label, kind, params in roster():
        params = dict(params)
        gmu = params.pop("ground_mu", 0.8)
        dt = params.pop("dt", ENGINE_DT["chrono" if kind == "chrono_scm" else kind])
        if kind == "chrono_scm":
            dt = 2e-3
        jobs = [bridge.make_job(f"{label}_{b}".replace(" ", "_"),
                                common.prepare_commands(gt, dt), dt,
                                params=params, ground_mu=gmu)
                for b, gt in bags.items()]
        results = run(kind, jobs)
        per_bag = {}
        for job, res, (b, gt) in zip(jobs, results, bags.items()):
            s = common.score_result(res, gt)
            per_bag[b] = {"combined_mean": s["combined_mean"],
                          "yaw_rmse_deg_mean": s["yaw_rmse_deg_mean"],
                          "stable": s["stable"]}
        gt_mean = float(np.mean([per_bag[b]["combined_mean"] for b in common.GT_BAGS]))
        row = {"label": label, "kind": kind, "params": {**params, "dt": dt},
               "gt_mean": gt_mean,
               "holdout": per_bag[common.HOLDOUT_BAG]["combined_mean"],
               "per_bag": per_bag}
        out_rows.append(row)
        print(f"{label}: GT-mean {gt_mean:.3f} m, holdout {row['holdout']:.3f} m",
              flush=True)

    with open(common.RESULTS_DIR / "final_eval.json", "w") as f:
        json.dump({"window_s": common.WINDOW_S, "rows": out_rows}, f, indent=1)
    print(f"wrote results/final_eval.json (window {common.WINDOW_S:g} s)")


if __name__ == "__main__":
    main()
