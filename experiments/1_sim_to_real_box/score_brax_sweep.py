"""Score the comprehensive Brax forward sweep (forward_brax_<pipeline>.npz) and
report the best config PER PIPELINE, with the exact 1_sim_to_real_box metric.

Writes:
  results/brax_best_per_pipeline.json  -- best config + per-run error per pipeline
  results/sweep_brax.json              -- the overall-best (for the bar panel),
                                          including downsampled trajectories
                                          (and per-pipeline best trajectories for plotting)

Run in the main ostrich venv:
    .venv/bin/python experiments/1_sim_to_real_box/score_brax_sweep.py
"""
import glob, json, pathlib, re
import numpy as np
from collections import defaultdict

from common_box import load_gt, score

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
RUN_PLOT = "run_2026_05_20-18_10_33"   # run shown in the paper xy/z panels


def dt_of(cfg):
    return float(re.search(r"dt([0-9.eE-]+)", cfg).group(1))


def main():
    gts = {}
    best_per_pipe = {}
    overall = None  # (err, pipeline, cfg, runs_scores)
    for npz in sorted(RES.glob("forward_brax_*.npz")):
        pipe = re.match(r"forward_brax_(.+)\.npz", npz.name).group(1)
        data = np.load(npz)
        # group keys by config (strip leading "run|")
        cfgs = defaultdict(dict)
        for key in data.files:
            run, cfg = key.split("|", 1)
            cfgs[cfg][run] = key
        rows = []
        for cfg, runkeys in cfgs.items():
            dt = dt_of(cfg)
            scores = {}
            for run, key in runkeys.items():
                if run not in gts:
                    gts[run] = load_gt(str(HERE / "data" / f"{run}.json"))
                pose = data[key]
                if not np.all(np.isfinite(pose)) or np.max(np.abs(pose[:, :3])) > 10:
                    scores[run] = {"combined_with_yaw": np.inf}
                    continue
                scores[run] = score(pose, dt, gts[run])
            # need both runs finite to count
            errs = [s["combined_with_yaw"] for s in scores.values()]
            if len(errs) < len(RUNS_EXPECTED) or not np.all(np.isfinite(errs)):
                mean = np.inf
            else:
                mean = float(np.mean(errs))
            rows.append((mean, cfg, runkeys, scores))
        rows.sort(key=lambda r: r[0])
        if rows and np.isfinite(rows[0][0]):
            mean, cfg, runkeys, scores = rows[0]
            best_per_pipe[pipe] = {
                "best_error": mean, "config": cfg,
                "per_run": {r: float(s["combined_with_yaw"]) for r, s in scores.items()},
            }
            print(f"{pipe:12s} best {mean:.3f} m   {cfg}")
            # keep best trajectory for the plotted run
            if RUN_PLOT in runkeys:
                pose = data[runkeys[RUN_PLOT]]
                s = score(pose, dt_of(cfg), gts[RUN_PLOT])
                best_per_pipe[pipe]["traj_xy"] = np.asarray(s["sim_rel"])[::5, :2].tolist()
                best_per_pipe[pipe]["traj_xyz"] = np.asarray(s["sim_rel"])[::5].tolist()
                best_per_pipe[pipe]["traj_t"] = np.asarray(s["sim_t_aligned"])[::5].tolist()
            if overall is None or mean < overall[0]:
                overall = (mean, pipe, cfg, scores, dt_of(cfg), runkeys, data)
        else:
            print(f"{pipe:12s} NO bounded config tracked")

    (RES / "brax_best_per_pipeline.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("traj")}
                    for k, v in best_per_pipe.items()}, indent=2))

    # overall best -> sweep_brax.json (bar panel + xy/z trajectory for plotted run)
    if overall:
        mean, pipe, cfg, scores, dt, runkeys, data = overall
        out = {"simulator": "Brax", "best_error": mean,
               "best_params": {"pipeline": pipe, "config": cfg, "dt": dt},
               "best_per_run": {}}
        for run, s in scores.items():
            entry = {"combined_with_yaw": s["combined_with_yaw"],
                     "yaw_rmse_deg": s.get("yaw_rmse_deg", float("nan"))}
            if np.isfinite(s["combined_with_yaw"]) and "sim_rel" in s:
                entry["sim_rel"] = np.asarray(s["sim_rel"])[::5].tolist()
                entry["sim_t_aligned"] = np.asarray(s["sim_t_aligned"])[::5].tolist()
            out["best_per_run"][run] = entry
        (RES / "sweep_brax.json").write_text(json.dumps(out, indent=2))
        print(f"\nOVERALL best Brax: {mean:.3f} m ({pipe}, {cfg})")
        print("compare: Ostrich 0.062 | MuJoCo 0.054 | Semi-Implicit 0.110")
        print(f"wrote sweep_brax.json + brax_best_per_pipeline.json")
    # also dump per-pipeline trajectories for a diagnostic plot
    np.savez(RES / "brax_best_traj.npz",
             **{f"{p}_xy": np.asarray(v["traj_xy"]) for p, v in best_per_pipe.items()
                if "traj_xy" in v})


RUNS_EXPECTED = ["run_2026_05_20-18_04_51", "run_2026_05_20-18_10_33"]

if __name__ == "__main__":
    main()
