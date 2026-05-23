"""Axion parameter sweep for the box sim-to-real benchmark.

Drives the helhest_junior model over the box with each run's recorded wheel
commands, scores the prism-tracked trajectory (XY + climb Z) against the real
total-station trajectory, and reports the best (mu_front, mu_rear, dt).

Reuses the validated box simulator from examples/helhest_junior/replay_real.py.

Usage:
    python experiments/1_sim_to_real_box/sweep_axion.py \
        --gt data/run_2026_05_20-18_04_51.json data/run_2026_05_20-18_10_33.json \
        --dt 0.05 --mu-front 0.8 --mu-rear 0.5 0.8 1.0 \
        --save results/sweep_axion.json
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # project root

import numpy as np
import warp as wp
from axion import (AxionEngineConfig, ComplianceConfig, ContactsConfig, LinearSolverConfig,
                   LinesearchConfig, LoggingConfig, NewtonRaphsonConfig, RenderingConfig,
                   SimulationConfig)

from common_box import DATA_DIR, RESULTS_DIR, load_gt, resample_setpoints, score
from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator

DURATION = 12.0


def make_engine_config():
    return AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-6, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )


def run_config(mu_front, mu_rear, dt, gts, use_graph):
    """Build one sim at these params, replay every GT run, return per-run scores."""
    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=int(round(1 / dt)), start_paused=False),
        make_engine_config(), LoggingConfig(),
        control_mode="velocity", mu_front=mu_front, mu_rear=mu_rear,
    )
    out = {}
    for name, gt in gts.items():
        setpoints = resample_setpoints(gt, dt, DURATION)
        sim.reset_state()
        if use_graph:
            pose, _ = sim.replay_graph(setpoints)
        else:
            pose, _ = sim.replay(setpoints)
        out[name] = score(pose, dt, gt)
    del sim
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", nargs="+", default=[
        str(DATA_DIR / "run_2026_05_20-18_04_51.json"),
        str(DATA_DIR / "run_2026_05_20-18_10_33.json")])
    ap.add_argument("--dt", type=float, nargs="+", default=[0.05])
    ap.add_argument("--mu-front", type=float, nargs="+", default=[0.8])
    ap.add_argument("--mu-rear", type=float, nargs="+", default=[0.35, 0.5, 0.7, 0.9])
    ap.add_argument("--save", default=str(RESULTS_DIR / "sweep_axion.json"))
    args = ap.parse_args()

    gts = {pathlib.Path(p).stem: load_gt(p) for p in args.gt}
    use_graph = wp.get_device().is_cuda
    configs = list(itertools.product(args.dt, args.mu_front, args.mu_rear))
    print(f"Axion box sweep: {len(configs)} configs x {len(gts)} runs "
          f"({'cuda-graph' if use_graph else 'python-loop'})")

    best = None
    rows = []
    for dt, mf, mr in configs:
        t0 = time.perf_counter()
        scores = run_config(mf, mr, dt, gts, use_graph)
        combined = float(np.mean([s["combined"] for s in scores.values()]))
        rows.append({"dt": dt, "mu_front": mf, "mu_rear": mr, "combined": combined,
                     "per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"]}
                                 for n, s in scores.items()}})
        print(f"  dt={dt} mu_front={mf} mu_rear={mr}: combined={combined:.3f} m "
              f"({time.perf_counter()-t0:.1f}s)")
        if best is None or combined < best["combined"]:
            best = {"dt": dt, "mu_front": mf, "mu_rear": mr, "combined": combined,
                    "scores": scores}

    # Persist best params + best trajectories (for plotting, no re-sim needed).
    bp = {"dt": best["dt"], "mu_front": best["mu_front"], "mu_rear": best["mu_rear"]}
    out = {
        "simulator": "Axion",
        "best_params": bp,
        "best_error": best["combined"],
        "best_per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"],
                             "sim_rel": s["sim_rel"].tolist(),
                             "sim_t_aligned": s["sim_t_aligned"].tolist()}
                         for n, s in best["scores"].items()},
        "grid": rows,
    }
    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(out, f)
    print(f"\nBest: {bp}  combined={best['combined']:.3f} m  -> {args.save}")


if __name__ == "__main__":
    main()
