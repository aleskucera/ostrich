"""Reproducible accuracy-vs-dt sweep for the box scene.

For each engine (Axion, MuJoCo) at its yaw-tuned best params from
experiments/1_sim_to_real_box, runs a dt grid and records per-(engine, dt)
results into results/accuracy_vs_dt.json. plot_dt_vs_error.py loads that
JSON and produces the headline figure; no hand-copied numbers anywhere.

Usage:
    python experiments/2_dt_stability_box/run_accuracy_vs_dt.py
    python experiments/2_dt_stability_box/run_accuracy_vs_dt.py --engine MuJoCo
    python experiments/2_dt_stability_box/run_accuracy_vs_dt.py --save other.json
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "1_sim_to_real_box"))

import numpy as np

from common_box import DATA_DIR, load_gt, resample_setpoints, score

DURATION = 12.0
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# ----------------------------- engine configs --------------------------------
# These are the yaw-aware-tuned bests from 1_sim_to_real_box/README.md.

AXION_PARAMS = dict(
    mu_front=0.8, mu_rear=1.5, mu_rolling=0.7,
    compliance_contact=1e-7,
)
MUJOCO_PARAMS = dict(
    mu=1.5, tor=2.0, kv=1000.0, solref0=0.005,
    condim=6, integrator="implicitfast",
)
SEMI_IMPLICIT_PARAMS = dict(
    mu=0.05, ke=8e4, kd=2e3, kf=1500.0, k_d_act=200.0,
    joint_attach_ke=1e6, joint_attach_kd=1e2,
)

# dt grids — chosen to span each engine's full "interesting" range AND
# to push past the wall on each side (so the figure shows where each
# engine actually crashes, not just where its accuracy plateau ends).
AXION_DT_GRID  = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10,
                  0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]
MUJOCO_DT_GRID = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2,
                  2e-2, 3e-2, 5e-2, 7e-2, 1e-1]
# SemiImplicit: small grid because the small-dt end is expensive
# (12 s / 5e-4 = 24k steps per run) and the stable region is narrow.
SI_DT_GRID = [2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3]


# ------------------------------- runners -------------------------------------
def axion_runner(gt, dt):
    from axion import (AxionEngineConfig, ComplianceConfig, ContactsConfig, LinearSolverConfig,
                       LinesearchConfig, LoggingConfig, NewtonRaphsonConfig, RenderingConfig,
                       SimulationConfig)
    from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator

    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=max(1, int(round(1 / dt))),
                        start_paused=False),
        AxionEngineConfig(
            nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
            linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
            compliance=ComplianceConfig(joint=6e-8, contact=AXION_PARAMS["compliance_contact"],
                                        friction=1e-6),
            linesearch=LinesearchConfig(enabled=False),
            contacts=ContactsConfig(max_per_world=256)),
        LoggingConfig(), control_mode="velocity",
        mu_front=AXION_PARAMS["mu_front"], mu_rear=AXION_PARAMS["mu_rear"],
        mu_rolling=AXION_PARAMS["mu_rolling"])
    sp = resample_setpoints(gt, dt, DURATION)
    sim.reset_state()
    pose, _ = sim.replay_graph(sp)
    del sim
    return pose


def mujoco_runner(gt, dt):
    from sweep_mujoco import BASE_PARAMS, simulate
    box = gt["box"]
    bg = dict(box_x=box["center"][0], box_y=box["center"][1], box_z=box["center"][2],
              box_hx=box["half_extents"][0], box_hy=box["half_extents"][1],
              box_hz=box["half_extents"][2])
    p = {**BASE_PARAMS, **bg, "dt": dt, "kv": MUJOCO_PARAMS["kv"],
         "ground_friction": MUJOCO_PARAMS["mu"], "box_friction": MUJOCO_PARAMS["mu"],
         "front_friction": MUJOCO_PARAMS["mu"], "rear_friction": MUJOCO_PARAMS["mu"],
         "ground_torsional": MUJOCO_PARAMS["tor"], "front_torsional": MUJOCO_PARAMS["tor"],
         "rear_torsional": MUJOCO_PARAMS["tor"],
         "solref0": MUJOCO_PARAMS["solref0"], "condim": MUJOCO_PARAMS["condim"],
         "integrator": MUJOCO_PARAMS["integrator"]}
    return simulate(p, gt)


def semi_implicit_runner(gt, dt):
    from axion import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                       SimulationConfig)
    from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator

    P = SEMI_IMPLICIT_PARAMS
    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=max(1, int(round(1 / dt))),
                        start_paused=False),
        SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1,
                                 joint_attach_ke=P["joint_attach_ke"],
                                 joint_attach_kd=P["joint_attach_kd"]),
        LoggingConfig(), control_mode="velocity",
        k_p=0.0, k_d=P["k_d_act"],
        mu_front=P["mu"], mu_rear=P["mu"], mu_rolling=0.7,
        ground_ke=P["ke"], ground_kd=P["kd"], ground_kf=P["kf"],
        box_ke=P["ke"],    box_kd=P["kd"],    box_kf=P["kf"],
        wheel_ke=P["ke"],  wheel_kd=P["kd"],  wheel_kf=P["kf"])
    sp = resample_setpoints(gt, dt, DURATION)
    sim.reset_state()
    pose, _ = sim.replay_graph(sp)
    del sim
    return pose


ENGINES = {
    "Axion":         {"runner": axion_runner,         "dt_grid": AXION_DT_GRID,  "params": AXION_PARAMS},
    "MuJoCo":        {"runner": mujoco_runner,        "dt_grid": MUJOCO_DT_GRID, "params": MUJOCO_PARAMS},
    "Semi-Implicit": {"runner": semi_implicit_runner, "dt_grid": SI_DT_GRID,     "params": SEMI_IMPLICIT_PARAMS},
}


# ------------------------------ stability ------------------------------------
def stability(pose, gt, scored) -> tuple[bool, str]:
    """no-NaN AND chassis z in bounds AND robot crossed the box."""
    if not np.isfinite(scored["combined_with_yaw"]):
        return False, "NaN/inf"
    if not np.all(np.isfinite(pose)):
        return False, "NaN/inf pose"
    z = pose[:, 2]
    if not (z.min() > 0.05 and z.max() < 2.0):
        return False, f"z out [{z.min():.2f},{z.max():.2f}]"
    box_far = gt["box"]["center"][0] + gt["box"]["half_extents"][0]
    if pose[-1, 0] < box_far + 0.5:
        return False, f"didn't pass box (x_final={pose[-1, 0]:.2f})"
    return True, "ok"


# ------------------------------ main loop ------------------------------------
def sweep_engine(name, gts):
    runner = ENGINES[name]["runner"]
    grid = ENGINES[name]["dt_grid"]
    rows = []
    print(f"\n=== {name} ({len(grid)} dts × {len(gts)} runs) ===")
    for dt in grid:
        per_gt = {}
        for gn, gt in gts.items():
            t0 = time.perf_counter()
            note = "ok"
            try:
                pose = runner(gt, dt)
                s = score(pose, dt, gt)
                stable, reason = stability(pose, gt, s)
            except Exception as e:
                pose, stable, reason, note = None, False, f"FAIL:{type(e).__name__}", "exception"
                s = {"combined": float("nan"), "combined_with_yaw": float("nan"),
                     "xy": float("nan"), "z": float("nan"), "yaw_rmse_deg": float("nan")}
            per_gt[gn] = {
                "combined": s["combined"],
                "combined_with_yaw": s["combined_with_yaw"],
                "yaw_rmse_deg": s["yaw_rmse_deg"],
                "stable": stable, "reason": reason,
                "wall_s": time.perf_counter() - t0, "note": note,
            }
        finite = [v["combined_with_yaw"] for v in per_gt.values()
                  if np.isfinite(v["combined_with_yaw"])]
        mean_cwy = float(np.mean(finite)) if finite else float("nan")
        all_stable = all(v["stable"] for v in per_gt.values())
        rows.append({"dt": dt, "per_gt": per_gt,
                     "mean_combined_with_yaw": mean_cwy, "all_stable": all_stable})
        flag = "STABLE  " if all_stable else "unstable"
        print(f"  dt={dt:8.5f} | {flag} | mean_err={mean_cwy:9.3f} m | " +
              " ".join(f"{gn[-8:]}={v['combined_with_yaw']:7.3f}({'S' if v['stable'] else 'U'})"
                      for gn, v in per_gt.items()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", nargs="+", default=[
        str(DATA_DIR / "run_2026_05_20-18_04_51.json"),
        str(DATA_DIR / "run_2026_05_20-18_10_33.json")])
    ap.add_argument("--engine", nargs="+", choices=list(ENGINES) + ["all"], default=["all"])
    ap.add_argument("--save", default=str(RESULTS_DIR / "accuracy_vs_dt.json"))
    args = ap.parse_args()

    gts = {pathlib.Path(p).stem: load_gt(p) for p in args.gt}
    targets = list(ENGINES) if "all" in args.engine else args.engine

    out = {
        "scene": "helhest_junior over box (matches experiments/1_sim_to_real_box)",
        "metric": "combined_with_yaw (position + L*yaw RMSE with L=0.5) [m]",
        "accuracy_threshold_m": 0.5,
        "duration_s": DURATION,
        "gts": list(gts),
        "stability_criterion": ("no NaN AND chassis z in [0.05, 2.0] m AND robot drove "
                                "at least 0.5 m past the far edge of the box"),
        "params": {name: ENGINES[name]["params"] for name in targets},
        "results": {},
    }
    for name in targets:
        out["results"][name] = sweep_engine(name, gts)

    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.save}")


if __name__ == "__main__":
    main()
