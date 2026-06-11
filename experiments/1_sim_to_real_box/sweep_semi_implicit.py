"""Newton Semi-Implicit parameter sweep for the box sim-to-real benchmark.

Same junior model + box scene as the Ostrich sweep, but built on top of
``SemiImplicitEngineConfig``. Semi-implicit Euler is explicit on the contact
forces, so it needs:
  - small dt (~1e-3 or finer for stability),
  - explicit spring-damper contact stiffness (ke/kd) — *not* ignored like in
    Ostrich. wheel_ke/wheel_kd/wheel_kf are exposed for this reason.
  - kf (friction stiffness) — separate tunable.

Drive signal and scoring are identical to the other engines (yaw-aware
combined metric in common_box.score).

Usage:
    python experiments/1_sim_to_real_box/sweep_semi_implicit.py \
        --dt 0.001 --mu 0.05 0.5 --ke 4e4 1e5 --kd 1e3 4e3 --kf 500 1500 \
        --save results/sweep_semi_implicit.json
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp
from ostrich import LoggingConfig, RenderingConfig, SemiImplicitEngineConfig, SimulationConfig

from common_box import DATA_DIR, RESULTS_DIR, load_gt, resample_setpoints, score
from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator

DURATION = 12.0


def run_config(dt, mu, ke, kd, kf, k_d_act, joint_ke, joint_kd, gts, use_graph):
    """Build one sim at these params, replay every GT run, return per-run scores.
    `k_d_act` is the wheel actuator velocity-feedback gain — SemiImplicit needs
    this nonzero to apply torque in TARGET_VELOCITY mode (Ostrich doesn't).
    `joint_ke/joint_kd` are SemiImplicit's joint-constraint stiffness/damping
    (engine-level); the library defaults (1e4/1e2) are too soft for our heavy
    chassis hitting the box, but pushing them too high destabilises the
    harder runs — sweep them per-scenario."""
    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=int(round(1 / dt)), start_paused=False),
        SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1,
                                 joint_attach_ke=joint_ke, joint_attach_kd=joint_kd),
        LoggingConfig(),
        control_mode="velocity",
        k_p=0.0, k_d=k_d_act,  # SemiImplicit-friendly velocity gain
        mu_front=mu, mu_rear=mu, mu_rolling=0.7,
        ground_ke=ke, ground_kd=kd, ground_kf=kf,
        box_ke=ke,    box_kd=kd,    box_kf=kf,
        wheel_ke=ke,  wheel_kd=kd,  wheel_kf=kf,
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
    # Defaults are this box scene's tuned best (combined = 0.169 m at floor).
    # Notes:
    #  - mu is ~10-20x smaller than physical rubber: SemiImplicit's spring-damper
    #    friction model lets the same effective force come from kf*slip*mu, so
    #    "physical" mu would overshoot. mu=0.1 is fragile (blows up on harder
    #    runs at high joint_attach_ke); mu=0.05 holds across both runs.
    #  - ke (contact) had to be ~10x exp-1's helhest value (8000) because the
    #    junior chassis is heavier (90 kg) and we need a stiffer contact to
    #    keep it from sinking into the box during the climb.
    #  - joint_attach_ke=1e6 (vs library default 1e4) tightens the wheel/
    #    chassis attachment; the heavy chassis was effectively decoupling on
    #    impact. Raising further (1e7) NaNs.
    #  - k_d_act = wheel actuator velocity gain; SemiImplicit needs >0 to apply
    #    torque in TARGET_VELOCITY mode (Ostrich drives fine with kd=0).
    #  - dt=5e-4 is at the stability edge; smaller dt is also stable, larger
    #    blows up. Tuning is fragile — many neighbouring configs NaN.
    ap.add_argument("--dt", type=float, nargs="+", default=[5e-4])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.05])
    ap.add_argument("--ke", type=float, nargs="+", default=[8e4])
    ap.add_argument("--kd", type=float, nargs="+", default=[2e3])
    ap.add_argument("--kf", type=float, nargs="+", default=[1500.0])
    ap.add_argument("--k-d-act", type=float, nargs="+", default=[200.0],
                    help="wheel actuator velocity-feedback gain (target_kd); "
                         "SemiImplicit needs >0 to apply torque in velocity mode")
    ap.add_argument("--joint-attach-ke", type=float, nargs="+", default=[1e6],
                    help="SemiImplicit joint constraint stiffness — library "
                         "default 1e4 is too soft for heavy chassis impact "
                         "but high values destabilise harder runs")
    ap.add_argument("--joint-attach-kd", type=float, nargs="+", default=[1e2],
                    help="joint constraint damping (higher destabilises)")
    ap.add_argument("--no-graph", action="store_true",
                    help="disable CUDA graph capture (use the slow Python loop)")
    ap.add_argument("--save", default=str(RESULTS_DIR / "sweep_semi_implicit.json"))
    args = ap.parse_args()

    gts = {pathlib.Path(p).stem: load_gt(p) for p in args.gt}
    use_graph = (not args.no_graph) and wp.get_device().is_cuda
    configs = list(itertools.product(args.dt, args.mu, args.ke, args.kd, args.kf,
                                     args.k_d_act, args.joint_attach_ke, args.joint_attach_kd))
    print(f"Semi-Implicit box sweep: {len(configs)} configs x {len(gts)} runs "
          f"({'cuda-graph' if use_graph else 'python-loop'})")

    best, rows = None, []
    for dt, mu, ke, kd, kf, k_d_act, joint_ke, joint_kd in configs:
        t0 = time.perf_counter()
        try:
            scores = run_config(dt, mu, ke, kd, kf, k_d_act, joint_ke, joint_kd, gts, use_graph)
            combined = float(np.mean([s["combined_with_yaw"] for s in scores.values()]))
        except Exception as e:
            print(f"  dt={dt} mu={mu} ke={ke:g} kd={kd:g} kf={kf:g} k_d_act={k_d_act:g} "
                  f"jke={joint_ke:g} jkd={joint_kd:g}: FAILED ({e})")
            continue
        cfg = {"dt": dt, "mu": mu, "ke": ke, "kd": kd, "kf": kf, "k_d_act": k_d_act,
               "joint_attach_ke": joint_ke, "joint_attach_kd": joint_kd}
        rows.append({**cfg, "combined": combined,
                     "per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"],
                                     "combined_with_yaw": s["combined_with_yaw"],
                                     "yaw_rmse_deg": s["yaw_rmse_deg"]}
                                 for n, s in scores.items()}})
        per_run = " ".join(f"{n[-8:]}={s['combined_with_yaw']:6.3f}"
                           for n, s in scores.items())
        print(f"  dt={dt} mu={mu} ke={ke:g} kd={kd:g} kf={kf:g} k_d_act={k_d_act:g} "
              f"jke={joint_ke:g} jkd={joint_kd:g}: mean={combined:.3f} m  ({per_run}) "
              f"({time.perf_counter()-t0:.1f}s)")
        if best is None or combined < best["combined"]:
            best = {**cfg, "combined": combined, "scores": scores}

    if best is None:
        print("All configs failed.")
        return
    bp = {k: best[k] for k in ("dt", "mu", "ke", "kd", "kf", "k_d_act",
                                "joint_attach_ke", "joint_attach_kd")}
    out = {
        "simulator": "Semi-Implicit",
        "best_params": bp,
        "best_error": best["combined"],
        "best_per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"],
                             "combined_with_yaw": s["combined_with_yaw"],
                             "yaw_rmse_deg": s["yaw_rmse_deg"],
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
