"""Interactive Semi-Implicit tuner for the box scene.

Drives the helhest_junior + box scene with the recorded wheel setpoints of
one real run, on the Newton SemiImplicit solver. Every knob the solver cares
about is a CLI arg so you can iterate manually — watch the viewer, adjust
params, re-run.

Defaults are the current-best SI config from our sweeps; override what you
want to vary:

    # GL viewer, default best:
    python experiments/1_sim_to_real_box/tune_semi_implicit.py

    # try softer contact + lower friction, headless, print score:
    python experiments/1_sim_to_real_box/tune_semi_implicit.py \
        --vis headless --ke 4e4 --mu 0.05 --k-d-act 200

    # different run, smaller dt:
    python experiments/1_sim_to_real_box/tune_semi_implicit.py \
        --run 18_10_33 --dt 2.5e-4 --vis gl

Reports the same combined-with-yaw score as the sweeps (common_box.score)
when a real GT is loaded, so iterative results are directly comparable to
sweep_semi_implicit.py's numbers.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp
from ostrich import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                   SimulationConfig)

from common_box import DATA_DIR, load_gt, resample_setpoints, score
from examples.helhest_junior.replay_real import (HelhestJuniorReplaySimulator,
                                                  PRISM_OFFSET, prism_track)

DEFAULT_RUN = "18_04_51"
DEFAULT_DURATION = 12.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help=f"run id suffix (default {DEFAULT_RUN}); --run NONE for "
                         "no GT, just watch")
    ap.add_argument("--vis", choices=["gl", "headless"], default="gl",
                    help="GL viewer (default) or headless (for scoring only)")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    # solver / contact knobs
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--mu", type=float, default=0.05,
                    help="friction (applied to ground + box + wheels) — "
                         "kept low; mu=0.1 blows up on harder runs at high jke")
    ap.add_argument("--ke", type=float, default=8e4,
                    help="contact normal stiffness (applied to ground+box+wheels)")
    ap.add_argument("--kd", type=float, default=2e3,
                    help="contact damping")
    ap.add_argument("--kf", type=float, default=1500.0,
                    help="friction stiffness")
    # actuator knobs (SI: k_p typically 0, k_d is the velocity-feedback gain)
    ap.add_argument("--k-p", type=float, default=0.0,
                    help="wheel actuator position gain (target_ke)")
    ap.add_argument("--k-d-act", type=float, default=200.0,
                    help="wheel actuator velocity-feedback gain (target_kd) — "
                         "SI needs >0 in TARGET_VELOCITY mode")
    # engine knobs
    ap.add_argument("--angular-damping", type=float, default=0.05)
    ap.add_argument("--friction-smoothing", type=float, default=0.1)
    # Joint constraint stiffness — library default is 1e4 (too soft for the
    # heavy chassis); 1e6 is the sweet spot when paired with mu=0.05.
    # 1e6 + mu=0.1 destabilises the harder runs; 1e7 NaNs.
    ap.add_argument("--joint-attach-ke", type=float, default=1e6,
                    help="joint constraint stiffness (lib default 1e4; "
                         "1e6 + mu=0.05 is robust; 1e7 NaNs)")
    ap.add_argument("--joint-attach-kd", type=float, default=1e2,
                    help="joint constraint damping (default 1e2; higher destabilizes)")
    # output
    ap.add_argument("--out", default=None,
                    help="optional prefix to save the comparison .npz/.png "
                         "after the run (same format as replay_real.py --out)")
    args = ap.parse_args()

    # Load GT (or synthetic constant-velocity if --run NONE)
    if args.run.upper() == "NONE":
        # 2 rad/s straight forward, ramp 0.3 s
        N = 2000
        t = np.linspace(0, args.duration, N)
        ramp = np.clip(t / 0.3, 0, 1)
        cmd = (2.0 * ramp)[:, None] * np.ones((1, 3), dtype=np.float32)
        gt = {"control": {"t": t, "lrr": cmd.astype(np.float32)},
              "box": {"center": [1.37, 0, 0.06], "half_extents": [0.37, 0.575, 0.06]},
              "real": {"t": np.array([0., args.duration]),
                       "x": np.array([0., 0.]), "y": np.array([0., 0.]),
                       "z": np.array([0., 0.])}}
        run_id = "synthetic 2 rad/s"
    else:
        gt = load_gt(DATA_DIR / f"run_2026_05_20-{args.run}.json")
        run_id = gt["run_id"]

    setpoints = resample_setpoints(gt, args.dt, args.duration)
    print(f"Run {run_id}: {setpoints.shape[0]} steps @ dt={args.dt}s")
    print(f"  mu={args.mu}  ke={args.ke:g}  kd={args.kd:g}  kf={args.kf:g}  "
          f"k_p={args.k_p}  k_d_act={args.k_d_act}")
    print(f"  joint_attach_ke={args.joint_attach_ke:g}  "
          f"joint_attach_kd={args.joint_attach_kd:g}")

    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=args.duration,
                         target_timestep_seconds=args.dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="gl" if args.vis == "gl" else "null",
                        target_fps=max(1, int(round(1 / args.dt))),
                        start_paused=False),
        SemiImplicitEngineConfig(angular_damping=args.angular_damping,
                                 friction_smoothing=args.friction_smoothing,
                                 joint_attach_ke=args.joint_attach_ke,
                                 joint_attach_kd=args.joint_attach_kd),
        LoggingConfig(),
        control_mode="velocity",
        k_p=args.k_p, k_d=args.k_d_act,
        mu_front=args.mu, mu_rear=args.mu, mu_rolling=0.7,
        ground_ke=args.ke, ground_kd=args.kd, ground_kf=args.kf,
        box_ke=args.ke,    box_kd=args.kd,    box_kf=args.kf,
        wheel_ke=args.ke,  wheel_kd=args.kd,  wheel_kf=args.kf,
    )

    t0 = time.perf_counter()
    use_graph = (args.vis == "headless") and wp.get_device().is_cuda
    if use_graph:
        sim_pose, _ = sim.replay_graph(setpoints)
    else:
        sim_pose, _ = sim.replay(setpoints)
    elapsed = time.perf_counter() - t0
    print(f"\nRan {setpoints.shape[0]} steps in {elapsed:.2f} s "
          f"({'cuda-graph' if use_graph else 'python-loop / GL'}, "
          f"{1000 * elapsed / max(1, setpoints.shape[0]):.2f} ms/step)")

    if args.run.upper() != "NONE":
        s = score(sim_pose, args.dt, gt)
        print(f"\n=== Score on {run_id} ===")
        print(f"  combined_with_yaw : {s['combined_with_yaw']:.4f} m  (sweep best 0.204)")
        print(f"  pos L2            : {s['combined']:.4f} m")
        print(f"  yaw RMSE          : {s['yaw_rmse_deg']:.2f} deg")
        print(f"  shift             : {s['shift']:+.3f} s")

    if args.out:
        from examples.helhest_junior.replay_real import align_real_to_sim
        real = {k: np.asarray(gt["real"][k]) for k in ("t", "x", "y", "z")}
        # Reconstruct an "aligned" real dict the comparison plotter expects.
        # When --run is a real h5, common_box.load_gt has already aligned real
        # to start-at-origin; just put it back in the array shape replay_real's
        # save_comparison wants.
        real_aligned = np.column_stack([real["x"], real["y"], real["z"]])
        from examples.helhest_junior.replay_real import save_comparison
        sim_prism = prism_track(sim_pose, PRISM_OFFSET)
        save_comparison(args.out, sim_prism, real_aligned, real["t"], args.dt)


if __name__ == "__main__":
    main()
