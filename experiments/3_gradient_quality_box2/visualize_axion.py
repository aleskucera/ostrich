"""Interactive GL visualisation of a single Axion optimisation trial.

Builds the same `HelhestJuniorBoxFinalPoseOptimizer` as `optimize_axion.py`
but with ``vis_type="gl"`` so the Newton GL viewer pops up and you can
*watch* the trajectory improve as the optimiser iterates.

Each call to `sim.render_episode()` replays the stored simulation state
in the viewer at the configured `playback_speed`. We invoke it every
`--render-every` iterations (default 5) so the on-screen replay shows
the progression instead of every single iter (which would crawl).

The target XY is marked as a small green cross at z=0.05 m via a viewer
``callback`` passed to `render_episode`.

Usage:
    python experiments/3_gradient_quality_box2/visualize_axion.py
    # one seed, 100 iters, render every 5 iters at 2x playback
    python experiments/3_gradient_quality_box2/visualize_axion.py \
        --seed 42 --iterations 100 --render-every 5 --playback-speed 2.0
"""
import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import warp as wp
from axion import LoggingConfig, RenderingConfig, SimulationConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Reuse the optimiser + helpers from the headless script — only change is
# vis_type, render_episode calls, and the target-marker callback.
from optimize_axion import (HelhestJuniorBoxFinalPoseOptimizer, SplineAdam,
                              initial_spline, sample_ic_target, make_configs,
                              NUM_WHEEL_DOFS, INIT_TYPES)
from axion import (AxionEngineConfig, ComplianceConfig, ContactsConfig,
                   LinearSolverConfig, LinesearchConfig, NewtonRaphsonConfig)


def make_gl_configs(duration, dt):
    """Same engine config as make_configs() but with vis_type='gl'."""
    sim_cfg = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt,
                                num_worlds=1, use_cuda_graph=True)
    rc = RenderingConfig(vis_type="gl", target_fps=max(1, int(1 / dt)),
                          start_paused=False)
    ec = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))
    return sim_cfg, rc, ec


def make_target_callback(target_xy, color=(0.0, 1.0, 0.0)):
    """Return a callback that draws a green cross at the target XY on every
    frame. Used as the `callback` arg to render_episode.
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    cross_a = np.array([[tx - 0.2, ty, 0.05], [tx, ty - 0.2, 0.05]],
                        dtype=np.float32)
    cross_b = np.array([[tx + 0.2, ty, 0.05], [tx, ty + 0.2, 0.05]],
                        dtype=np.float32)
    cross_a_wp = wp.array(cross_a, dtype=wp.vec3)
    cross_b_wp = wp.array(cross_b, dtype=wp.vec3)

    def callback(viewer, step_idx, state):
        viewer.log_lines("/target_marker", cross_a_wp, cross_b_wp, color)

    return callback


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42,
                    help="task seed (selects which (IC, target) pair from the "
                         "deterministic sample_ic_target stream is used)")
    ap.add_argument("--trial-index", type=int, default=0,
                    help="which trial in the sample_ic_target stream to render "
                         "(0 = first)")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--ic-xy", type=float, default=0.1)
    ap.add_argument("--ic-yaw-deg", type=float, default=5.0)
    ap.add_argument("--target-x", type=float, default=3.0)
    ap.add_argument("--target-y", type=float, default=0.0)
    ap.add_argument("--target-xy-jitter", type=float, default=0.3)
    ap.add_argument("--target-yaw-deg", type=float, default=15.0)
    ap.add_argument("--w-track", type=float, default=1.0,
                    help="Per-step chassis-xy tracking against linear IC→target ref")
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-vel", type=float, default=0.3)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--init-type", choices=INIT_TYPES, default="distance-aware")
    ap.add_argument("--init-noise-std", type=float, default=0.2)
    ap.add_argument("--beta1", type=float, default=0.9,
                    help="Adam first-moment decay. 0.0 = no momentum (RMSprop)")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lr-min-ratio", type=float, default=0.2,
                    help="Cosine LR decay endpoint as fraction of lr_init")
    ap.add_argument("--amsgrad", action="store_true",
                    help="Use AMSGrad (running max of v in Adam denominator)")
    ap.add_argument("--render-every", type=int, default=5,
                    help="render after every N iters (lower = smoother "
                         "animation, longer total wall time)")
    ap.add_argument("--playback-speed", type=float, default=2.0,
                    help="GL playback speed multiplier (1=real time, 2=2x)")
    ap.add_argument("--start-paused", action="store_true",
                    help="pause the very first replay so you can orient the camera")
    args = ap.parse_args()

    ic_perturb = {"xy": args.ic_xy, "yaw_rad": np.deg2rad(args.ic_yaw_deg)}
    target_perturb = {"xy_center": (args.target_x, args.target_y),
                       "xy_jitter": args.target_xy_jitter,
                       "yaw_rad": np.deg2rad(args.target_yaw_deg)}
    weights = {"track": args.w_track, "pos": args.w_pos, "yaw": args.w_yaw, "vel": args.w_vel,
               "smooth": args.w_smooth, "reg": args.w_reg}

    # Match the same task stream as optimize_axion.py — draw `trial_index`+1
    # ICs/targets from the seed-based RNG so the GL viz exercises the exact
    # same task as one of the headless trials.
    task_rng = np.random.default_rng(args.seed)
    for _ in range(args.trial_index + 1):
        ic, target = sample_ic_target(task_rng, ic_perturb, target_perturb)
    print(f"=== Visualising trial {args.trial_index} (seed {args.seed}) ===")
    print(f"IC:     xy=({ic['xy'][0]:+.2f}, {ic['xy'][1]:+.2f}) m, "
          f"yaw={np.rad2deg(ic['yaw']):+.1f}°")
    print(f"Target: xy=({target['xy'][0]:+.2f}, {target['xy'][1]:+.2f}) m, "
          f"yaw={np.rad2deg(target['yaw']):+.1f}°")
    print(f"K={args.K}  iters={args.iterations}  dt={args.dt}  horizon={args.horizon_s}s")
    print(f"Render every {args.render_every} iters at {args.playback_speed}x speed")

    sim_cfg, rc, ec = make_gl_configs(args.horizon_s, args.dt)
    sim = HelhestJuniorBoxFinalPoseOptimizer(
        sim_cfg, rc, ec, LoggingConfig(),
        ic_xy=ic["xy"], ic_yaw=ic["yaw"],
        target_xy=target["xy"], target_yaw=target["yaw"],
        weights=weights, K=args.K)
    # Track chassis (body 0) so its trajectory is drawn as a coloured line
    # in the viewer. Different `iteration` values produce separate lines
    # so the user can see the path evolve.
    sim.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    spline_seed = args.seed + args.trial_index + 1000
    sim.spline_params = initial_spline(
        args.K, NUM_WHEEL_DOFS, spline_seed, args.init_type,
        target["xy"], args.horizon_s, noise_std=args.init_noise_std)
    sim.spline_adam = SplineAdam(K=args.K, num_dofs=NUM_WHEEL_DOFS,
                                  lr=args.lr, lr_min_ratio=args.lr_min_ratio,
                                  total_steps=args.iterations,
                                  betas=(args.beta1, args.beta2),
                                  amsgrad=args.amsgrad)
    sim._apply_params(sim.spline_params)

    target_cb = make_target_callback(target["xy"])

    t0 = time.perf_counter()
    for it in range(args.iterations):
        loss = sim.opt_step()
        stats = sim._last_step_stats
        # Mark overshoot: cos(g, m) < 0 means momentum opposes the current
        # gradient — that's when Adam keeps walking the "wrong" way.
        overshoot = " <--OVERSHOOT" if stats["cos_gm"] < 0 else ""
        if it % args.render_every == 0 or it == args.iterations - 1:
            metrics = sim.final_metrics()
            print(f"  iter {it:3d}: loss={loss:.4f}  "
                  f"pos_err={metrics['pos_error_m']:.3f}m  "
                  f"term_vel={metrics['terminal_speed_mps']:.2f}m/s  "
                  f"|g|={stats['gnorm']:.3f} |m|={stats['mnorm']:.3f} "
                  f"cos(g,m)={stats['cos_gm']:+.2f}{overshoot}",
                  flush=True)
            sim.render_episode(iteration=it, callback=target_cb,
                                playback_speed=args.playback_speed,
                                start_paused=(args.start_paused and it == 0))
    elapsed = time.perf_counter() - t0
    print(f"\nDone — total wall: {elapsed:.1f}s")
    metrics = sim.final_metrics()
    print(f"Final pos_err: {metrics['pos_error_m']:.3f} m")
    print(f"Final term_vel: {metrics['terminal_speed_mps']:.3f} m/s")
    print(f"Final yaw_err: {np.rad2deg(metrics['yaw_error_rad']):.1f}°")
    print(f"Control jerk: {metrics['control_jerk']:.1f}")

    # Final endless replay — close the window to exit.
    print("\nReplaying final trajectory (close the GL window to exit) ...")
    sim.render_episode(iteration=args.iterations, callback=target_cb,
                        playback_speed=args.playback_speed,
                        loop=True, loops_count=1000)
    sim.close()


if __name__ == "__main__":
    main()
