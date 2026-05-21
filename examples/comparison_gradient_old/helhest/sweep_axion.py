"""Hyperparameter sweep for Axion helhest — minimize trajectory error vs ground truth.

Each config runs in a subprocess to avoid GPU memory leaks between runs.

For Axion the sweepable parameters are:
  - k_p (velocity servo proportional gain, = target_ke)
  - friction_left_right (front wheel friction)
  - friction_rear (rear wheel friction)
  (dt is fixed at 0.05 — Axion's implicit solver handles large timesteps)

Usage:
    python examples/comparison_gradient/helhest/sweep_axion.py \
        --ground-truth results/helhest_chrono.json \
        --save results/sweep_axion.json
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np


DURATION = 3.0
DT = 0.05


def run_single_config(dt, k_p, friction_lr, friction_rear):
    """Run one Axion config in a subprocess, return trajectory as list of [x, y]."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    worker_code = f"""
import os, json, sys
os.environ["PYOPENGL_PLATFORM"] = "glx"
import newton, numpy as np, warp as wp
from axion import (
    AxionDifferentiableSimulator, AxionEngineConfig, ComplianceConfig,
    ContactsConfig, LinearSolverConfig, LinesearchConfig,
    LoggingConfig, NewtonRaphsonConfig, RenderingConfig, SimulationConfig,
)
from axion.simulation.sim_config import SyncMode
from examples.helhest.common import create_helhest_model, HelhestConfig

DT = {dt}
K_P = {k_p}
FRICTION_LR = {friction_lr}
FRICTION_REAR = {friction_rear}
DURATION = {DURATION}
TARGET_CTRL = (1.0, 6.0, 0.0)
WHEEL_DOF_OFFSET = 6

class SweepSim(AxionDifferentiableSimulator):
    def build_model(self):
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=K_P, k_d=HelhestConfig.TARGET_KD,
            friction_left_right=FRICTION_LR, friction_rear=FRICTION_REAR,
        )
        return self.builder.finalize_replicated(num_worlds=1, requires_grad=False)
    def compute_loss(self): pass
    def update(self): pass

sim = SweepSim(
    SimulationConfig(duration_seconds=DURATION, target_timestep_seconds=DT, num_worlds=1, sync_mode=SyncMode.ALIGN_FPS_TO_DT, use_cuda_graph=False),
    RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False),
    AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=12, backtrack_min_iter=8, atol=1e-1),
        linear=LinearSolverConfig(max_iters=12, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-6, friction=1e-6),
        contacts=ContactsConfig(max_per_world=8),
        linesearch=LinesearchConfig(enabled=False),
    ),
    LoggingConfig(),
)

model = sim.model
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.states[0])
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.target_states[0])

T = sim.clock.total_sim_steps
num_dofs = sim.trajectory.joint_target_vel.shape[-1]
for i in range(T):
    ctrl = np.zeros(num_dofs, dtype=np.float32)
    ctrl[WHEEL_DOF_OFFSET + 0] = TARGET_CTRL[0]
    ctrl[WHEEL_DOF_OFFSET + 1] = TARGET_CTRL[1]
    ctrl[WHEEL_DOF_OFFSET + 2] = TARGET_CTRL[2]
    wp.copy(sim.target_controls[i].joint_target_vel,
            wp.array(ctrl, dtype=wp.float32, device=model.device))
sim.run_target_episode()

num_steps = sim.trajectory.target_body_pose.shape[0]
traj = []
for t in range(num_steps):
    bp = sim.trajectory.target_body_pose[t].numpy()[0, 0]
    traj.append([float(bp[0]), float(bp[1])])

import json, pathlib
pathlib.Path("{tmp_path}").write_text(json.dumps(traj))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:] if result.stderr else "unknown error")
        with open(tmp_path) as f:
            traj = json.load(f)
        return traj
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def trajectory_error(traj_sim, traj_gt):
    """Mean L2 distance between two trajectories, resampled to common length."""
    sim_np = np.array(traj_sim)
    gt_np = np.array(traj_gt)
    n_sim, n_gt = len(sim_np), len(gt_np)
    n = min(n_sim, n_gt, 500)

    t_sim = np.linspace(0, 1, n_sim)
    t_gt = np.linspace(0, 1, n_gt)
    t_common = np.linspace(0, 1, n)

    sim_x = np.interp(t_common, t_sim, sim_np[:, 0])
    sim_y = np.interp(t_common, t_sim, sim_np[:, 1])
    gt_x = np.interp(t_common, t_gt, gt_np[:, 0])
    gt_y = np.interp(t_common, t_gt, gt_np[:, 1])

    return float(np.mean(np.sqrt((sim_x - gt_x)**2 + (sim_y - gt_y)**2)))


def run_sweep(configs, traj_gt, label=""):
    results = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        try:
            traj = run_single_config(**cfg)
            err = trajectory_error(traj, traj_gt)
            results.append({"params": cfg, "error": err, "final_xy": traj[-1]})
            print(f"  {label} [{i+1}/{n}] err={err:.4f} "
                  f"dt={cfg['dt']} k_p={cfg['k_p']} "
                  f"ff={cfg['friction_lr']} rf={cfg['friction_rear']}")
        except Exception as e:
            err_msg = str(e)[:200]
            results.append({"params": cfg, "error": float("inf"),
                            "final_xy": [0, 0], "exception": err_msg})
            print(f"  {label} [{i+1}/{n}] FAILED: {err_msg}")

    results.sort(key=lambda r: r["error"])
    return results


def build_coarse_configs():
    configs = []
    for dt in [DT]:
        for k_p in [50.0, 100.0, 150.0, 200.0]:
            for friction_lr in [0.5, 0.7, 1.0, 1.5]:
                for friction_rear in [0.2, 0.35, 0.5]:
                    configs.append(dict(
                        dt=dt, k_p=k_p,
                        friction_lr=friction_lr,
                        friction_rear=friction_rear,
                    ))
    return configs


def build_fine_configs(best):
    configs = []
    dt_b = best["dt"]
    kp_b = best["k_p"]
    ff_b = best["friction_lr"]
    rf_b = best["friction_rear"]

    for dt in [dt_b]:
        for k_p in [kp_b * 0.8, kp_b * 0.9, kp_b, kp_b * 1.1, kp_b * 1.2]:
            for ff_mult in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]:
                for rf_mult in [0.85, 0.95, 1.0, 1.05, 1.15]:
                    configs.append(dict(
                        dt=dt, k_p=k_p,
                        friction_lr=ff_b * ff_mult,
                        friction_rear=rf_b * rf_mult,
                    ))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True,
                        help="Path to ground truth trajectory JSON")
    parser.add_argument("--save", metavar="PATH",
                        help="Save sweep results to JSON")
    parser.add_argument("--top", type=int, default=10,
                        help="Print top N results (default: 10)")
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        gt = json.load(f)
    traj_gt = gt["target_trajectory"]
    gt_np = np.array(traj_gt)
    print(f"Ground truth: {gt.get('simulator', '?')}, "
          f"dt={gt['dt']}, T={gt['T']}, "
          f"final=({gt_np[-1, 0]:.3f}, {gt_np[-1, 1]:.3f})")

    # Stage 1: coarse sweep
    print(f"\n=== Stage 1: Coarse sweep ===")
    coarse_configs = build_coarse_configs()
    print(f"Running {len(coarse_configs)} configurations...")
    t0 = time.perf_counter()
    coarse_results = run_sweep(coarse_configs, traj_gt, "coarse")
    t_coarse = time.perf_counter() - t0
    print(f"Coarse sweep done in {t_coarse:.1f}s")

    print(f"\nTop {args.top} coarse results:")
    for i, r in enumerate(coarse_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} k_p={p['k_p']} "
              f"ff={p['friction_lr']} rf={p['friction_rear']} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    # Stage 2: fine sweep
    print(f"\n=== Stage 2: Fine sweep ===")
    best_coarse = coarse_results[0]["params"]
    fine_configs = build_fine_configs(best_coarse)
    print(f"Running {len(fine_configs)} configurations...")
    t0 = time.perf_counter()
    fine_results = run_sweep(fine_configs, traj_gt, "fine")
    t_fine = time.perf_counter() - t0
    print(f"Fine sweep done in {t_fine:.1f}s")

    print(f"\nTop {args.top} fine results:")
    for i, r in enumerate(fine_results[:args.top]):
        p = r["params"]
        print(f"  {i+1}. err={r['error']:.4f} | "
              f"dt={p['dt']} k_p={p['k_p']:.1f} "
              f"ff={p['friction_lr']:.2f} rf={p['friction_rear']:.2f} "
              f"| final=({r['final_xy'][0]:.3f}, {r['final_xy'][1]:.3f})")

    if args.save:
        best = fine_results[0]
        output = {
            "simulator": "Axion",
            "ground_truth": args.ground_truth,
            "best_params": best["params"],
            "best_error": best["error"],
            "best_final_xy": best["final_xy"],
            "coarse_sweep_size": len(coarse_configs),
            "fine_sweep_size": len(fine_configs),
            "top_10_coarse": [{"error": r["error"], "params": r["params"],
                               "final_xy": r["final_xy"]}
                              for r in coarse_results[:10]],
            "top_10_fine": [{"error": r["error"], "params": r["params"],
                             "final_xy": r["final_xy"]}
                            for r in fine_results[:10]],
        }
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
