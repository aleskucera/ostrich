"""Helhest_junior box trajectory optimization using Axion adjoint gradients.

Box-obstacle counterpart of experiments/3_gradient_quality. Optimizes K spline
control points to match a real-robot trajectory while crossing the box, using
the *yaw-tuned best* physics params from experiments/1_sim_to_real_box.

Mirrors examples/helhest/gradient/trajectory_spline_box.py (which uses the
full helhest + a *simulated* target) but uses the junior model + a *real* GT
trajectory as the target — same pattern the original experiments/3 follows
for the flat-ground turn/accel scenes.

Outputs results/axion[_<gt>].json with per-trial loss curves so the gradient
quality can be compared once additional engines (semi-implicit, mjx, ...)
are added.

Usage:
    python experiments/3_gradient_quality_box/optimize_axion.py \
        --gt experiments/1_sim_to_real_box/data/run_2026_05_20-18_10_33.json
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import newton
import numpy as np
import warp as wp
from axion import (AxionDifferentiableSimulator, AxionEngineConfig, ComplianceConfig,
                   ContactsConfig, LinearSolverConfig, LinesearchConfig, LoggingConfig,
                   NewtonRaphsonConfig, RenderingConfig, SimulationConfig)

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# DOF layout: [0..5] free base, [6] left, [7] right, [8] rear
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


# ------------------------------ helpers --------------------------------------
def make_interp_matrix(T: int, K: int):
    """Linear interpolation matrix W [T, K] (one knot becomes a triangle)."""
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    """Adam on a [K, num_dofs] numpy parameter array with cosine LR decay."""
    def __init__(self, K, num_dofs, lr=0.1, lr_min_ratio=0.2, total_steps=50,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return params - self._cosine_lr() * mh / (np.sqrt(vh) + self.eps)


@wp.kernel
def chassis_xy_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),         # [T, W, B]
    target_pose: wp.array(dtype=wp.transform, ndim=3),       # [T, W, B]
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Mean squared XY distance between chassis (body 0) and the target."""
    t = wp.tid()
    p = wp.transform_get_translation(body_pose[t, 0, 0])
    q = wp.transform_get_translation(target_pose[t, 0, 0])
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def fill_target_chassis_kernel(
    target_xyz: wp.array(dtype=wp.vec3),                     # [T] world chassis pos
    target_pose: wp.array(dtype=wp.transform, ndim=3),       # [T, W, B]
):
    """Write target chassis position into target_body_pose[t, 0, 0]; leave
    other bodies as identity transforms (they're not consumed by the loss)."""
    t = wp.tid()
    target_pose[t, 0, 0] = wp.transform(target_xyz[t], wp.quat_identity())


# ------------------------------ optimizer ------------------------------------
class HelhestJuniorBoxOptimizer(AxionDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t, K=10,
                 mu_front=0.8, mu_rear=1.2, mu_rolling=0.7):
        self.K = K
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.mu_rolling = mu_rolling
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)  # [T,K], [K]
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        self._setup_target(target_xyz_rel, target_t)

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0))
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0, k_d=0.0,
            friction_left_right=self.mu_front, friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling)
        return self.builder.finalize_replicated(num_worlds=1, requires_grad=True)

    def _setup_target(self, target_xyz_rel, target_t):
        """Resample real GT onto sim timesteps and write into target_body_pose.
        GT is start-aligned to origin; we shift to sim's chassis-start frame."""
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T) * dt
        target = np.zeros((T, 3), dtype=np.float32)
        for c in range(3):
            target[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])

        # Sim chassis world origin = body 0 at initial state.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        sim_origin = self.states[0].body_q.numpy()[0, :3]  # [x,y,z]
        target_world = target + sim_origin.astype(np.float32)

        target_vec3 = wp.array(target_world, dtype=wp.vec3, device=self.model.device)
        wp.launch(fill_target_chassis_kernel, dim=T,
                  inputs=[target_vec3], outputs=[self.trajectory.target_body_pose],
                  device=self.model.device)

    # ---- spline expand / contract / apply ----
    def _expand(self, params):
        return self.W @ params  # [T,K]@[K,3] = [T,3]

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    # ---- AxionDifferentiableSimulator hooks ----
    def compute_loss(self):
        T = self.trajectory.body_pose.shape[0]
        wp.launch(chassis_xy_loss_kernel, dim=T,
                  inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose, 1.0 / T],
                  outputs=[self.loss], device=self.solver.model.device)

    def update(self):
        # update is called after diff_step; spline_adam is owned by the trainer
        pass

    # ---- one optimisation iteration ----
    def opt_step(self):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]   # [T,3]
        grad_params = self._contract(grad_v)                            # [K,3]
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val


# ------------------------------ trial driver ---------------------------------
def make_configs(duration, dt):
    sim_cfg = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt,
                                num_worlds=1, use_cuda_graph=False)
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / dt)), start_paused=False)
    ec = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))
    return sim_cfg, rc, ec


def run_trial(target_xyz_rel, target_t, K, lr, iterations, seed, duration, dt):
    sim_cfg, rc, ec = make_configs(duration, dt)
    sim = HelhestJuniorBoxOptimizer(sim_cfg, rc, ec, LoggingConfig(),
                                     target_xyz_rel, target_t, K=K)
    # Random initial spline params: small forward bias + per-trial noise
    rng = np.random.default_rng(seed)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_WHEEL_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS,
                                  lr=lr, lr_min_ratio=0.2, total_steps=iterations)
    sim._apply_params(sim.spline_params)

    losses = []
    t0_total = time.perf_counter()
    for it in range(iterations):
        t0 = time.perf_counter()
        loss = sim.opt_step()
        losses.append(loss)
        if it % 5 == 0 or it == iterations - 1:
            print(f"    iter {it:3d}: loss={loss:.4f}  ({time.perf_counter()-t0:.2f}s)")
    elapsed = time.perf_counter() - t0_total
    sim.close()
    del sim
    return {"seed": int(seed), "losses": losses, "wall_s": elapsed,
            "best_loss": float(min(losses))}


# --------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"),
                    help="ground-truth JSON (from 1_sim_to_real_box/data/)")
    ap.add_argument("--K", type=int, default=10, help="spline control points")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0,
                    help="trajectory horizon (s). Box crossing happens around t≈3-5 s")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xyz = np.column_stack([gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]])
    # Clip target to the horizon (only the part the sim will cover).
    m = (real_t >= 0) & (real_t <= args.horizon_s)
    real_t = real_t[m]; real_xyz = real_xyz[m]
    print(f"Loaded GT {gt['run_id']}: {len(real_t)} target points over t in "
          f"[{real_t.min():.2f}, {real_t.max():.2f}] s")
    print(f"K={args.K}  iters={args.iterations}  lr={args.lr}  trials={args.num_trials}  "
          f"dt={args.dt}  horizon={args.horizon_s}s")

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        print(f"\n--- trial {k + 1}/{args.num_trials} (seed={seed}) ---")
        trials.append(run_trial(real_xyz, real_t, args.K, args.lr, args.iterations,
                                  seed, args.horizon_s, args.dt))

    out = {
        "simulator": "Axion",
        "gt": gt["run_id"],
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "axion.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    best = min(t["best_loss"] for t in trials)
    print(f"\nBest loss across {args.num_trials} trials: {best:.4f}   -> {save_path}")


if __name__ == "__main__":
    main()
