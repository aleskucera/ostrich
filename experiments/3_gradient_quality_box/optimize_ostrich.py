"""Helhest_junior box trajectory optimization using Ostrich adjoint gradients.

Box-obstacle counterpart of experiments/3_gradient_quality. Optimizes K spline
control points to match a real-robot trajectory while crossing the box, using
the *yaw-tuned best* physics params from experiments/1_sim_to_real_box.

Mirrors examples/helhest/gradient/trajectory_spline_box.py (which uses the
full helhest + a *simulated* target) but uses the junior model + a *real* GT
trajectory as the target — same pattern the original experiments/3 follows
for the flat-ground turn/accel scenes.

Outputs results/ostrich[_<gt>].json with per-trial loss curves so the gradient
quality can be compared once additional engines (semi-implicit, mjx, ...)
are added.

Usage:
    python experiments/3_gradient_quality_box/optimize_ostrich.py \
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
from ostrich import (OstrichDifferentiableSimulator, OstrichEngineConfig, ComplianceConfig,
                   ContactsConfig, LinearSolverConfig, LinesearchConfig, LoggingConfig,
                   NewtonRaphsonConfig, RenderingConfig, SimulationConfig)

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# DOF layout: [0..5] free base, [6] left, [7] right, [8] rear
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3       # number of driven wheels (L, R, rear)
NUM_OPT_DOFS = 2         # we optimize only L and R; rear = (L+R)/2 (skid-steer)


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
def chassis_yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),         # [T, W, B]
    target_yaw_rel: wp.array(dtype=wp.float32),              # [T]
    yaw_offset: float,                                       # sim's initial yaw (world frame)
    weight: float,                                           # L^2 / T
    loss: wp.array(dtype=wp.float32),
):
    """Mean squared (L * yaw_rel error). target_yaw_rel is yaw relative to sim's
    starting yaw (matches the GT pre-processing convention)."""
    t = wp.tid()
    q = wp.transform_get_rotation(body_pose[t, 0, 0])
    # Extract yaw from quaternion (z-up world, planar motion dominant).
    qw = q[3]; qx = q[0]; qy = q[1]; qz = q[2]
    yaw_sim = wp.atan2(2.0 * (qw * qz + qx * qy),
                       1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_rel = yaw_sim - yaw_offset
    # Wrap difference to [-pi, pi].
    d = yaw_rel - target_yaw_rel[t]
    PI = 3.14159265358979323846
    if d > PI:
        d = d - 2.0 * PI
    if d < -PI:
        d = d + 2.0 * PI
    wp.atomic_add(loss, 0, weight * d * d)


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
class HelhestJuniorBoxOptimizer(OstrichDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t, target_yaw_rel=None,
                 K=10, yaw_lever=0.5, smooth_lambda=0.0,
                 mu_front=0.8, mu_rear=1.2, mu_rolling=0.7):
        self.K = K
        self.yaw_lever = float(yaw_lever)  # L in (L * RMSE(Δyaw))
        self.smooth_lambda = float(smooth_lambda)
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.mu_rolling = mu_rolling
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)  # [T,K], [K]
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        self._setup_target(target_xyz_rel, target_t, target_yaw_rel)

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

    def _setup_target(self, target_xyz_rel, target_t, target_yaw_rel=None):
        """Resample real GT onto sim timesteps and write into target_body_pose.
        GT is start-aligned to origin; we shift to sim's chassis-start frame.
        If target_yaw_rel is provided, also stores it for the yaw loss kernel."""
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T) * dt
        target = np.zeros((T, 3), dtype=np.float32)
        for c in range(3):
            target[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])

        # Sim chassis world origin and yaw = body 0 at initial state.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        body_q0 = self.states[0].body_q.numpy()[0]
        sim_origin = body_q0[:3]
        qx, qy, qz, qw = body_q0[3], body_q0[4], body_q0[5], body_q0[6]
        sim_yaw0 = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                    1.0 - 2.0 * (qy * qy + qz * qz)))
        target_world = target + sim_origin.astype(np.float32)

        target_vec3 = wp.array(target_world, dtype=wp.vec3, device=self.model.device)
        wp.launch(fill_target_chassis_kernel, dim=T,
                  inputs=[target_vec3], outputs=[self.trajectory.target_body_pose],
                  device=self.model.device)

        if target_yaw_rel is not None:
            target_yaw = np.interp(t_sim, target_t, target_yaw_rel).astype(np.float32)
            self.target_yaw_rel = wp.array(target_yaw, dtype=wp.float32,
                                           device=self.model.device)
            self.yaw_offset = sim_yaw0
        else:
            self.target_yaw_rel = None
            self.yaw_offset = sim_yaw0

    # ---- spline expand / contract / apply ----
    def _expand(self, params):
        return self.W @ params  # [T,K]@[K,3] = [T,3]

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        """params is [K, 2] (left, right). Rear wheel velocity is
        constrained to (left + right) / 2 by skid-steer kinematics."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded_lr = self._expand(params)                        # [T, 2]
        wheel_vel = np.zeros((T, NUM_WHEEL_DOFS), dtype=np.float32)
        wheel_vel[:, 0] = expanded_lr[:, 0]                       # left
        wheel_vel[:, 1] = expanded_lr[:, 1]                       # right
        wheel_vel[:, 2] = 0.5 * (expanded_lr[:, 0] + expanded_lr[:, 1])  # rear
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = wheel_vel
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    # ---- OstrichDifferentiableSimulator hooks ----
    def compute_loss(self):
        T = self.trajectory.body_pose.shape[0]
        wp.launch(chassis_xy_loss_kernel, dim=T,
                  inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose, 1.0 / T],
                  outputs=[self.loss], device=self.solver.model.device)
        if self.target_yaw_rel is not None and self.yaw_lever > 0.0:
            wp.launch(chassis_yaw_loss_kernel, dim=T,
                      inputs=[self.trajectory.body_pose,
                              self.target_yaw_rel,
                              self.yaw_offset,
                              (self.yaw_lever * self.yaw_lever) / T],
                      outputs=[self.loss],
                      device=self.solver.model.device)

    def update(self):
        # update is called after diff_step; spline_adam is owned by the trainer
        pass

    # ---- one optimisation iteration ----
    def opt_step(self):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        grad_w = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]   # [T,3]
        # Project [T,3] wheel gradient back to [T,2] (L, R) under rear=(L+R)/2:
        # d_loss/d_L += 0.5 * d_loss/d_rear; d_loss/d_R += 0.5 * d_loss/d_rear.
        grad_v = np.zeros((grad_w.shape[0], NUM_OPT_DOFS), dtype=grad_w.dtype)
        grad_v[:, 0] = grad_w[:, 0] + 0.5 * grad_w[:, 2]
        grad_v[:, 1] = grad_w[:, 1] + 0.5 * grad_w[:, 2]
        grad_params = self._contract(grad_v)                            # [K,2]
        # Smoothness regularisation on the spline knots (Python-side; the term
        # only depends on params, not on the Warp graph).
        if self.smooth_lambda > 0.0:
            p = self.spline_params
            diff = p[1:] - p[:-1]                                       # [K-1, 3]
            smooth_loss = self.smooth_lambda * float((diff * diff).sum())
            loss_val += smooth_loss
            grad_smooth = np.zeros_like(p)
            grad_smooth[:-1] -= 2.0 * self.smooth_lambda * diff
            grad_smooth[1:]  += 2.0 * self.smooth_lambda * diff
            grad_params = grad_params + grad_smooth
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val


# ------------------------------ trial driver ---------------------------------
def make_configs(duration, dt):
    # use_cuda_graph=True is essential: OstrichDifferentiableSimulator.diff_step()
    # captures the full forward+backward into a CUDA graph and replays it each
    # iteration. Without it every iter pays full kernel-launch overhead — we
    # saw ~160 ms/step in that mode vs ~9 ms/step in original exp-3 (which
    # uses CUDA graph).
    sim_cfg = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt,
                                num_worlds=1, use_cuda_graph=True)
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / dt)), start_paused=False)
    ec = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))
    return sim_cfg, rc, ec


def run_trial(target_xyz_rel, target_t, K, lr, iterations, seed, duration, dt,
              target_yaw_rel=None, yaw_lever=0.5, smooth_lambda=0.0):
    sim_cfg, rc, ec = make_configs(duration, dt)
    sim = HelhestJuniorBoxOptimizer(sim_cfg, rc, ec, LoggingConfig(),
                                     target_xyz_rel, target_t,
                                     target_yaw_rel=target_yaw_rel,
                                     K=K, yaw_lever=yaw_lever,
                                     smooth_lambda=smooth_lambda)
    # Random initial spline params: small forward bias + per-trial noise.
    # Shape [K, NUM_OPT_DOFS] = [K, 2] (left, right). Rear coupled at apply time.
    rng = np.random.default_rng(seed)
    init = 2.0 + 0.5 * rng.standard_normal((K, NUM_OPT_DOFS))
    sim.spline_params = init.astype(np.float64)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_OPT_DOFS,
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
    final_params = np.asarray(sim.spline_params).tolist()
    sim.close()
    del sim
    return {"seed": int(seed), "losses": losses, "wall_s": elapsed,
            "best_loss": float(min(losses)),
            "init_params": init.tolist(),
            "final_params": final_params}


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
    # dt=0.1 sits in Ostrich's accuracy plateau (~0.066 m mean vs 0.063 at
    # dt=0.05; see experiments/2_dt_stability_box). Halves the step count
    # for free. dt=0.2 would also be fine (~0.076 m), 0.3 starts costing
    # accuracy (~0.088 m).
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--yaw-lever", type=float, default=0.5,
                    help="L in yaw-aware loss (m); 0 disables yaw term")
    ap.add_argument("--smooth-lambda", type=float, default=0.01,
                    help="L2 smoothness reg on spline knot differences")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xyz = np.column_stack([gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]])
    real_yaw = np.asarray(gt["real"]["yaw_rel"]) if "yaw_rel" in gt["real"] else None
    # Clip target to the horizon (only the part the sim will cover).
    m = (real_t >= 0) & (real_t <= args.horizon_s)
    real_t = real_t[m]; real_xyz = real_xyz[m]
    if real_yaw is not None:
        real_yaw = real_yaw[m]
    print(f"Loaded GT {gt['run_id']}: {len(real_t)} target points over t in "
          f"[{real_t.min():.2f}, {real_t.max():.2f}] s")
    print(f"K={args.K}  iters={args.iterations}  lr={args.lr}  trials={args.num_trials}  "
          f"dt={args.dt}  horizon={args.horizon_s}s  "
          f"yaw_L={args.yaw_lever}  smooth_lambda={args.smooth_lambda}")

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        print(f"\n--- trial {k + 1}/{args.num_trials} (seed={seed}) ---")
        trials.append(run_trial(real_xyz, real_t, args.K, args.lr, args.iterations,
                                  seed, args.horizon_s, args.dt,
                                  target_yaw_rel=real_yaw,
                                  yaw_lever=args.yaw_lever,
                                  smooth_lambda=args.smooth_lambda))

    out = {
        "simulator": "Ostrich",
        "gt": gt["run_id"],
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "ostrich.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    best = min(t["best_loss"] for t in trials)
    print(f"\nBest loss across {args.num_trials} trials: {best:.4f}   -> {save_path}")


if __name__ == "__main__":
    main()
