"""Helhest_junior box trajectory optimization using Newton Semi-Implicit
+ Warp tape BPTT.

SemiImplicit counterpart of optimize_axion.py / optimize_mjx.py. Optimises
a K-knot wheel-velocity spline so the helhest_junior matches a recorded
real trajectory while crossing the box, with gradients via backpropagation
through Warp's computation tape over Newton's SemiImplicit solver.

Calibrated SI physics + joint stiffness from experiments/1_sim_to_real_box
(mu=0.05, ke=8e4, kd=2e3, kf=1500, k_d_act=200, joint_attach_ke=1e6,
dt=5e-4 — at SI's stability edge on this scene, larger dt NaNs).

Outputs results/semi_implicit.json with the same per-trial loss-curve
schema as optimize_axion.py, so plot_results.py overlays both engines.

Usage:
    python experiments/3_gradient_quality_box/optimize_semi_implicit.py
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
from axion import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                   SimulationConfig)
from axion.simulation.differentiable_simulator import NewtonDifferentiableSimulator

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# DOF layout: [0..5] free base, [6] left, [7] right, [8] rear
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# SI tuned best on this scene (from 1_sim_to_real_box re-tune).
SI_MU = 0.05
SI_KE = 8e4
SI_KD = 2e3
SI_KF = 1500.0
SI_K_D_ACT = 200.0
SI_JOINT_ATTACH_KE = 1e6
SI_JOINT_ATTACH_KD = 1e2


# ------------------------------ helpers --------------------------------------
def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.05, lr_min_ratio=0.2, total_steps=50,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr; self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps; self.b1, self.b2 = betas; self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return params - self._lr() * mh / (np.sqrt(vh) + self.eps)


@wp.kernel
def chassis_xy_loss_step_kernel(
    body_q: wp.array(dtype=wp.transform),      # state.body_q at one timestep
    target_xy: wp.array(dtype=wp.vec2),        # [T+1] target XY per step
    step_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """One-timestep XY-distance contribution: state[step_idx] vs target_xy[step_idx]."""
    p = wp.transform_get_translation(body_q[0])   # body 0 = chassis
    q = target_xy[step_idx]
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


# ------------------------------ optimizer ------------------------------------
class HelhestJuniorBoxSIOptimizer(NewtonDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 target_xyz_rel, target_t, K=10):
        self.K = K
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self._setup_target(target_xyz_rel, target_t)

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF))
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=0.0, k_d=SI_K_D_ACT,   # SI needs nonzero k_d in TARGET_VELOCITY mode
            friction_left_right=SI_MU, friction_rear=SI_MU,
            mu_rolling=0.7,
            ke=SI_KE, kd=SI_KD, kf=SI_KF)
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True)

    def _setup_target(self, target_xyz_rel, target_t):
        """Resample real GT into sim-frame XY (shifted by chassis spawn)."""
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        t_sim = np.arange(T + 1) * dt
        target_xy_rel = np.zeros((T + 1, 2), dtype=np.float32)
        for c in range(2):
            target_xy_rel[:, c] = np.interp(t_sim, target_t, target_xyz_rel[:, c])

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        sim_origin = self.states[0].body_q.numpy()[0, :2]
        target_xy_world = target_xy_rel + sim_origin.astype(np.float32)
        self.target_xy = wp.array(target_xy_world, dtype=wp.vec2, requires_grad=False,
                                   device=self.model.device)

    # ---- spline expand / contract / apply ----
    def _expand(self, params):
        return self.W @ params  # [T,K]@[K,3] -> [T,3]

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        """Write expanded per-step controls into self.controls[i].joint_target_vel."""
        T = self.clock.total_sim_steps
        num_dofs = self.controls[0].joint_target_vel.shape[-1]
        expanded = self._expand(params)
        for i in range(T):
            ctrl_np = np.zeros(num_dofs, dtype=np.float32)
            ctrl_np[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(self.controls[i].joint_target_vel,
                    wp.array(ctrl_np, dtype=wp.float32, device=self.model.device))

    # ---- loss (called by diff_step) ----
    def compute_loss(self):
        T = self.clock.total_sim_steps
        for t in range(T):
            wp.launch(
                chassis_xy_loss_step_kernel, dim=1,
                inputs=[self.states[t + 1].body_q, self.target_xy, t + 1, 1.0 / T],
                outputs=[self.loss], device=self.model.device)

    def update(self):
        # Owned by the trainer (see opt_step) — keep this a no-op so the
        # NewtonDifferentiableSimulator hook contract is satisfied.
        pass

    # ---- one optimisation iteration ----
    def opt_step(self):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        # NewtonDifferentiableSimulator exposes per-step controls with .grad;
        # collect per-step grads on joint_target_vel into a [T,3] array.
        T = self.clock.total_sim_steps
        grad_v = np.zeros((T, NUM_WHEEL_DOFS), dtype=np.float32)
        for i in range(T):
            g = self.controls[i].joint_target_vel.grad.numpy()
            grad_v[i] = g[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
            self.controls[i].joint_target_vel.grad.zero_()
        grad_params = self._contract(grad_v)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val


# ------------------------------ trial driver ---------------------------------
def make_configs(duration, dt):
    sim_cfg = SimulationConfig(
        duration_seconds=duration, target_timestep_seconds=dt,
        num_worlds=1, use_cuda_graph=True)  # graph capture also helps SI BPTT
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / dt)),
                          start_paused=False)
    ec = SemiImplicitEngineConfig(
        angular_damping=0.05, friction_smoothing=0.1,
        joint_attach_ke=SI_JOINT_ATTACH_KE, joint_attach_kd=SI_JOINT_ATTACH_KD)
    return sim_cfg, rc, ec


def run_trial(target_xyz_rel, target_t, K, lr, iterations, seed, duration, dt):
    sim_cfg, rc, ec = make_configs(duration, dt)
    sim = HelhestJuniorBoxSIOptimizer(sim_cfg, rc, ec, LoggingConfig(),
                                       target_xyz_rel, target_t, K=K)
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
            print(f"    iter {it:3d}: loss={loss:.4f}  ({time.perf_counter() - t0:.2f}s)")
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
                                         / "run_2026_05_20-18_10_33.json"))
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.05,
                    help="lower than Axion's 0.1 — SI gradients can be noisy")
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42)
    # 6 s would be 12k SI steps and a heavy Warp tape; default 4 s.
    ap.add_argument("--horizon-s", type=float, default=4.0)
    # dt=5e-4 is the stability edge on this scene (2_dt_stability_box: 7e-4
    # already NaNs at our tuned k_d_act). Don't bump this without re-tuning.
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xyz = np.column_stack([gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]])
    m = (real_t >= 0) & (real_t <= args.horizon_s)
    real_t = real_t[m]; real_xyz = real_xyz[m]

    print(f"Loaded GT {gt['run_id']}: {len(real_t)} target points over "
          f"t in [{real_t.min():.2f}, {real_t.max():.2f}] s")
    print(f"K={args.K}  iters={args.iterations}  lr={args.lr}  "
          f"trials={args.num_trials}  dt={args.dt}  horizon={args.horizon_s}s")
    print(f"-> {int(args.horizon_s / args.dt)} SI steps per iter "
          f"(BPTT tape ~ steps × per-step kernels)")

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        print(f"\n--- trial {k + 1}/{args.num_trials} (seed={seed}) ---")
        trials.append(run_trial(real_xyz, real_t, args.K, args.lr, args.iterations,
                                  seed, args.horizon_s, args.dt))

    out = {
        "simulator": "Semi-Implicit",
        "gradient_method": "warp tape (BPTT)",
        "gt": gt["run_id"],
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "semi_implicit.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    best = min(t["best_loss"] for t in trials)
    print(f"\nBest loss across {args.num_trials} trials: {best:.4f}   -> {save_path}")


if __name__ == "__main__":
    main()
