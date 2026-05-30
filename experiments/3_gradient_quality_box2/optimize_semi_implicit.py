"""Helhest_junior box random-IC final-pose optimisation — Semi-Implicit.

Newton SI counterpart of optimize_axion.py / optimize_mjx.py. Same task
(random IC + random target + weighted 5-term loss), same calibrated
physics (mu=0.05, ke=8e4, kd=2e3, kf=1500, k_d_act=200, joint_attach_ke=1e6,
dt=5e-4).

Gradient clipping at 1.0 is kept on by default — same safety net the
gradient-quality experiment proved essential for SI at horizon=6 s.

Usage:
    python experiments/3_gradient_quality_box2/optimize_semi_implicit.py
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
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# SI tuned best from experiments/1_sim_to_real_box / 3_gradient_quality_box.
SI_MU = 0.05
SI_KE = 8e4
SI_KD = 2e3
SI_KF = 1500.0
SI_K_D_ACT = 200.0
SI_JOINT_ATTACH_KE = 1e6
SI_JOINT_ATTACH_KD = 1e2


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    def __init__(self, K, num_dofs, lr=0.02, lr_min_ratio=0.2, total_steps=100,
                 betas=(0.9, 0.999), eps=1e-8):
        self.lr_init = lr; self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps; self.b1, self.b2 = betas; self.eps = eps
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


# ------------------------------ loss kernels ---------------------------------
@wp.kernel
def chassis_track_step_kernel(
    body_q: wp.array(dtype=wp.transform),    # state.body_q (single timestep)
    ref_x: float, ref_y: float,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Per-step chassis-xy tracking loss against a reference waypoint.

    L_track = w * Σ_t ‖xy(t) − ref_xy(t)‖²  (chassis = body 0). Launched once
    per timestep on that state's body_q (SI keeps per-step State objects, so
    there is no batched body_pose array to launch over in one go like Axion).

    Gives each spline knot a direct gradient signal at every timestep instead
    of relying on the terminal loss to compound back through thousands of BPTT
    steps — the same shaping term used in optimize_axion.py / optimize_mjx.py.
    """
    p = wp.transform_get_translation(body_q[0])
    dx = p[0] - ref_x
    dy = p[1] - ref_y
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def final_pos_loss_step_kernel(
    body_q: wp.array(dtype=wp.transform),    # state.body_q (single timestep)
    target_x: float, target_y: float,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Final XY position loss (body 0 = chassis). Launched on the last state."""
    p = wp.transform_get_translation(body_q[0])
    dx = p[0] - target_x
    dy = p[1] - target_y
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def final_yaw_loss_step_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_yaw: float,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_yaw = w * (1 - cos(yaw - target_yaw)). Wrap-around safe."""
    q = wp.transform_get_rotation(body_q[0])
    qx = q[0]; qy = q[1]; qz = q[2]; qw = q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    delta = yaw - target_yaw
    wp.atomic_add(loss, 0, weight * (1.0 - wp.cos(delta)))


@wp.kernel
def terminal_vel_pair_kernel(
    body_q_t: wp.array(dtype=wp.transform),
    body_q_t1: wp.array(dtype=wp.transform),
    inv_dt: float,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Terminal-velocity contribution from one pair (t, t+1): w * ‖(xy[t+1]-xy[t])/dt‖²"""
    p0 = wp.transform_get_translation(body_q_t[0])
    p1 = wp.transform_get_translation(body_q_t1[0])
    vx = (p1[0] - p0[0]) * inv_dt
    vy = (p1[1] - p0[1]) * inv_dt
    wp.atomic_add(loss, 0, weight * (vx * vx + vy * vy))


@wp.kernel
def smoothness_pair_kernel(
    u_t: wp.array(dtype=wp.float32),     # controls[i].joint_target_vel
    u_t1: wp.array(dtype=wp.float32),    # controls[i+1].joint_target_vel
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_smooth contribution from one pair (t, t+1): w * Σ_k (u[t+1,k] - u[t,k])²"""
    k = wp.tid()
    dof = wheel_dof_offset + k
    du = u_t1[dof] - u_t[dof]
    wp.atomic_add(loss, 0, weight * du * du)


@wp.kernel
def magnitude_reg_step_kernel(
    u_t: wp.array(dtype=wp.float32),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_reg contribution from one timestep: w * Σ_k u[t,k]²"""
    k = wp.tid()
    dof = wheel_dof_offset + k
    u = u_t[dof]
    wp.atomic_add(loss, 0, weight * u * u)


# ------------------------------ optimizer ------------------------------------
class HelhestJuniorBoxFinalPoseSI(NewtonDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 ic_xy, ic_yaw, target_xy, target_yaw, weights, K=10,
                 terminal_tail_frac=0.10):
        self.K = K
        self.ic_xy = tuple(ic_xy)
        self.ic_yaw = float(ic_yaw)
        self.target_xy = tuple(target_xy)
        self.target_yaw = float(target_yaw)
        self.weights = weights
        self.terminal_tail_frac = float(terminal_tail_frac)
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Waypoint reference path: linear interpolation in xy from IC to
        # target, sampled at every state timestep (T_states = T+1). Used by
        # chassis_track_step_kernel for the per-step shaping signal.
        T_states = T + 1
        t_norm = np.arange(T_states, dtype=np.float32) / max(T_states - 1, 1)
        ref_xy_np = np.zeros((T_states, 2), dtype=np.float32)
        ref_xy_np[:, 0] = self.ic_xy[0] + t_norm * (self.target_xy[0] - self.ic_xy[0])
        ref_xy_np[:, 1] = self.ic_xy[1] + t_norm * (self.target_xy[1] - self.ic_xy[1])
        self.ref_xy_np = ref_xy_np

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=SI_MU, ke=SI_KE, kd=SI_KD, kf=SI_KF))
        chassis_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self.ic_yaw)
        chassis_xform = wp.transform(
            wp.vec3(float(self.ic_xy[0]), float(self.ic_xy[1]), 0.5), chassis_q)
        create_helhest_junior_model(
            self.builder, xform=chassis_xform,
            control_mode="velocity",
            k_p=0.0, k_d=SI_K_D_ACT,
            friction_left_right=SI_MU, friction_rear=SI_MU,
            mu_rolling=0.7,
            ke=SI_KE, kd=SI_KD, kf=SI_KF)
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, requires_grad=True)

    def _expand(self, params):
        return self.W @ params

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.controls[0].joint_target_vel.shape[-1]
        expanded = self._expand(params)
        for i in range(T):
            ctrl_np = np.zeros(num_dofs, dtype=np.float32)
            ctrl_np[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(self.controls[i].joint_target_vel,
                    wp.array(ctrl_np, dtype=wp.float32, device=self.model.device))

    def compute_loss(self):
        T = self.clock.total_sim_steps
        tail_len = max(2, int(round(T * self.terminal_tail_frac)))
        tail_start = T - tail_len
        device = self.solver.model.device
        w = self.weights

        # 0. Per-step waypoint tracking — dense shaping signal. One launch per
        #    POST-STEP state (states[1..T]), each reading that step's chassis
        #    xy. We deliberately skip states[0]: it is the eval_fk initial leaf
        #    (not produced by a sim step), and backpropping a loss through it
        #    NaNs the semi-implicit adjoint. box1 (3_gradient_quality_box) does
        #    the same. Normalise by T so the term is a mean over steps —
        #    dt-independent and comparable to Axion/MJX.
        if w.get("track", 0.0) > 0.0:
            T_states = T + 1
            per_step_track = float(w["track"] / T)
            for i in range(1, T_states):
                wp.launch(chassis_track_step_kernel, dim=1,
                          inputs=[self.states[i].body_q,
                                  float(self.ref_xy_np[i, 0]),
                                  float(self.ref_xy_np[i, 1]),
                                  per_step_track],
                          outputs=[self.loss], device=device)

        # 1. Final position (single launch on state[T])
        wp.launch(final_pos_loss_step_kernel, dim=1,
                  inputs=[self.states[T].body_q,
                          float(self.target_xy[0]), float(self.target_xy[1]),
                          float(w["pos"])],
                  outputs=[self.loss], device=device)

        # 2. Final yaw
        wp.launch(final_yaw_loss_step_kernel, dim=1,
                  inputs=[self.states[T].body_q,
                          float(self.target_yaw), float(w["yaw"])],
                  outputs=[self.loss], device=device)

        # 3. Terminal velocity (finite-diff on consecutive states in the tail)
        if w["vel"] > 0.0:
            inv_dt = float(1.0 / self.clock.dt)
            per_pair_weight = float(w["vel"] / tail_len)
            for i in range(tail_start, T):
                wp.launch(terminal_vel_pair_kernel, dim=1,
                          inputs=[self.states[i].body_q,
                                  self.states[i + 1].body_q,
                                  inv_dt, per_pair_weight],
                          outputs=[self.loss], device=device)

        # 4. Smoothness (Σ_t ‖u[t+1]-u[t]‖²) — one launch per consecutive pair
        if w["smooth"] > 0.0:
            per_step_weight = float(w["smooth"] / (T - 1))
            for i in range(T - 1):
                wp.launch(smoothness_pair_kernel, dim=NUM_WHEEL_DOFS,
                          inputs=[self.controls[i].joint_target_vel,
                                  self.controls[i + 1].joint_target_vel,
                                  WHEEL_DOF_OFFSET, per_step_weight],
                          outputs=[self.loss], device=device)

        # 5. Magnitude reg — one launch per timestep
        if w["reg"] > 0.0:
            per_step_weight = float(w["reg"] / T)
            for i in range(T):
                wp.launch(magnitude_reg_step_kernel, dim=NUM_WHEEL_DOFS,
                          inputs=[self.controls[i].joint_target_vel,
                                  WHEEL_DOF_OFFSET, per_step_weight],
                          outputs=[self.loss], device=device)

    def update(self):
        pass

    def opt_step(self, clip_grad_norm=1.0):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        T = self.clock.total_sim_steps
        grad_v = np.zeros((T, NUM_WHEEL_DOFS), dtype=np.float32)
        for i in range(T):
            g = self.controls[i].joint_target_vel.grad.numpy()
            grad_v[i] = g[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
            self.controls[i].joint_target_vel.grad.zero_()

        # NaN/Inf salvage. The SI adjoint over thousands of stiff-contact
        # steps can emit non-finite per-step gradients — typically a blow-up
        # originating at the violent box-contact step that then propagates
        # backward. If we let even one NaN through, the clip check
        # (`nan > clip` is False) passes it unclipped, Adam folds NaN into
        # m/v/params, and EVERY later iter is NaN — one bad backward kills the
        # whole run. Instead, zero only the non-finite entries (keeping the
        # finite components, often the post-contact timesteps) so the step
        # direction stays usable. Count them so the caller can report how
        # degenerate the gradient was.
        n_bad_grad = int((~np.isfinite(grad_v)).sum())
        if n_bad_grad > 0:
            grad_v = np.nan_to_num(grad_v, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_n_bad_grad = n_bad_grad

        grad_params = self._contract(grad_v)
        gnorm = float(np.linalg.norm(grad_params))
        # Backstop: if salvage still left a non-finite norm (or the whole
        # gradient was non-finite), skip the update so params/Adam stay clean.
        if not np.isfinite(gnorm):
            self.tape.zero()
            self.loss.zero_()
            return loss_val, gnorm
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            grad_params = grad_params * (clip_grad_norm / gnorm)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val, gnorm

    def final_metrics(self):
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        bp_T = self.states[T].body_q.numpy()[0]    # chassis pose: [x,y,z,qx,qy,qz,qw]
        bp_Tm1 = self.states[T - 1].body_q.numpy()[0]
        xy_T = bp_T[:2]; z_T = bp_T[2]
        q = bp_T[3:7]
        yaw_T = float(np.arctan2(
            2.0 * (q[3] * q[2] + q[0] * q[1]),
            1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])))
        v_xy = (bp_T[:2] - bp_Tm1[:2]) / dt
        terminal_speed = float(np.linalg.norm(v_xy))
        # Reconstruct u[t] from spline_params for jerk computation
        u = self._expand(self.spline_params)        # [T, 3]
        du = np.diff(u, axis=0)
        control_jerk = float(np.linalg.norm(du, axis=1).sum())
        pos_error = float(np.linalg.norm(xy_T - np.asarray(self.target_xy)))
        yaw_error = float(abs(np.arctan2(np.sin(yaw_T - self.target_yaw),
                                           np.cos(yaw_T - self.target_yaw))))
        return {
            "xy_final": [float(xy_T[0]), float(xy_T[1])],
            "z_final": float(z_T),
            "yaw_final": yaw_T,
            "pos_error_m": pos_error,
            "yaw_error_rad": yaw_error,
            "terminal_speed_mps": terminal_speed,
            "control_jerk": control_jerk,
        }


# ------------------------------ trial driver ---------------------------------
def make_configs(duration, dt):
    sim_cfg = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt,
                                num_worlds=1, use_cuda_graph=True)
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / dt)), start_paused=False)
    ec = SemiImplicitEngineConfig(
        angular_damping=0.05, friction_smoothing=0.1,
        joint_attach_ke=SI_JOINT_ATTACH_KE, joint_attach_kd=SI_JOINT_ATTACH_KD)
    return sim_cfg, rc, ec


def sample_ic_target(rng, ic_perturb, target_perturb):
    ic = {
        "xy": [
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
        ],
        "yaw": float(rng.uniform(-ic_perturb["yaw_rad"], ic_perturb["yaw_rad"])),
    }
    target = {
        "xy": [
            float(target_perturb["xy_center"][0]
                  + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])),
            float(target_perturb["xy_center"][1]
                  + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])),
        ],
        "yaw": float(rng.uniform(-target_perturb["yaw_rad"], target_perturb["yaw_rad"])),
    }
    return ic, target


WHEEL_RADIUS = 0.35  # m — matches examples/helhest_junior/common.py
INIT_TYPES = ("constant", "distance-aware")


def initial_spline(K, num_dofs, seed, init_type, target_xy, horizon,
                   noise_std=0.2, const_mean=2.0, const_std=0.5):
    """K×num_dofs initial spline guess — same helper as optimize_axion.py.

    'distance-aware' starts as a decaying ramp from 2·ω_avg to 0, where
    ω_avg = |target_x| / (horizon · wheel_radius); 'constant' is the old
    mean=2.0 init. Distance-aware bakes in the "drive then stop" shape.
    """
    rng = np.random.default_rng(seed)
    if init_type == "constant":
        return (const_mean + const_std * rng.standard_normal((K, num_dofs))).astype(np.float64)
    if init_type == "distance-aware":
        v_avg = float(abs(target_xy[0])) / max(horizon, 1e-6)
        omega_avg = v_avg / WHEEL_RADIUS
        ramp = np.linspace(2.0 * omega_avg, 0.0, K)
        init = ramp[:, None] * np.ones(num_dofs)
        init = init + noise_std * rng.standard_normal((K, num_dofs))
        return init.astype(np.float64)
    raise ValueError(f"unknown init_type: {init_type}. Choices: {INIT_TYPES}")


def run_trial(seed, K, lr, iterations, horizon, dt, ic, target, weights,
              clip_grad_norm=1.0, init_type="distance-aware", init_noise_std=0.2):
    sim_cfg, rc, ec = make_configs(horizon, dt)
    sim = HelhestJuniorBoxFinalPoseSI(
        sim_cfg, rc, ec, LoggingConfig(),
        ic_xy=ic["xy"], ic_yaw=ic["yaw"],
        target_xy=target["xy"], target_yaw=target["yaw"],
        weights=weights, K=K)
    sim.spline_params = initial_spline(
        K, NUM_WHEEL_DOFS, seed, init_type, target["xy"], horizon,
        noise_std=init_noise_std)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS,
                                  lr=lr, lr_min_ratio=0.2, total_steps=iterations)
    sim._apply_params(sim.spline_params)

    losses, grad_norms, n_clipped = [], [], 0
    n_nan_grad_iters, total_bad_grad = 0, 0
    best_loss = float("inf"); best_iter = 0
    best_params = sim.spline_params.copy()
    t0 = time.perf_counter()
    for it in range(iterations):
        loss, gnorm = sim.opt_step(clip_grad_norm=clip_grad_norm)
        losses.append(loss); grad_norms.append(gnorm)
        if clip_grad_norm is not None and gnorm > clip_grad_norm:
            n_clipped += 1
        n_bad = getattr(sim, "_last_n_bad_grad", 0)
        if n_bad > 0:
            n_nan_grad_iters += 1; total_bad_grad += n_bad
        # Snapshot best-iter params — the noisy contact gradient makes Adam
        # wander past discovered minima, so the deployable params are the
        # best Adam visited, not iter N-1. Matches optimize_axion / _mjx.
        if loss < best_loss:
            best_loss = float(loss); best_iter = it
            best_params = sim.spline_params.copy()
        if it % 10 == 0 or it == iterations - 1:
            star = " *" if (clip_grad_norm is not None and gnorm > clip_grad_norm) else ""
            nanflag = f" nan_grad={n_bad}" if n_bad > 0 else ""
            print(f"    iter {it:3d}: loss={loss:.4f} |g|={gnorm:.2f}{star}{nanflag}", flush=True)
    wall = time.perf_counter() - t0
    if n_nan_grad_iters > 0:
        print(f"    [nan-grad] {n_nan_grad_iters}/{iterations} iters had "
              f"non-finite grad entries ({total_bad_grad} total, salvaged)", flush=True)

    # Restore best-iter params, re-roll out, then read metrics from that.
    sim.spline_params = best_params
    sim._apply_params(best_params)
    sim.diff_step(); wp.synchronize()
    metrics = sim.final_metrics()
    metrics["success"] = bool(metrics["pos_error_m"] < 0.2
                              and metrics["terminal_speed_mps"] < 0.3)
    print(f"    BEST-iter restored: iter {best_iter}, loss {best_loss:.4f}  "
          f"-> pos_err={metrics['pos_error_m']:.3f}m  "
          f"vel={metrics['terminal_speed_mps']:.3f}m/s  "
          f"{'OK' if metrics['success'] else 'FAIL'}", flush=True)
    sim.close(); del sim
    return {"seed": int(seed), "ic": ic, "target": target,
            "losses": losses, "grad_norms": grad_norms,
            "n_clipped": int(n_clipped), "wall_s": wall,
            "n_nan_grad_iters": int(n_nan_grad_iters),
            "total_bad_grad": int(total_bad_grad),
            "best_loss": float(best_loss), "best_iter": int(best_iter),
            "final_loss": float(losses[-1]),
            "final_metrics": metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--ic-xy", type=float, default=0.1,
                    help="±range for IC xy perturbation (m). Matched to "
                    "optimize_axion / _mjx (reduced from the original ±0.3).")
    ap.add_argument("--ic-yaw-deg", type=float, default=5.0,
                    help="±range for IC yaw perturbation (deg). Matched to "
                    "optimize_axion / _mjx (reduced from ±15°).")
    ap.add_argument("--target-x", type=float, default=3.0)
    ap.add_argument("--target-y", type=float, default=0.0)
    ap.add_argument("--target-xy-jitter", type=float, default=0.3)
    ap.add_argument("--target-yaw-deg", type=float, default=15.0)
    ap.add_argument("--w-track", type=float, default=1.0,
                    help="Per-step chassis-xy tracking weight against a linear "
                    "IC→target reference (the shaping term). Matches "
                    "optimize_axion / _mjx. Set to 0 to disable.")
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-vel", type=float, default=0.3)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument("--init-type", choices=INIT_TYPES, default="constant",
                    help="initial spline guess; matched to optimize_axion "
                    "(constant mean=2.0 — the canonical apples-to-apples init).")
    ap.add_argument("--init-noise-std", type=float, default=0.2)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ic_perturb = {"xy": args.ic_xy, "yaw_rad": np.deg2rad(args.ic_yaw_deg)}
    target_perturb = {"xy_center": (args.target_x, args.target_y),
                       "xy_jitter": args.target_xy_jitter,
                       "yaw_rad": np.deg2rad(args.target_yaw_deg)}
    weights = {"track": args.w_track, "pos": args.w_pos, "yaw": args.w_yaw,
               "vel": args.w_vel, "smooth": args.w_smooth, "reg": args.w_reg}

    print(f"K={args.K} iters={args.iterations} lr={args.lr} "
          f"trials={args.num_trials} dt={args.dt} horizon={args.horizon_s}s")
    print(f"IC perturb: xy±{args.ic_xy}m, yaw±{args.ic_yaw_deg}°")
    print(f"Target: ({args.target_x}, {args.target_y}) ± "
          f"{args.target_xy_jitter}m, yaw±{args.target_yaw_deg}°")
    print(f"Weights: {weights}  (clip={args.clip_grad_norm})")

    trials = []
    task_rng = np.random.default_rng(args.seed_base)
    for k in range(args.num_trials):
        ic, target = sample_ic_target(task_rng, ic_perturb, target_perturb)
        spline_seed = args.seed_base + k + 1000
        print(f"\n--- trial {k + 1}/{args.num_trials}  "
              f"IC=({ic['xy'][0]:+.2f},{ic['xy'][1]:+.2f},{np.rad2deg(ic['yaw']):+.1f}°)  "
              f"target=({target['xy'][0]:.2f},{target['xy'][1]:+.2f},{np.rad2deg(target['yaw']):+.1f}°) ---")
        t = run_trial(spline_seed, args.K, args.lr, args.iterations,
                      args.horizon_s, args.dt, ic, target, weights,
                      clip_grad_norm=args.clip_grad_norm,
                      init_type=args.init_type,
                      init_noise_std=args.init_noise_std)
        m = t["final_metrics"]
        print(f"  -> pos_err={m['pos_error_m']:.3f}m  vel={m['terminal_speed_mps']:.2f}m/s  "
              f"jerk={m['control_jerk']:.1f}  {'OK' if m['success'] else 'FAIL'}", flush=True)
        trials.append(t)

    success_rate = sum(t["final_metrics"]["success"] for t in trials) / len(trials)
    median_pos = float(np.median([t["final_metrics"]["pos_error_m"] for t in trials]))
    median_vel = float(np.median([t["final_metrics"]["terminal_speed_mps"] for t in trials]))

    out = {
        "simulator": "Semi-Implicit",
        "task": "random-IC final-pose",
        "K": args.K, "lr": args.lr, "iterations": args.iterations,
        "horizon_s": args.horizon_s, "dt": args.dt,
        "clip_grad_norm": args.clip_grad_norm,
        "num_trials": args.num_trials, "seed_base": args.seed_base,
        "ic_perturbation": ic_perturb,
        "target_perturbation": {**target_perturb,
                                 "xy_center": list(target_perturb["xy_center"])},
        "loss_weights": weights,
        "init_type": args.init_type,
        "init_noise_std": args.init_noise_std,
        "aggregate": {
            "success_rate": success_rate,
            "median_pos_error_m": median_pos,
            "median_terminal_speed_mps": median_vel,
        },
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "semi_implicit.json")
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f)
    print(f"\n=== Aggregate ===")
    print(f"  Success rate: {success_rate:.0%}")
    print(f"  Median pos error: {median_pos:.3f} m")
    print(f"  Median terminal speed: {median_vel:.3f} m/s")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
