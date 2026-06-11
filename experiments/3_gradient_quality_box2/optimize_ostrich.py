"""Helhest_junior box random-IC final-pose optimisation — Ostrich.

Different from experiments/3_gradient_quality_box (which optimises to match a
recorded GT trajectory at every timestep). Here each trial:

  - Spawns the chassis at a randomly perturbed IC (xy + yaw).
  - Picks a random target xy + yaw past the box.
  - Optimises a K-knot wheel-velocity spline that drives the chassis from the
    IC to the target, stopping at zero velocity at the end, with smooth
    controls. The optimiser sees no GT trajectory.

Loss is a weighted sum of:
  L = w_pos * ‖xy_T − target_xy‖²
    + w_yaw * (1 − cos(ψ_T − target_yaw))
    + w_vel * Σ_{tail} ‖v_xy(t)‖²           (terminal-vel penalty via finite-diff)
    + w_smooth * Σ_t ‖u[t+1] − u[t]‖²       (control smoothness)
    + w_reg * Σ_t ‖u[t]‖²                   (control magnitude)

The terminal-velocity term is summed over the last ``terminal_tail_frac`` of
the horizon (default 0.10) rather than just at t=T. That broadens the
gradient signal and encourages deceleration rather than a sudden braking
step at the end.

Outputs results/ostrich[_<save>].json with per-trial loss curves AND derived
metrics (final-pose error, terminal speed, control jerk, success flag).

Usage:
    python experiments/3_gradient_quality_box2/optimize_ostrich.py
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
from ostrich import (
    OstrichDifferentiableSimulator,
    OstrichEngineConfig,
    ComplianceConfig,
    ContactsConfig,
    LinearSolverConfig,
    LinesearchConfig,
    LoggingConfig,
    NewtonRaphsonConfig,
    RenderingConfig,
    SimulationConfig,
)
from ostrich.collision.config import ContactReductionConfig

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


def make_interp_matrix(T: int, K: int):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x)
        hi = min(lo + 1, K - 1)
        a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W, W.sum(axis=0)


class SplineAdam:
    """Adam (Kingma & Ba, 2014) with optional cosine LR decay and AMSGrad
    (Reddi et al., 2018).

    AMSGrad replaces the running second-moment v̂ with its running MAXIMUM
    v_max in the denominator:
        v_max ← max(v_max, v_t)
        step  = lr · m̂ / (√v_max + ε)
    This prevents the "shrinking effective step size near minima" pathology
    that breaks vanilla Adam on sharp non-convex minima — the denominator
    can only grow, so step size monotonically decreases as the optimiser
    encounters larger gradients (early descent) and never resets to a
    larger value during fine-tuning.
    """

    def __init__(self, K, num_dofs, lr=0.1, lr_min_ratio=0.2, total_steps=100,
                 betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.b1, self.b2 = betas
        self.eps = eps
        self.amsgrad = bool(amsgrad)
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.v_max = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        p = min(self.t / max(1, self.total_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * p))

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        # b1=0 → no momentum (RMSprop). Skip bias correction (denominator → 0).
        mh = self.m / (1 - self.b1**self.t) if self.b1 > 0 else self.m
        # Bias-correct v BEFORE comparing with running max. The original Reddi
        # paper omitted the bias correction, which makes the very first step
        # huge (v is heavily biased toward zero at t=1, so √v ≪ |g|, and the
        # step ≈ lr·sign(g)/√(1-β2) ~ 32·lr blows up). PyTorch / Keras both
        # bias-correct first.
        vh = self.v / (1 - self.b2**self.t)
        if self.amsgrad:
            np.maximum(self.v_max, vh, out=self.v_max)
            v_denom = self.v_max
        else:
            v_denom = vh
        return params - self._cosine_lr() * mh / (np.sqrt(v_denom) + self.eps)


# ------------------------------ loss kernels ---------------------------------
@wp.kernel
def chassis_track_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),    # [T+1, W, B]
    ref_xy: wp.array(dtype=wp.vec2),                     # [T+1]
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Per-step chassis-xy tracking loss against a reference waypoint path.

    L_track = w * Σ_t ‖xy(t) − ref_xy(t)‖²  (chassis = body 0)

    Why this term: terminal-only losses give the optimiser no gradient signal
    until the contact-event noise from later timesteps propagates 60 steps
    back to the spline knots — which it does poorly because each step adds
    ~1e-3 of floating-point noise. A per-step tracking loss puts a small
    gradient signal at EVERY timestep, so each spline knot directly affects
    the loss without needing the noise to compound across 60 backward steps.
    """
    t = wp.tid()
    p = wp.transform_get_translation(body_pose[t, 0, 0])
    r = ref_xy[t]
    dx = p[0] - r[0]
    dy = p[1] - r[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def final_pos_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_x: float,
    target_y: float,
    final_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_pos = w * ‖xy(T) − target_xy‖² (body 0 = chassis)."""
    p = wp.transform_get_translation(body_pose[final_idx, 0, 0])
    dx = p[0] - target_x
    dy = p[1] - target_y
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def final_yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_yaw: float,
    final_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_yaw = w * (1 − cos(ψ(T) − target_yaw)). Wrap-around safe."""
    q = wp.transform_get_rotation(body_pose[final_idx, 0, 0])
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    delta = yaw - target_yaw
    wp.atomic_add(loss, 0, weight * (1.0 - wp.cos(delta)))


@wp.kernel
def terminal_vel_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    start_idx: int,
    end_idx: int,
    inv_dt: float,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_vel = w * Σ_{t ∈ [start, end)} ‖(xy[t+1] − xy[t]) / dt‖²

    Approximates chassis linear velocity via forward finite difference. Tail
    indices [start, end) cover the deceleration zone.
    """
    t_local = wp.tid()
    t = start_idx + t_local
    if t >= end_idx:
        return
    p0 = wp.transform_get_translation(body_pose[t, 0, 0])
    p1 = wp.transform_get_translation(body_pose[t + 1, 0, 0])
    vx = (p1[0] - p0[0]) * inv_dt
    vy = (p1[1] - p0[1]) * inv_dt
    wp.atomic_add(loss, 0, weight * (vx * vx + vy * vy))


@wp.kernel
def smoothness_loss_kernel(
    joint_target_vel: wp.array(dtype=wp.float32, ndim=3),  # [T, W, D]
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_smooth = w * Σ_t Σ_k (u[t+1, k] − u[t, k])²"""
    t, k = wp.tid()  # k ∈ {0..NUM_WHEEL_DOFS-1}
    dof = wheel_dof_offset + k
    u0 = joint_target_vel[t, 0, dof]
    u1 = joint_target_vel[t + 1, 0, dof]
    du = u1 - u0
    wp.atomic_add(loss, 0, weight * du * du)


@wp.kernel
def magnitude_reg_kernel(
    joint_target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L_reg = w * Σ_t Σ_k u[t, k]²"""
    t, k = wp.tid()
    dof = wheel_dof_offset + k
    u = joint_target_vel[t, 0, dof]
    wp.atomic_add(loss, 0, weight * u * u)


# ------------------------------ optimizer ------------------------------------
class HelhestJuniorBoxFinalPoseOptimizer(OstrichDifferentiableSimulator):
    """One trial: build scene with random IC, owns spline, runs diff_step."""

    def __init__(
        self,
        sim_config,
        render_config,
        engine_config,
        logging_config,
        ic_xy,
        ic_yaw,
        target_xy,
        target_yaw,
        weights,
        K=10,
        mu_front=0.8,
        mu_rear=1.2,
        mu_rolling=0.7,
        terminal_tail_frac=0.10,
    ):
        self.K = K
        self.ic_xy = tuple(ic_xy)
        self.ic_yaw = float(ic_yaw)
        self.target_xy = tuple(target_xy)
        self.target_yaw = float(target_yaw)
        self.weights = weights
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.mu_rolling = mu_rolling
        self.terminal_tail_frac = float(terminal_tail_frac)
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Build the waypoint reference path: linear interpolation in xy from
        # IC to target, sampled at every body_pose timestep (T_states = T+1).
        # Used by chassis_track_loss_kernel to give the optimiser per-step
        # gradient signal — see the kernel docstring for the why.
        T_states = T + 1
        t_norm = np.arange(T_states, dtype=np.float32) / max(T_states - 1, 1)
        ref_xy_np = np.zeros((T_states, 2), dtype=np.float32)
        ref_xy_np[:, 0] = self.ic_xy[0] + t_norm * (self.target_xy[0] - self.ic_xy[0])
        ref_xy_np[:, 1] = self.ic_xy[1] + t_norm * (self.target_xy[1] - self.ic_xy[1])
        self.ref_xy_buf = wp.array(ref_xy_np, dtype=wp.vec2,
                                     device=self.model.device,
                                     requires_grad=False)

    def build_model(self):
        """Same scene as exp 3 box but chassis spawned at the randomised IC."""
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0],
            hy=BOX_HALF_EXTENTS[1],
            hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0),
        )
        chassis_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self.ic_yaw)
        chassis_xform = wp.transform(
            wp.vec3(float(self.ic_xy[0]), float(self.ic_xy[1]), 0.5), chassis_q
        )
        create_helhest_junior_model(
            self.builder,
            xform=chassis_xform,
            control_mode="velocity",
            k_p=250.0,
            k_d=0.0,
            friction_left_right=self.mu_front,
            friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
        )
        return self.builder.finalize_replicated(num_worlds=1, requires_grad=True)

    # ---- spline expand / contract / apply ----
    def _expand(self, params):
        return self.W @ params  # [T,K] @ [K,3] = [T,3]

    def _contract(self, grad_v):
        safe = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe[:, None]

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    # ---- compute_loss is the heart of the new task ----
    def compute_loss_track(self, T_states, weight, device):
        """Per-step waypoint tracking — main shaping signal."""
        wp.launch(chassis_track_loss_kernel, dim=T_states,
                  inputs=[self.trajectory.body_pose, self.ref_xy_buf,
                          float(weight / T_states)],
                  outputs=[self.loss], device=device)

    def compute_loss(self):
        # NB: body_pose has T_sim+1 slots (slot 0 is the initial state,
        # slots 1..T_sim are post-step states). joint_target_vel has only
        # T_sim slots (one control per step). Use the right T for each
        # kernel — mixing them up reads/writes one slot out-of-bounds and
        # corrupts the gradient buffer.
        T_states = self.trajectory.body_pose.shape[0]  # = T_sim + 1
        T_steps = self.clock.total_sim_steps  # = T_sim
        final_idx = T_states - 1  # = T_sim
        tail_len = max(2, int(round(T_states * self.terminal_tail_frac)))
        tail_start = T_states - 1 - tail_len
        device = self.solver.model.device
        w = self.weights

        # 0. Per-step waypoint tracking — dense gradient signal
        if w.get("track", 0.0) > 0.0:
            self.compute_loss_track(T_states, w["track"], device)

        # 1. Final position
        wp.launch(
            final_pos_loss_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_pose,
                float(self.target_xy[0]),
                float(self.target_xy[1]),
                final_idx,
                float(w["pos"]),
            ],
            outputs=[self.loss],
            device=device,
        )

        # 2. Final yaw
        wp.launch(
            final_yaw_loss_kernel,
            dim=1,
            inputs=[self.trajectory.body_pose, float(self.target_yaw), final_idx, float(w["yaw"])],
            outputs=[self.loss],
            device=device,
        )

        # 3. Terminal velocity (reads body_pose[t] and body_pose[t+1] for
        #    t in [tail_start, T_states-1) — all in bounds since body_pose
        #    has T_states slots).
        if w["vel"] > 0.0:
            wp.launch(
                terminal_vel_loss_kernel,
                dim=tail_len,
                inputs=[
                    self.trajectory.body_pose,
                    tail_start,
                    final_idx,
                    float(1.0 / self.clock.dt),
                    float(w["vel"] / tail_len),
                ],
                outputs=[self.loss],
                device=device,
            )

        # 4. Control smoothness — reads joint_target_vel[t] and [t+1],
        #    so dim is T_steps-1, not T_states-1, to keep t+1 in bounds.
        if w["smooth"] > 0.0:
            wp.launch(
                smoothness_loss_kernel,
                dim=(T_steps - 1, NUM_WHEEL_DOFS),
                inputs=[
                    self.trajectory.joint_target_vel,
                    WHEEL_DOF_OFFSET,
                    float(w["smooth"] / (T_steps - 1)),
                ],
                outputs=[self.loss],
                device=device,
            )

        # 5. Magnitude regularisation — reads joint_target_vel[t], so
        #    dim is T_steps (not T_states which would step OOB).
        if w["reg"] > 0.0:
            wp.launch(
                magnitude_reg_kernel,
                dim=(T_steps, NUM_WHEEL_DOFS),
                inputs=[
                    self.trajectory.joint_target_vel,
                    WHEEL_DOF_OFFSET,
                    float(w["reg"] / T_steps),
                ],
                outputs=[self.loss],
                device=device,
            )

    def update(self):
        pass  # spline_adam owned by trainer

    def opt_step(self):
        self.diff_step()
        wp.synchronize()
        loss_val = float(self.loss.numpy()[0])
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]
        grad_params = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        # ---- Diagnostics, BEFORE Adam updates m. ----
        # |g|: current-step gradient L2 norm
        # |m|: Adam's accumulated momentum L2 norm (smoothed from past
        #      gradients; with β1=0.9 has ~10-iter memory)
        # cos(g, m):
        #     +1 → momentum perfectly aligned with current gradient (descent)
        #      0 → orthogonal (transitioning)
        #     -1 → momentum directly OPPOSES current gradient (overshoot —
        #          Adam keeps stepping in old direction even though current
        #          gradient says to reverse)
        g_flat = grad_params.flatten().astype(np.float64)
        m_flat = self.spline_adam.m.flatten()
        g_norm = float(np.linalg.norm(g_flat))
        m_norm = float(np.linalg.norm(m_flat))
        denom = g_norm * m_norm
        cos_gm = float(np.dot(g_flat, m_flat) / denom) if denom > 1e-12 else 0.0
        self._last_step_stats = {
            "gnorm": g_norm,
            "mnorm": m_norm,
            "cos_gm": cos_gm,
        }
        # ----
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)
        self.tape.zero()
        self.loss.zero_()
        return loss_val

    # ---- metrics extraction (after training) ----
    def final_metrics(self):
        """Read final chassis pose + estimate terminal speed + control jerk."""
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        bp = self.trajectory.body_pose.numpy()  # [T, 1, B, 7]: x,y,z,qx,qy,qz,qw
        chassis_T = bp[T - 1, 0, 0]
        chassis_Tm1 = bp[max(T - 2, 0), 0, 0]
        xy_T = chassis_T[:2]
        z_T = chassis_T[2]
        q = chassis_T[3:7]  # x,y,z,w
        yaw_T = float(
            np.arctan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        )
        # Terminal speed = ‖xy_T − xy_{T-1}‖ / dt
        v_xy = (chassis_T[:2] - chassis_Tm1[:2]) / dt
        terminal_speed = float(np.linalg.norm(v_xy))
        # Control jerk = Σ_t ‖u[t+1] − u[t]‖
        jtv = self.trajectory.joint_target_vel.numpy()  # [T, 1, D]
        u = jtv[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        du = np.diff(u, axis=0)
        control_jerk = float(np.linalg.norm(du, axis=1).sum())
        # Position error
        pos_error = float(np.linalg.norm(xy_T - np.asarray(self.target_xy)))
        yaw_error = float(
            abs(np.arctan2(np.sin(yaw_T - self.target_yaw), np.cos(yaw_T - self.target_yaw)))
        )
        return {
            "xy_final": [float(xy_T[0]), float(xy_T[1])],
            "z_final": float(z_T),
            "yaw_final": yaw_T,
            "pos_error_m": pos_error,
            "yaw_error_rad": yaw_error,
            "terminal_speed_mps": terminal_speed,
            "control_jerk": control_jerk,
        }


# ------------------------------ initial-control helper ----------------------
WHEEL_RADIUS = 0.35  # m — matches examples/helhest_junior/common.py
INIT_TYPES = ("constant", "distance-aware")


def initial_spline(
    K, num_dofs, seed, init_type, target_xy, horizon, noise_std=0.2, const_mean=2.0, const_std=0.5
):
    """Build the K×num_dofs initial spline-parameter guess.

    ``init_type``:
      * ``"constant"`` — original behaviour: ``const_mean + const_std *
        N(0, 1)``. Mean wheel angular velocity 2 rad/s ≈ 0.7 m/s linear,
        which over the default 6 s horizon overshoots a 3 m target by ~1 m.
        The optimiser then has to learn to slow down → 40% of trials fail.

      * ``"distance-aware"`` — decaying ramp from ``2·ω_avg`` to ~0 across
        the K knots, where ``ω_avg = |target_x| / (horizon · wheel_radius)``
        is the average wheel angular velocity needed to reach the target
        in the horizon. Drives forward fast initially, decays naturally
        toward zero — so the "drive then stop" shape is built in. Adds
        ``noise_std·N(0,1)`` so per-trial inits remain diverse.
    """
    rng = np.random.default_rng(seed)
    if init_type == "constant":
        return (const_mean + const_std * rng.standard_normal((K, num_dofs))).astype(np.float64)
    if init_type == "distance-aware":
        v_avg = float(abs(target_xy[0])) / max(horizon, 1e-6)  # m/s
        omega_avg = v_avg / WHEEL_RADIUS  # rad/s
        ramp = np.linspace(2.0 * omega_avg, 0.0, K)  # [K]
        init = ramp[:, None] * np.ones(num_dofs)  # [K, num_dofs]
        init = init + noise_std * rng.standard_normal((K, num_dofs))
        return init.astype(np.float64)
    raise ValueError(f"unknown init_type: {init_type}. Choices: {INIT_TYPES}")


# ------------------------------ trial driver ---------------------------------
def make_configs(duration, dt):
    sim_cfg = SimulationConfig(
        duration_seconds=duration, target_timestep_seconds=dt, num_worlds=1, use_cuda_graph=True
    )
    rc = RenderingConfig(vis_type="null", target_fps=max(1, int(1 / dt)), start_paused=False)
    ec = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-7, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        # Contact filtering: cluster reduction (the default in
        # examples/conf/engine/ostrich.yaml). Collapses redundant wheel-on-box
        # contact points (which a contact patch normally produces as 8-16
        # individual points, each with its own gradient direction —
        # producing the gradient noise we diagnosed). With clustering,
        # contacts with similar normals AND positions become one
        # representative point per cluster, capped at max_per_pair.
        contacts=ContactsConfig(
            max_per_world=256,
            reduction=ContactReductionConfig(policy="cluster", max_per_pair=8),
        ),
    )
    return sim_cfg, rc, ec


def sample_ic_target(rng, ic_perturb, target_perturb):
    """Draw a random IC + target pair using two RNG calls."""
    ic = {
        "xy": [
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
            float(rng.uniform(-ic_perturb["xy"], ic_perturb["xy"])),
        ],
        "yaw": float(rng.uniform(-ic_perturb["yaw_rad"], ic_perturb["yaw_rad"])),
    }
    target = {
        "xy": [
            float(
                target_perturb["xy_center"][0]
                + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])
            ),
            float(
                target_perturb["xy_center"][1]
                + rng.uniform(-target_perturb["xy_jitter"], target_perturb["xy_jitter"])
            ),
        ],
        "yaw": float(rng.uniform(-target_perturb["yaw_rad"], target_perturb["yaw_rad"])),
    }
    return ic, target


def run_trial(
    seed,
    K,
    lr,
    iterations,
    horizon,
    dt,
    ic,
    target,
    weights,
    init_type="distance-aware",
    init_noise_std=0.2,
    beta1=0.9,
    beta2=0.999,
    lr_min_ratio=0.2,
    amsgrad=False,
):
    sim_cfg, rc, ec = make_configs(horizon, dt)
    sim = HelhestJuniorBoxFinalPoseOptimizer(
        sim_cfg,
        rc,
        ec,
        LoggingConfig(),
        ic_xy=ic["xy"],
        ic_yaw=ic["yaw"],
        target_xy=target["xy"],
        target_yaw=target["yaw"],
        weights=weights,
        K=K,
    )
    sim.spline_params = initial_spline(
        K, NUM_WHEEL_DOFS, seed, init_type, target["xy"], horizon, noise_std=init_noise_std
    )
    sim.spline_adam = SplineAdam(
        K=K,
        num_dofs=NUM_WHEEL_DOFS,
        lr=lr,
        lr_min_ratio=lr_min_ratio,
        total_steps=iterations,
        betas=(beta1, beta2),
        amsgrad=amsgrad,
    )
    sim._apply_params(sim.spline_params)

    losses, gnorms, mnorms, cos_gms = [], [], [], []
    best_loss = float("inf")
    best_iter = 0
    best_params = sim.spline_params.copy()
    t0 = time.perf_counter()
    for it in range(iterations):
        loss = sim.opt_step()
        s = sim._last_step_stats
        losses.append(loss); gnorms.append(s["gnorm"])
        mnorms.append(s["mnorm"]); cos_gms.append(s["cos_gm"])
        # Snapshot best-iter params. Adam wobbles near minima even with all
        # the other fixes (shaping, cluster filter, AMSGrad) — best-iter
        # captures the optimum the optimiser found and ignores subsequent
        # drift.
        if loss < best_loss:
            best_loss = float(loss)
            best_iter = it
            best_params = sim.spline_params.copy()
        if it % 10 == 0 or it == iterations - 1:
            overshoot = " <-- overshoot" if s["cos_gm"] < 0 else ""
            print(f"    iter {it:3d}: loss={loss:.4f}  "
                  f"|g|={s['gnorm']:.3f} |m|={s['mnorm']:.3f} "
                  f"cos(g,m)={s['cos_gm']:+.2f}{overshoot}", flush=True)
    wall = time.perf_counter() - t0

    # Restore best-iter params before computing the deployable trajectory's
    # metrics. final_metrics now reflects the BEST point Adam visited, not
    # whatever it wandered to at iter N-1.
    sim.spline_params = best_params
    sim._apply_params(best_params)
    sim.diff_step(); wp.synchronize()
    metrics = sim.final_metrics()
    metrics["success"] = bool(metrics["pos_error_m"] < 0.2 and metrics["terminal_speed_mps"] < 0.3)
    print(f"    BEST-iter restored: iter {best_iter}, loss {best_loss:.4f}  "
          f"-> pos_err={metrics['pos_error_m']:.3f}m  "
          f"vel={metrics['terminal_speed_mps']:.3f}m/s  "
          f"{'OK' if metrics['success'] else 'FAIL'}", flush=True)
    sim.close()
    del sim
    return {
        "seed": int(seed),
        "ic": ic,
        "target": target,
        "losses": losses,
        "grad_norms": gnorms,
        "momentum_norms": mnorms,
        "cos_grad_momentum": cos_gms,
        "wall_s": wall,
        "best_loss": float(best_loss),
        "best_iter": int(best_iter),
        "final_loss": float(losses[-1]),
        "final_metrics": metrics,
    }


def check_gradient_finite_difference(seed, K, horizon, dt, ic, target, weights,
                                       init_type="distance-aware",
                                       init_noise_std=0.2, eps=1e-3):
    """Compare Ostrich's autodiff gradient ∇_θ L to the central finite-difference
    gradient on the SAME spline init. Cheap: 2·30 = 60 sim evals (~30 s).

    For each spline-param dim i:
        fd_grad[i] = (L(θ + eps·e_i) − L(θ − eps·e_i)) / (2·eps)
    Then compare to autodiff grad:
        rel_err[i] = |fd_grad[i] − ad_grad[i]| / (|fd_grad[i]| + 1e-12)
        cos_global = (fd · ad) / (‖fd‖·‖ad‖)

    If rel_err is large (>10%) or cos < 0.95, the autodiff gradient is wrong
    and EVERY optimizer is downstream of that.
    """
    sim_cfg, rc, ec = make_configs(horizon, dt)
    sim = HelhestJuniorBoxFinalPoseOptimizer(
        sim_cfg, rc, ec, LoggingConfig(),
        ic_xy=ic["xy"], ic_yaw=ic["yaw"],
        target_xy=target["xy"], target_yaw=target["yaw"],
        weights=weights, K=K,
    )
    init = initial_spline(K, NUM_WHEEL_DOFS, seed, init_type, target["xy"],
                          horizon, noise_std=init_noise_std)
    init = init.astype(np.float64)

    def eval_loss_and_grad(params):
        sim.spline_params = params.copy()
        sim._apply_params(sim.spline_params)
        sim.diff_step()
        wp.synchronize()
        loss = float(sim.loss.numpy()[0])
        grad_v = sim.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        grad_params = sim._contract(grad_v)
        sim.trajectory.joint_target_vel.grad.zero_()
        sim.tape.zero()
        sim.loss.zero_()
        return loss, grad_params.copy()

    def eval_loss_only(params):
        sim.spline_params = params.copy()
        sim._apply_params(sim.spline_params)
        sim.diff_step()  # cheaper to also use diff_step (already CUDA-graph'd)
        wp.synchronize()
        loss = float(sim.loss.numpy()[0])
        sim.trajectory.joint_target_vel.grad.zero_()
        sim.tape.zero()
        sim.loss.zero_()
        return loss

    print(f"\n=== GRADIENT CHECK ===")
    print(f"K={K}, NUM_WHEEL_DOFS={NUM_WHEEL_DOFS}, total params={K*NUM_WHEEL_DOFS}")
    print(f"eps={eps}\n")

    loss0, ad_grad = eval_loss_and_grad(init)
    print(f"baseline loss = {loss0:.6f}")
    print(f"autodiff grad: shape={ad_grad.shape}  "
          f"|grad|={np.linalg.norm(ad_grad):.6f}  "
          f"max|.|={np.max(np.abs(ad_grad)):.6f}\n")

    fd_grad = np.zeros_like(ad_grad)
    print(f"{'idx':>3s}  {'(k,d)':>6s}  {'autodiff':>11s}  {'finite-diff':>11s}  "
          f"{'abs_err':>9s}  {'rel_err':>8s}")
    print("-" * 70)
    n_bad = 0
    for k in range(K):
        for d in range(NUM_WHEEL_DOFS):
            p_plus = init.copy(); p_plus[k, d] += eps
            p_minus = init.copy(); p_minus[k, d] -= eps
            l_plus = eval_loss_only(p_plus)
            l_minus = eval_loss_only(p_minus)
            fd = (l_plus - l_minus) / (2.0 * eps)
            fd_grad[k, d] = fd
            ad = ad_grad[k, d]
            abs_err = abs(fd - ad)
            rel_err = abs_err / (abs(fd) + 1e-12)
            bad = (abs_err > 1e-4) and (rel_err > 0.1)
            mark = "  *" if bad else ""
            n_bad += int(bad)
            idx = k * NUM_WHEEL_DOFS + d
            print(f"{idx:>3d}  {f'({k},{d})':>6s}  {ad:>+11.6f}  {fd:>+11.6f}  "
                  f"{abs_err:>9.6f}  {rel_err:>8.4f}{mark}")
    print("-" * 70)
    cos_sim = float(np.dot(ad_grad.flatten(), fd_grad.flatten()) /
                    (np.linalg.norm(ad_grad) * np.linalg.norm(fd_grad) + 1e-12))
    rel_norm_diff = np.linalg.norm(fd_grad - ad_grad) / (np.linalg.norm(fd_grad) + 1e-12)
    print(f"\nGlobal cosine(ad, fd) = {cos_sim:+.6f}")
    print(f"‖fd − ad‖ / ‖fd‖ = {rel_norm_diff:.4f}")
    print(f"# bad components  = {n_bad} / {K*NUM_WHEEL_DOFS}")
    print()
    if cos_sim > 0.99 and rel_norm_diff < 0.05:
        print("VERDICT: autodiff gradient is CORRECT. The optimization "
              "issues are NOT due to a broken gradient.")
    elif cos_sim > 0.9:
        print("VERDICT: autodiff gradient is MOSTLY correct but has measurable "
              "error. Some components are off; could explain optimizer issues.")
    else:
        print("VERDICT: autodiff gradient is WRONG. Optimization can't possibly "
              "converge cleanly until this is fixed.")

    sim.close(); del sim
    return ad_grad, fd_grad, cos_sim, rel_norm_diff


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--num-trials", type=int, default=25)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--ic-xy", type=float, default=0.1,
                    help="±range for IC xy perturbation (m). Reduced from "
                    "the original ±0.3 m because larger random IC offsets "
                    "made many trials infeasible — the robot started too "
                    "far off-axis to reach the target across the box in 6s "
                    "with K=10 controls.")
    ap.add_argument("--ic-yaw-deg", type=float, default=5.0,
                    help="±range for IC yaw perturbation (deg). Reduced "
                    "from ±15° for the same reason as --ic-xy.")
    ap.add_argument(
        "--target-x",
        type=float,
        default=3.0,
        help="target x (m, past the box at x≈1.37+half_extent)",
    )
    ap.add_argument("--target-y", type=float, default=0.0)
    ap.add_argument(
        "--target-xy-jitter", type=float, default=0.3, help="±range for target xy jitter (m)"
    )
    ap.add_argument(
        "--target-yaw-deg", type=float, default=15.0, help="±range for target yaw jitter (deg)"
    )
    ap.add_argument("--w-track", type=float, default=1.0,
                    help="Per-step chassis-xy tracking weight against a linear "
                    "IC→target reference. The 'shaping' term: gives each spline "
                    "knot a direct gradient signal at every timestep instead of "
                    "relying on terminal loss to compound back through 60 BPTT "
                    "steps (which is dominated by simulator noise). Set to 0 "
                    "to disable.")
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-vel", type=float, default=0.3)
    ap.add_argument("--w-smooth", type=float, default=1e-3)
    ap.add_argument("--w-reg", type=float, default=1e-5)
    ap.add_argument(
        "--init-type",
        choices=INIT_TYPES,
        default="constant",
        help="initial spline guess. 'constant' (default) is the "
        "mean=2.0 init — the canonical apples-to-apples init shared "
        "with MJX/SI (also yields higher success than distance-aware "
        "under the matched task). 'distance-aware' starts as a "
        "decaying ramp from 2·ω_avg to 0 based on target distance.",
    )
    ap.add_argument(
        "--init-noise-std",
        type=float,
        default=0.2,
        help="std of N(0,1) noise added on top of the init",
    )
    ap.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam first-moment decay. Set to 0 to disable "
        "momentum (RMSprop) — useful when the contact "
        "gradient is noisy and momentum carries the "
        "optimiser past discovered minima.",
    )
    ap.add_argument("--beta2", type=float, default=0.999, help="Adam second-moment decay.")
    ap.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.2,
        help="Cosine LR decay endpoint as a fraction of lr_init. "
        "0.2 (default) decays to 20%% of lr; 0.05 decays to "
        "5%% — smaller late-iter step, less divergence after "
        "convergence.",
    )
    ap.add_argument(
        "--amsgrad", action="store_true",
        help="Use AMSGrad (Reddi et al. 2018): replace Adam's running v̂ "
        "with its running MAX in the denominator. Prevents the 'effective "
        "step size doesn't shrink near sharp minima' pathology — denominator "
        "can only grow, so per-step magnitude is monotonically non-increasing.",
    )
    ap.add_argument("--check-gradient", action="store_true",
                    help="Instead of optimising, run ONE finite-difference "
                    "gradient check on the initial spline params. Prints the "
                    "autodiff gradient and the central-difference gradient "
                    "per-parameter so you can see if Ostrich's adjoint is "
                    "actually computing what we think it is.")
    ap.add_argument("--check-eps", type=float, default=1e-3,
                    help="Finite-difference step size for --check-gradient.")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    ic_perturb = {"xy": args.ic_xy, "yaw_rad": np.deg2rad(args.ic_yaw_deg)}
    target_perturb = {
        "xy_center": (args.target_x, args.target_y),
        "xy_jitter": args.target_xy_jitter,
        "yaw_rad": np.deg2rad(args.target_yaw_deg),
    }
    weights = {
        "track": args.w_track,
        "pos": args.w_pos,
        "yaw": args.w_yaw,
        "vel": args.w_vel,
        "smooth": args.w_smooth,
        "reg": args.w_reg,
    }

    print(
        f"K={args.K} iters={args.iterations} lr={args.lr} "
        f"trials={args.num_trials} dt={args.dt} horizon={args.horizon_s}s"
    )
    print(f"IC perturb: xy±{args.ic_xy}m, yaw±{args.ic_yaw_deg}°")
    print(
        f"Target: ({args.target_x}, {args.target_y}) ± "
        f"{args.target_xy_jitter}m, yaw±{args.target_yaw_deg}°"
    )
    print(f"Weights: {weights}")

    # --check-gradient mode: run the FD check on trial 0 and exit.
    if args.check_gradient:
        task_rng = np.random.default_rng(args.seed_base)
        ic, target = sample_ic_target(task_rng, ic_perturb, target_perturb)
        spline_seed = args.seed_base + 0 + 1000
        check_gradient_finite_difference(
            spline_seed, args.K, args.horizon_s, args.dt, ic, target, weights,
            init_type=args.init_type, init_noise_std=args.init_noise_std,
            eps=args.check_eps,
        )
        return

    trials = []
    # Pre-draw all ICs+targets with a deterministic stream so the same
    # seed-base produces the same task set across engines.
    task_rng = np.random.default_rng(args.seed_base)
    for k in range(args.num_trials):
        ic, target = sample_ic_target(task_rng, ic_perturb, target_perturb)
        spline_seed = args.seed_base + k + 1000  # offset to avoid RNG collision
        print(f"\n--- trial {k + 1}/{args.num_trials}  IC={ic}  target={target} ---")
        t = run_trial(
            spline_seed, args.K, args.lr, args.iterations,
            args.horizon_s, args.dt, ic, target, weights,
            init_type=args.init_type,
            init_noise_std=args.init_noise_std,
            beta1=args.beta1, beta2=args.beta2,
            lr_min_ratio=args.lr_min_ratio,
            amsgrad=args.amsgrad,
        )
        t["metrics_summary"] = (
            f"pos_err={t['final_metrics']['pos_error_m']:.3f}m  "
            f"vel={t['final_metrics']['terminal_speed_mps']:.2f}m/s  "
            f"jerk={t['final_metrics']['control_jerk']:.1f}  "
            f"{'OK' if t['final_metrics']['success'] else 'FAIL'}"
        )
        print(f"  -> {t['metrics_summary']}")
        trials.append(t)

    success_rate = sum(t["final_metrics"]["success"] for t in trials) / len(trials)
    median_pos = float(np.median([t["final_metrics"]["pos_error_m"] for t in trials]))
    median_vel = float(np.median([t["final_metrics"]["terminal_speed_mps"] for t in trials]))

    out = {
        "simulator": "Ostrich",
        "task": "random-IC final-pose",
        "K": args.K,
        "lr": args.lr,
        "iterations": args.iterations,
        "horizon_s": args.horizon_s,
        "dt": args.dt,
        "num_trials": args.num_trials,
        "seed_base": args.seed_base,
        "ic_perturbation": ic_perturb,
        "target_perturbation": {**target_perturb, "xy_center": list(target_perturb["xy_center"])},
        "loss_weights": weights,
        "init_type": args.init_type,
        "init_noise_std": args.init_noise_std,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "lr_min_ratio": args.lr_min_ratio,
        "amsgrad": args.amsgrad,
        "aggregate": {
            "success_rate": success_rate,
            "median_pos_error_m": median_pos,
            "median_terminal_speed_mps": median_vel,
        },
        "trials": trials,
    }
    save_path = args.save or str(RESULTS_DIR / "ostrich.json")
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
