"""Bundled-gradient balance optimization for the Helhest robot.

Resurrects the original ``helhest_balance_axion.py`` (lost in fe18754
and the subsequent file deletion), modernizes for the current API,
parameterizes wheel-velocity controls as a spline (K knots × 3 wheels
instead of T × 3 per-step), and adds the bundled-gradient path
mirroring ``gradient/trajectory_spline_surface_fast_bundled.py``.

Scenario: chassis spawned tilted backward at ``BALANCE_PITCH``, sitting
on the rear wheel; loss penalizes orientation drift (up-vector) and
position drift over the whole trajectory. Optimization variable: K
spline knots × 3 wheel velocities. With ``num_worlds > 1`` and
``sigma > 0`` the optimizer averages exact gradients over N noisy
rollouts (Suh, Pang, Tedrake 2021 — "Bundled Gradients through
Contact via Randomized Smoothing"); with N=1, σ=0 it falls back to the
exact gradient (the original "disaster" baseline).

This is the natural test problem for bundled gradients: balancing on
two wheels involves continuous contact-mode flips (rear wheel lift-on/
off, friction sticking↔sliding) that make the exact gradient
non-smooth. Smoothing across an ensemble should give a gradient that
points toward stable balance instead of the wild local-discontinuity
gradient at the current rocking phase.
"""
import argparse
import math
import os
import pathlib
import time

import numpy as np
import warp as wp
import newton
from newton import Model

from axion import (
    AxionDifferentiableSimulator,
    AxionEngineConfig,
    LoggingConfig,
    RenderingConfig,
    SimulationConfig,
    NewtonRaphsonConfig,
    LinearSolverConfig,
    ComplianceConfig,
    LinesearchConfig,
    ContactsConfig,
)

from examples.helhest.common import HelhestConfig, create_helhest_model

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# Geometry: chassis CoM is at body-frame (~-0.047, 0, ~0.6); rear wheel pivot at
# (-0.697, 0, 0). The line pivot→CoM makes atan2(0.65, 0.6) ≈ 47° with vertical,
# so tilting the body back by ~47° puts the CoM directly above the rear-wheel
# contact patch — the geometric balance point. The original commit had
# math.pi/2 (90°) which would have laid the chassis flat on its back; that's
# almost certainly a typo for math.pi/4 (45°).
BALANCE_PITCH = 0.825  # ≈47°


# -----------------------------------------------------------------------------
# Spline + Adam (cloned from trajectory_spline_surface_fast_bundled.py).
# -----------------------------------------------------------------------------


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, K] linear interpolation weight matrix + per-column normalization."""
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    col_sums = W.sum(axis=0)
    return W, col_sums


class SplineAdam:
    def __init__(self, K: int, num_dofs: int, lr: float, total_steps: int = 200,
                 lr_min_ratio: float = 0.05, betas=(0.9, 0.999), eps: float = 1e-8):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self) -> float:
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        lr = self._cosine_lr()
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


# -----------------------------------------------------------------------------
# Loss kernels.
# -----------------------------------------------------------------------------


@wp.kernel
def balance_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),  # [T+1, N, num_bodies]
    target_rot: wp.quat,
    target_pos: wp.vec3,
    weight_rot: float,
    weight_pos: float,
    loss: wp.array(dtype=wp.float32),
):
    """Quadratic orientation loss: penalizes any deviation from target up,
    even small wobbles. Original formulation."""
    sim_step, w = wp.tid()

    xform = body_pose[sim_step, w, 0]  # body 0 = chassis
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    pos_err = wp.vec3(pos[0] - target_pos[0], pos[1] - target_pos[1], 0.0)
    p_loss = wp.dot(pos_err, pos_err)

    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    up_err = current_up - target_up
    r_loss = wp.dot(up_err, up_err)

    wp.atomic_add(loss, 0, weight_pos * p_loss + weight_rot * r_loss)


@wp.kernel
def threshold_balance_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    target_pos: wp.vec3,
    weight_rot: float,
    weight_pos: float,
    alive_threshold: float,                # e.g. 0.85
    loss: wp.array(dtype=wp.float32),
):
    """Threshold orientation loss: zero when chassis-up is within the
    alive_threshold of target up; quadratic ramp below.

      r_loss = max(0, alive_threshold - dot(up, target_up))²

    With alive_threshold = 0.85 (≈ 32° from target), the optimizer gets
    NO gradient signal once the robot is balanced — it can stop pushing
    toward perfect alignment and instead focus on edge cases. Position
    loss is unchanged.
    """
    sim_step, w = wp.tid()

    xform = body_pose[sim_step, w, 0]
    pos = wp.transform_get_translation(xform)
    rot = wp.transform_get_rotation(xform)

    pos_err = wp.vec3(pos[0] - target_pos[0], pos[1] - target_pos[1], 0.0)
    p_loss = wp.dot(pos_err, pos_err)

    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    similarity = wp.dot(current_up, target_up)
    deficit = wp.max(0.0, alive_threshold - similarity)
    r_loss = deficit * deficit

    wp.atomic_add(loss, 0, weight_pos * p_loss + weight_rot * r_loss)


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),  # [T, N, num_dofs]
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, w, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, w, dof_idx] - target_vel[sim_step, w, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


# -----------------------------------------------------------------------------
# Diagnostic kernels — computed on the same trajectory after diff_step,
# but without contributing to the backward-pass gradient. They give us a
# decomposition of the loss + a "survival time" metric (how many timesteps
# the robot was at least half-upright) that correlates better with the
# qualitative "did the robot stay up?" question than the cumulative
# pose-error sum.
# -----------------------------------------------------------------------------


@wp.kernel
def diag_pos_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    weight_pos: float,
    out: wp.array(dtype=wp.float32),
):
    sim_step, w = wp.tid()
    xform = body_pose[sim_step, w, 0]
    pos = wp.transform_get_translation(xform)
    pos_err = wp.vec3(pos[0] - target_pos[0], pos[1] - target_pos[1], 0.0)
    wp.atomic_add(out, 0, weight_pos * wp.dot(pos_err, pos_err))


@wp.kernel
def diag_rot_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    weight_rot: float,
    out: wp.array(dtype=wp.float32),
):
    sim_step, w = wp.tid()
    xform = body_pose[sim_step, w, 0]
    rot = wp.transform_get_rotation(xform)
    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    up_err = current_up - target_up
    wp.atomic_add(out, 0, weight_rot * wp.dot(up_err, up_err))


@wp.kernel
def diag_survival_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_rot: wp.quat,
    threshold: float,                       # 0.5 ≈ within 60° of target up
    out: wp.array(dtype=wp.int32),          # count of "alive" timesteps
):
    """Counts timesteps where dot(chassis_up, target_up) > threshold —
    a permissive 'did the robot stay roughly upright' check."""
    sim_step, w = wp.tid()
    xform = body_pose[sim_step, w, 0]
    rot = wp.transform_get_rotation(xform)
    up_local = wp.vec3(0.0, 0.0, 1.0)
    current_up = wp.quat_rotate(rot, up_local)
    target_up = wp.quat_rotate(target_rot, up_local)
    if wp.dot(current_up, target_up) > threshold:
        wp.atomic_add(out, 0, 1)


@wp.kernel
def diag_smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    out: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, w, dof_idx] - target_vel[sim_step, w, dof_idx]
    wp.atomic_add(out, 0, weight * diff * diff)


@wp.kernel
def diag_reg_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    out: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, w, dof_idx]
    wp.atomic_add(out, 0, weight * v * v)


# -----------------------------------------------------------------------------
# Optimizer class.
# -----------------------------------------------------------------------------


class HelhestBalanceBundledOptimizer(AxionDifferentiableSimulator):
    """Spline-parameterized balance with optional bundled-gradient smoothing.

    With ``num_worlds=1`` and ``sigma=0`` this is equivalent to the original
    exact-gradient setup (modulo the spline reparameterization). With N>1 and
    σ>0, gradients are averaged across N noisy rollouts.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 10,
        sigma: float = 0.5,
        sigma_min_ratio: float = 0.1,
        antithetic: bool = True,
        lr: float = 0.05,
        total_steps: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.999,
        orient_loss: str = "threshold",
        weight_pos: float = 1.0,
        weight_rot: float = 200.0,
    ):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.K = num_control_points
        self.N = sim_config.num_worlds
        self.sigma_init = sigma
        self.sigma_min = sigma * sigma_min_ratio
        self.antithetic = antithetic and (sigma > 0) and (self.N > 1)
        if self.antithetic and self.N % 2 != 0:
            raise ValueError(f"antithetic=True requires even num_worlds, got {self.N}")

        self.lr = lr
        self.total_adam_steps = total_steps
        self.beta1 = beta1
        self.beta2 = beta2

        if orient_loss not in ("quadratic", "threshold"):
            raise ValueError(
                f"orient_loss must be 'quadratic' or 'threshold', got {orient_loss!r}")
        self.orient_loss = orient_loss

        # Visualization controls (set by main()).
        self.render_every: int = 0    # 0 = headless, K = replay every K-th iter
        self.render_loops: int = 1
        self.render_speed: float = 1.0

        # Track chassis (body 0) so render_episode can draw its trajectory.
        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.weight_rot = float(weight_rot)
        self.weight_pos = float(weight_pos)
        self.smoothness_weight = 1e-1
        self.regularization_weight = 1e-4

        # Optimization state: K knots × 3 wheel velocities.
        self.spline_params = np.zeros((self.K, NUM_WHEEL_DOFS), dtype=np.float64)
        self.spline_adam: SplineAdam | None = None
        self.W: np.ndarray | None = None
        self.W_col_sums: np.ndarray | None = None
        self.noise: np.ndarray | None = None  # [N, K, 3]

        # Diagnostics.
        self.iter_losses: list[float] = []
        self.iter_walls: list[float] = []
        self.iter_pos_losses: list[float] = []
        self.iter_rot_losses: list[float] = []
        self.iter_smooth_losses: list[float] = []
        self.iter_reg_losses: list[float] = []
        self.iter_alive_frac: list[float] = []  # fraction of timesteps "upright"

        # Diagnostic buffers (no grad — recomputed each iter from the
        # already-populated trajectory; do not affect backward pass).
        self._diag_pos = wp.zeros(1, dtype=wp.float32)
        self._diag_rot = wp.zeros(1, dtype=wp.float32)
        self._diag_smooth = wp.zeros(1, dtype=wp.float32)
        self._diag_reg = wp.zeros(1, dtype=wp.float32)
        self._diag_alive = wp.zeros(1, dtype=wp.int32)
        # dot(chassis_up, target_up) > 0.85 ≈ within ~32° of the target pose.
        # Stricter than just "didn't flip over" — counts only timesteps where the
        # chassis is *close to balanced*, which is the qualitative criterion the
        # eye uses when watching the GL viewer.
        self.alive_threshold = 0.85

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Spawn tilted backward by BALANCE_PITCH around +y (lift the front
        # wheels off, balance over the rear wheel).
        initial_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), initial_rot),
            control_mode="velocity",
            k_p=HelhestConfig.TARGET_KE,
            k_d=HelhestConfig.TARGET_KD,
            friction_left_right=0.7,
            friction_rear=0.35,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    # --- Bundled-gradient plumbing ------------------------------------------

    def _sample_noise(self, sigma: float) -> np.ndarray:
        if sigma <= 0.0 or self.N <= 1:
            return np.zeros((self.N, self.K, NUM_WHEEL_DOFS), dtype=np.float64)
        if self.antithetic:
            half = self.N // 2
            base = np.random.randn(half, self.K, NUM_WHEEL_DOFS).astype(np.float64) * sigma
            return np.concatenate([base, -base], axis=0)
        return np.random.randn(self.N, self.K, NUM_WHEEL_DOFS).astype(np.float64) * sigma

    def _current_sigma(self) -> float:
        if self.sigma_init <= 0.0:
            return 0.0
        progress = min(self.spline_adam.t / self.spline_adam.total_steps, 1.0)
        return self.sigma_min + 0.5 * (self.sigma_init - self.sigma_min) * (
            1.0 + np.cos(np.pi * progress)
        )

    def _expand_per_world(self, params: np.ndarray) -> np.ndarray:
        """[K, 3] params + [N, K, 3] noise -> [T, N, 3] per-step per-world wheel vels."""
        perturbed = params[None, :, :] + self.noise           # [N, K, 3]
        return np.einsum("tk,nkd->tnd", self.W, perturbed)    # [T, N, 3]

    def _apply_params(self, params: np.ndarray):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        self.noise = self._sample_noise(self._current_sigma())
        expanded = self._expand_per_world(params)
        vel_np = np.zeros((T, self.N, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded.astype(
            np.float32
        )
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    # --- Loss / update -------------------------------------------------------

    def compute_loss(self):
        T_plus_1 = self.trajectory.body_pose.shape[0]
        target_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        target_pos = wp.vec3(0.0, 0.0, 0.0)

        if self.orient_loss == "threshold":
            wp.launch(
                kernel=threshold_balance_loss_kernel,
                dim=(T_plus_1, self.N),
                inputs=[
                    self.trajectory.body_pose,
                    target_rot, target_pos,
                    self.weight_rot, self.weight_pos,
                    float(self.alive_threshold),
                ],
                outputs=[self.loss],
                device=self.solver.model.device,
            )
        else:
            wp.launch(
                kernel=balance_loss_kernel,
                dim=(T_plus_1, self.N),
                inputs=[
                    self.trajectory.body_pose,
                    target_rot, target_pos,
                    self.weight_rot, self.weight_pos,
                ],
                outputs=[self.loss],
                device=self.solver.model.device,
            )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, self.N, NUM_WHEEL_DOFS),
            inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET, self.regularization_weight],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=smoothness_kernel,
            dim=(self.clock.total_sim_steps - 1, self.N, NUM_WHEEL_DOFS),
            inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET, self.smoothness_weight],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def _compute_diagnostics(self) -> dict:
        """Recompute per-component losses + survival on the just-finished
        rollout, without touching the backward-pass tape. Returns dict of
        per-world averages."""
        for buf in (self._diag_pos, self._diag_rot, self._diag_smooth, self._diag_reg):
            buf.zero_()
        self._diag_alive.zero_()

        T_plus_1 = self.trajectory.body_pose.shape[0]
        target_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), BALANCE_PITCH)
        target_pos = wp.vec3(0.0, 0.0, 0.0)
        device = self.solver.model.device

        wp.launch(diag_pos_loss_kernel, dim=(T_plus_1, self.N),
                  inputs=[self.trajectory.body_pose, target_pos, self.weight_pos],
                  outputs=[self._diag_pos], device=device)
        wp.launch(diag_rot_loss_kernel, dim=(T_plus_1, self.N),
                  inputs=[self.trajectory.body_pose, target_rot, self.weight_rot],
                  outputs=[self._diag_rot], device=device)
        wp.launch(diag_survival_kernel, dim=(T_plus_1, self.N),
                  inputs=[self.trajectory.body_pose, target_rot, float(self.alive_threshold)],
                  outputs=[self._diag_alive], device=device)
        wp.launch(diag_smoothness_kernel,
                  dim=(self.clock.total_sim_steps - 1, self.N, NUM_WHEEL_DOFS),
                  inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                          self.smoothness_weight],
                  outputs=[self._diag_smooth], device=device)
        wp.launch(diag_reg_kernel,
                  dim=(self.clock.total_sim_steps, self.N, NUM_WHEEL_DOFS),
                  inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                          self.regularization_weight],
                  outputs=[self._diag_reg], device=device)
        wp.synchronize()

        # All component buffers sum across N worlds → divide for per-world avg.
        pos = float(self._diag_pos.numpy()[0]) / self.N
        rot = float(self._diag_rot.numpy()[0]) / self.N
        smooth = float(self._diag_smooth.numpy()[0]) / self.N
        reg = float(self._diag_reg.numpy()[0]) / self.N
        alive = int(self._diag_alive.numpy()[0])
        alive_frac = alive / float(T_plus_1 * self.N)
        return dict(pos=pos, rot=rot, smooth=smooth, reg=reg, alive_frac=alive_frac)

    def update(self):
        # gradient w.r.t. joint_target_vel: [T, N, num_dofs]
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]  # [T, N, 3]

        # Per-world spline contraction: [T, N, 3] -> [K, N, 3]
        safe_sums = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        grad_per_world = np.einsum("tk,tnd->knd", self.W, grad_v) / safe_sums[:, None, None]

        # Bundled gradient = average across N worlds.
        grad_params = grad_per_world.mean(axis=1)

        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def _build_target_episode(self):
        """Run a target rollout where wheels are stationary — that's our 'reference'.
        Not actually used by the loss (target is BALANCE_PITCH/origin), but the
        AxionDifferentiableSimulator base expects target trajectories to exist.
        """
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        T = self.clock.total_sim_steps
        zeros = np.zeros((self.N, num_dofs), dtype=np.float32)
        for i in range(T):
            wp.copy(self.target_controls[i].joint_target_vel, wp.array(zeros, dtype=wp.float32))
        self.run_target_episode()

    def train(self, iterations: int = 60):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        self._build_target_episode()

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.zeros((self.K, NUM_WHEEL_DOFS), dtype=np.float64)
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS,
            lr=self.lr, lr_min_ratio=0.2, total_steps=self.total_adam_steps,
            betas=(self.beta1, self.beta2),
        )
        self._apply_params(self.spline_params)

        for i in range(iterations):
            wp.synchronize()
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            self.iter_walls.append(time.perf_counter() - t0)
            curr_loss = float(self.loss.numpy()[0]) / self.N
            self.iter_losses.append(curr_loss)
            sigma_now = self._current_sigma()

            diag = self._compute_diagnostics()
            self.iter_pos_losses.append(diag["pos"])
            self.iter_rot_losses.append(diag["rot"])
            self.iter_smooth_losses.append(diag["smooth"])
            self.iter_reg_losses.append(diag["reg"])
            self.iter_alive_frac.append(diag["alive_frac"])

            print(f"Iter {i:3d}: loss={curr_loss:>9.2f}  "
                  f"[rot={diag['rot']:>7.1f} pos={diag['pos']:>6.1f} "
                  f"smooth={diag['smooth']:>5.2f} reg={diag['reg']:>5.2f}]  "
                  f"alive={diag['alive_frac']*100:>4.1f}%  σ={sigma_now:.2f}  "
                  f"wall={self.iter_walls[-1]:.2f}s", flush=True)

            # Replay world 0's trajectory in the GL viewer at the requested cadence.
            if self.render_every > 0 and (i % self.render_every == 0):
                self.render_episode(
                    iteration=i,
                    loop=True, loops_count=self.render_loops,
                    playback_speed=self.render_speed,
                )

            self.update()
            self.tape.zero()
            self.loss.zero_()


# -----------------------------------------------------------------------------
# CLI entry point.
# -----------------------------------------------------------------------------


def _make_default_engine_config() -> AxionEngineConfig:
    return AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, atol=1e-3, tol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-10, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=64),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-worlds", type=int, default=8,
                        help="Bundled samples (parallel worlds). Use 1 for exact gradient.")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Initial perturbation σ on spline knots (rad/s); 0 disables bundling.")
    parser.add_argument("--sigma-min-ratio", type=float, default=0.1)
    parser.add_argument("--no-antithetic", action="store_true")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--knots", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam β1. 0.0 = no momentum (RMSprop-like).")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam β2 (running squared-gradient avg).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis", choices=["gl", "headless"], default="headless")
    parser.add_argument("--render-every", type=int, default=0,
                        help="Replay the trajectory in the GL viewer every K Adam iterations (0=disabled).")
    parser.add_argument("--render-loops", type=int, default=1,
                        help="How many times to loop each replay before continuing training.")
    parser.add_argument("--render-speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0=realtime, 2.0=2x).")
    parser.add_argument("--orient-loss", choices=["quadratic", "threshold"], default="threshold",
                        help="Orientation loss formulation: 'quadratic' = original (penalizes any "
                             "deviation); 'threshold' = zero loss when chassis-up is within "
                             "alive_threshold of target (default).")
    parser.add_argument("--weight-pos", type=float, default=1.0,
                        help="Position-drift loss weight. Original was 10.0; 1.0 lets the robot "
                             "translate freely while still penalizing far-drift.")
    parser.add_argument("--weight-rot", type=float, default=200.0,
                        help="Orientation loss weight (multiplies the chosen rot-loss formulation).")
    args = parser.parse_args()

    np.random.seed(args.seed)

    sim_config = SimulationConfig(
        duration_seconds=4.0,
        target_timestep_seconds=5e-2,
        num_worlds=args.num_worlds,
    )
    render_config = RenderingConfig(
        vis_type=("null" if args.vis == "headless" else "gl"),
        target_fps=30, usd_file=None, world_offset_x=20.0, world_offset_y=20.0,
    )
    engine_config = _make_default_engine_config()
    logging_config = LoggingConfig()

    sim = HelhestBalanceBundledOptimizer(
        sim_config, render_config, engine_config, logging_config,
        num_control_points=args.knots, sigma=args.sigma,
        sigma_min_ratio=args.sigma_min_ratio, antithetic=not args.no_antithetic,
        lr=args.lr, total_steps=args.iterations,
        beta1=args.beta1, beta2=args.beta2,
        orient_loss=args.orient_loss,
        weight_pos=args.weight_pos, weight_rot=args.weight_rot,
    )
    sim.render_every = args.render_every
    sim.render_loops = args.render_loops
    sim.render_speed = args.render_speed
    sim.train(iterations=args.iterations)


if __name__ == "__main__":
    main()
