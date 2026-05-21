"""Waypoint-based trajectory optimization on flat ground.

Instead of tracking a full reference trajectory at every timestep, the robot
must pass through a sparse set of (timestep, x, y) waypoints.  Waypoints are
sampled from a target-spline simulation so the experiment stays reproducible
and seed-controlled.

Usage:
    # Single seed, 6 waypoints:
    python -m examples.terrain_traversal.optimize_waypoints --seed 42

    # Fewer / more waypoints:
    python -m examples.terrain_traversal.optimize_waypoints --seed 42 --num-waypoints 3

    # With visualization:
    python -m examples.terrain_traversal.optimize_waypoints --seed 42 --visualize

    # Batch over N seeds:
    python -m examples.terrain_traversal.optimize_waypoints --num-seeds 20 \
        --save results/waypoints_batch.json
"""

import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.sim_config import SyncMode
from newton import Model

from examples.terrain_traversal.helhest_model import create_helhest_model
from examples.terrain_traversal.helhest_model import HelhestConfig
from examples.terrain_traversal.optimize import (
    generate_splines,
    make_interp_matrix,
    WHEEL_DOF_OFFSET,
    NUM_WHEEL_DOFS,
    DT,
)

os.environ["PYOPENGL_PLATFORM"] = "glx"


# ---------------------------------------------------------------------------
# Waypoint generation
# ---------------------------------------------------------------------------

def generate_waypoints_from_spline(
    target_spline: np.ndarray,
    num_waypoints: int,
    dt: float,
    duration: float,
    spawn_pos: tuple[float, float, float],
    engine_config: AxionEngineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the target spline on flat ground and sample waypoints.

    Returns:
        waypoint_steps: (N,) int array of timestep indices.
        waypoint_xy:    (N, 2) float array of (x, y) positions.
        waypoint_quat:  (N, 4) float array of quaternions (x, y, z, w).
        waypoint_vel_xy: (N, 2) float array of linear velocity (vx, vy).
        full_trajectory_xy: (T+1, 2) for plotting reference.
    """
    T = int(duration / dt)
    K = target_spline.shape[0]
    W, _ = make_interp_matrix(T, K)
    expanded = W @ target_spline  # (T, 3)

    # Build a throwaway 1-world model
    from axion.core.engine import AxionEngine
    from axion.core.logging_config import LoggingConfig as _LoggingConfig
    from axion.core.model_builder import AxionModelBuilder

    builder = AxionModelBuilder()
    builder.rigid_gap = 0.1
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0),
    )
    create_helhest_model(
        builder,
        xform=wp.transform(wp.vec3(*spawn_pos), wp.quat_identity()),
        control_mode="velocity",
        k_p=250.0, k_d=0.0,
        friction_left_right=0.8, friction_rear=0.35,
    )
    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)
    engine = AxionEngine(
        model=model, sim_steps=T, config=engine_config,
        logging_config=_LoggingConfig(), differentiable_simulation=False,
    )
    ndof = engine.dims.joint_dof_count
    control = model.control()

    # Forward simulate
    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    state_out = model.state()

    # Store full body_q: (x, y, z, qx, qy, qz, qw)
    poses = np.zeros((T + 1, 7), dtype=np.float32)
    poses[0] = state_in.body_q.numpy()[0, :7]

    ctrl_np = np.zeros((1, ndof), dtype=np.float32)
    for step in range(T):
        ctrl_np[0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[step]
        wp.copy(
            control.joint_target_vel,
            wp.array(ctrl_np, dtype=wp.float32, device=model.device),
        )
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
        poses[step + 1] = state_in.body_q.numpy()[0, :7]

    poses_xy = poses[:, :2]

    # Compute velocity from finite differences of position
    vel_xy = np.zeros((T + 1, 2), dtype=np.float32)
    vel_xy[1:] = (poses_xy[1:] - poses_xy[:-1]) / dt
    vel_xy[0] = vel_xy[1]  # copy first step

    # Sample waypoints at evenly-spaced timesteps, starting after ~10% of the horizon
    t_start = max(1, T // 10)
    step_indices = np.linspace(t_start, T, num_waypoints, dtype=int)
    # Avoid duplicates at boundaries
    step_indices = np.unique(step_indices)
    waypoint_xy = poses_xy[step_indices]
    waypoint_quat = poses[step_indices, 3:7]  # (qx, qy, qz, qw)
    waypoint_vel_xy = vel_xy[step_indices]

    return step_indices.astype(np.int32), waypoint_xy.astype(np.float32), waypoint_quat.astype(np.float32), waypoint_vel_xy.astype(np.float32), poses_xy


# ---------------------------------------------------------------------------
# L-BFGS optimizer
# ---------------------------------------------------------------------------

class SplineLBFGS:
    """L-BFGS optimizer for small spline parameter spaces.

    Uses the two-loop recursion to build an approximate inverse Hessian
    from the last `m` (s, y) pairs.  No line search — uses a fixed step
    size with gradient clipping and a safeguard that rejects steps that
    increase the loss.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        lr: float = 1.0,
        m: int = 10,
        grad_clip: float = 5.0,
        max_step_norm: float = 0.5,
    ):
        self.lr = lr
        self.m = m
        self.grad_clip = grad_clip
        self.max_step_norm = max_step_norm

        # History buffers (stored as flat vectors)
        self.n = shape[0] * shape[1]
        self.shape = shape
        self.s_hist: list[np.ndarray] = []  # param differences
        self.y_hist: list[np.ndarray] = []  # grad differences
        self.rho_hist: list[float] = []

        self.prev_params: np.ndarray | None = None
        self.prev_grad: np.ndarray | None = None
        self.step_count = 0

    def reset_history(self):
        """Clear L-BFGS history (call after NaN or rollback)."""
        self.s_hist.clear()
        self.y_hist.clear()
        self.rho_hist.clear()
        self.prev_params = None
        self.prev_grad = None

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        g = grad.ravel().astype(np.float64)
        p = params.ravel().astype(np.float64)

        # Skip if gradient is NaN
        if not np.all(np.isfinite(g)):
            self.reset_history()
            return params

        # Clip gradient
        g_norm = np.linalg.norm(g)
        if g_norm > self.grad_clip:
            g = g * (self.grad_clip / g_norm)

        # Update history with (s, y) pair from last step
        if self.prev_params is not None:
            s = p - self.prev_params
            y = g - self.prev_grad
            sy = np.dot(s, y)
            if sy > 1e-10:  # curvature condition
                self.s_hist.append(s)
                self.y_hist.append(y)
                self.rho_hist.append(1.0 / sy)
                if len(self.s_hist) > self.m:
                    self.s_hist.pop(0)
                    self.y_hist.pop(0)
                    self.rho_hist.pop(0)

        self.prev_params = p.copy()
        self.prev_grad = g.copy()

        # Two-loop recursion
        q = g.copy()
        k = len(self.s_hist)

        if k == 0:
            # First iteration: steepest descent
            direction = g
        else:
            alphas = np.zeros(k)
            for i in range(k - 1, -1, -1):
                alphas[i] = self.rho_hist[i] * np.dot(self.s_hist[i], q)
                q = q - alphas[i] * self.y_hist[i]

            # Initial Hessian approximation: H0 = (s^T y / y^T y) * I
            s_last = self.s_hist[-1]
            y_last = self.y_hist[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            r = gamma * q

            for i in range(k):
                beta = self.rho_hist[i] * np.dot(self.y_hist[i], r)
                r = r + self.s_hist[i] * (alphas[i] - beta)

            direction = r

        # Clip the step direction norm to prevent blowup
        step = self.lr * direction
        step_norm = np.linalg.norm(step)
        if step_norm > self.max_step_norm:
            step = step * (self.max_step_norm / step_norm)

        new_params = p - step
        self.step_count += 1
        return new_params.reshape(self.shape)


# ---------------------------------------------------------------------------
# Loss kernels
# ---------------------------------------------------------------------------

@wp.kernel
def waypoint_pos_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    waypoint_pos: wp.array(dtype=wp.vec3),
    waypoint_step: wp.array(dtype=wp.int32),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 position loss at sparse waypoint timesteps."""
    i = wp.tid()
    t = waypoint_step[i]
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    target = waypoint_pos[i]
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def waypoint_yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    waypoint_quat: wp.array(dtype=wp.quat),
    waypoint_step: wp.array(dtype=wp.int32),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Yaw alignment loss at sparse waypoint timesteps."""
    i = wp.tid()
    t = waypoint_step[i]
    q = wp.transform_get_rotation(body_pose[t, 0, 0])
    q_target = waypoint_quat[i]
    fwd = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    fwd_target = wp.quat_rotate(q_target, wp.vec3(1.0, 0.0, 0.0))
    dot_fwd = wp.dot(fwd, fwd_target)
    wp.atomic_add(loss, 0, weight * (1.0 - dot_fwd * dot_fwd))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Finite-difference smoothness on wheel velocities."""
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, 0, dof_idx] - target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

SPAWN_POS = (-8.0, 0.0, 0.5)


class WaypointOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=10,
        target_spline=None,
        init_spline=None,
        waypoint_steps=None,
        waypoint_xy=None,
        waypoint_quat=None,
        full_target_xy=None,
        lr=0.1,
        visualize=False,
    ):
        self.K = num_control_points
        self._target_spline = target_spline
        self._init_spline = init_spline
        self._lr = lr
        self._visualize = visualize
        self._render_frame = 0

        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.waypoint_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7
        self.smoothness_weight = 1e-3

        # Store waypoints as warp arrays
        N = len(waypoint_steps)
        self._waypoint_steps_np = waypoint_steps
        self._waypoint_xy_np = waypoint_xy
        self._waypoint_quat_np = waypoint_quat
        self._full_target_xy = full_target_xy
        self._num_waypoints = N

        self.wp_steps = wp.array(waypoint_steps, dtype=wp.int32, device=self.model.device)
        self.wp_pos = wp.array(
            [wp.vec3(float(p[0]), float(p[1]), 0.0) for p in waypoint_xy],
            dtype=wp.vec3, device=self.model.device,
        )
        self.wp_quat = wp.array(
            [wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in waypoint_quat],
            dtype=wp.quat, device=self.model.device,
        )

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        if self._visualize:
            self.viewer.set_camera(
                pos=wp.vec3(-12.0, -12.0, 14.0),
                pitch=-35.0,
                yaw=45.0,
            )

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1

        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0),
        )

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(*SPAWN_POS), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0, k_d=0.0,
            friction_left_right=0.8, friction_rear=0.35,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def _expand(self, params):
        return self.W @ params

    def _contract(self, grad_v):
        return self.W.T @ grad_v

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        T = self.clock.total_sim_steps
        N = self._num_waypoints

        wp.launch(
            kernel=waypoint_pos_loss_kernel,
            dim=N,
            inputs=[
                self.trajectory.body_pose,
                self.wp_pos,
                self.wp_steps,
                self.waypoint_weight / N,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=waypoint_yaw_loss_kernel,
            dim=N,
            inputs=[
                self.trajectory.body_pose,
                self.wp_quat,
                self.wp_steps,
                self.yaw_weight / N,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(T, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / T,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        if T > 1:
            wp.launch(
                kernel=smoothness_kernel,
                dim=(T - 1, NUM_WHEEL_DOFS),
                inputs=[
                    self.trajectory.joint_target_vel,
                    WHEEL_DOF_OFFSET,
                    self.smoothness_weight / T,
                ],
                outputs=[self.loss],
                device=self.solver.model.device,
            )

    def update(self):
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]
        grad_params = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.optimizer.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def _compute_waypoint_errors(self):
        """Return per-waypoint L2 distance (m) from current trajectory."""
        poses = self.trajectory.body_pose.numpy()[:, 0, 0, :3]  # (T+1, 3)
        errors = []
        for i, t in enumerate(self._waypoint_steps_np):
            dx = poses[t, 0] - self._waypoint_xy_np[i, 0]
            dy = poses[t, 1] - self._waypoint_xy_np[i, 1]
            errors.append(float(np.sqrt(dx * dx + dy * dy)))
        return np.array(errors)

    def render(self, train_iter):
        if not self._visualize:
            return
        if self._render_frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]

        # Draw waypoints as small boxes
        waypoint_xforms = wp.array(
            [
                wp.transform(
                    wp.vec3(float(self._waypoint_xy_np[i, 0]),
                            float(self._waypoint_xy_np[i, 1]),
                            float(SPAWN_POS[2])),
                    wp.quat_identity(),
                )
                for i in range(self._num_waypoints)
            ],
            dtype=wp.transform,
        )
        waypoint_colors = wp.array(
            [wp.vec3(1.0, 0.2, 0.0)] * self._num_waypoints,
            dtype=wp.vec3,
        )
        half = (0.15, 0.15, 0.15)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/waypoints",
                newton.GeoType.BOX,
                half,
                waypoint_xforms,
                waypoint_colors,
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")
        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=3.0,
        )
        self._render_frame += 1

    def train(self, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps
        self.W, _ = make_interp_matrix(T, self.K)

        # Initialize with perturbed spline
        self.spline_params = self._init_spline.copy().astype(np.float64)
        self.optimizer = SplineLBFGS(
            shape=(self.K, NUM_WHEEL_DOFS),
            lr=self._lr,
            m=10,
            grad_clip=5.0,
        )
        self._apply_params(self.spline_params)

        print(
            f"Waypoint optimization: T={T}, dt={self.clock.dt:.4f}, "
            f"K={self.K}, waypoints={self._num_waypoints}, "
            f"duration={T * self.clock.dt:.1f}s"
        )
        print(f"  Waypoint timesteps: {self._waypoint_steps_np.tolist()}")

        results = {
            "simulator": "Axion",
            "problem": "waypoint_flat",
            "K": self.K,
            "T": T,
            "dt": self.clock.dt,
            "duration": T * self.clock.dt,
            "num_waypoints": self._num_waypoints,
            "waypoint_steps": self._waypoint_steps_np.tolist(),
            "waypoint_xy": self._waypoint_xy_np.tolist(),
            "full_target_xy": {
                "x": self._full_target_xy[:, 0].tolist(),
                "y": self._full_target_xy[:, 1].tolist(),
            },
            "iterations": [],
            "loss": [],
            "mean_wp_error_m": [],
            "max_wp_error_m": [],
            "time_ms": [],
            "trajectories": {},
        }

        best_mean_err = float("inf")
        best_params = self.spline_params.copy()
        nan_recoveries = 0

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            # NaN recovery: roll back to best params, reset optimizer
            if not np.isfinite(curr_loss):
                nan_recoveries += 1
                print(f"  Iter {i:3d}: NaN — rollback to best (recovery #{nan_recoveries})")
                self.spline_params = best_params.copy()
                self._apply_params(self.spline_params)
                self.optimizer.reset_history()
                self.tape.zero()
                self.loss.zero_()
                continue

            wp_errors = self._compute_waypoint_errors()
            mean_err = float(np.mean(wp_errors))
            max_err = float(np.max(wp_errors))

            if mean_err < best_mean_err:
                best_mean_err = mean_err
                best_params = self.spline_params.copy()

            print(
                f"  Iter {i:3d}: loss={curr_loss:.4f} | "
                f"wp_mean={mean_err:.3f}m wp_max={max_err:.3f}m | "
                f"best_mean={best_mean_err:.3f}m | "
                f"t={t_iter * 1000:.0f}ms"
            )

            poses = self.trajectory.body_pose.numpy()[:, 0, 0, :3]
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["mean_wp_error_m"].append(mean_err)
            results["max_wp_error_m"].append(max_err)
            results["time_ms"].append(t_iter * 1000)
            results["trajectories"][str(i)] = {
                "x": poses[:, 0].tolist(),
                "y": poses[:, 1].tolist(),
            }

            self.render(i)
            self.update()
            self.tape.zero()
            self.loss.zero_()

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_single(args, seed):
    engine_config = AxionEngineConfig(
        max_newton_iters=14,
        max_linear_iters=16,
        backtrack_min_iter=10,
        newton_atol=1e-3,
        linear_atol=1e-3,
        linear_tol=1e-3,
        enable_linesearch=False,
        joint_compliance=6e-8,
        contact_compliance=0.1,
        friction_compliance=1e-6,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=256,
    )

    target_spline, init_spline = generate_splines(
        seed, K=args.K, sigma=args.sigma, wildness=args.curvature,
    )
    print(
        f"  Target spline[0]: L={target_spline[0,0]:.2f} R={target_spline[0,1]:.2f} "
        f"Rear={target_spline[0,2]:.2f}"
    )
    print(
        f"  Init   spline[0]: L={init_spline[0,0]:.2f} R={init_spline[0,1]:.2f} "
        f"Rear={init_spline[0,2]:.2f}"
    )

    # Generate waypoints by simulating the target spline
    waypoint_steps, waypoint_xy, waypoint_quat, _, full_target_xy = generate_waypoints_from_spline(
        target_spline,
        num_waypoints=args.num_waypoints,
        dt=args.dt,
        duration=args.duration,
        spawn_pos=SPAWN_POS,
        engine_config=engine_config,
    )
    print(f"  Waypoints ({len(waypoint_steps)}):")
    for i, (t, xy, q) in enumerate(zip(waypoint_steps, waypoint_xy, waypoint_quat)):
        # Extract yaw from quaternion for display
        qx, qy, qz, qw = q
        fx = 1 - 2 * (qy * qy + qz * qz)
        fy = 2 * (qx * qy + qw * qz)
        yaw_deg = float(np.degrees(np.arctan2(fy, fx)))
        print(f"    [{i}] t={t} ({t * args.dt:.2f}s)  ->  ({xy[0]:.2f}, {xy[1]:.2f})  yaw={yaw_deg:.1f}°")

    visualize = getattr(args, "visualize", False)

    sim_config = SimulationConfig(
        duration_seconds=args.duration,
        target_timestep_seconds=args.dt,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="gl" if visualize else "null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=1,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = WaypointOptimizer(
        sim_config, render_config, exec_config, engine_config, logging_config,
        num_control_points=args.K,
        target_spline=target_spline,
        init_spline=init_spline,
        waypoint_steps=waypoint_steps,
        waypoint_xy=waypoint_xy,
        waypoint_quat=waypoint_quat,
        full_target_xy=full_target_xy,
        lr=args.lr,
        visualize=visualize,
    )
    result = sim.train(iterations=args.iterations)
    result["seed"] = seed
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint-based trajectory optimization on flat ground",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Run over N seeds (0..N-1)")
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--K", type=int, default=10,
                        help="Spline control points (default: 10)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-waypoints", type=int, default=6,
                        help="Number of waypoints to sample from target (default: 6)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Initial guess perturbation std (default: 1.5)")
    parser.add_argument("--curvature", type=float, default=0.8,
                        help="Target trajectory curvature (default: 0.8)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.num_seeds is not None:
        all_results = []
        for seed in range(args.num_seeds):
            print(f"\n{'=' * 60}")
            print(f"  SEED {seed}/{args.num_seeds - 1}")
            print(f"{'=' * 60}")
            result = run_single(args, seed=seed)
            all_results.append(result)

        final_errs = [r["mean_wp_error_m"][-1] for r in all_results]
        best_errs = [min(r["mean_wp_error_m"]) for r in all_results]

        summary = {
            "num_seeds": args.num_seeds,
            "num_waypoints": args.num_waypoints,
            "final_mean_wp_error_median": float(np.median(final_errs)),
            "final_mean_wp_error_mean": float(np.mean(final_errs)),
            "best_mean_wp_error_median": float(np.median(best_errs)),
            "best_mean_wp_error_mean": float(np.mean(best_errs)),
            "per_seed": all_results,
        }

        print(f"\n{'=' * 60}")
        print(f"  BATCH SUMMARY ({args.num_seeds} seeds, {args.num_waypoints} waypoints)")
        print(f"{'=' * 60}")
        print(f"  Final mean wp error: {summary['final_mean_wp_error_median']:.3f}m median")
        print(f"  Best  mean wp error: {summary['best_mean_wp_error_median']:.3f}m median")

        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(summary, indent=2))
            print(f"  Saved to {args.save}")
    else:
        result = run_single(args, seed=args.seed)
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(result, indent=2))
            print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
