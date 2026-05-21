"""Terrain traversal benchmark — Axion only.

Optimises K spline control knots for the Helhest robot to follow a target
trajectory over a procedurally generated triangle mesh surface.

Both the target trajectory and the initial guess are splines generated from
the seed, with the initial guess being a perturbed version of the target.
This ensures the experiment is fully reproducible and eliminates any
cherry-picking of controls.

Usage:
    # Single random terrain with seed:
    python -m examples.terrain_traversal.optimize --seed 42

    # Batch over N random terrains:
    python -m examples.terrain_traversal.optimize --num-seeds 50 \
        --save results/terrain_batch.json

    # With visualization:
    python -m examples.terrain_traversal.optimize --seed 42 --visualize
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
from examples.terrain_traversal.terrain import generate_terrain_mesh

os.environ["PYOPENGL_PLATFORM"] = "glx"

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

DT = 8e-2

# Control ranges for random spline generation (rad/s)
CTRL_RANGES = {
    "left": (1.5, 4.0),
    "right": (0.5, 3.0),
    "rear": (0.5, 3.0),
}


def generate_splines(seed: int, K: int, sigma: float = 0.5, wildness: float = 0.8):
    """Generate a random target spline and a perturbed initial guess.

    The target spline has K knots with per-wheel velocities drawn uniformly.
    Left and right wheels are correlated (base + differential) to avoid pure
    spinning and produce natural driving trajectories.

    The initial guess is the target + Gaussian noise, clamped to valid range.

    Args:
        seed: Random seed.
        K: Number of spline control points.
        sigma: Std dev of perturbation noise (rad/s).
        wildness: Max differential between left/right wheels (rad/s).
            0.0 = straight driving, 0.8 = moderate turns, 2.0 = aggressive.

    Returns:
        Tuple of (target_spline, init_spline), each shape (K, 3).
    """
    rng = np.random.default_rng(seed)

    # Generate correlated left/right: base speed + differential
    base_speed = rng.uniform(1.5, 3.5, size=K)  # forward component
    differential = rng.uniform(-wildness, wildness, size=K)  # turning component
    left = base_speed + differential
    right = base_speed - differential
    rear = rng.uniform(*CTRL_RANGES["rear"], size=K)

    target_spline = np.column_stack([left, right, rear])

    # Smooth the target slightly to avoid jerky controls
    if K > 2:
        for col in range(3):
            padded = np.pad(target_spline[:, col], 1, mode="edge")
            target_spline[:, col] = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

    # Clamp to valid ranges
    target_spline[:, 0] = np.clip(target_spline[:, 0], 0.5, 5.0)
    target_spline[:, 1] = np.clip(target_spline[:, 1], 0.1, 5.0)
    target_spline[:, 2] = np.clip(target_spline[:, 2], 0.1, 5.0)

    # Perturbed initial guess — resample if perturbation is too large
    max_knot_dist = 2.0 * sigma  # reject if any knot moves more than this
    for _ in range(100):
        noise = rng.normal(0.0, sigma, size=(K, 3))
        init_spline = np.clip(target_spline + noise, 0.1, 5.0)
        knot_dists = np.linalg.norm(init_spline - target_spline, axis=1)
        if np.max(knot_dists) <= max_knot_dist:
            break

    return target_spline, init_spline


def make_interp_matrix(T: int, K: int):
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
    def __init__(
        self,
        K,
        num_dofs,
        lr,
        total_steps=200,
        lr_min_ratio=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
        grad_clip=None,
    ):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.grad_clip = grad_clip
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params, grad):
        self.t += 1
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        lr = self._cosine_lr()
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    q = wp.transform_get_rotation(body_pose[t, 0, 0])
    q_target = wp.transform_get_rotation(target_body_pose[t, 0, 0])
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


class TerrainTraversalOptimizer(AxionDifferentiableSimulator):
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
        terrain_seed=None,
        roughness=1.0,
        terrain_freq=1.0,
        lr=0.005,
        check_grad=False,
        visualize=False,
    ):
        self.K = num_control_points
        self._target_spline = target_spline  # (K, 3)
        self._init_spline = init_spline  # (K, 3)
        self._terrain_seed = terrain_seed
        self._roughness = roughness
        self._terrain_freq = terrain_freq
        self._lr = lr
        self._check_grad = check_grad
        self._visualize = visualize
        self._render_frame = 0
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7
        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        if self._visualize:
            # 3/4 view from behind-left, elevated to see full 24m terrain + robot
            self.viewer.set_camera(
                pos=wp.vec3(-15.0, -15.0, 18.0),
                pitch=-35.0,
                yaw=45.0,
            )

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.2

        if self._terrain_seed is not None:
            surface_mesh, terrain_h = generate_terrain_mesh(
                seed=self._terrain_seed,
                roughness=self._roughness,
                terrain_freq=self._terrain_freq,
            )
            spawn_z = terrain_h + HelhestConfig.WHEEL_RADIUS + 0.05
        else:
            raise ValueError("terrain_seed is required")

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(-8.0, 0.0, spawn_z), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0,
            k_d=0.0,
            friction_left_right=0.8,
            friction_rear=0.35,
        )

        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.5,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
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
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def _apply_spline_to_controls(self, spline, controls, T):
        """Expand a (K, 3) spline and write into a list of controls."""
        expanded = self.W @ spline  # (T, 3)
        num_dofs = controls[0].joint_target_vel.shape[-1]
        for i in range(T):
            ctrl = np.zeros(num_dofs, dtype=np.float32)
            ctrl[WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(
                controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )

    def compute_loss(self):
        T = self.trajectory.body_pose.shape[0]
        wp.launch(
            kernel=loss_kernel,
            dim=T,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.trajectory_weight / T,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=yaw_loss_kernel,
            dim=T,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.yaw_weight / T,
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

    def update(self):
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]
        grad_params = self._contract(grad_v)

        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def finite_diff_check(self, eps=1e-3):
        """Compare adjoint gradient with centered finite differences on current params."""
        print("\n  === Finite Difference Gradient Check ===")

        # Get adjoint gradient
        self._apply_params(self.spline_params)
        self.diff_step()
        wp.synchronize()
        adjoint_loss = self.loss.numpy()[0]

        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]
        adjoint_grad = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        self.tape.zero()
        self.loss.zero_()

        # Finite differences on a subset of params (first 3 knots, all wheels)
        fd_grad = np.zeros_like(adjoint_grad)
        check_knots = min(3, self.K)
        for k in range(check_knots):
            for d in range(NUM_WHEEL_DOFS):
                params_plus = self.spline_params.copy()
                params_plus[k, d] += eps
                self._apply_params(params_plus)
                self.diff_step()
                wp.synchronize()
                loss_plus = self.loss.numpy()[0]
                self.tape.zero()
                self.loss.zero_()

                params_minus = self.spline_params.copy()
                params_minus[k, d] -= eps
                self._apply_params(params_minus)
                self.diff_step()
                wp.synchronize()
                loss_minus = self.loss.numpy()[0]
                self.tape.zero()
                self.loss.zero_()

                fd_grad[k, d] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original params
        self._apply_params(self.spline_params)

        # Print comparison
        print(f"  Loss at current params: {adjoint_loss:.6f}")
        print(f"  {'Knot':>4} {'DOF':>3} | {'Adjoint':>10} {'FD':>10} {'Ratio':>8} {'Match':>5}")
        print(f"  {'-'*50}")
        for k in range(check_knots):
            for d in range(NUM_WHEEL_DOFS):
                a = adjoint_grad[k, d]
                f = fd_grad[k, d]
                if abs(f) > 1e-8:
                    ratio = a / f
                    match = "OK" if 0.5 < abs(ratio) < 2.0 else "BAD"
                else:
                    ratio = float("nan")
                    match = "~0"
                dof_name = ["L", "R", "Re"][d]
                print(f"  {k:4d} {dof_name:>3} | {a:10.4f} {f:10.4f} {ratio:8.2f} {match:>5}")
        print()

    def render(self, train_iter):
        if not self._visualize:
            return
        if self._render_frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]

        target_poses = self.trajectory.target_body_pose.numpy()
        num_steps = target_poses.shape[0]

        waypoint_stride = max(1, num_steps // 20)
        waypoint_indices = list(range(0, num_steps, waypoint_stride))

        waypoint_xforms = wp.array(
            [target_poses[i, 0, 0] for i in waypoint_indices],
            dtype=wp.transform,
        )
        waypoint_colors = wp.array(
            [wp.vec3(1.0, 0.2, 0.0)] * len(waypoint_indices),
            dtype=wp.vec3,
        )

        half = (
            HelhestConfig.CHASSIS_SIZE[0] / 8.0,
            HelhestConfig.CHASSIS_SIZE[1] / 8.0,
            HelhestConfig.CHASSIS_SIZE[2] / 8.0,
        )

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target_trajectory",
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

    def _settle(self, num_steps=10):
        """Run non-differentiable settling steps to let the robot rest on the terrain."""
        print(f"Settling for {num_steps} steps...")
        settle_state_a = self.model.state()
        settle_state_b = self.model.state()
        settle_control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, settle_state_a)

        for i in range(num_steps):
            self.collision_pipeline.collide(settle_state_a, self.contacts)
            self.solver.step(
                state_in=settle_state_a,
                state_out=settle_state_b,
                control=settle_control,
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            settle_state_a, settle_state_b = settle_state_b, settle_state_a

        wp.copy(self.model.joint_q, settle_state_a.joint_q)
        wp.copy(self.model.joint_qd, settle_state_a.joint_qd)
        self.solver.reset_timestep_counter()

    def train(self, iterations=200):
        self._settle()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps

        # Build interpolation matrix
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        # Target episode: expand target spline into per-step controls and simulate
        self._apply_spline_to_controls(self._target_spline, self.target_controls, T)
        self.run_target_episode()

        if self._visualize:
            print("Rendering target episode...")
            self.states, self.target_states = self.target_states, self.states
            self.render_episode(
                iteration=-1, loop=True, loops_count=1, playback_speed=1.0, start_paused=True
            )
            self.states, self.target_states = self.target_states, self.states

        # Initialize optimizer with perturbed spline
        self.spline_params = self._init_spline.copy().astype(np.float64)
        self.spline_adam = SplineAdam(
            K=self.K,
            num_dofs=NUM_WHEEL_DOFS,
            lr=self._lr,
            lr_min_ratio=0.1,
            total_steps=iterations,
        )
        self._apply_params(self.spline_params)

        # Optimization
        print(
            f"Terrain traversal: T={T}, dt={self.clock.dt:.4f}, "
            f"K={self.K}, duration={T * self.clock.dt:.1f}s, "
            f"seed={self._terrain_seed}"
        )

        if self._check_grad:
            self.finite_diff_check()

        results = {
            "simulator": "Axion",
            "problem": "terrain_traversal",
            "seed": self._terrain_seed,
            "target_spline": self._target_spline.tolist(),
            "init_spline": self._init_spline.tolist(),
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "duration": T * self.clock.dt,
            "roughness": self._roughness,
            "terrain_freq": self._terrain_freq,
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "time_ms": [],
            "trajectories": {},
            "best_iters": [],
        }

        target_pos = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]
        results["target_trajectory"] = {
            "x": target_pos[:, 0].tolist(),
            "y": target_pos[:, 1].tolist(),
        }

        best_rmse = float("inf")

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            poses = self.trajectory.body_pose.numpy()[:, 0, 0, :3]
            targets = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]
            rmse_m = float(np.sqrt(np.mean(np.sum((poses - targets) ** 2, axis=-1))))

            if rmse_m < best_rmse:
                best_rmse = rmse_m
                results["best_iters"].append(i)

            print(
                f"  Iter {i:3d}: loss={curr_loss:.4f} | "
                f"RMSE={rmse_m:.3f}m | "
                f"best={best_rmse:.3f}m | "
                f"t={t_iter * 1000:.0f}ms"
            )

            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["rmse_m"].append(rmse_m)
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


def run_single(args, seed):
    """Run optimization for a single terrain seed. Returns results dict."""
    visualize = getattr(args, "visualize", False)

    target_spline, init_spline = generate_splines(
        seed,
        K=args.K,
        sigma=args.sigma,
        wildness=args.curvature,
    )
    print(
        f"  Target spline[0]: L={target_spline[0,0]:.2f} R={target_spline[0,1]:.2f} "
        f"Rear={target_spline[0,2]:.2f}"
    )
    print(
        f"  Init   spline[0]: L={init_spline[0,0]:.2f} R={init_spline[0,1]:.2f} "
        f"Rear={init_spline[0,2]:.2f}"
    )

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
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = TerrainTraversalOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=args.K,
        target_spline=target_spline,
        init_spline=init_spline,
        terrain_seed=seed,
        roughness=args.roughness,
        terrain_freq=args.terrain_freq,
        lr=args.lr,
        check_grad=getattr(args, "check_grad", False),
        visualize=visualize,
    )
    return sim.train(iterations=args.iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10.0)",
    )
    parser.add_argument("--dt", type=float, default=DT, help=f"Timestep (default: {DT})")
    parser.add_argument(
        "--K", type=int, default=10, help="Number of spline control points (default: 10)"
    )
    parser.add_argument(
        "--iterations", type=int, default=200, help="Optimisation iterations (default: 200)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Terrain/control seed (default: 0)")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Run over N random seeds (0..N-1) and save batch results",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Perturbation std dev for initial guess (rad/s, default: 0.5)",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.8,
        help="Max left/right wheel differential — controls how curvy the target is "
        "(0=straight, 0.8=moderate, 2.0=aggressive, default: 0.8)",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=1.0,
        help="Terrain amplitude multiplier (0.5=gentle, 1.0=default, 2.0=rugged)",
    )
    parser.add_argument(
        "--terrain-freq",
        type=float,
        default=1.0,
        help="Terrain frequency multiplier (0.5=broad hills, 1.0=default, 2.0=tight ripples)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Adam learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--check-grad",
        action="store_true",
        help="Run finite-difference gradient check before optimization",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable OpenGL visualization of optimization progress",
    )
    args = parser.parse_args()

    if args.num_seeds is not None:
        all_results = []
        for seed in range(args.num_seeds):
            print(f"\n{'='*60}")
            print(f"  SEED {seed}/{args.num_seeds - 1}")
            print(f"{'='*60}")
            result = run_single(args, seed=seed)
            all_results.append(result)

        final_rmses = [r["rmse_m"][-1] for r in all_results]
        best_rmses = [min(r["rmse_m"]) for r in all_results]
        median_times = [float(np.median(r["time_ms"][1:])) for r in all_results]

        summary = {
            "num_seeds": args.num_seeds,
            "sigma": args.sigma,
            "dt": args.dt,
            "duration": args.duration,
            "K": args.K,
            "iterations": args.iterations,
            "final_rmse_median": float(np.median(final_rmses)),
            "final_rmse_mean": float(np.mean(final_rmses)),
            "final_rmse_std": float(np.std(final_rmses)),
            "final_rmse_min": float(np.min(final_rmses)),
            "final_rmse_max": float(np.max(final_rmses)),
            "best_rmse_median": float(np.median(best_rmses)),
            "best_rmse_mean": float(np.mean(best_rmses)),
            "best_rmse_std": float(np.std(best_rmses)),
            "median_iter_time_ms": float(np.median(median_times)),
            "per_seed": all_results,
        }

        print(f"\n{'='*60}")
        print(f"  BATCH SUMMARY ({args.num_seeds} seeds, sigma={args.sigma})")
        print(f"{'='*60}")
        print(
            f"  Final RMSE: {summary['final_rmse_median']:.3f}m median, "
            f"{summary['final_rmse_mean']:.3f} +/- {summary['final_rmse_std']:.3f}m"
        )
        print(
            f"  Best  RMSE: {summary['best_rmse_median']:.3f}m median, "
            f"{summary['best_rmse_mean']:.3f} +/- {summary['best_rmse_std']:.3f}m"
        )
        print(f"  Iter  time: {summary['median_iter_time_ms']:.0f}ms median")

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
