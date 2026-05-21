"""Helhest trajectory optimization using Axion (Newton/Warp, implicit differentiation).

Comparable to examples/comparison/helhest_mjx.py.

Optimizes K spline control points linearly interpolated to per-timestep wheel
velocities. Gradients flow through Axion's custom implicit backward pass.
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
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.sim_config import SyncMode
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig
from newton import Model

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 5e-2
DURATION = 3.0
K = 30  # number of spline control points

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

TARGET_CTRL = (1.0, 6.0, 0.0)
INIT_CTRL = (2.0, 5.0, 0.0)


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (T, K) linear interpolation matrix and per-column normalization."""
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
    """Adam optimizer for a (K, num_dofs) numpy parameter array."""

    def __init__(
        self, K: int, num_dofs: int, lr: float, betas=(0.9, 0.999), eps=1e-8, clip_grad=100.0
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.clip_grad = clip_grad
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        grad = np.clip(grad, -self.clip_grad, self.clip_grad)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance between chassis position summed over all timesteps."""
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 magnitude regularization: weight * Σ_t ||v_t||² over wheel DOFs."""
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


class HelhestTrajectorySplineOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = K,
        save_path: str = None,
    ):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.save_path = save_path
        self.K = num_control_points
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.regularization_weight = 1e-7
        self.frame = 0

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
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

    def _expand(self, params: np.ndarray) -> np.ndarray:
        return self.W @ params  # (T, K) @ (K, 3) = (T, 3)

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        return (self.W.T @ grad_v) / self.W_col_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]
        wp.launch(
            kernel=loss_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.trajectory_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight,
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

    def _extract_target_trajectory(self):
        """Extract (T+1, 2) xy target trajectory from trajectory buffer."""
        num_steps = self.trajectory.target_body_pose.shape[0]
        traj = []
        for t in range(num_steps):
            body_pose = self.trajectory.target_body_pose[t].numpy()[0, 0]
            traj.append([float(body_pose[0]), float(body_pose[1])])
        return traj

    def train(self, iterations=200, target_only=False):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        # --- Target episode ---
        for i in range(T):
            ctrl = np.zeros(num_dofs, dtype=np.float32)
            ctrl[WHEEL_DOF_OFFSET + 0] = TARGET_CTRL[0]
            ctrl[WHEEL_DOF_OFFSET + 1] = TARGET_CTRL[1]
            ctrl[WHEEL_DOF_OFFSET + 2] = TARGET_CTRL[2]
            wp.copy(
                self.target_controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )

        self.run_target_episode()

        if target_only:
            traj = self._extract_target_trajectory()
            print(f"Target final xy: ({traj[-1][0]:.3f}, {traj[-1][1]:.3f})")
            traj_result = {
                "simulator": "Axion",
                "problem": "helhest",
                "dt": self.clock.dt,
                "T": T,
                "target_trajectory": traj,
            }
            if self.save_path:
                pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(self.save_path).write_text(json.dumps(traj_result, indent=2))
                print(f"Saved to {self.save_path}")
            return

        if not self.save_path:
            print("Rendering target episode...")
            self.states, self.target_states = self.target_states, self.states
            self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=1.0)
            self.states, self.target_states = self.target_states, self.states

        # --- Spline setup ---
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array(
            [[INIT_CTRL[0], INIT_CTRL[1], INIT_CTRL[2]]] * self.K, dtype=np.float64
        )
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.15, betas=(0.5, 0.999)
        )
        self._apply_params(self.spline_params)

        # --- Optimization ---
        print(f"\nOptimizing: T={T}, dt={self.clock.dt:.4f}, K={self.K}, lr=0.3 (SplineAdam)")
        results = {
            "simulator": "Axion",
            "problem": "helhest",
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "iterations": [],
            "loss": [],
            "time_ms": [],
        }
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            p0, pm, pN = (
                self.spline_params[0],
                self.spline_params[self.K // 2],
                self.spline_params[-1],
            )
            print(
                f"Iter {i:3d}: loss={curr_loss:.4f} | "
                f"cp[0]=({p0[0]:.2f},{p0[1]:.2f}) "
                f"cp[{self.K//2}]=({pm[0]:.2f},{pm[1]:.2f}) "
                f"cp[-1]=({pN[0]:.2f},{pN[1]:.2f}) | "
                f"t={t_iter * 1000:.0f}ms"
            )
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_iter * 1000)

            self.update()
            self.tape.zero()
            self.loss.zero_()

        if self.save_path:
            pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path).write_text(json.dumps(results, indent=2))
            print(f"Saved to {self.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON and run headless")
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Only compute and save the target trajectory, skip optimization",
    )
    parser.add_argument("--dt", type=float, default=DT, help=f"Timestep override (default: {DT})")
    args = parser.parse_args()

    # Default save path for --target-only without explicit --save
    if args.target_only and not args.save:
        args.save = "/tmp/helhest_axion.json"

    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=args.dt,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null" if (args.save or args.target_only) else "gl",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    engine_config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=12, backtrack_min_iter=8, atol=0.1),
        linear=LinearSolverConfig(max_iters=12, atol=0.001, tol=0.001, regularization=1e-06),
        compliance=ComplianceConfig(joint=6e-08, contact=1e-06, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False, conservative_step_count=16, conservative_upper_bound=0.05, min_step=1e-06, optimistic_step_count=48, optimistic_window=0.4),
        contacts=ContactsConfig(max_per_world=8),
    )
    logging_config = LoggingConfig()

    sim = HelhestTrajectorySplineOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=K,
        save_path=args.save,
    )
    sim.train(iterations=50, target_only=args.target_only)


if __name__ == "__main__":
    main()
