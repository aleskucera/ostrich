"""Terrain traversal benchmark — Axion only.

Optimises K=10 spline control knots for the Helhest robot to follow a target
trajectory over a triangle mesh surface. This task is inaccessible to all
evaluated baselines because none support differentiable mesh collision.

Reports per-iteration time (forward + backward) and loss curve.

Usage:
    python examples/comparison_gradient/helhest/terrain_traversal.py \
        --save results/terrain_traversal.json
    python examples/comparison_gradient/helhest/terrain_traversal.py \
        --duration 10.0 --save results/terrain_traversal_10s.json
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import openmesh
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
ASSETS_DIR = pathlib.Path(__file__).parent.parent.parent.joinpath("assets")

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

DT = 8e-2


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
        self, K, num_dofs, lr, total_steps=200, lr_min_ratio=0.05, betas=(0.9, 0.999), eps=1e-8
    ):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params, grad):
        self.t += 1
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
        engine_config,
        logging_config,
        num_control_points=10,
        save_path=None,
        target_ctrl=(2.8, 1.2, 2.0),
        init_ctrl=(2.0, 2.0, 1.0),
    ):
        self.save_path = save_path
        self.K = num_control_points
        self._target_ctrl = target_ctrl
        self._init_ctrl = init_ctrl
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7
        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.2
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0,
            k_d=0.0,
            friction_left_right=0.8,
            friction_rear=0.35,
        )

        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
        scale = np.array([6.0, 6.0, 5.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])
        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

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
        safe_sums = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe_sums[:, None]

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

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

    def train(self, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        # Target episode
        for i in range(T):
            ctrl = np.zeros(num_dofs, dtype=np.float32)
            ctrl[WHEEL_DOF_OFFSET + 0] = self._target_ctrl[0]
            ctrl[WHEEL_DOF_OFFSET + 1] = self._target_ctrl[1]
            ctrl[WHEEL_DOF_OFFSET + 2] = self._target_ctrl[2]
            wp.copy(
                self.target_controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )
        self.run_target_episode()

        # Spline setup
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array([list(self._init_ctrl)] * self.K, dtype=np.float64)
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.1, lr_min_ratio=0.2, total_steps=iterations
        )
        self._apply_params(self.spline_params)

        # Optimization
        print(
            f"Terrain traversal: T={T}, dt={self.clock.dt:.4f}, "
            f"K={self.K}, duration={T * self.clock.dt:.1f}s"
        )
        # Iterations at which to snapshot the full xy trajectory
        snapshot_iters = {0, 10, 30, 50, 100, iterations - 1}

        results = {
            "simulator": "Axion",
            "problem": "terrain_traversal",
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "duration": T * self.clock.dt,
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "time_ms": [],
            "trajectories": {},  # iter -> {"x": [...], "y": [...]}
        }

        # Save target trajectory (computed once from target episode)
        target_pos = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]  # [T, 3]
        results["target_trajectory"] = {
            "x": target_pos[:, 0].tolist(),
            "y": target_pos[:, 1].tolist(),
        }

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            # RMSE in meters: sqrt( (1/T) * Σ_t ||p_t - p*_t||² )
            # wp.transform numpy layout: [px, py, pz, qx, qy, qz, qw]
            poses = self.trajectory.body_pose.numpy()[:, 0, 0, :3]  # [T, 3]
            targets = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]  # [T, 3]
            rmse_m = float(np.sqrt(np.mean(np.sum((poses - targets) ** 2, axis=-1))))

            print(
                f"Iter {i:3d}: loss={curr_loss:.4f} | "
                f"RMSE={rmse_m:.3f}m | "
                f"t={t_iter * 1000:.0f}ms"
            )

            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["rmse_m"].append(rmse_m)
            results["time_ms"].append(t_iter * 1000)

            if i in snapshot_iters:
                results["trajectories"][str(i)] = {
                    "x": poses[:, 0].tolist(),
                    "y": poses[:, 1].tolist(),
                }

            self.update()
            self.tape.zero()
            self.loss.zero_()

        if self.save_path:
            pathlib.Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.save_path).write_text(json.dumps(results, indent=2))
            print(f"Saved to {self.save_path}")


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
    args = parser.parse_args()

    sim_config = SimulationConfig(
        duration_seconds=args.duration,
        target_timestep_seconds=args.dt,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    engine_config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=14, backtrack_min_iter=10, atol=0.001),
        linear=LinearSolverConfig(max_iters=16, atol=0.001, tol=0.001, regularization=1e-06),
        compliance=ComplianceConfig(joint=6e-08, contact=0.1, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    logging_config = LoggingConfig()

    sim = TerrainTraversalOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=args.K,
        save_path=args.save,
    )
    sim.train(iterations=args.iterations)


if __name__ == "__main__":
    main()
