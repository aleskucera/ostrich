"""Helhest trajectory optimization — Axion, 100 parallel worlds.

All worlds share the same control parameters (K spline points).
Loss and gradients are averaged over worlds so the optimization
is scale-independent of NUM_WORLDS.

Usage:
    python examples/comparison/helhest_batch/axion_sim.py --save examples/comparison/helhest_batch/results/axion.json
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
from config import DURATION
from config import INIT_CTRL
from config import K
from config import NUM_WHEEL_DOFS
from config import NUM_WORLDS
from config import TARGET_CTRL
from config import WHEEL_DOF_OFFSET
from newton import Model

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

DT = 7e-2


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, K, num_dofs, lr, betas=(0.9, 0.999), eps=1e-8, clip_grad=100.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.clip_grad = clip_grad
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def step(self, params, grad):
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
    """L2 loss averaged over timesteps and worlds."""
    t, w = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, w, 0])
    target_pos = wp.transform_get_translation(target_body_pose[t, w, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 regularization averaged over timesteps and worlds."""
    sim_step, world_idx, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, world_idx, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


class HelhestBatchOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=K,
        save_path=None,
    ):
        self.save_path = save_path
        self.K = num_control_points
        self.num_worlds = sim_config.num_worlds
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.regularization_weight = 1e-7
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
            num_worlds=self.num_worlds,
            requires_grad=True,
        )

    def _expand(self, params):
        return self.W @ params  # (T, K) @ (K, 3) = (T, 3)

    def _contract(self, grad_v):
        return (self.W.T @ grad_v) / self.W_col_sums[:, None]

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)  # (T, 3)
        vel_np = np.zeros((T, self.num_worlds, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[:, None, :]
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]
        # Weight normalizes over both timesteps and worlds
        wp.launch(
            kernel=loss_kernel,
            dim=(num_steps, self.num_worlds),
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.trajectory_weight / (num_steps * self.num_worlds),
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, self.num_worlds, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / self.num_worlds,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # Average gradient over worlds before contracting
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ].mean(
            axis=1
        )  # (T, 3)
        grad_params = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def train(self, iterations=50):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        # Set target controls for all worlds
        ctrl = np.zeros((self.num_worlds, num_dofs), dtype=np.float32)
        ctrl[:, WHEEL_DOF_OFFSET + 0] = TARGET_CTRL[0]
        ctrl[:, WHEEL_DOF_OFFSET + 1] = TARGET_CTRL[1]
        ctrl[:, WHEEL_DOF_OFFSET + 2] = TARGET_CTRL[2]
        for i in range(T):
            wp.copy(
                self.target_controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )

        self.run_target_episode()

        # Spline setup
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array(
            [[INIT_CTRL[0], INIT_CTRL[1], INIT_CTRL[2]]] * self.K, dtype=np.float64
        )
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.15, betas=(0.5, 0.999)
        )
        self._apply_params(self.spline_params)

        print(
            f"\nOptimizing: T={T}, dt={self.clock.dt:.4f}, K={self.K}, num_worlds={self.num_worlds}"
        )
        peak_mem_mb = 0.0
        results = {
            "simulator": "Axion",
            "problem": "helhest_batch",
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "num_worlds": self.num_worlds,
            "iterations": [],
            "loss": [],
            "time_ms": [],
            "peak_gpu_mb": None,
        }
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            used_mb = wp.get_mempool_used_bytes() / 1024**2
            peak_mem_mb = max(peak_mem_mb, used_mb)

            curr_loss = self.loss.numpy()[0]
            print(f"Iter {i:3d}: loss={curr_loss:.4f} | t={t_iter * 1000:.0f}ms | mem={used_mb:.0f}MB")
            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["time_ms"].append(t_iter * 1000)

        results["peak_gpu_mb"] = peak_mem_mb

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
    args = parser.parse_args()

    sim_config = SimulationConfig(
        duration_seconds=DURATION,
        target_timestep_seconds=DT,
        num_worlds=NUM_WORLDS,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    engine_config = AxionEngineConfig(
        max_newton_iters=12,
        max_linear_iters=12,
        backtrack_min_iter=8,
        newton_atol=1e-3,
        linear_atol=1e-3,
        linear_tol=1e-3,
        enable_linesearch=False,
        joint_compliance=6e-8,
        contact_compliance=1e-6,
        friction_compliance=1e-6,
        regularization=1e-6,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig()

    sim = HelhestBatchOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=K,
        save_path=args.save,
    )
    sim.train(iterations=50)


if __name__ == "__main__":
    main()
