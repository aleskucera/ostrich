import os
import pathlib
import time

import hydra
import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from newton import Model
from omegaconf import DictConfig

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("conf")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, K] linear interpolation weight matrix and per-column normalization.

    W[t, k] is the weight of control point k at timestep t.
    Expanding:   v = W @ params                          ([T,3] = [T,K] @ [K,3])
    Contracting: grad_params = (W.T @ grad_v) / col_sums  (average, not sum)
    """
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    col_sums = W.sum(axis=0)  # [K] — total weight each control point receives
    return W, col_sums


class SplineAdam:
    """Adam optimizer for a [K, num_dofs] numpy parameter array."""

    def __init__(self, K: int, num_dofs: int, lr: float, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
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
def yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Yaw heading error via forward vector dot product, summed over all timesteps."""
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
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 10,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.K = num_control_points

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 1.0
        self.regularization_weight = 1e-6

        self.frame = 0

        # Initial guess (Left, Right, Rear)
        self.init_wheel_vel = (3.0, 3.0, 0.0)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.7,
            ke=50.0,
            kd=50.0,
            kf=50.0,
        )
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
        """Expand [K, 3] control points → [T, 3] per-step wheel velocities."""
        return self.W @ params  # [T, K] @ [K, 3] = [T, 3]

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        """Contract [T, 3] velocity gradients → [K, 3] control point gradients (normalized average)."""
        return (self.W.T @ grad_v) / self.W_col_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        """Write expanded spline params into joint_target_vel and per-step controls."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)  # [T, 3]

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
            kernel=yaw_loss_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.yaw_weight / num_steps,
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
        # Contract: [T, 3] gradients → [K, 3] control point gradients
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]  # [T, 3]
        grad_params = self._contract(grad_v)  # [K, 3]

        self.trajectory.joint_target_vel.grad.zero_()

        # Adam step on control points
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)

        # Expand and sync back
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 10 != 0:
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
            playback_speed=5.0,
        )

        self.frame += 1

    def train(self, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        # --- Target episode ---
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        T = self.clock.total_sim_steps

        for i in range(T):
            target_ctrl = np.zeros(num_dofs, dtype=np.float32)
            target_ctrl[WHEEL_DOF_OFFSET + 0] = 1.0
            target_ctrl[WHEEL_DOF_OFFSET + 1] = 6.0
            target_ctrl[WHEEL_DOF_OFFSET + 2] = 0.0

            target_ctrl_wp = wp.array(target_ctrl, dtype=wp.float32, device=self.model.device)
            wp.copy(self.target_controls[i].joint_target_vel, target_ctrl_wp)

        self.run_target_episode()

        print("Rendering target episode...")
        self.states, self.target_states = self.target_states, self.states
        self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=1.0)
        self.states, self.target_states = self.target_states, self.states

        # --- Spline setup ---
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)  # [T, K], [K]

        # Initialize all control points to the constant initial guess
        self.spline_params = np.array(
            [[self.init_wheel_vel[0], self.init_wheel_vel[1], self.init_wheel_vel[2]]] * self.K,
            dtype=np.float64,
        )  # [K, 3]

        self.spline_adam = SplineAdam(K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.2)

        self._apply_params(self.spline_params)

        # --- Optimization ---
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            p0 = self.spline_params[0]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | cp[0] L={p0[0]:.3f} R={p0[1]:.3f} | K={self.K} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="helhest_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestTrajectorySplineOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=30,
    )
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
