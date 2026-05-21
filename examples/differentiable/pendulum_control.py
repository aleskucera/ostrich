"""
Pendulum swing-up via spline trajectory optimization.

Goal: optimize a spline of target velocities so the pendulum swings
up to the top position and balances there.
"""
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
from axion.core.types import JointMode
from newton import Model
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

HX = 1.0
HY = 0.1
HZ = 0.1


def bourke_color_map(v_min, v_max, v):
    c = wp.vec3(1.0, 1.0, 1.0)
    v = np.clip(v, v_min, v_max)
    dv = v_max - v_min

    if v < (v_min + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - v_min) / dv
    elif v < (v_min + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (v_min + 0.25 * dv - v) / dv
    elif v < (v_min + 0.75 * dv):
        c[0] = 4.0 * (v - v_min - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (v_min + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c


# --- SPLINE OPTIMIZER UTILITIES ---


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


# --- LOSS KERNELS ---


@wp.kernel
def swing_up_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize distance from the CoM to the target position."""
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize high control velocities (encourages stopping at the top)."""
    t = wp.tid()
    v = target_vel[t, 0, 0]
    wp.atomic_add(loss, 0, weight * v * v)


class PendulumSwingUp(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 15,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Spline parameters
        self.K = num_control_points
        self.num_dofs = 1

        # Loss weights
        self.position_weight = 10.0
        self.regularization_weight = 1e-4

        self.frame = 0
        self.track_body(body_idx=0, name="pendulum", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=HX, hy=HY, hz=HZ)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-HX, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=500.0,
            target_kd=100.0,
            label="pendulum_joint",
            custom_attributes={
                "joint_dof_mode": [JointMode.TARGET_VELOCITY],
            },
        )

        self.builder.add_articulation([j0], label="pendulum")

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    # --- SPLINE MAPPING ---

    def _expand(self, params: np.ndarray) -> np.ndarray:
        return self.W @ params  # [T, K] @ [K, 1] = [T, 1]

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        return (self.W.T @ grad_v) / self.W_col_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        T = self.clock.total_sim_steps
        expanded = self._expand(params)  # [T, 1]

        vel_np = np.zeros((T, 1, self.num_dofs), dtype=np.float32)
        vel_np[:, 0, :] = expanded

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.clock.total_sim_steps

        # The pivot is at Z=5. The CoM is 1.0 units away from the pivot.
        # So straight UP means the CoM is at (0, 0, 6.0).
        target_pos = wp.vec3(0.0, 0.0, 6.0)

        wp.launch(
            kernel=swing_up_loss_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.body_pose,
                target_pos,
                self.position_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

        wp.launch(
            kernel=regularization_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.joint_target_vel,
                self.regularization_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # 1. Get raw gradients from the trajectory buffer
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[:, 0, :]  # [T, 1]

        # 2. Map gradients down to the K spline control points
        grad_params = self._contract(grad_v)  # [K, 1]
        self.trajectory.joint_target_vel.grad.zero_()

        # 3. Adam step on the control points
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)

        # 4. Map the updated control points back to the full simulation trajectory
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 20.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            # Draw a visual marker for the goal position
            viewer.log_shapes(
                "/goal",
                newton.GeoType.SPHERE,
                (0.2, 0.2, 0.2),  # sphere radius
                wp.array(
                    [wp.transform(wp.vec3(0.0, 0.0, 6.0), wp.quat_identity())], dtype=wp.transform
                ),
                wp.array([wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3),
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,
        )

        self.frame += 1

    def train(self, iterations=300):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps

        # Initialize Spline Matrices
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        # Initialize parameters slightly off-zero to give gradients a nudge
        self.spline_params = np.ones((self.K, self.num_dofs), dtype=np.float64) * 0.1
        self.spline_adam = SplineAdam(K=self.K, num_dofs=self.num_dofs, lr=0.2)

        self._apply_params(self.spline_params)

        # Optimization loop
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            p_start = self.spline_params[0, 0]
            p_mid = self.spline_params[self.K // 2, 0]
            p_end = self.spline_params[-1, 0]

            print(
                f"Iter {i:3d}: Loss={curr_loss:.4f} | Spline(start={p_start:.2f}, mid={p_mid:.2f}, end={p_end:.2f}) | eps={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = PendulumSwingUp(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=30,  # Using 15 control points for a smooth swing
    )
    sim.train(iterations=300)


if __name__ == "__main__":
    main()
