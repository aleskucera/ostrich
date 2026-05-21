import os
import pathlib
import time

import hydra
import newton
import numpy as np
import warp as wp
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.differentiable_simulator import NewtonDifferentiableSimulator
from newton import Model
from omegaconf import DictConfig

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("conf")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


class ControlAdam:
    """Adam optimizer for per-step controls stored in a [T, num_dofs] numpy array."""

    def __init__(self, T: int, num_dofs: int, lr: float, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros((T, num_dofs), dtype=np.float64)
        self.v = np.zeros((T, num_dofs), dtype=np.float64)
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
    body_q: wp.array(dtype=wp.transform),
    target_body_q: wp.array(dtype=wp.transform),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance between chassis position at the final timestep."""
    pos = wp.transform_get_translation(body_q[0])
    target_pos = wp.transform_get_translation(target_body_q[0])
    delta = pos - target_pos
    loss[0] = weight * wp.dot(delta, delta)


@wp.kernel
def smoothness_kernel(
    vel_a: wp.array(dtype=wp.float32),
    vel_b: wp.array(dtype=wp.float32),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 finite-difference smoothness penalty between two consecutive control steps."""
    wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    diff = vel_b[dof_idx] - vel_a[dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


class HelhestEndpointFeatherstoneOptimizer(NewtonDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        k_p: float = 0.0,
        k_d: float = 100.0,
    ):
        self.k_p = k_p
        self.k_d = k_d

        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.endpoint_weight = 10.0
        self.smoothness_weight = 1e-2

        self.frame = 0

        # Initial guess (Left, Right, Rear)
        self.init_wheel_vel = (2.0, 5.0, 0.0)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.0

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.7,
            ke=1e4,
            kd=1e2,
            kf=1e3,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
            control_mode="velocity",
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=0.7,
            friction_rear=0.35,
            ke=5e4,
            kd=1e3,
            kf=1e3,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def run_target_episode(self):
        self.episode_trajectory.save_state(self.target_states[0], 0)
        for i in range(self.clock.total_sim_steps):
            self.collision_pipeline.collide(self.target_states[i], self.contacts)
            self.solver.step(
                state_in=self.target_states[i],
                state_out=self.target_states[i + 1],
                control=self.target_controls[i],
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            self.episode_trajectory.save_state(self.target_states[i + 1], i + 1)

    def compute_loss(self):
        # Endpoint position loss against final target state
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.states[-1].body_q,
                self.episode_trajectory.body_q[-1],
                self.endpoint_weight,
            ],
            outputs=[self.loss],
            device=self.model.device,
        )
        # Smoothness across consecutive control steps
        for i in range(self.clock.total_sim_steps - 1):
            wp.launch(
                kernel=smoothness_kernel,
                dim=NUM_WHEEL_DOFS,
                inputs=[
                    self.controls[i].joint_target_vel,
                    self.controls[i + 1].joint_target_vel,
                    WHEEL_DOF_OFFSET,
                    self.smoothness_weight,
                ],
                outputs=[self.loss],
                device=self.model.device,
            )

    def update(self):
        T = self.clock.total_sim_steps
        num_dofs = self.model.joint_dof_count

        # Collect gradients from all per-step controls
        grads = np.zeros((T, num_dofs), dtype=np.float64)
        for i in range(T):
            grads[i] = self.controls[i].joint_target_vel.grad.numpy()

        # Adam step on wheel DOFs
        grad_wheels = grads[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        self.wheel_vel_params[:, :] = self.control_adam.step(self.wheel_vel_params, grad_wheels)

        # Zero gradients and write updated velocities back into controls
        for i in range(T):
            self.controls[i].joint_target_vel.grad.zero_()
            vel_np = np.zeros(num_dofs, dtype=np.float32)
            vel_np[WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = self.wheel_vel_params[i]
            wp.copy(self.controls[i].joint_target_vel, wp.array(vel_np, dtype=wp.float32))

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 10 != 0:
            return

        loss_val = self.loss.numpy()[0]

        target_q = self.episode_trajectory.body_q.numpy()  # [T+1, num_bodies]

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (
                    HelhestConfig.CHASSIS_SIZE[0] / 2.0,
                    HelhestConfig.CHASSIS_SIZE[1] / 2.0,
                    HelhestConfig.CHASSIS_SIZE[2] / 2.0,
                ),
                wp.array([target_q[-1, 0]], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.2, 0.0)], dtype=wp.vec3),
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
        num_dofs = self.model.joint_dof_count
        T = self.clock.total_sim_steps

        for i in range(T):
            target_ctrl = np.zeros(num_dofs, dtype=np.float32)
            target_ctrl[WHEEL_DOF_OFFSET + 0] = 1.0
            target_ctrl[WHEEL_DOF_OFFSET + 1] = 6.0
            target_ctrl[WHEEL_DOF_OFFSET + 2] = 0.0
            wp.copy(
                self.target_controls[i].joint_target_vel,
                wp.array(target_ctrl, dtype=wp.float32, device=self.model.device),
            )

        self.run_target_episode()

        print("Rendering target episode...")
        self.states, self.target_states = self.target_states, self.states
        self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=0.1)
        self.states, self.target_states = self.target_states, self.states

        # --- Optimization setup ---
        # wheel_vel_params: [T, 3] numpy array of (left, right, rear) velocities
        self.wheel_vel_params = np.zeros((T, NUM_WHEEL_DOFS), dtype=np.float64)
        self.wheel_vel_params[:, 0] = self.init_wheel_vel[0]
        self.wheel_vel_params[:, 1] = self.init_wheel_vel[1]
        self.wheel_vel_params[:, 2] = self.init_wheel_vel[2]

        self.control_adam = ControlAdam(T=T, num_dofs=NUM_WHEEL_DOFS, lr=1.0)

        # Initialize controls
        for i in range(T):
            vel_np = np.zeros(num_dofs, dtype=np.float32)
            vel_np[WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = self.wheel_vel_params[i]
            wp.copy(
                self.controls[i].joint_target_vel,
                wp.array(vel_np, dtype=wp.float32, device=self.model.device),
            )

        # --- Optimization loop ---
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            vel_l = self.wheel_vel_params[0, 0]
            vel_r = self.wheel_vel_params[0, 1]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | L={vel_l:.3f} R={vel_r:.3f} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="helhest_featherstone_diff"
)
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestEndpointFeatherstoneOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    )
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
