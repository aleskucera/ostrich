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
from axion.optim.trajectory_adam import TrajectoryAdam
from newton import Model
from omegaconf import DictConfig

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("conf")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


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


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance between chassis position at the final timestep only."""
    tid = wp.tid()
    if tid > 0:
        return

    last = body_pose.shape[0] - 1
    pos = wp.transform_get_translation(body_pose[last, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[last, 0, 0])

    delta = pos - target_pos
    loss[0] = weight * wp.dot(delta, delta)


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


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 finite-difference penalty: weight * Σ_t ||v_{t+1} - v_t||² over wheel DOFs."""
    sim_step, wheel_idx = wp.tid()

    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, 0, dof_idx] - target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


@wp.kernel
def update_wheel_vel_kernel(
    target_vel_grad: wp.array(dtype=wp.float32, ndim=3),
    alpha: float,
    wheel_dof_offset: int,
    target_vel: wp.array(dtype=wp.float32, ndim=3),
):
    """Gradient descent step on wheel DOFs only (offset into the full DOF array)."""
    sim_step, world_idx, wheel_idx = wp.tid()

    dof_idx = wheel_dof_offset + wheel_idx
    g = target_vel_grad[sim_step, world_idx, dof_idx]

    max_grad = 100.0
    grad_clamped = wp.clamp(g, -max_grad, max_grad)

    wp.atomic_add(target_vel, sim_step, world_idx, dof_idx, -alpha * grad_clamped)


class HelhestEndpointOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 1.0
        self.endpoint_weight = 10.0
        self.smoothness_weight = 1e-2
        self.regularization_weight = 1e-7

        self.frame = 0

        # Straight-line drive initial guess (Left, Right, Rear)
        self.init_wheel_vel = (2.0, 5.0, 0.0)

        # Track chassis (body 0)
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

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.endpoint_weight,
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
        wp.launch(
            kernel=smoothness_kernel,
            dim=(self.clock.total_sim_steps - 1, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.smoothness_weight,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # Apply Adam step directly
        self.optimizer.step(self.trajectory.joint_target_vel.grad)

        self.trajectory.joint_target_vel.grad.zero_()

        # Sync updated trajectory values back into per-step controls
        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 10 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 50.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            # Draw target final chassis position as a reference box
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (
                    HelhestConfig.CHASSIS_SIZE[0] / 8.0,
                    HelhestConfig.CHASSIS_SIZE[1] / 8.0,
                    HelhestConfig.CHASSIS_SIZE[2] / 8.0,
                ),
                wp.array(
                    [self.trajectory.target_body_pose.numpy()[-1, 0, 0]],
                    dtype=wp.transform,
                ),
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

    def _make_wheel_vel_array(self, vels: tuple) -> np.ndarray:
        """Build a full joint_target_vel numpy array with `vels` at the 3 wheel DOFs."""
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        arr = np.zeros((self.clock.total_sim_steps, 1, num_dofs), dtype=np.float32)
        arr[:, :, WHEEL_DOF_OFFSET + 0] = vels[0]  # left
        arr[:, :, WHEEL_DOF_OFFSET + 1] = vels[1]  # right
        arr[:, :, WHEEL_DOF_OFFSET + 2] = vels[2]  # rear
        return arr

    def train(self, iterations=60):
        # Evaluate forward kinematics to place bodies at their rest pose
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        # --- Target episode ---
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        # Create a time-varying, swerving target control
        for i in range(self.clock.total_sim_steps):
            target_ctrl = np.zeros(num_dofs, dtype=np.float32)
            target_ctrl[WHEEL_DOF_OFFSET + 0] = 1.0
            target_ctrl[WHEEL_DOF_OFFSET + 1] = 6.0
            target_ctrl[WHEEL_DOF_OFFSET + 2] = 0.0

            target_ctrl_wp = wp.array(target_ctrl, dtype=wp.float32, device=self.model.device)
            wp.copy(self.target_controls[i].joint_target_vel, target_ctrl_wp)

        self.run_target_episode()

        # Render target trajectory before optimization starts
        print("Rendering target episode...")
        self.states, self.target_states = self.target_states, self.states
        self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=1.0)
        self.states, self.target_states = self.target_states, self.states

        # --- Optimization ---
        # Initialize trajectory buffer and per-step controls with the initial guess
        init_arr = self._make_wheel_vel_array(self.init_wheel_vel)
        wp.copy(self.trajectory.joint_target_vel, wp.array(init_arr, dtype=wp.float32))

        init_ctrl = np.zeros(num_dofs, dtype=np.float32)
        init_ctrl[WHEEL_DOF_OFFSET + 0] = self.init_wheel_vel[0]
        init_ctrl[WHEEL_DOF_OFFSET + 1] = self.init_wheel_vel[1]
        init_ctrl[WHEEL_DOF_OFFSET + 2] = self.init_wheel_vel[2]
        init_ctrl_wp = wp.array(init_ctrl, dtype=wp.float32, device=self.model.device)

        self.optimizer = TrajectoryAdam(
            params=self.trajectory.joint_target_vel,
            lr=1.0,
            betas=(0.9, 0.999),
            dof_offset=WHEEL_DOF_OFFSET,
            num_dofs=NUM_WHEEL_DOFS,
        )

        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, init_ctrl_wp)

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            current_vel_l = self.trajectory.joint_target_vel.numpy()[0, 0, WHEEL_DOF_OFFSET]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | init left_vel={current_vel_l:.3f} | episode={t_episode:.3f}s"
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

    sim = HelhestEndpointOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
