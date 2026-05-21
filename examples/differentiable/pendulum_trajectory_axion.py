"""
Pendulum target velocity optimization using implicit differentiation.

Goal: find the constant target joint velocity of a single pendulum that reproduces
a target trajectory. The optimizer starts from a different target velocity
and uses the Axion implicit gradient to recover the true target velocity.

Scene: single pendulum, pivot fixed at (0, 0, 5), link swings in the XZ plane
       (revolute joint around world Y axis).
Optimization parameter: target angular velocity around Y (scalar, constant over time).
Loss: L2 trajectory matching across all timesteps.
"""
import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.types import JointMode  # Need this for control modes
from newton import Model
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


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
    loss: wp.array(dtype=wp.float32),
):
    """L2 trajectory loss: sum of squared position errors across all timesteps."""
    tid = wp.tid()

    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])

    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


@wp.kernel
def update_control_kernel(
    target_vel_grad: wp.array(dtype=wp.float32, ndim=3),
    alpha: float,
    total_steps: int,
    target_vel: wp.array(dtype=wp.float32, ndim=3),
):
    sim_step, world_idx, dof_idx = wp.tid()

    g = target_vel_grad[sim_step, world_idx, dof_idx]
    max_grad = 100.0
    grad_clamped = wp.clamp(g, -max_grad, max_grad)

    # if world_idx == 0 and dof_idx == 0:
    #     wp.printf("Target Vel Grad: %f\n", g)

    wp.atomic_add(target_vel, sim_step, world_idx, dof_idx, -alpha * grad_clamped)


class PendulumControlOptimizer(AxionDifferentiableSimulator):
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
        # Learning rate might need tuning depending on how long your trajectory is
        self.learning_rate = 1.0

        self.frame = 0

        # Optimization targets
        self.init_target_vel = 1.0  # Starting guess
        self.true_target_vel = 3.0  # The trajectory we want to mimic

        self.track_body(body_idx=0, name="link", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        hx = 1.0
        hy = 0.1
        hz = 0.1

        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=1000.0,
            target_kd=1000.0,
            label="pendulum_joint",
            custom_attributes={
                "joint_dof_mode": [JointMode.TARGET_VELOCITY],
            },
        )

        self.builder.add_articulation([j0], label="pendulum")

        model = self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

        return model

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_control_kernel,
            dim=(self.clock.total_sim_steps, self.simulation_config.num_worlds, 1),
            inputs=[
                self.trajectory.joint_target_vel.grad,
                self.learning_rate,
                self.clock.total_sim_steps,
                self.trajectory.joint_target_vel,
            ],
            device=self.solver.model.device,
        )
        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 3 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 20.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,
        )

        self.frame += 1

    def train(self, iterations=60):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        self.trajectory.joint_target_vel.fill_(self.true_target_vel)

        for i in range(self.clock.total_sim_steps):
            self.target_controls[i].joint_target_vel.fill_(self.true_target_vel)
            self.controls[i].joint_target_vel.fill_(self.init_target_vel)

        self.run_target_episode()

        # 2. Reset back to the initial guess for optimization
        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            # Peek at the target velocity assigned for the first timestep
            current_vel = self.trajectory.joint_target_vel.numpy()[0][0][0]
            print(
                f"Iter {i}: Loss={curr_loss:.6f} | target_vel={current_vel:.4f} (true_target={self.true_target_vel})"
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

    sim = PendulumControlOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=60)


if __name__ == "__main__":
    main()
