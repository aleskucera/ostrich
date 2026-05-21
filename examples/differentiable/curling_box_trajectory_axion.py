import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
import warp.optim
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
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
    tid = wp.tid()

    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])

    delta = pos - target_pos
    l2_loss = wp.dot(delta, delta)
    wp.atomic_add(loss, 0, l2_loss)


@wp.kernel
def update_kernel(
    initial_qd_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    initial_qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return

    # Only update the Y (sliding) velocity component
    qd_y = initial_qd[0][1]
    qd_y_grad = initial_qd_grad[0, 0][1]

    max_grad = 5.0
    qd_y_grad = wp.clamp(qd_y_grad, -max_grad, max_grad)

    wp.printf("Gradient: %f\n", qd_y_grad)

    qd_y_new = qd_y - alpha * qd_y_grad
    initial_qd[0] = wp.spatial_vector(0.0, qd_y_new, 0.0, 0.0, 0.0, 0.0)


class CurlingBoxTrajectoryOptimizer(AxionDifferentiableSimulator):
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
        self.learning_rate = 0.3

        self.frame = 0

        # Initial velocity guess (Y = sliding direction)
        self.init_vel = wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        # Target velocity to recover
        self.target_init_vel = wp.spatial_vector(0.0, 2.5, 0.0, 0.0, 0.0, 0.0)

        self.track_body(body_idx=0, name="box", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.5
        shape_config = newton.ModelBuilder.ShapeConfig(
            ke=1e5, kd=1e2, kf=1e3, mu=0.15, density=100.0
        )

        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.21), wp.quat_identity()),
        )
        self.builder.add_shape_box(body=0, hx=0.2, hy=0.2, hz=0.2, cfg=shape_config)

        self.builder.add_ground_plane(cfg=shape_config)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
            ],
            outputs=[
                self.loss,
            ],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_vel.grad[0],
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_qd,
            ],
        )

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 10.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.18, 0.18, 0.18),
                wp.array(
                    [self.trajectory.target_body_pose.numpy()[-1, 0, 0]],
                    dtype=wp.transform,
                ),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
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

    def train(self, iterations=50):
        # Run target episode to populate target trajectory
        wp.copy(
            self.target_states[0].body_qd,
            wp.array([self.target_init_vel], dtype=wp.spatial_vector),
        )
        self.run_target_episode()

        # Set initial velocity guess
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            vel = self.states[0].body_qd.numpy()[0][0:3]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | Init Vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"
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

    sim = CurlingBoxTrajectoryOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=50)


if __name__ == "__main__":
    main()
