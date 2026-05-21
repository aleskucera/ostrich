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
    trajectory_body_q: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid > 0:
        return

    # Use the final state from the trajectory
    pos = wp.transform_get_translation(trajectory_body_q[trajectory_body_q.shape[0] - 1, 0, 0])
    delta = pos - target_pos
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def update_kernel(
    qd_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return

    max_grad = 20.0
    g = qd_grad[0, 0]
    g_clamped = wp.spatial_vector(
        wp.clamp(g[0], -max_grad, max_grad),
        wp.clamp(g[1], -max_grad, max_grad),
        wp.clamp(g[2], -max_grad, max_grad),
        wp.clamp(g[3], -max_grad, max_grad),
        wp.clamp(g[4], -max_grad, max_grad),
        wp.clamp(g[5], -max_grad, max_grad),
    )

    # gradient descent step
    qd[0] = qd[0] - g_clamped * alpha


class BallThrowOptimizerImplicit(AxionDifferentiableSimulator):
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

        # 2. Optimization Setup
        self.target_pos = wp.vec3(0.0, 5.0, 1.0)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 1e-1

        self.frame = 0

        # Initial velocity guessing (angular, linear)
        self.init_vel = wp.spatial_vector(0.0, 0.5, 3.0, 0.0, 0.0, 0.0)

        # 3. Setup Automatic Trajectory Tracking
        self.track_body(body_idx=0, name="ball", color=(0.0, 1.0, 0.0))

    def build_model(self) -> Model:
        shape_config = newton.ModelBuilder.ShapeConfig(
            ke=1e6, kf=1e3, kd=1e3, mu=0.5, collision_group=1
        )
        ground_config = newton.ModelBuilder.ShapeConfig(
            ke=1e6, kf=1e3, kd=1e3, mu=0.5, collision_group=1
        )
        self.builder.rigid_gap = 1.0

        # Initialize the ball
        ball = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_sphere(body=ball, radius=0.2, cfg=shape_config)

        self.builder.add_ground_plane(cfg=ground_config)
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_pose,
                self.target_pos,
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
                self.trajectory.body_vel.grad[0],  # Initial velocity gradient from trajectory
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_qd,  # Update initial state for next episode
            ],
        )

    def render(self, train_iter):
        # Only render every 5 iterations
        if self.frame > 0 and train_iter % 3 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 10.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=2.0,
        )

        self.frame += 1

    def train(self, iterations=30):
        # Set initial velocity
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            vel = self.states[0].body_qd.numpy()[0][0:3]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | Init Linear Vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"
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

    sim = BallThrowOptimizerImplicit(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=30)


if __name__ == "__main__":
    main()
