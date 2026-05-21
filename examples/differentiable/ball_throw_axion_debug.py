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
    trajectory_body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    target_pos: wp.vec3,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid > 0:
        return

    l = wp.float32(0.0)
    # vel = trajectory_body_vel[trajectory_body_vel.shape[0] - 1, 0, 0][2]
    # l = l + wp.pow(vel, 2.0)
    for i in range(trajectory_body_vel.shape[0]):
        vel = trajectory_body_vel[i, 0, 0][2]
        l = l + wp.pow(vel, 2.0)
    loss[0] = l


@wp.kernel
def update_kernel(
    qd_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return

    # Gradient clipping to prevent divergence
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

    wp.printf("Gradient: [%f %f %f %f %f %f]\n", g[0], g[1], g[2], g[3], g[4], g[5])


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
        self.learning_rate = 0.01

        self.frame = 0

        # Initial velocity guessing (angular, linear)
        self.init_vel = wp.spatial_vector(0.0, 2.0, -0.1, 0.0, 0.0, 0.0)

        # 3. Setup Automatic Trajectory Tracking
        self.track_body(body_idx=0, name="ball", color=(0.0, 1.0, 0.0))

    def build_model(self) -> Model:
        shape_config = newton.ModelBuilder.ShapeConfig(ke=1e6, kf=1e3, kd=1e3, mu=0.5, density=0.0)

        # Initialize the ball
        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_sphere(body=0, radius=0.2, cfg=shape_config)

        # self.builder.add_ground_plane(cfg=shape_config)
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
                self.trajectory.body_vel,
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

    def debug_train(self):
        # Set initial velocity
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        self.diff_step()

        curr_loss = self.loss.numpy()[0]
        vel = self.states[0].body_qd.numpy()[0][3:6]

        print(f"Loss={curr_loss:.4f} | Init Linear Vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")
        self.solver.save_logs()


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
    sim.debug_train()


if __name__ == "__main__":
    main()
