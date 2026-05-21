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

    # Sum Y velocities across all trajectory steps to verify gradient flow
    l = wp.float32(0.0)
    for i in range(trajectory_body_vel.shape[0]):
        vel = trajectory_body_vel[i, 0, 0][1]
        l = l + vel
    loss[0] = l


@wp.kernel
def update_kernel(
    initial_qd_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    alpha: float,
    initial_qd: wp.array(dtype=wp.spatial_vector, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return

    qd_y = initial_qd[0][1]
    qd_y_grad = initial_qd_grad[0, 0][1]

    wp.printf("Gradient: %f\n", qd_y_grad)

    max_grad = 5.0
    qd_y_grad = wp.clamp(qd_y_grad, -max_grad, max_grad)

    qd_y_new = qd_y - alpha * qd_y_grad
    initial_qd[0] = wp.spatial_vector(0.0, qd_y_new, 0.0, 0.0, 0.0, 0.0)


class CurlingOptimizerImplicitDebug(AxionDifferentiableSimulator):
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

        self.target_pos = wp.vec3(0.0, 3.0, 0.1)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 1e-1

        # Initial velocity guess (Y = sliding direction)
        self.init_vel = wp.spatial_vector(0.0, 0.5, 0.0, 0.0, 0.0, 0.0)

        self.track_body(body_idx=0, name="stone", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        shape_config = newton.ModelBuilder.ShapeConfig(ke=1e5, kd=1e2, kf=1e3, mu=1.0, density=10.0)

        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_cylinder(body=0, radius=0.3, half_height=0.1, cfg=shape_config)

        self.builder.add_ground_plane(cfg=shape_config)

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
                self.trajectory.body_vel.grad[0],
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_qd,
            ],
        )

    def debug_train(self):
        # Set initial velocity
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        self.diff_step()

        curr_loss = self.loss.numpy()[0]
        vel = self.states[0].body_qd.numpy()[0][0:3]

        print(f"Loss={curr_loss:.4f} | Init Vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})")
        self.solver.save_logs()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = CurlingOptimizerImplicitDebug(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.debug_train()


if __name__ == "__main__":
    main()
