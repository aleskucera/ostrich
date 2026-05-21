import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import DatasetSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class HelhestObstacleSimulator(DatasetSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "position",
        k_p: float = 0.0,
        k_d: float = 0.0,
        friction: float = 0.7,
    ):
        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction
        self.pos_min = wp.vec3(-0.2, -0.2, 1.0)
        self.pos_max = wp.vec3(0.2, 0.2, 2.0)
        self.lin_vel_min, self.lin_vel_max = -1.0, 1.0
        self.ang_vel_min, self.ang_vel_max = -3.14, 3.14
        self.joint_target_lower_bound = -10.0
        self.joint_target_upper_bound = 10.0
        self.seed = 42
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # 1. Internal Buffers for Position Control
        # Stores [v_left, v_right, v_rear] - Sized for num_worlds
        self.target_velocities = wp.zeros(
            3 * self.simulation_config.num_worlds, dtype=wp.float32, device=self.model.device
        )
        # Stores [angle_left, angle_right, angle_rear]
        self.wheel_angles = wp.zeros(
            3 * self.simulation_config.num_worlds, dtype=wp.float32, device=self.model.device
        )

        # Buffer for the full joint array
        self.joint_target_buffer = wp.zeros_like(self.model.joint_target_pos)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model with an obstacle.
        """
        self.builder.rigid_gap = 1.0

        # Robot position
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 0.6

        create_helhest_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,
        )

        # Ground Plane parameters
        FRICTION = 1.0
        RESTITUTION = 0.0
        KE = 1.0e4
        KD = 1.0e3
        KF = 1.0e3

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        # Reduced size: hx=0.5 (1m length), hy=1.0 (2m width), hz=0.1 (0.2m height)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.25, 0.25, 0.0), wp.quat_identity()),
            hx=2.0,
            hy=2.0,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # Ground plane
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=KE,
                kd=KD,
                kf=KF,
                mu=FRICTION,
                restitution=RESTITUTION,
            )
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_obstacle_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestObstacleSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_obstacle_example()
