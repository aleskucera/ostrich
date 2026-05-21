import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import LoggingConfig
from omegaconf import DictConfig

try:
    from examples.taros_4.common import create_taros4_model
except ImportError:
    from common import create_taros4_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


@wp.kernel
def integrate_wheel_position_kernel(
    current_wheel_angles: wp.array(dtype=wp.float32),
    target_velocities: wp.array(dtype=wp.float32),
    dt: float,
    joint_target_pos: wp.array(dtype=wp.float32),
    fl_idx: int,
    fr_idx: int,
    rl_idx: int,
    rr_idx: int,
):
    # Drive forward command
    v = 6.0
    
    # Integrate
    new_ang_fl = current_wheel_angles[0] + v * dt
    new_ang_fr = current_wheel_angles[1] + v * dt
    new_ang_rl = current_wheel_angles[2] + v * dt
    new_ang_rr = current_wheel_angles[3] + v * dt

    # Store state
    current_wheel_angles[0] = new_ang_fl
    current_wheel_angles[1] = new_ang_fr
    current_wheel_angles[2] = new_ang_rl
    current_wheel_angles[3] = new_ang_rr

    # Write to global array
    joint_target_pos[fl_idx] = new_ang_fl
    joint_target_pos[fr_idx] = new_ang_fr
    joint_target_pos[rl_idx] = new_ang_rl
    joint_target_pos[rr_idx] = new_ang_rr


class TarosObstacleSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "velocity",
        k_p: float = 1000.0,
        k_d: float = 0.0,
        friction: float = 0.5,
    ):
        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # Taros-4 DOFs: 6 (Base) + 4 (Wheels) = 10
        if self.control_mode == "velocity":
            # Drive forward: All wheels
            robot_joint_target = np.array([0.0] * 6 + [6.0, 6.0, 6.0, 6.0], dtype=np.float32)
            joint_target = np.tile(robot_joint_target, self.simulation_config.num_worlds)
            self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)
        else:
            self.wheel_angles = wp.zeros(4, dtype=wp.float32, device=self.model.device)
            self.joint_target = wp.zeros(10, dtype=wp.float32, device=self.model.device)

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        if self.control_mode == "velocity":
            wp.copy(self.control.joint_target_vel, self.joint_target)
        else:
            wp.launch(
                kernel=integrate_wheel_position_kernel,
                dim=1,
                inputs=[
                    self.wheel_angles,
                    wp.zeros(1, dtype=wp.float32, device=self.model.device), # dummy
                    self.clock.dt,
                    self.joint_target,
                    6, 7, 8, 9,
                ],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Taros-4 model with an obstacle.
        """

        # Robot position
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 1.0

        create_taros4_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction=self.friction,
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
            xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.16,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((8.0, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.0,
            hz=0.22,
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # Ground plane
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.1,
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


@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def taros4_obstacle_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = TarosObstacleSimulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
    )
    simulator.run()


if __name__ == "__main__":
    taros4_obstacle_example()
