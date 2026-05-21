import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
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


@wp.kernel
def integrate_wheel_position_kernel(
    current_wheel_angles: wp.array(dtype=wp.float32),
    dt: float,
    joint_target_pos: wp.array(dtype=wp.float32),
    l_idx: int,
    r_idx: int,
    rear_idx: int,
):
    # High speed for flipping: 15.0 rad/s
    v = -15.0

    # Integrate
    new_ang_l = current_wheel_angles[0] + v * dt
    new_ang_r = current_wheel_angles[1] + v * dt
    new_ang_rear = current_wheel_angles[2] + 0.0 * dt

    # Store state
    current_wheel_angles[0] = new_ang_l
    current_wheel_angles[1] = new_ang_r
    current_wheel_angles[2] = new_ang_rear

    # Write to global array
    joint_target_pos[l_idx] = new_ang_l
    joint_target_pos[r_idx] = new_ang_r
    joint_target_pos[rear_idx] = new_ang_rear


class HelhestFlipSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "position",
        k_p: float = 50.0,
        k_d: float = 0.1,
        friction: float = 0.7,
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

        # Helhest DOFs: 6 (Base) + 1 (Left) + 1 (Right) + 1 (Rear) = 9
        if self.control_mode == "velocity":
            # High speed for flipping: 15.0 rad/s
            robot_joint_target = np.array([0.0] * 6 + [-10.0, -10.0, 0.0], dtype=np.float32)
            joint_target = np.tile(robot_joint_target, self.simulation_config.num_worlds)
            self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)
        else:
            self.wheel_angles = wp.zeros(3, dtype=wp.float32, device=self.model.device)
            self.joint_target = wp.zeros(9, dtype=wp.float32, device=self.model.device)

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
                    self.clock.dt,
                    self.joint_target,
                    6,
                    7,
                    8,
                ],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model for the flip scenario.
        """

        # Robot position
        robot_x = -3.0
        robot_y = 0.0
        robot_z = 0.5

        create_helhest_model(
            self.builder,
            xform=wp.transform(
                (robot_x, robot_y, robot_z),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            ),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,
        )

        # Environment parameters from original flip.py
        FRICTION = 0.8
        RESTITUTION = 0.0
        KE = 500.0
        KD = 500.0
        KF = 500.0

        # --- Add Static Obstacles and Ground ---

        # Obstacle 1
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=1.75,
            hy=1.5,
            hz=0.10,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
                ke=KE,
                kd=KD,
                kf=KF,
            ),
        )

        # Obstacle 2
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=0.5,
            hy=1.75,
            hz=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
                ke=KE,
                kd=KD,
                kf=KF,
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
def helhest_flip_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestFlipSimulator(
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
    helhest_flip_example()
