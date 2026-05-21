"""Headless deterministic Helhest-vs-obstacles benchmark scene.

Mirrors `obstacle.py` (constant 6.0 rad/s wheel velocity, four static
box obstacles of increasing height) but in headless mode so the
workload is reproducible across optimization branches. The robot
unavoidably impacts an obstacle at some point in the run, which makes
this scene the natural target for evaluating contact-reduction
policies (rank-deficiency at impact is exactly the failure mode
top-K, FPS, cluster, hull are meant to fix).
"""
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


class HelhestObstacleBenchmark(InteractiveSimulator):
    """Constant forward drive into a row of static obstacles.

    Velocity-mode control with all three wheels commanded to the same
    rad/s, so the scene runs deterministically without any keyboard
    input. ViewerNull rendering keeps it headless and terminates after
    `simulation.duration_seconds`.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "velocity",
        k_p: float = 150.0,
        k_d: float = 0.0,
        friction: float = 0.5,
        drive_velocity: float = 6.0,
    ):
        if control_mode != "velocity":
            raise ValueError(
                "obstacle_benchmark only supports control_mode='velocity'; "
                f"got {control_mode!r}"
            )

        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction
        self.drive_velocity = drive_velocity

        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # 9 joint DOFs: 6 free-joint base + 3 wheels (left, right, rear).
        targets = np.zeros(9, dtype=np.float32)
        targets[6] = drive_velocity
        targets[7] = drive_velocity
        targets[8] = drive_velocity
        self.joint_target = wp.array(targets, dtype=wp.float32, device=self.model.device)

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
        wp.copy(self.control.joint_target_vel, self.joint_target)

    def build_model(self) -> newton.Model:
        """Helhest in front of a row of four box obstacles on a ground plane."""
        self.builder.rigid_gap = 1.0

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

        FRICTION = 0.4
        RESTITUTION = 0.0
        KE = 1.0e4
        KD = 1.0e3
        KF = 1.0e3

        # Four static box obstacles of increasing height in the robot's path.
        # Heights selected so the robot can clear the first two and impact the
        # third around step ~60-70 (matches the original obstacle.py scene).
        for x, hz in [(2.0, 0.10), (5.0, 0.25), (8.0, 0.40), (11.0, 0.65)]:
            self.builder.add_shape_box(
                body=-1,
                xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                hx=0.5,
                hy=1.0,
                hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(
                    mu=FRICTION,
                    restitution=RESTITUTION,
                ),
            )

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


@hydra.main(
    config_path=str(CONFIG_PATH),
    config_name="helhest_obstacle_benchmark",
    version_base=None,
)
def helhest_obstacle_benchmark(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestObstacleBenchmark(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
        drive_velocity=cfg.drive_velocity,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_obstacle_benchmark()
