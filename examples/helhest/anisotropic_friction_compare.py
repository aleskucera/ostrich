"""Side-by-side comparison of isotropic vs anisotropic wheel friction.

Two Helhest robots in one scene receive identical skid-steer commands.
- Robot A (at y=+1.5): anisotropic — high lateral mu, low longitudinal mu.
  Wheels roll easily forward but resist sideways skid.
- Robot B (at y=-1.5): isotropic — same scalar mu in all tangent directions.

Both robots execute a constant differential drive (left wheel slower than
right) which is a continuous skid-steer turn. The anisotropic robot
tracks a cleaner arc; the isotropic robot scrubs and drifts more.
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


class HelhestAnisotropicCompare(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        k_p: float = 1000.0,
        k_d: float = 0.0,
        friction_lat: float = 1.4,
        friction_long: float = 0.15,
        drive_speed: float = 8.0,
        turn_ratio: float = 0.4,
    ):
        self.k_p = k_p
        self.k_d = k_d
        self.friction_lat = friction_lat
        self.friction_long = friction_long
        self.drive_speed = drive_speed
        self.turn_ratio = turn_ratio
        super().__init__(sim_config, render_config, engine_config, logging_config)

        # Constant skid-steer drive command for both robots. DOF layout (one
        # world, two helhests, 9 DOFs each):
        #   Robot A: base 0..5, left=6, right=7, rear=8
        #   Robot B: base 9..14, left=15, right=16, rear=17
        # Using a CONSTANT target lets the CUDA-graph capture work — a
        # time-varying Python schedule would get baked at capture time.
        # Positive wheel speed drives the chassis forward (in body +x).
        v_left = self.drive_speed
        v_right = self.drive_speed * self.turn_ratio
        v_rear = 0.0
        target = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        target[6] = v_left
        target[7] = v_right
        target[8] = v_rear
        target[15] = v_left
        target[16] = v_right
        target[17] = v_rear
        self.joint_target = wp.from_numpy(target, dtype=wp.float32, device=self.model.device)

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
        # Robot A: anisotropic — high lateral mu, low longitudinal mu
        create_helhest_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(0.0, 1.5, 0.5),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            ),
            control_mode="velocity",
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_lat,
            friction_rear=self.friction_lat,
            friction_long_left_right=self.friction_long,
            friction_long_rear=self.friction_long,
        )

        # Robot B: isotropic baseline — same scalar mu in all directions
        create_helhest_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(0.0, -1.5, 0.5),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            ),
            control_mode="velocity",
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_lat,
            friction_rear=self.friction_lat,
            # friction_long_* omitted -> isotropic
        )

        # Ground plane. Low mu so the wheels' anisotropic coefficients dominate
        # the contact (combine rule averages per-axis with ground's scalar mu).
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.05)
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_anisotropic_compare(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestAnisotropicCompare(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_anisotropic_compare()
