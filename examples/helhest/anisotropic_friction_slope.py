"""Side-slope test: how much do the robots slip sideways under gravity alone?

Two Helhest robots rest on a slope tilted around the world X-axis (i.e. the
downhill direction is along world Y). With the robots oriented so body-Y
(the wheel spin axis = the lateral friction axis) aligns with world Y,
the downhill force acts in the wheels' LATERAL friction direction.

- Robot A (anisotropic): high lateral mu, low longitudinal mu. The high
  lateral mu pins it against the slope — no slip.
- Robot B (isotropic): same scalar mu = robot A's *longitudinal* mu. This
  is the "fair" isotropic baseline that wants easy rolling — but it
  gets equally low friction in the lateral direction, so it slides
  downhill.

Wheels are velocity-controlled to a target of 0 (locked) so the test is
purely about contact friction, not rolling.
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


class HelhestSlopeCompare(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        k_p: float = 2000.0,
        k_d: float = 0.0,
        slope_angle_deg: float = 15.0,
        friction_lat: float = 1.4,
        friction_long: float = 0.15,
    ):
        self.k_p = k_p
        self.k_d = k_d
        self.slope_angle_rad = float(np.deg2rad(slope_angle_deg))
        self.friction_lat = friction_lat
        self.friction_long = friction_long
        super().__init__(sim_config, render_config, engine_config, logging_config)

        # Lock wheels: target velocity = 0. With high k_p the wheels resist
        # rotation, so any motion is pure sliding on the contact.
        target = np.zeros(self.model.joint_dof_count, dtype=np.float32)
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
        theta = self.slope_angle_rad
        slope_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), theta)

        # Slope: a wide, thin static box tilted around the X-axis. Tilt is
        # around X => surface normal is (0, -sin θ, cos θ) and "downhill"
        # in world is along -Y (and slightly -Z).
        self.builder.add_shape_box(
            body=-1,  # static (attached to world)
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), slope_rot),
            hx=15.0,
            hy=10.0,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.0),
        )

        # Robots are tilted to match the slope so all three wheels contact
        # the surface flat. Body-Y after this rotation is (0, cos θ, sin θ),
        # which projects onto the slope tangent plane as itself (perpendicular
        # to the slope normal) — i.e. exactly the downhill/uphill axis.
        # So the wheel's lateral friction acts in the downhill direction.
        # Place each robot ~0.6 m above the slope along the slope-normal.
        n_slope = wp.vec3(0.0, -np.sin(theta), np.cos(theta))
        spawn_height = 1.0
        offset = n_slope * spawn_height

        # Robot A (anisotropic) on the left in X
        create_helhest_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(-2.5, 0.0, 0.0) + offset,
                slope_rot,
            ),
            control_mode="velocity",
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_lat,
            friction_rear=self.friction_lat,
            friction_long_left_right=self.friction_long,
            friction_long_rear=self.friction_long,
        )

        # Robot B (isotropic baseline using the LOW = rolling-direction value).
        # Same scalar mu everywhere — the "fair" isotropic equivalent of a robot
        # that wants easy rolling. It gets cheap rolling AND cheap sideways slip.
        create_helhest_model(
            self.builder,
            xform=wp.transform(
                wp.vec3(2.5, 0.0, 0.0) + offset,
                slope_rot,
            ),
            control_mode="velocity",
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction_long,
            friction_rear=self.friction_long,
            # friction_long_* omitted => isotropic
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_slope_compare(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestSlopeCompare(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_slope_compare()
