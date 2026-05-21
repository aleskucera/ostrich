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
    from examples.marv.common import create_marv_model
except ImportError:
    from common import create_marv_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Use parent/parent/conf to access shared examples config if needed,
# or just parent/conf if marv has its own.
# unified_model.py used: pathlib.Path(__file__).parent.parent.joinpath("conf")
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class MarvControlSimulator(InteractiveSimulator):
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

        # DOFs: 6 (Base) + 4 legs * 5 joints/leg = 26
        # Base: 0-5
        # FL: 6 (Flip), 7-10 (Wheels)
        # FR: 11 (Flip), 12-15 (Wheels)
        # RL: 16 (Flip), 17-20 (Wheels)
        # RR: 21 (Flip), 22-25 (Wheels)
        self.num_dofs = 26
        self.joint_target = wp.zeros(self.num_dofs, dtype=wp.float32, device=self.model.device)

        # State tracking for smooth control
        self.flipper_pos_front = 0.0
        self.flipper_pos_rear = 0.0
        self.wheel_pos_left = 0.0
        self.wheel_pos_right = 0.0

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        # Constants
        MAX_SPEED = 600.0  # Mujoco 600
        TURN_SPEED = 500.0  # Mujoco 500
        FLIPPER_SPEED = 0.1

        # Reset drive speeds
        target_fwd = 0.0
        target_turn = 0.0

        if hasattr(self.viewer, "is_key_down"):
            # Drive: I/K for Fwd/Back
            if self.viewer.is_key_down("i"):
                target_fwd += MAX_SPEED
            if self.viewer.is_key_down("k"):
                target_fwd -= MAX_SPEED

            # Turn: J/L for Left/Right
            if self.viewer.is_key_down("j"):
                target_turn += TURN_SPEED
            if self.viewer.is_key_down("l"):
                target_turn -= TURN_SPEED

            # Flippers
            # Front: M (Up), N (Down)
            if self.viewer.is_key_down("m"):
                self.flipper_pos_front += FLIPPER_SPEED
            if self.viewer.is_key_down("n"):
                self.flipper_pos_front -= FLIPPER_SPEED

            # Rear: U (Up), O (Down)
            if self.viewer.is_key_down("u"):
                self.flipper_pos_rear += FLIPPER_SPEED
            if self.viewer.is_key_down("o"):
                self.flipper_pos_rear -= FLIPPER_SPEED

            # Normalize angles to [-pi, pi]
            self.flipper_pos_front = (self.flipper_pos_front + np.pi) - np.pi
            self.flipper_pos_rear = (self.flipper_pos_rear + np.pi) - np.pi

        # Apply logic
        # Differential drive:
        # Left side = fwd - turn
        # Right side = fwd + turn
        left_vel = target_fwd - target_turn
        right_vel = target_fwd + target_turn

        # Integrate wheel positions
        dt = self.clock.dt
        self.wheel_pos_left += left_vel * dt
        self.wheel_pos_right += right_vel * dt

        # Build target array on CPU
        targets = np.zeros(self.num_dofs, dtype=np.float32)

        # Indices mapping
        # FL (Left): 6-10
        targets[6] = self.flipper_pos_front
        targets[7:11] = self.wheel_pos_left

        # FR (Right): 11-15
        targets[11] = self.flipper_pos_front
        targets[12:16] = self.wheel_pos_right

        # RL (Left): 16-20
        targets[16] = self.flipper_pos_rear
        targets[17:21] = self.wheel_pos_left

        # RR (Right): 21-25
        targets[21] = self.flipper_pos_rear
        targets[22:26] = self.wheel_pos_right

        # Copy to GPU
        # Note: We tile this for num_worlds if we support parallel envs,
        # but for simple control usually num_worlds=1.
        # If num_worlds > 1, we should replicate.

        if self.simulation_config.num_worlds > 1:
            full_targets = np.tile(targets, self.simulation_config.num_worlds)
            wp.copy(
                self.joint_target,
                wp.array(full_targets, dtype=wp.float32, device=self.model.device),
            )
        else:
            wp.copy(
                self.joint_target, wp.array(targets, dtype=wp.float32, device=self.model.device)
            )

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
        wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2
        # Create Marv
        create_marv_model(self.builder, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))

        # Add Ground
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=1.0)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Obstacle 1: Stairs (Stepped boxes)
        num_steps = 6
        step_depth = 0.6
        step_height = 0.08  # Smaller steps for Helhest
        step_width = 4.0
        start_x = 5.0
        start_y = -4.0

        for i in range(num_steps):
            h_curr = (i + 1) * step_height
            z_curr = h_curr / 2.0
            x_curr = start_x + i * step_depth

            self.builder.add_shape_box(
                -1,
                wp.transform(wp.vec3(x_curr, start_y, z_curr), wp.quat_identity()),
                hx=step_depth / 2.0,
                hy=step_width / 2.0,
                hz=h_curr / 2.0,
                cfg=ground_cfg,
            )

        # Obstacle 2: Ramp
        ramp_length = 5.0
        ramp_width = 4.0
        ramp_height = 1.0
        ramp_angle = float(np.arctan2(ramp_height, ramp_length))

        ramp_x = 7.0
        ramp_y = 4.0
        ramp_z = ramp_height / 2.0 - 0.05

        q_ramp = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -ramp_angle)

        self.builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(ramp_x, ramp_y, ramp_z), q_ramp),
            hx=ramp_length / 2.0,
            hy=ramp_width / 2.0,
            hz=0.05,
            cfg=ground_cfg,
        )

        # Obstacle 3: Uneven terrain (Small boulders)
        import random

        random.seed(42)
        for _ in range(30):
            rx = random.uniform(3.0, 12.0)
            ry = random.uniform(-2.0, 2.0)
            rz = 0.05
            self.builder.add_shape_box(
                -1,
                wp.transform(
                    wp.vec3(rx, ry, rz),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random.uniform(0, 3.14)),
                ),
                hx=random.uniform(0.15, 0.5),
                hy=random.uniform(0.15, 0.5),
                hz=random.uniform(0.05, 0.15),
                cfg=ground_cfg,
            )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def marv_control_example(cfg: DictConfig):
    sim_config = hydra.utils.instantiate(cfg.simulation)
    render_config = hydra.utils.instantiate(cfg.rendering)
    engine_config = hydra.utils.instantiate(cfg.engine)
    logging_config = hydra.utils.instantiate(cfg.logging)

    simulator = MarvControlSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    marv_control_example()
