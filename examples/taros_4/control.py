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
    from examples.taros_4.common import create_taros4_model
except ImportError:
    from common import create_taros4_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


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
    # Read command velocities
    v_l = target_velocities[0]
    v_r = target_velocities[1]

    # Integrate: Angle = Angle + Velocity * dt
    new_ang_fl = current_wheel_angles[0] + v_l * dt
    new_ang_fr = current_wheel_angles[1] + v_r * dt
    new_ang_rl = current_wheel_angles[2] + v_l * dt
    new_ang_rr = current_wheel_angles[3] + v_r * dt

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


class Taros4ControlSimulator(InteractiveSimulator):
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
        max_accel: float = 8.0,
    ):
        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction
        self.max_accel = max_accel
        self._cmd_left_v = 0.0
        self._cmd_right_v = 0.0
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # 4 Wheel DOFs (Front Left, Front Right, Rear Left, Rear Right)
        # For input we just store [left_v, right_v]
        self.target_velocities = wp.zeros(2, dtype=wp.float32, device=self.model.device)

        if self.control_mode == "position":
            # Stores [angle_fl, angle_fr, angle_rl, angle_rr]
            self.wheel_angles = wp.zeros(4, dtype=wp.float32, device=self.model.device)

        # Initialize full target array
        # First 6 are for the free joint (base)
        self.joint_target = wp.zeros(10, dtype=wp.float32, device=self.model.device)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Check keyboard input and update wheel velocities."""
        base_speed = 6.0
        turn_speed = 3.0

        left_v = 0.0
        right_v = 0.0

        # Simple WASD/Arrow style logic
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):  # Forward
                left_v += base_speed
                right_v += base_speed
            if self.viewer.is_key_down("k"):  # Backward
                left_v -= base_speed
                right_v -= base_speed

            # Turn Left/Right
            if self.viewer.is_key_down("j"):  # Left
                left_v -= turn_speed
                right_v += turn_speed
            if self.viewer.is_key_down("l"):  # Right
                left_v += turn_speed
                right_v -= turn_speed

        # Slew-rate limit toward the desired command (acceleration ramp).
        segment_dt = self.clock.steps_per_segment * self.clock.dt
        dv_max = self.max_accel * segment_dt
        self._cmd_left_v += float(np.clip(left_v - self._cmd_left_v, -dv_max, dv_max))
        self._cmd_right_v += float(np.clip(right_v - self._cmd_right_v, -dv_max, dv_max))
        left_v = self._cmd_left_v
        right_v = self._cmd_right_v

        # Update targets
        if self.control_mode == "velocity":
            targets_cpu = np.zeros(10, dtype=np.float32)
            targets_cpu[6] = left_v
            targets_cpu[7] = right_v
            targets_cpu[8] = left_v
            targets_cpu[9] = right_v

            wp.copy(
                self.joint_target, wp.array(targets_cpu, dtype=wp.float32, device=self.model.device)
            )
        else:
            # For position control, we just copy the velocity commands to the GPU
            vels_cpu = np.array([left_v, right_v], dtype=np.float32)
            wp.copy(self.target_velocities, wp.array(vels_cpu, device=self.model.device))

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
            # INTEGRATE: Convert Velocity -> Position
            wp.launch(
                kernel=integrate_wheel_position_kernel,
                dim=1,
                inputs=[
                    self.wheel_angles,
                    self.target_velocities,
                    self.clock.dt,
                    self.joint_target,
                    6,
                    7,
                    8,
                    9,  # Indices
                ],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.8
        # --- 1. Ground ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Obstacle 1: Stairs (Stepped boxes)
        num_steps = 6
        step_depth = 0.6
        step_height = 0.08
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

        # --- 2. Robot ---
        robot_x = 0.0
        robot_y = 0.0
        robot_z = 1.0  # Slightly above ground

        create_taros4_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction=self.friction,
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def taros4_control_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Taros4ControlSimulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
        max_accel=cfg.control.get("max_accel", 2.0),
    )

    simulator.run()


if __name__ == "__main__":
    taros4_control_example()
