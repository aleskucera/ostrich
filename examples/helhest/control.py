import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import AxionEngine
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
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


@wp.kernel
def integrate_wheel_position_kernel(
    current_wheel_angles: wp.array(dtype=wp.float32),
    target_velocities: wp.array(dtype=wp.float32),
    dt: float,
    joint_target_pos: wp.array(dtype=wp.float32),
    left_idx: int,
    right_idx: int,
    rear_idx: int,
):
    # Read command velocities
    v_l = target_velocities[0]
    v_r = target_velocities[1]
    v_rear = target_velocities[2]

    # Integrate: Angle = Angle + Velocity * dt
    new_ang_l = current_wheel_angles[0] + v_l * dt
    new_ang_r = current_wheel_angles[1] + v_r * dt
    new_ang_rear = current_wheel_angles[2] + v_rear * dt

    # Store state
    current_wheel_angles[0] = new_ang_l
    current_wheel_angles[1] = new_ang_r
    current_wheel_angles[2] = new_ang_rear

    # # Wrap to [-PI, PI] to preserve precision
    # PI = 3.14159265
    # TWO_PI = 6.2831853
    #
    # out_l = new_ang_l - TWO_PI * wp.floor((new_ang_l + PI) / TWO_PI)
    # out_r = new_ang_r - TWO_PI * wp.floor((new_ang_r + PI) / TWO_PI)
    # out_rear = new_ang_rear - TWO_PI * wp.floor((new_ang_rear + PI) / TWO_PI)

    # Write to global array
    joint_target_pos[left_idx] = new_ang_l
    joint_target_pos[right_idx] = new_ang_r
    joint_target_pos[rear_idx] = new_ang_rear


@wp.kernel
def apply_wheel_velocity_kernel(
    target_velocities: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
    left_idx: int,
    right_idx: int,
    rear_idx: int,
):
    # Write command velocities to the global array at specified indices
    joint_target_vel[left_idx] = target_velocities[0]
    joint_target_vel[right_idx] = target_velocities[1]
    joint_target_vel[rear_idx] = target_velocities[2]


class HelhestControlSimulator(InteractiveSimulator):
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
        self.left_indices_cpu = []
        self.right_indices_cpu = []
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # 1. Internal Buffers for Position Control
        # Stores [v_left, v_right, v_rear]
        self.target_velocities = wp.zeros(3, dtype=wp.float32, device=self.model.device)
        # Stores [angle_left, angle_right, angle_rear]
        self.wheel_angles = wp.zeros(3, dtype=wp.float32, device=self.model.device)

        # Buffer for the full joint array
        self.joint_target_buffer = wp.zeros_like(self.model.joint_target_pos)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Check keyboard input and update wheel velocities."""
        base_speed = 7.0
        turn_speed = 3.0

        left_v = 0.0
        right_v = 0.0

        # Using I/J/K/L with the correct 'is_key_down' method
        if self.viewer and hasattr(self.viewer, "is_key_down"):
            # Forward (I)
            if self.viewer.is_key_down("i"):
                left_v += base_speed
                right_v += base_speed
            # Backward (K)
            if self.viewer.is_key_down("k"):
                left_v -= base_speed
                right_v -= base_speed
            # Left (J)
            if self.viewer.is_key_down("j"):
                left_v -= turn_speed
                right_v += turn_speed
            # Right (L)
            if self.viewer.is_key_down("l"):
                left_v += turn_speed
                right_v -= turn_speed

        rear_v = (left_v + right_v) / 2.0

        # Update the GPU velocity buffer
        # We assume indices 0=Left, 1=Right, 2=Rear for this small buffer
        vels_cpu = np.array([left_v, right_v, rear_v], dtype=np.float32)
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
            # Buffer for the full joint array
            wp.launch(
                kernel=apply_wheel_velocity_kernel,
                dim=1,
                inputs=[
                    self.target_velocities,
                    self.joint_target_buffer,
                    6,
                    7,
                    8,
                ],
                device=self.model.device,
            )

            wp.copy(self.control.joint_target_vel, self.joint_target_buffer)
        else:
            # INTEGRATE: Convert Velocity -> Position
            # Indices in Helhest model: 6=Left, 7=Right, 8=Rear
            wp.launch(
                kernel=integrate_wheel_position_kernel,
                dim=1,
                inputs=[
                    self.wheel_angles,  # State (read/write)
                    self.target_velocities,  # Input (from keyboard)
                    self.clock.dt,  # dt
                    self.joint_target_buffer,  # Output
                    6,
                    7,
                    8,  # Indices
                ],
                device=self.model.device,
            )

            # Apply to Position Control Target
            wp.copy(self.control.joint_target_pos, self.joint_target_buffer)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 1.0
        # --- 1. Ground ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.3,
            ke=4e4,
            kd=4e3,
            kf=1e3,
        )
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
        robot_z = 0.5  # Slightly above ground

        create_helhest_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,  # Keep rear wheel slippery
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_control_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    # Force GL viewer
    render_config.vis_type = "gl"

    simulator = HelhestControlSimulator(
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
    helhest_control_example()
