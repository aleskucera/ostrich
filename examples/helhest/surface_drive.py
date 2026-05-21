import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import openmesh
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
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


@wp.kernel
def integrate_wheel_position_kernel(
    current_wheel_angles: wp.array(dtype=wp.float32),
    target_velocities: wp.array(dtype=wp.float32),
    dt: float,
    joint_target_pos: wp.array(dtype=wp.float32),
    l_idx: int,
    r_idx: int,
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

    # Write to global array
    joint_target_pos[l_idx] = new_ang_l
    joint_target_pos[r_idx] = new_ang_r
    joint_target_pos[rear_idx] = new_ang_rear


class HelhestSurfaceSimulator(InteractiveSimulator):
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

        # 3 Target velocities [left, right, rear]
        self.target_velocities = wp.zeros(3, dtype=wp.float32, device=self.model.device)

        if self.control_mode == "position":
            # Stores [angle_l, angle_r, angle_rear]
            self.wheel_angles = wp.zeros(3, dtype=wp.float32, device=self.model.device)

        # Helhest DOFs: 6 (Base) + 3 (Left, Right, Rear)
        self.joint_target = wp.zeros(9, dtype=wp.float32, device=self.model.device)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Check keyboard input and update wheel velocities."""
        base_speed = 5.0
        turn_speed = 2.5

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

        rear_v = (left_v + right_v) / 2.0

        # Update targets
        if self.control_mode == "velocity":
            targets_cpu = np.zeros(9, dtype=np.float32)
            targets_cpu[6] = left_v
            targets_cpu[7] = right_v
            targets_cpu[8] = rear_v

            wp.copy(
                self.joint_target, wp.array(targets_cpu, dtype=wp.float32, device=self.model.device)
            )
        else:
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
                ],
                device=self.model.device,
            )
            wp.copy(self.control.joint_target_pos, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model on a surface.
        """

        self.builder.rigid_gap = 0.5

        # Robot position
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 1.7

        create_helhest_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,
        )

        # Surface Mesh
        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()

        scale = np.array([6.0, 6.0, 4.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])

        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        # Add the surface mesh to a separate builder so it gets shape_world=-1
        # (Newton's "global" sentinel). The mesh is then stored once and the
        # broadphase tests it against shapes from every world without duplication.
        globals_builder = newton.ModelBuilder()
        globals_builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.5,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            global_builder=globals_builder,
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_surface_drive_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestSurfaceSimulator(
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
    helhest_surface_drive_example()
