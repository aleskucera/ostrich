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
    from examples.marv_tracked.common import create_marv_tracked_model
except ImportError:
    from common import create_marv_tracked_model

# Reuse track kernels from control module
try:
    from examples.marv_tracked.control import integrate_track_kernel
    from examples.marv_tracked.control import set_flipper_targets_kernel
    from examples.marv_tracked.control import update_track_joints_kernel
except ImportError:
    from control import integrate_track_kernel
    from control import set_flipper_targets_kernel
    from control import update_track_joints_kernel

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class MarvTrackedSurfaceSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        self.track_info_cpu = {}

        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # Initialize Track States
        self.track_states = {}
        all_offsets = self.model.track_u_offset.numpy()

        for code in ["FL", "FR", "RL", "RR"]:
            info = self.track_info_cpu[code]
            indices = info["indices"]

            u_arr = wp.zeros(1, dtype=wp.float32, device=self.model.device)
            vel_arr = wp.zeros(1, dtype=wp.float32, device=self.model.device)

            indices_wp = wp.array(indices, dtype=int, device=self.model.device)
            offsets_wp = wp.array(all_offsets[indices], dtype=wp.float32, device=self.model.device)

            self.track_states[code] = {
                "u": u_arr,
                "vel": vel_arr,
                "indices": indices_wp,
                "offsets": offsets_wp,
                "helper": info["helper"],
                "xform": info["xform"],
            }

        # Flipper control state
        self.flipper_vel_front = 0.0
        self.flipper_vel_rear = 0.0

        # Flipper DOF indices
        qd_start_wp = self.model.joint_qd_start
        qd_start_np = qd_start_wp.numpy()

        if len(qd_start_np.shape) == 2:
            joint_start_dof = qd_start_np[0]
        else:
            joint_start_dof = qd_start_np

        self.flipper_dof_indices = {}
        for code in ["FL", "FR", "RL", "RR"]:
            j_idx = self.track_info_cpu[code]["joint_idx"]
            dof_idx = joint_start_dof[j_idx]
            self.flipper_dof_indices[code] = dof_idx

        self.joint_targets = wp.zeros_like(self.model.joint_target_vel)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        MAX_SPEED = 0.75
        TURN_SPEED = 0.5
        FLIPPER_SPEED = 1.0

        target_fwd = 0.0
        target_turn = 0.0

        self.flipper_vel_front = 0.0
        self.flipper_vel_rear = 0.0

        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):
                target_fwd += MAX_SPEED
            if self.viewer.is_key_down("k"):
                target_fwd -= MAX_SPEED
            if self.viewer.is_key_down("j"):
                target_turn += TURN_SPEED
            if self.viewer.is_key_down("l"):
                target_turn -= TURN_SPEED

            if self.viewer.is_key_down("m"):
                self.flipper_vel_front += FLIPPER_SPEED
            if self.viewer.is_key_down("n"):
                self.flipper_vel_front -= FLIPPER_SPEED
            if self.viewer.is_key_down("u"):
                self.flipper_vel_rear += FLIPPER_SPEED
            if self.viewer.is_key_down("o"):
                self.flipper_vel_rear -= FLIPPER_SPEED

        left_vel = target_fwd - target_turn
        right_vel = target_fwd + target_turn

        wp.copy(
            self.track_states["FL"]["vel"],
            wp.array([left_vel], dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.track_states["RL"]["vel"],
            wp.array([left_vel], dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.track_states["FR"]["vel"],
            wp.array([right_vel], dtype=wp.float32, device=self.model.device),
        )
        wp.copy(
            self.track_states["RR"]["vel"],
            wp.array([right_vel], dtype=wp.float32, device=self.model.device),
        )

        wp.launch(
            kernel=set_flipper_targets_kernel,
            dim=1,
            inputs=[
                self.joint_targets,
                self.flipper_dof_indices["FL"],
                self.flipper_vel_front,
                self.flipper_dof_indices["FR"],
                self.flipper_vel_front,
                self.flipper_dof_indices["RL"],
                self.flipper_vel_rear,
                self.flipper_dof_indices["RR"],
                self.flipper_vel_rear,
            ],
            device=self.model.device,
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
        # 1. Update Track Physics
        for code, state in self.track_states.items():
            wp.launch(
                kernel=integrate_track_kernel,
                dim=1,
                inputs=[state["u"], state["vel"], self.clock.dt],
                device=self.model.device,
            )

            helper = state["helper"]
            wp.launch(
                kernel=update_track_joints_kernel,
                dim=len(state["indices"]),
                inputs=[
                    state["indices"],
                    state["offsets"],
                    state["u"],
                    state["xform"],
                    helper.r1,
                    helper.r2,
                    wp.vec2(helper.p1_top),
                    wp.vec2(helper.p2_top),
                    wp.vec2(helper.p1_bot),
                    wp.vec2(helper.p2_bot),
                    wp.vec2(helper.c1),
                    wp.vec2(helper.c2),
                    helper.L1,
                    helper.L2,
                    helper.L3,
                    helper.total_len,
                    helper.ang_f_start,
                    helper.ang_r_start,
                    self.model.joint_X_p,
                ],
                device=self.model.device,
            )

        # 2. Apply Flipper Targets
        wp.copy(self.control.joint_target_vel, self.joint_targets)

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2

        # Place robot elevated on the surface
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 1.7

        chassis_id, track_info = create_marv_tracked_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
        )

        self.track_info_cpu = track_info

        # Surface Mesh
        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()

        scale = np.array([6.0, 6.0, 4.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])

        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.8,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="marv_tracked", version_base=None)
def marv_tracked_surface_drive_example(cfg: DictConfig):
    sim_config = hydra.utils.instantiate(cfg.simulation)
    render_config = hydra.utils.instantiate(cfg.rendering)
    engine_config = hydra.utils.instantiate(cfg.engine)
    logging_config = hydra.utils.instantiate(cfg.logging)

    simulator = MarvTrackedSurfaceSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    marv_tracked_surface_drive_example()
