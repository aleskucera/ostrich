import os
import pathlib
import random
from typing import Any
from typing import Dict
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
    from examples.marv_tracked.common import create_marv_tracked_model
except ImportError:
    from common import create_marv_tracked_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


@wp.kernel
def integrate_track_kernel(
    track_u: wp.array(dtype=wp.float32), velocity: wp.array(dtype=wp.float32), dt: wp.float32
):
    track_u[0] = track_u[0] + velocity[0] * dt


@wp.kernel
def update_track_joints_kernel(
    joint_indices: wp.array(dtype=int),
    u_offsets: wp.array(dtype=wp.float32),
    global_u: wp.array(dtype=wp.float32),
    X_track: wp.transform,
    # Track Geometry Constants
    r1: wp.float32,
    r2: wp.float32,
    p1_top: wp.vec2,
    p2_top: wp.vec2,
    p1_bot: wp.vec2,
    p2_bot: wp.vec2,
    c1: wp.vec2,
    c2: wp.vec2,
    L1: wp.float32,
    L2: wp.float32,
    L3: wp.float32,
    total_len: wp.float32,
    ang_f_start: wp.float32,
    ang_r_start: wp.float32,
    # Output
    joint_X_p: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    joint_idx = joint_indices[tid]
    u_offset = u_offsets[tid]
    u_curr = wp.mod(global_u[0] + u_offset, total_len)
    if u_curr < 0.0:
        u_curr = u_curr + total_len

    pos_2d = wp.vec2(0.0, 0.0)
    tan_2d = wp.vec2(0.0, 0.0)

    # Parametric Track Logic
    if u_curr < L1:
        t = u_curr / L1
        pos_2d = (1.0 - t) * p1_top + t * p2_top
        tan_2d = wp.normalize(p2_top - p1_top)
    elif u_curr < L2:
        arc_u = u_curr - L1
        angle = ang_f_start - (arc_u / r2)
        pos_2d = c2 + r2 * wp.vec2(wp.cos(angle), wp.sin(angle))
        tan_2d = wp.vec2(wp.sin(angle), -wp.cos(angle))
    elif u_curr < L3:
        line_u = u_curr - L2
        t = line_u / (L3 - L2)
        pos_2d = (1.0 - t) * p2_bot + t * p1_bot
        tan_2d = wp.normalize(p1_bot - p2_bot)
    else:
        arc_u = u_curr - L3
        angle = ang_r_start - (arc_u / r1)
        pos_2d = c1 + r1 * wp.vec2(wp.cos(angle), wp.sin(angle))
        tan_2d = wp.vec2(wp.sin(angle), -wp.cos(angle))

    # Construct 3D orientation
    tangent = wp.vec3(tan_2d[0], tan_2d[1], 0.0)
    normal = wp.vec3(-tan_2d[1], tan_2d[0], 0.0)
    binormal = wp.vec3(0.0, 0.0, 1.0)

    # Orientation matrix to quaternion
    rot_matrix = wp.mat33(
        tangent[0],
        normal[0],
        binormal[0],
        tangent[1],
        normal[1],
        binormal[1],
        tangent[2],
        normal[2],
        binormal[2],
    )
    q_local = wp.quat_from_matrix(rot_matrix)
    pos_local = wp.vec3(pos_2d[0], pos_2d[1], 0.0)

    X_local = wp.transform(pos_local, q_local)
    # X_track is the track frame RELATIVE to the Parent (Flipper Arm)
    X_in_parent = wp.transform_multiply(X_track, X_local)

    joint_X_p[joint_idx] = X_in_parent


class MarvTrackedSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        # We need to defer track initialization until AFTER build_model is called
        # but build_model is called inside super().__init__.
        # So we initialize containers here, but populate them later?
        # No, InteractiveSimulator.__init__ calls build_model, then creates the model, then we can init our stuff.
        # But we need to capture the track info from build_model.
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

            # Warp Arrays
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

        # Joint targets for flippers
        # We need to know which indices are the flipper joints.
        # They are stored in track_info_cpu as 'joint_idx'.
        # But wait, 'joint_idx' is an integer ID returned by add_joint.
        # We need to map this to the index in the joint_target array?
        # If we use a single articulation, the joint indices are sequential in the articulation.
        # Axion's model.joint_target corresponds to the articulation DOFs.
        # The base is 0-5.
        # Then we added legs.
        # We should find the index of the flipper joint in the articulation.
        # Since we added them in order, we can probably deduce it, or use the mapping if available.
        # However, newton.Model doesn't easily expose "joint_id -> dof_index" map in Python yet?
        # Assuming the order in common.py:
        # Base (6)
        # FL: Flipper (1 DOF) + Tracks (0 DOF in joint_target? Tracks use update_track_joints_kernel)
        # Wait, track elements are connected by joints. What kind?
        # In tank_control.py:
        # box_shape -> add_track -> creates joints.
        # The joints created by add_track usually don't have DOFs that are controlled via joint_target (they are constrained/kinematic-like driven by the kernel?).
        # Actually, add_track usually creates PRISMATIC or SLIDER joints if they move along a line?
        # Or maybe they are FREE joints but constrained?
        # In tank_control.py, there is NO control applied to track joints in `control_policy` via `joint_target`.
        # So track joints consume DOFs but we don't actuate them normally.
        # So we only care about Flipper Joints for `joint_target`.

        # Order:
        # Base (6 DOFs)
        # FL Flipper (1 DOF)
        # FL Track Elements...
        # FR Flipper (1 DOF)
        # ...

        # We need the DOF index for each flipper.
        # self.model.joint_qd_start maps joint_index -> start_dof.
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
        # Constants
        MAX_SPEED = 0.75
        TURN_SPEED = 0.5
        FLIPPER_SPEED = 1.0

        target_fwd = 0.0
        target_turn = 0.0

        # Reset flipper velocities
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

        # Differential Drive
        left_vel = target_fwd - target_turn
        right_vel = target_fwd + target_turn

        # Update Track Velocities
        # FL, RL -> Left
        # FR, RR -> Right

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

        # Update Flipper Targets
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
            # Integrate U
            wp.launch(
                kernel=integrate_track_kernel,
                dim=1,
                inputs=[state["u"], state["vel"], self.clock.dt],
                device=self.model.device,
            )

            # Update Joints
            helper = state["helper"]
            wp.launch(
                kernel=update_track_joints_kernel,
                dim=len(state["indices"]),
                inputs=[
                    state["indices"],
                    state["offsets"],
                    state["u"],
                    state["xform"],
                    # Geometry
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
        # Create Marv Tracked
        chassis_id, track_info = create_marv_tracked_model(
            self.builder, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity())
        )

        self.track_info_cpu = track_info

        # Add Ground
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=0.5)
        self.builder.add_ground_plane(cfg=ground_cfg)

        # Obstacles (Same as before)
        # Stairs
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

        # Obstacle 4: Large box in the middle
        self.builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(3.0, 0.0, 0.10), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.20,
            cfg=ground_cfg,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


@wp.kernel
def set_flipper_targets_kernel(
    targets: wp.array(dtype=wp.float32),
    idx_fl: int,
    val_fl: wp.float32,
    idx_fr: int,
    val_fr: wp.float32,
    idx_rl: int,
    val_rl: wp.float32,
    idx_rr: int,
    val_rr: wp.float32,
):
    targets[idx_fl] = val_fl
    targets[idx_fr] = val_fr
    targets[idx_rl] = val_rl
    targets[idx_rr] = val_rr


@hydra.main(config_path=str(CONFIG_PATH), config_name="marv_tracked", version_base=None)
def marv_tracked_example(cfg: DictConfig):
    sim_config = hydra.utils.instantiate(cfg.simulation)
    render_config = hydra.utils.instantiate(cfg.rendering)
    engine_config = hydra.utils.instantiate(cfg.engine)
    logging_config = hydra.utils.instantiate(cfg.logging)

    simulator = MarvTrackedSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    marv_tracked_example()
