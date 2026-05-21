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

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class Track2D:
    """Python helper for initial track geometry calculation."""

    def __init__(self, r_rear, r_front, dist):
        self.r1 = r_rear
        self.r2 = r_front
        self.dist = dist
        self.c1 = np.array([0.0, 0.0])
        self.c2 = np.array([dist, 0.0])

        D_vec = self.c2 - self.c1
        L = np.linalg.norm(D_vec)
        val = np.clip((self.r1 - self.r2) / L, -1.0, 1.0)
        beta = np.arccos(val)

        dir_vec = np.array([1.0, 0.0])

        def rotate(v, angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

        n_top = rotate(dir_vec, beta)
        self.p1_top = self.c1 + self.r1 * n_top
        self.p2_top = self.c2 + self.r2 * n_top
        n_bot = rotate(dir_vec, -beta)
        self.p1_bot = self.c1 + self.r1 * n_bot
        self.p2_bot = self.c2 + self.r2 * n_bot

        self.len_top = np.linalg.norm(self.p2_top - self.p1_top)
        self.len_bot = np.linalg.norm(self.p1_bot - self.p2_bot)

        ang_f_start = np.arctan2(n_top[1], n_top[0])
        ang_f_end = np.arctan2(n_bot[1], n_bot[0])
        self.sweep_front = (ang_f_start - ang_f_end) % (2 * np.pi)
        self.len_arc_front = self.sweep_front * self.r2

        ang_r_start = np.arctan2(n_bot[1], n_bot[0])
        ang_r_end = np.arctan2(n_top[1], n_top[0])
        self.sweep_rear = (ang_r_start - ang_r_end) % (2 * np.pi)
        self.len_arc_rear = self.sweep_rear * self.r1

        self.ang_f_start = ang_f_start
        self.ang_r_start = ang_r_start

        self.L1 = self.len_top
        self.L2 = self.L1 + self.len_arc_front
        self.L3 = self.L2 + self.len_bot
        self.total_len = self.L3 + self.len_arc_rear

    def get_frame(self, u_in):
        u = u_in % self.total_len
        if u < self.L1:
            t = u / self.len_top
            pos = (1 - t) * self.p1_top + t * self.p2_top
            tan = (self.p2_top - self.p1_top) / self.len_top
        elif u < self.L2:
            arc_u = u - self.L1
            angle = self.ang_f_start - (arc_u / self.r2)
            pos = self.c2 + self.r2 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])
        elif u < self.L3:
            line_u = u - self.L2
            t = line_u / self.len_bot
            pos = (1 - t) * self.p2_bot + t * self.p1_bot
            tan = (self.p1_bot - self.p2_bot) / self.len_bot
        else:
            arc_u = u - self.L3
            angle = self.ang_r_start - (arc_u / self.r1)
            pos = self.c1 + self.r1 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])
        return pos, tan


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

    # Parametric Track Logic ported to Warp
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
    # X_track is the track frame RELATIVE to the Base
    X_in_base = wp.transform_multiply(X_track, X_local)

    joint_X_p[joint_idx] = X_in_base


def generate_track_data(r1, r2, dist, tube_radius=0.1, segments=200, sides=12):
    track = Track2D(r1, r2, dist)
    vertices = []
    indices = []
    u_vals = np.linspace(0, track.total_len, segments, endpoint=False)
    for i, u in enumerate(u_vals):
        pos_2d, tan_2d = track.get_frame(u)
        tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
        normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
        binormal = np.array([0.0, 0.0, 1.0])
        center = np.array([pos_2d[0], pos_2d[1], 0.0])
        for s in range(sides):
            theta = 2.0 * np.pi * s / sides
            off_n = np.cos(theta) * tube_radius
            off_b = np.sin(theta) * tube_radius
            pt = center + off_n * normal + off_b * binormal
            vertices.append(pt)
    for i in range(segments):
        i_next = (i + 1) % segments
        base = i * sides
        base_next = i_next * sides
        for s in range(sides):
            s_next = (s + 1) % sides
            indices.extend([base + s, base_next + s, base_next + s_next])
            indices.extend([base + s, base_next + s_next, base + s_next])
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


class TankObstacleSimulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        self.left_indices_cpu = []
        self.right_indices_cpu = []
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.track_left_u = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.track_right_u = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        # Initial velocities (drive forward)
        self.track_left_velocity = wp.array([1.0], dtype=wp.float32, device=self.model.device)
        self.track_right_velocity = wp.array([1.0], dtype=wp.float32, device=self.model.device)

        # Upload indices
        self.left_joint_indices = wp.array(
            self.left_indices_cpu, dtype=int, device=self.model.device
        )
        self.right_joint_indices = wp.array(
            self.right_indices_cpu, dtype=int, device=self.model.device
        )

        # Get offsets
        all_offsets = self.model.track_u_offset.numpy()
        self.left_u_offsets = wp.array(
            all_offsets[self.left_indices_cpu], dtype=wp.float32, device=self.model.device
        )
        self.right_u_offsets = wp.array(
            all_offsets[self.right_indices_cpu], dtype=wp.float32, device=self.model.device
        )

    def build_model(self) -> newton.Model:
        # --- 1. Ground ---
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            -1,
            wp.transform(wp.vec3(1.5, 0.0, 0.0), wp.quat_identity()),
            hx=2.0,
            hy=3.0,
            hz=0.6,
            cfg=ground_cfg,
        )

        # --- 2. Track Parameters ---
        r_rear, r_front, dist = 0.6, 0.6, 2.5
        self.track_helper = Track2D(r_rear, r_front, dist)

        verts, indices = generate_track_data(r_rear, r_front, dist, tube_radius=0.05)
        mesh_obj = newton.Mesh(vertices=verts, indices=indices)

        # Rotated to be upright (rolling on XZ plane)
        q_tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 3.1415 / 2.0)

        # Track Centers relative to chassis center
        # Chassis is at (0,0,0) relative to itself.
        # Tracks at y = +/- 1.2
        # Track length runs along X.
        # Track 2D geometry starts at (0,0) and goes to (dist, 0).
        # We want to center the track along X. Center is at (dist/2, 0) in local 2D track frame.
        # So we shift by -dist/2 in X.

        track_x_shift = -dist / 2.0

        # Left Track (+Y)
        p_track_left_local = wp.vec3(track_x_shift, 1.2, 0.0)
        self.X_track_left = wp.transform(p_track_left_local, q_tilt)

        # Right Track (-Y)
        p_track_right_local = wp.vec3(track_x_shift, -1.2, 0.0)
        self.X_track_right = wp.transform(p_track_right_local, q_tilt)

        # Start position of Tank
        start_z = 1.0  # Clear ground
        start_xform = wp.transform(wp.vec3(-3.5, 0.0, start_z), wp.quat_identity())

        # --- 3. Dynamic Base (Chassis) ---
        # Box chassis
        chassis_hx, chassis_hy, chassis_hz = 1.4, 1.0, 0.3
        chassis_cfg = newton.ModelBuilder.ShapeConfig(
            density=1000.0, is_visible=True, contact_margin=0.2
        )

        base = self.builder.add_link(key="chassis", mass=0.0, xform=start_xform)
        self.builder.add_shape_box(
            body=base,
            xform=wp.transform_identity(),
            hx=chassis_hx,
            hy=chassis_hy,
            hz=chassis_hz,
            cfg=chassis_cfg,
        )

        base_joint = self.builder.add_joint(
            newton.JointType.FREE, -1, base, parent_xform=start_xform
        )

        # Visuals for Track Paths (optional, but helpful)
        visual_cfg = newton.ModelBuilder.ShapeConfig(is_visible=True, has_shape_collision=False)
        self.builder.add_shape_mesh(
            body=base,
            mesh=mesh_obj,
            xform=self.X_track_left,
            cfg=visual_cfg,
            key="track_visual_left",
        )
        self.builder.add_shape_mesh(
            body=base,
            mesh=mesh_obj,
            xform=self.X_track_right,
            cfg=visual_cfg,
            key="track_visual_right",
        )

        # --- 4. Track Elements ---
        box_shape = newton.ModelBuilder.ShapeConfig(
            is_visible=True, density=100.0, mu=1.0, contact_margin=0.2
        )
        num_boxes = 16
        element_radius = 0.25
        element_half_width = 0.6

        # Add Left Track
        self.left_indices_cpu = self.builder.add_track(
            parent_body=base,
            num_elements=num_boxes,
            element_radius=element_radius,
            element_half_width=element_half_width,
            shape_config=box_shape,
            track_helper=self.track_helper,
            track_center=p_track_left_local,
            track_rotation=q_tilt,
            parent_world_xform=start_xform,
            name_prefix="track_left",
        )

        # Add Right Track
        self.right_indices_cpu = self.builder.add_track(
            parent_body=base,
            num_elements=num_boxes,
            element_radius=element_radius,
            element_half_width=element_half_width,
            shape_config=box_shape,
            track_helper=self.track_helper,
            track_center=p_track_right_local,
            track_rotation=q_tilt,
            parent_world_xform=start_xform,
            name_prefix="track_right",
        )
        # Create Articulation
        self.builder.add_articulation([base_joint] + self.left_indices_cpu + self.right_indices_cpu)

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    @override
    def control_policy(self, current_state: newton.State):

        # --- Left Track ---
        wp.launch(
            kernel=integrate_track_kernel,
            dim=1,
            inputs=[self.track_left_u, self.track_left_velocity, self.clock.dt],
            device=self.model.device,
        )

        if len(self.left_indices_cpu) > 0:
            wp.launch(
                kernel=update_track_joints_kernel,
                dim=len(self.left_indices_cpu),
                inputs=[
                    self.left_joint_indices,
                    self.left_u_offsets,
                    self.track_left_u,
                    self.X_track_left,
                    # Geometry
                    self.track_helper.r1,
                    self.track_helper.r2,
                    wp.vec2(self.track_helper.p1_top),
                    wp.vec2(self.track_helper.p2_top),
                    wp.vec2(self.track_helper.p1_bot),
                    wp.vec2(self.track_helper.p2_bot),
                    wp.vec2(self.track_helper.c1),
                    wp.vec2(self.track_helper.c2),
                    self.track_helper.L1,
                    self.track_helper.L2,
                    self.track_helper.L3,
                    self.track_helper.total_len,
                    self.track_helper.ang_f_start,
                    self.track_helper.ang_r_start,
                    self.model.joint_X_p,
                ],
                device=self.model.device,
            )

        # --- Right Track ---
        wp.launch(
            kernel=integrate_track_kernel,
            dim=1,
            inputs=[self.track_right_u, self.track_right_velocity, self.clock.dt],
            device=self.model.device,
        )

        if len(self.right_indices_cpu) > 0:
            wp.launch(
                kernel=update_track_joints_kernel,
                dim=len(self.right_indices_cpu),
                inputs=[
                    self.right_joint_indices,
                    self.right_u_offsets,
                    self.track_right_u,
                    self.X_track_right,
                    # Geometry (Same as left)
                    self.track_helper.r1,
                    self.track_helper.r2,
                    wp.vec2(self.track_helper.p1_top),
                    wp.vec2(self.track_helper.p2_top),
                    wp.vec2(self.track_helper.p1_bot),
                    wp.vec2(self.track_helper.p2_bot),
                    wp.vec2(self.track_helper.c1),
                    wp.vec2(self.track_helper.c2),
                    self.track_helper.L1,
                    self.track_helper.L2,
                    self.track_helper.L3,
                    self.track_helper.total_len,
                    self.track_helper.ang_f_start,
                    self.track_helper.ang_r_start,
                    self.model.joint_X_p,
                ],
                device=self.model.device,
            )


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def tank_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = TankObstacleSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    tank_example()
