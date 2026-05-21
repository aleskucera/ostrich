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
    X_world = wp.transform_multiply(X_track, X_local)

    joint_X_p[joint_idx] = X_world


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


class TrackControlSimulator(InteractiveSimulator):
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
        self.track_global_u = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.track_velocity = wp.array([2.0], dtype=wp.float32, device=self.model.device)

        # Identify and upload track joint data to GPU once
        is_track = self.model.is_track_joint.numpy()
        self.track_joint_indices = wp.array(
            np.where(is_track == 1)[0], dtype=int, device=self.model.device
        )
        self.track_u_offsets = wp.array(
            self.model.track_u_offset.numpy()[self.track_joint_indices.numpy()],
            dtype=wp.float32,
            device=self.model.device,
        )

    def build_model(self) -> newton.Model:
        r_rear, r_front, dist = 1.5, 0.8, 5.0
        self.track_helper = Track2D(r_rear, r_front, dist)

        verts, indices = generate_track_data(r_rear, r_front, dist, tube_radius=0.05)
        mesh_obj = newton.Mesh(vertices=verts, indices=indices)

        q_tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5)
        p_track_origin = wp.vec3(0.0, 0.0, 1.0)
        self.X_track = wp.transform(p_track_origin, q_tilt)

        # Static Base
        base = self.builder.add_body(key="base", mass=0.0)
        base_joint = self.builder.add_joint(newton.JointType.FIXED, -1, base)
        # We will add base_joint to articulation later

        visual_cfg = newton.ModelBuilder.ShapeConfig(is_visible=True, has_shape_collision=False)
        self.builder.add_shape_mesh(
            body=base, mesh=mesh_obj, xform=self.X_track, cfg=visual_cfg, key="track_visual"
        )

        # Add Track Elements using the new builder method
        box_shape = newton.ModelBuilder.ShapeConfig(
            is_visible=True,
            density=100.0,
        )

        track_joints = self.builder.add_track(
            parent_body=base,
            num_elements=14,
            element_radius=0.1,
            element_half_width=0.4,
            shape_config=box_shape,
            track_helper=self.track_helper,
            track_center=p_track_origin,
            track_rotation=q_tilt,
        )

        self.builder.add_articulation([base_joint] + track_joints)

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

    @override
    def control_policy(self, current_state: newton.State):
        # Update scalar state on GPU
        wp.launch(
            kernel=integrate_track_kernel,
            dim=1,
            inputs=[self.track_global_u, self.track_velocity, self.clock.dt],
            device=self.model.device,
        )

        num_joints = len(self.track_joint_indices)
        if num_joints == 0:
            return

        # Execute update entirely on GPU
        wp.launch(
            kernel=update_track_joints_kernel,
            dim=num_joints,
            inputs=[
                self.track_joint_indices,
                self.track_u_offsets,
                self.track_global_u,
                self.X_track,
                # Pass geometry constants
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
def track_chain_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = TrackControlSimulator(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    track_chain_example()
