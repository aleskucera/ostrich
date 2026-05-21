import pathlib
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import newton
import numpy as np
import warp as wp
from axion import JointMode

ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


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
        # Handle case where circles are too close or r difference is large
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


class MarvConfig:
    """Configuration for the Marv robot model based on marv_flippers.xml"""

    # Chassis
    # XML: size="0.315 0.185 0.1" (half-extents) -> Full: 0.63 x 0.37 x 0.2
    CHASSIS_DIMS = [0.63, 0.37, 0.2]
    CHASSIS_MASS = 60.0
    CHASSIS_POS_OFFSET = wp.vec3(0.0, 0.0, 0.0)

    # Approximated Box Inertia for Chassis (m=40, size=0.63x0.37x0.2)
    # I = m/12 * (d1^2 + d2^2)
    _Ix = (40.0 / 12.0) * (0.37**2 + 0.2**2)
    _Iy = (40.0 / 12.0) * (0.63**2 + 0.2**2)
    _Iz = (40.0 / 12.0) * (0.63**2 + 0.37**2)
    CHASSIS_I = wp.mat33(_Ix, 0.0, 0.0, 0.0, _Iy, 0.0, 0.0, 0.0, _Iz)

    # Flipper & Wheel Configuration
    # XML: Flipper pos relative to chassis center
    FLIPPER_OFFSETS = {
        "FL": wp.vec3(0.25, 0.25, -0.095),
        "FR": wp.vec3(0.25, -0.25, -0.095),
        "RL": wp.vec3(-0.25, 0.25, -0.095),
        "RR": wp.vec3(-0.25, -0.25, -0.095),
    }

    # XML: Flipper Rotation euler="1.57 0 0" (90 deg around X)
    FLIPPER_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.57)

    # Track parameters approximating the wheel setup
    # r_rear ~ 0.12 (wheel 1)
    # r_front ~ 0.095 (wheel 4)
    # dist ~ 0.29 (dist between wheel 1 and 4)
    TRACK_R_REAR = 0.12
    TRACK_R_FRONT = 0.095
    TRACK_DIST = 0.29

    # Track Elements
    TRACK_NUM_ELEMENTS = 12
    TRACK_ELEM_RADIUS = 0.03
    TRACK_ELEM_WIDTH = 0.1  # Width of the track pad

    # Control Gains
    FLIPPER_KP = 300.0
    FLIPPER_KD = 50.0
    FLIPPER_KV = 4000.0


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    chassis = builder.add_link(
        xform=xform,
        label="chassis",
        mass=MarvConfig.CHASSIS_MASS,
        inertia=MarvConfig.CHASSIS_I,
        com=MarvConfig.CHASSIS_POS_OFFSET,
    )

    builder.add_shape_box(
        body=chassis,
        xform=wp.transform_identity(),
        hx=MarvConfig.CHASSIS_DIMS[0] / 2.0,
        hy=MarvConfig.CHASSIS_DIMS[1] / 2.0,
        hz=MarvConfig.CHASSIS_DIMS[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )
    return chassis


def _create_tracked_flipper_leg(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_xform: wp.transform,
    name_prefix: str,
    side_code: str,
    is_visible: bool,
) -> Tuple[int, List[int], Track2D, wp.transform]:
    """Creates one flipper arm with a track."""

    flipper_offset = MarvConfig.FLIPPER_OFFSETS[side_code]

    flipper_pos_world = wp.transform_point(parent_xform, flipper_offset)
    flipper_rot_world = wp.mul(parent_xform.q, MarvConfig.FLIPPER_ROT)

    # 1. Flipper Arm (Dynamic)
    flipper_link = builder.add_link(
        xform=wp.transform(flipper_pos_world, flipper_rot_world),
        label=f"{name_prefix}_arm",
        mass=5.0,
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
    )

    # Flipper Joint
    j_flipper = builder.add_joint_revolute(
        parent=parent_body,
        child=flipper_link,
        parent_xform=wp.transform(flipper_offset, MarvConfig.FLIPPER_ROT),
        child_xform=wp.transform_identity(),
        axis=(0.0, 0.0, 1.0),
        target_ke=MarvConfig.FLIPPER_KV,
        target_kd=0.0,
        label=f"{name_prefix}_joint",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
    )

    # 2. Track
    track_helper = Track2D(MarvConfig.TRACK_R_REAR, MarvConfig.TRACK_R_FRONT, MarvConfig.TRACK_DIST)

    # Track visual
    verts, indices = generate_track_data(
        MarvConfig.TRACK_R_REAR, MarvConfig.TRACK_R_FRONT, MarvConfig.TRACK_DIST, tube_radius=0.02
    )
    mesh_obj = newton.Mesh(vertices=verts, indices=indices)

    # Track geometry is defined in XY plane of the Track2D.
    # We want it to be upright?
    # Marv Flipper Arm: Z axis is the hinge axis.
    # The arm extends along X?
    # Original Marv wheel positions are along X (0.0 to 0.29).
    # Track2D goes from (0,0) to (dist, 0) along X.
    # So we align Track2D X with Flipper X.
    # We need to offset it so that the wheels align with the Track2D shape.
    # Wheel 1 is at 0.0. Track2D starts with rear circle at 0.0.
    # So no extra offset needed for X, unless we want to center it differently.

    # Visual
    builder.add_shape_mesh(
        body=flipper_link,
        mesh=mesh_obj,
        xform=wp.transform_identity(),  # Aligned with flipper link
        cfg=newton.ModelBuilder.ShapeConfig(is_visible=True, has_shape_collision=False),
        label=f"{name_prefix}_visual",
    )

    box_shape = newton.ModelBuilder.ShapeConfig(is_visible=True, density=100.0, mu=0.05)

    # We need to pass the parent_world_xform of the FLIPPER LINK, not the chassis.
    # This is used by add_track to set initial positions of track elements.
    flipper_xform_curr = wp.transform(flipper_pos_world, flipper_rot_world)

    track_indices = builder.add_track(
        parent_body=flipper_link,
        num_elements=MarvConfig.TRACK_NUM_ELEMENTS,
        element_radius=MarvConfig.TRACK_ELEM_RADIUS,
        element_half_width=MarvConfig.TRACK_ELEM_WIDTH,
        shape_config=box_shape,
        track_helper=track_helper,
        track_center=wp.vec3(0.0, 0.0, 0.0),  # Relative to flipper link
        track_rotation=wp.quat_identity(),  # Aligned with flipper link
        parent_world_xform=flipper_xform_curr,
        name_prefix=f"{name_prefix}_track",
    )

    # Return info
    return j_flipper, track_indices, track_helper, wp.transform_identity()


def create_marv_tracked_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    all_joints = [j_base]
    track_info = {}

    legs = [
        ("flipper_front_left", "FL"),
        ("flipper_front_right", "FR"),
        ("flipper_rear_left", "RL"),
        ("flipper_rear_right", "RR"),
    ]

    for name, code in legs:
        j_flip, t_indices, t_helper, t_xform = _create_tracked_flipper_leg(
            builder, chassis, xform, name, code, is_visible
        )
        all_joints.append(j_flip)
        all_joints.extend(t_indices)  # Track indices are also joint indices

        track_info[code] = {
            "indices": t_indices,
            "helper": t_helper,
            "xform": t_xform,  # Local to flipper
            "joint_idx": j_flip,  # Flipper joint index (if needed)
        }

    # 3. Articulation
    builder.add_articulation(all_joints, label="marv")

    return chassis, track_info
