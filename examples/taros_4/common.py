import pathlib
from typing import Dict
from typing import List
from typing import Tuple

import newton
import numpy as np
import openmesh
import warp as wp
from axion import JointMode

ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class Taros4Config:
    """Configuration for the Taros-4 robot model (Realistic V2 Specs)."""

    # Chassis
    # Dimensions (L x W x H) = 2.74 m x 1.77 m x 2.04 m
    CHASSIS_SIZE = [2.74, 1.5, 1.0]  # Reduced height for main body box to look reasonable
    CHASSIS_MASS = 1200.0
    CHASSIS_OFFSET = wp.vec3(0.0, 0.0, 0.2)
    # Approx Inertia for box of this size/mass
    # Ixx = m/12 * (w^2 + h^2)
    # Iyy = m/12 * (l^2 + h^2)
    # Izz = m/12 * (l^2 + w^2)
    CHASSIS_I = wp.mat33(600.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 1000.0)

    # Wheels
    # Approx 0.9m diameter -> 0.45m radius
    WHEEL_RADIUS = 0.45
    WHEEL_WIDTH = 0.3
    WHEEL_MASS = 50.0
    # Wheel Inertia (Cylinder)
    # I_axial = 0.5 * m * r^2
    # I_radial = m/12 * (3*r^2 + h^2)
    WHEEL_I = wp.mat33(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0)
    WHEEL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.0 * wp.pi / 2.0)
    # Independent rotation for the visual mesh
    WHEEL_MESH_ROT_LEFT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -1.0 * wp.pi / 2.0)
    WHEEL_MESH_ROT_RIGHT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.0 * wp.pi / 2.0)

    # Wheel Positions (Centered)
    # Wheelbase approx 2.0m -> x = +/- 1.0
    # Track approx 1.6m -> y = +/- 0.8
    FRONT_LEFT_WHEEL_POS = wp.vec3(1.0, 0.9, -0.3)
    FRONT_RIGHT_WHEEL_POS = wp.vec3(1.0, -0.9, -0.3)
    REAR_LEFT_WHEEL_POS = wp.vec3(-1.0, 0.9, -0.3)
    REAR_RIGHT_WHEEL_POS = wp.vec3(-1.0, -0.9, -0.3)

    # Joint Control
    TARGET_KE = 2000.0  # Stiffer for heavier robot
    TARGET_KD = 500.0


def _load_wheel_mesh():
    """Loads the wheel mesh and rescales it to exactly match the collision cylinder.

    offroad_wheel.obj is authored X-axial (radius along Y/Z). We measure the mesh
    bbox and pick a per-axis scale so the rendered mesh's extent matches
    ``WHEEL_WIDTH`` along the axial axis and ``2 * WHEEL_RADIUS`` radially.
    """
    try:
        wheel_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("taros-4/offroad_wheel.obj")))
        pts = np.array(wheel_m.points())
        size = pts.max(axis=0) - pts.min(axis=0)
        scale = np.array(
            [
                Taros4Config.WHEEL_WIDTH / size[0],
                (2.0 * Taros4Config.WHEEL_RADIUS) / size[1],
                (2.0 * Taros4Config.WHEEL_RADIUS) / size[2],
            ]
        )
        mesh_points = pts * scale
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        return newton.Mesh(mesh_points, mesh_indices)
    except Exception:
        print("Warning: Could not load wheel mesh, using fallback.")
        return None


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    """Adds the chassis link and shape."""
    chassis = builder.add_link(
        xform=xform,
        label="chassis",
        mass=Taros4Config.CHASSIS_MASS,
        inertia=Taros4Config.CHASSIS_I,
        com=Taros4Config.CHASSIS_OFFSET,
    )

    builder.add_shape_box(
        body=chassis,
        xform=wp.transform(Taros4Config.CHASSIS_OFFSET, wp.quat_identity()),
        hx=Taros4Config.CHASSIS_SIZE[0] / 2.0,
        hy=Taros4Config.CHASSIS_SIZE[1] / 2.0,
        hz=Taros4Config.CHASSIS_SIZE[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )
    return chassis


def _add_wheel(
    builder: newton.ModelBuilder,
    parent_xform: wp.transform,
    name: str,
    pos_local: wp.vec3,
    mu: float,
    wheel_mesh: newton.Mesh,
    is_visible: bool,
    mesh_rotation: wp.quat = wp.quat_identity(),
) -> int:
    """Adds a wheel link, shapes, and returns the link index."""
    pos_world = wp.transform_point(parent_xform, pos_local)
    rot_world = parent_xform.q

    wheel_link = builder.add_link(
        xform=wp.transform(pos_world, rot_world),
        label=name,
        mass=Taros4Config.WHEEL_MASS,
        inertia=Taros4Config.WHEEL_I,
        com=None,
    )

    # Visual Mesh
    if wheel_mesh and is_visible:
        builder.add_shape_mesh(
            body=wheel_link,
            mesh=wheel_mesh,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), mesh_rotation),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                collision_group=0,
                is_visible=is_visible,
            ),
        )
    else:
        # Fallback visual if mesh missing
        if is_visible:
            builder.add_shape_cylinder(
                body=wheel_link,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), Taros4Config.WHEEL_ROT),
                radius=Taros4Config.WHEEL_RADIUS,
                half_height=Taros4Config.WHEEL_WIDTH / 2.0,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    is_visible=True,
                    collision_group=0,
                ),
            )

    # Collision Shape
    builder.add_shape_cylinder(
        body=wheel_link,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), Taros4Config.WHEEL_ROT),
        radius=Taros4Config.WHEEL_RADIUS,
        half_height=Taros4Config.WHEEL_WIDTH / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=False,
            collision_group=-1,
            mu=mu,
        ),
    )
    return wheel_link


def create_taros4_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
    control_mode: str = "velocity",
    k_p: float = 1000.0,
    k_d: float = 0.0,
    friction: float = 0.5,
):
    """
    Creates a Taros-4 robot model using physical parameters from the URDF
    and visual assets from the examples.

    Args:
        builder: The model builder to add the robot to.
        xform: The world transform of the robot base.
        is_visible: Whether to enable visual shapes.
        control_mode: Actuation mode, either "velocity" or "position".
        k_p: Proportional gain (target_ke).
        k_d: Derivative gain (target_kd).
        friction: Friction coefficient for the wheels.
    """

    wheel_mesh_render = _load_wheel_mesh()

    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    # 2. Wheels
    front_left_wheel = _add_wheel(
        builder,
        xform,
        "front_left_wheel",
        Taros4Config.FRONT_LEFT_WHEEL_POS,
        friction,
        wheel_mesh_render,
        is_visible,
        mesh_rotation=Taros4Config.WHEEL_MESH_ROT_LEFT,
    )
    front_right_wheel = _add_wheel(
        builder,
        xform,
        "front_right_wheel",
        Taros4Config.FRONT_RIGHT_WHEEL_POS,
        friction,
        wheel_mesh_render,
        is_visible,
        mesh_rotation=Taros4Config.WHEEL_MESH_ROT_RIGHT,
    )
    rear_left_wheel = _add_wheel(
        builder,
        xform,
        "rear_left_wheel",
        Taros4Config.REAR_LEFT_WHEEL_POS,
        friction,
        wheel_mesh_render,
        is_visible,
        mesh_rotation=Taros4Config.WHEEL_MESH_ROT_LEFT,
    )
    rear_right_wheel = _add_wheel(
        builder,
        xform,
        "rear_right_wheel",
        Taros4Config.REAR_RIGHT_WHEEL_POS,
        friction,
        wheel_mesh_render,
        is_visible,
        mesh_rotation=Taros4Config.WHEEL_MESH_ROT_RIGHT,
    )

    # 3. Wheel Joints
    Y_AXIS = (0.0, 1.0, 0.0)

    mode = JointMode.TARGET_VELOCITY if control_mode == "velocity" else JointMode.TARGET_POSITION

    j_front_left = builder.add_joint_revolute(
        parent=chassis,
        child=front_left_wheel,
        parent_xform=wp.transform(Taros4Config.FRONT_LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="front_left_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    j_front_right = builder.add_joint_revolute(
        parent=chassis,
        child=front_right_wheel,
        parent_xform=wp.transform(Taros4Config.FRONT_RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="front_right_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    j_rear_left = builder.add_joint_revolute(
        parent=chassis,
        child=rear_left_wheel,
        parent_xform=wp.transform(Taros4Config.REAR_LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="rear_left_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    j_rear_right = builder.add_joint_revolute(
        parent=chassis,
        child=rear_right_wheel,
        parent_xform=wp.transform(Taros4Config.REAR_RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="rear_right_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    # 4. Articulation
    builder.add_articulation(
        [j_base, j_front_left, j_front_right, j_rear_left, j_rear_right],
        label="taros-4",
    )

    return chassis, [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]

    # 4. Articulation
    builder.add_articulation(
        [j_base, j_front_left, j_front_right, j_rear_left, j_rear_right],
        label="taros-4",
    )

    return chassis, [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
