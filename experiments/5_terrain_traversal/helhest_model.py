"""Helhest robot model builder — self-contained copy for terrain traversal."""

import pathlib

import newton
import numpy as np
import openmesh
import warp as wp
from axion import JointMode

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


class HelhestConfig:
    """Configuration for the Helhest robot model."""

    # Chassis
    CHASSIS_SIZE = [0.26, 0.6, 0.18]
    CHASSIS_MASS = 85.0
    CHASSIS_OFFSET = wp.vec3(-0.047, 0.0, 0.0)

    # Wheels
    WHEEL_RADIUS = 0.36
    WHEEL_WIDTH = 0.11
    WHEEL_MASS = 5.5
    WHEEL_I = wp.mat33(0.20045, 0.0, 0.0, 0.0, 0.20045, 0.0, 0.0, 0.0, 0.3888)
    WHEEL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)

    # Wheel Positions
    LEFT_WHEEL_POS = wp.vec3(0.0, 0.36, 0.0)
    RIGHT_WHEEL_POS = wp.vec3(0.0, -0.36, 0.0)
    REAR_WHEEL_POS = wp.vec3(-0.697, 0.0, 0.0)

    # Joint Control
    TARGET_KE = 150.0
    TARGET_KD = 0.0

    # Fixed Components Configuration
    # Format: name: (position, size, mass)
    FIXED_COMPONENTS = {
        "battery": (
            wp.vec3(-0.302, 0.165, 0.0),
            [0.25, 0.1, 0.19],
            2.0,
        ),
        "left_motor": (
            wp.vec3(-0.09, 0.14, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
        ),
        "right_motor": (
            wp.vec3(-0.09, -0.14, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
        ),
        "rear_motor": (
            wp.vec3(-0.22, -0.04, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
        ),
        "left_wheel_holder": (
            wp.vec3(-0.477, 0.095, 0.0),
            [0.6, 0.04, 0.18],
            3.0,
        ),
        "right_wheel_holder": (
            wp.vec3(-0.477, -0.095, 0.0),
            [0.6, 0.04, 0.18],
            3.0,
        ),
    }


def _load_wheel_mesh():
    """Loads and prepares the wheel mesh."""
    wheel_m = openmesh.read_trimesh(str(ASSETS_DIR / "helhest" / "wheel2.obj"))
    scale = np.array([0.78, 0.78, 0.78])
    mesh_points = np.array(wheel_m.points()) * scale
    mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
    return newton.Mesh(mesh_points, mesh_indices)


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    """Adds the chassis link with all fixed component shapes attached directly."""
    chassis = builder.add_link(
        xform=xform,
        label="chassis",
    )

    chassis_volume = (
        HelhestConfig.CHASSIS_SIZE[0]
        * HelhestConfig.CHASSIS_SIZE[1]
        * HelhestConfig.CHASSIS_SIZE[2]
    )
    builder.add_shape_box(
        body=chassis,
        xform=wp.transform(HelhestConfig.CHASSIS_OFFSET, wp.quat_identity()),
        hx=HelhestConfig.CHASSIS_SIZE[0] / 2.0,
        hy=HelhestConfig.CHASSIS_SIZE[1] / 2.0,
        hz=HelhestConfig.CHASSIS_SIZE[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=HelhestConfig.CHASSIS_MASS / chassis_volume,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )

    for name, (pos, size, mass) in HelhestConfig.FIXED_COMPONENTS.items():
        volume = size[0] * size[1] * size[2]
        builder.add_shape_box(
            body=chassis,
            xform=wp.transform(pos, wp.quat_identity()),
            hx=size[0] / 2.0,
            hy=size[1] / 2.0,
            hz=size[2] / 2.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=mass / volume,
                is_visible=is_visible,
                collision_group=0,
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
    ke: float = None,
    kd: float = None,
    kf: float = None,
) -> int:
    """Adds a wheel link, shapes, and returns the link index."""
    pos_world = wp.transform_point(parent_xform, pos_local)
    rot_world = parent_xform.q

    wheel_link = builder.add_link(
        xform=wp.transform(pos_world, rot_world),
        label=name,
        mass=HelhestConfig.WHEEL_MASS,
        inertia=HelhestConfig.WHEEL_I,
        com=None,
    )

    # Visual Mesh
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

    # Collision Shape
    collision_cfg_kwargs = {
        "density": 0.0,
        "is_visible": False,
        "collision_group": -1,
        "mu": mu,
        "mu_rolling": 0.7,
    }
    if ke is not None:
        collision_cfg_kwargs["ke"] = ke
    if kd is not None:
        collision_cfg_kwargs["kd"] = kd
    if kf is not None:
        collision_cfg_kwargs["kf"] = kf

    builder.add_shape_cylinder(
        body=wheel_link,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), HelhestConfig.WHEEL_ROT),
        radius=HelhestConfig.WHEEL_RADIUS,
        half_height=HelhestConfig.WHEEL_WIDTH / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(**collision_cfg_kwargs),
    )
    return wheel_link


def create_helhest_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
    control_mode: str = "position",
    k_p: float = 50.0,
    k_d: float = 0.1,
    friction_left_right: float = 0.7,
    friction_rear: float = 0.4,
    ke: float = None,
    kd: float = None,
    kf: float = None,
):
    """Creates a Helhest robot model.

    Args:
        builder: The model builder to add the robot to.
        xform: The world transform of the robot base.
        is_visible: Whether to enable visual shapes.
        control_mode: Actuation mode, either "velocity" or "position".
        k_p: Proportional gain (target_ke).
        k_d: Derivative gain (target_kd).
        friction_left_right: Friction coefficient for front wheels.
        friction_rear: Friction coefficient for the rear wheel.
    """

    wheel_mesh_render = _load_wheel_mesh()

    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    # 2. Wheels
    left_wheel = _add_wheel(
        builder, xform, "left_wheel", HelhestConfig.LEFT_WHEEL_POS,
        friction_left_right, wheel_mesh_render, is_visible, ke=ke, kd=kd, kf=kf,
    )
    right_wheel = _add_wheel(
        builder, xform, "right_wheel", HelhestConfig.RIGHT_WHEEL_POS,
        friction_left_right, wheel_mesh_render, is_visible, ke=ke, kd=kd, kf=kf,
    )
    rear_wheel = _add_wheel(
        builder, xform, "rear_wheel", HelhestConfig.REAR_WHEEL_POS,
        friction_rear, wheel_mesh_render, is_visible, ke=ke, kd=kd, kf=kf,
    )

    # 3. Wheel Joints
    Y_AXIS = (0.0, 1.0, 0.0)
    mode = JointMode.TARGET_VELOCITY if control_mode == "velocity" else JointMode.TARGET_POSITION

    j_left = builder.add_joint_revolute(
        parent=chassis, child=left_wheel,
        parent_xform=wp.transform(HelhestConfig.LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(), axis=Y_AXIS,
        target_ke=k_p, target_kd=k_d, label="left_wheel_j",
        custom_attributes={"joint_dof_mode": [mode]},
    )
    j_right = builder.add_joint_revolute(
        parent=chassis, child=right_wheel,
        parent_xform=wp.transform(HelhestConfig.RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(), axis=Y_AXIS,
        target_ke=k_p, target_kd=k_d, label="right_wheel_j",
        custom_attributes={"joint_dof_mode": [mode]},
    )
    j_rear = builder.add_joint_revolute(
        parent=chassis, child=rear_wheel,
        parent_xform=wp.transform(HelhestConfig.REAR_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(), axis=Y_AXIS,
        target_ke=k_p, target_kd=k_d, label="rear_wheel_j",
        custom_attributes={"joint_dof_mode": [mode]},
    )

    # 4. Articulation
    builder.add_articulation([j_base, j_left, j_right, j_rear], label="helhest")

    return chassis, [left_wheel]
