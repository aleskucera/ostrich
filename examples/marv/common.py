import pathlib
from typing import List

import newton
import numpy as np
import warp as wp
from ostrich import JointMode

ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


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

    # Wheels inside Flipper (positions relative to flipper origin)
    WHEEL_CONFIGS = [
        {"pos": 0.0, "radius": 0.12},  # wheel_1
        {"pos": 0.09666, "radius": 0.11166},  # wheel_2
        {"pos": 0.19333, "radius": 0.10333},  # wheel_3
        {"pos": 0.29, "radius": 0.095},  # wheel_4
    ]

    WHEEL_MASS = 0.5
    # Simple Sphere Inertia: 2/5 * m * r^2
    _Iw = 0.4 * 1.0 * (0.1**2)
    WHEEL_I = wp.mat33(_Iw, 0.0, 0.0, 0.0, _Iw, 0.0, 0.0, 0.0, _Iw)

    # Control Gains
    FLIPPER_KP = 1000.0
    FLIPPER_KD = 500.0  # Estimated damping
    WHEEL_KV = 90.0  # Velocity gain, mujoco 90
    WHEEL_KP = 600.0  # Position gain, mujoco 600


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


def _create_flipper_leg(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_xform: wp.transform,
    name_prefix: str,
    side_code: str,  # ADDED: Explicit code (FL, FR, etc) to fix labelError
    is_visible: bool,
) -> List[int]:
    """Creates one flipper arm with 4 attached wheels."""

    # 1. Create Flipper Arm (The carrier)
    # Use explicit side_code for lookup
    flipper_offset = MarvConfig.FLIPPER_OFFSETS[side_code]

    flipper_pos_world = wp.transform_point(parent_xform, flipper_offset)
    flipper_rot_world = wp.mul(parent_xform.q, MarvConfig.FLIPPER_ROT)

    flipper_link = builder.add_link(
        xform=wp.transform(flipper_pos_world, flipper_rot_world),
        label=f"{name_prefix}_arm",
        mass=0.1,
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
    )

    # Flipper Joint (Hinge on Z relative to Flipper)
    j_flipper = builder.add_joint_revolute(
        parent=parent_body,
        child=flipper_link,
        parent_xform=wp.transform(flipper_offset, MarvConfig.FLIPPER_ROT),
        child_xform=wp.transform_identity(),
        axis=(0.0, 0.0, 1.0),
        target_ke=MarvConfig.FLIPPER_KP,
        target_kd=MarvConfig.FLIPPER_KD,
        label=f"{name_prefix}_joint",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_POSITION]},
    )

    created_joints = [j_flipper]

    # 2. Add Wheels to Flipper
    for i, w_cfg in enumerate(MarvConfig.WHEEL_CONFIGS):
        w_pos_local = wp.vec3(w_cfg["pos"], 0.0, 0.0)
        w_pos_world = wp.transform_point(
            wp.transform(flipper_pos_world, flipper_rot_world), w_pos_local
        )

        wheel_link = builder.add_link(
            xform=wp.transform(w_pos_world, flipper_rot_world),
            label=f"{name_prefix}_wheel_{i+1}",
            mass=MarvConfig.WHEEL_MASS,
            inertia=MarvConfig.WHEEL_I,
        )

        # Wheel Sphere Shape
        builder.add_shape_sphere(
            body=wheel_link,
            radius=w_cfg["radius"],
            cfg=newton.ModelBuilder.ShapeConfig(
                density=1000.0,
                is_visible=is_visible,
                collision_group=-1,
                mu=1.0,
            ),
        )

        # Wheel Joint
        j_wheel = builder.add_joint_revolute(
            parent=flipper_link,
            child=wheel_link,
            parent_xform=wp.transform(w_pos_local, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
            target_ke=MarvConfig.WHEEL_KP,
            target_kd=MarvConfig.WHEEL_KV,
            label=f"{name_prefix}_wheel_joint_{i+1}",
            custom_attributes={
                "joint_dof_mode": [JointMode.TARGET_POSITION],
            },
        )
        created_joints.append(j_wheel)

    return created_joints


def create_marv_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
):
    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    all_joints = [j_base]

    # 2. Flippers - Pass explicit side codes (FL, FR, RL, RR)
    all_joints.extend(
        _create_flipper_leg(builder, chassis, xform, "flipper_front_left", "FL", is_visible)
    )
    all_joints.extend(
        _create_flipper_leg(builder, chassis, xform, "flipper_front_right", "FR", is_visible)
    )
    all_joints.extend(
        _create_flipper_leg(builder, chassis, xform, "flipper_rear_left", "RL", is_visible)
    )
    all_joints.extend(
        _create_flipper_leg(builder, chassis, xform, "flipper_rear_right", "RR", is_visible)
    )

    # 3. Articulation
    builder.add_articulation(all_joints, label="marv")

    return chassis
