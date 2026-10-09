"""Clearpath Husky A200 model for ostrich.

Built the way examples/helhest_junior/common.py builds the Helhest Junior: a
chassis link carrying box shapes, wheel links carrying cylinder shapes on
revolute joints, anisotropic wheel friction, and one articulation. Every
dimension, mass and inertia comes from Clearpath's husky_description; see
examples/assets/husky.urdf for the traceable copy and examples/husky/README.md
for the per-value provenance.
"""

import newton
import warp as wp
from ostrich import JointMode


class HuskyConfig:
    """Configuration for the Clearpath Husky A200 model.

    Axes: X = longitudinal (front = +X), Y = lateral (left wheel = +Y), Z = up.

    Body frame: the ORIGIN IS THE WHEEL-AXLE PLANE, not the URDF's `base_link`.
    The URDF hangs the wheels `wheel_vertical_offset` = 0.03282 m ABOVE base_link
    (the center of the bottom plate); shifting the frame down by that offset puts
    all four axles at z = 0, which is the convention the Helhest models use and
    what lets the campaign loader spawn at `ground + WHEEL_RADIUS + 0.02`.
    Every chassis z below is therefore the URDF value minus 0.03282.
    """

    # --- Wheels (husky_description/urdf/husky.urdf.xacro + wheel.urdf.xacro) ---
    WHEEL_RADIUS = 0.1651
    WHEEL_WIDTH = 0.1143
    WHEEL_MASS = 2.637
    # Straight from the husky_wheel macro's <inertia>, and already in the frame
    # we need: the wheel spins about body-local Y, and iyy = 0.04411 is the
    # axial term. (The measured wheel is rim-heavy; a uniform solid cylinder of
    # the same mass and radius would give 0.0359 axial / 0.0208 transverse.)
    WHEEL_I = wp.mat33(
        0.02467, 0.0, 0.0,
        0.0, 0.04411, 0.0,
        0.0, 0.0, 0.02467,
    )
    # The shape's own axis is +Z; rotating +90 deg about X lays it along body Y.
    WHEEL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)

    # Wheel mounts: (+-wheelbase/2, +-track/2, 0) with wheelbase 0.5120 and
    # track 0.5708. Ground clearance = WHEEL_RADIUS - 0.03282 = 0.13228 m.
    WHEELBASE = 0.5120
    TRACK = 0.5708
    FRONT_LEFT_WHEEL_POS = wp.vec3(0.2560, 0.2854, 0.0)
    FRONT_RIGHT_WHEEL_POS = wp.vec3(0.2560, -0.2854, 0.0)
    REAR_LEFT_WHEEL_POS = wp.vec3(-0.2560, 0.2854, 0.0)
    REAR_RIGHT_WHEEL_POS = wp.vec3(-0.2560, -0.2854, 0.0)

    # --- Chassis ---
    # The two stacked boxes that husky.urdf.xacro gives as base_link's collision
    # geometry, from base_x_size 0.98740, base_y_size 0.57090, base_z_size 0.24750:
    #   lower  base_x_size     x base_y_size x base_z_size/2        at base_z_size/4
    #   upper  base_x_size*4/5 x base_y_size x base_z_size/2 - 0.02 at base_z_size*3/4 - 0.01
    # minus the 0.03282 frame shift. Format: name: (center_pos, size [x, y, z]).
    CHASSIS_BOXES = {
        "lower": (wp.vec3(0.0, 0.0, 0.029055), [0.987400, 0.570900, 0.123750]),
        "upper": (wp.vec3(0.0, 0.0, 0.142805), [0.789920, 0.570900, 0.103750]),
    }
    # Mass properties are NOT derived from those boxes: husky.urdf.xacro carries a
    # measured tensor on `inertial_link`, so the shapes are added at density 0 and
    # the link is given these directly. The CoM is 8.5 cm off centre in -Y (the
    # battery); it is kept because it is what the real vehicle does.
    CHASSIS_MASS = 46.034
    CHASSIS_COM = wp.vec3(-0.00065, -0.085, 0.02918)  # 0.062 - 0.03282
    CHASSIS_I = wp.mat33(
        0.6022, -0.02364, -0.1197,
        -0.02364, 1.7386, -0.001544,
        -0.1197, -0.001544, 2.0296,
    )

    # Joint dof indices in the finalized model: the free base joint takes 6, then
    # the four revolutes in the order create_husky_model adds them.
    FRONT_LEFT_DOF = 6
    FRONT_RIGHT_DOF = 7
    REAR_LEFT_DOF = 8
    REAR_RIGHT_DOF = 9


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    """Adds the chassis link as the URDF's two stacked collision boxes."""
    chassis = builder.add_link(
        xform=xform,
        label="chassis",
        mass=HuskyConfig.CHASSIS_MASS,
        inertia=HuskyConfig.CHASSIS_I,
        com=HuskyConfig.CHASSIS_COM,
    )

    for name, (pos, size) in HuskyConfig.CHASSIS_BOXES.items():
        builder.add_shape_box(
            body=chassis,
            xform=wp.transform(pos, wp.quat_identity()),
            hx=size[0] / 2.0,
            hy=size[1] / 2.0,
            hz=size[2] / 2.0,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,  # mass/inertia are the measured ones on the link
                is_visible=is_visible,
                collision_group=-1,
            ),
            label=name,
        )

    return chassis


def _add_wheel(
    builder: newton.ModelBuilder,
    parent_xform: wp.transform,
    name: str,
    pos_local: wp.vec3,
    mu: float,
    is_visible: bool,
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu_rolling: float = 0.7,
    mu_long: float = None,
) -> int:
    """Adds a wheel link and its cylinder shape, and returns the link index.

    Friction model, identical to the Helhest's: the wheel spins about body-local
    Y, so the friction frame is anchored to that axis. `mu` is the lateral
    coefficient (resists skid along the spin axis); `mu_long` is the
    longitudinal coefficient (rolling direction, perpendicular to the spin axis
    in the contact plane). If `mu_long` is None, friction is isotropic at `mu`.
    """
    pos_world = wp.transform_point(parent_xform, pos_local)

    wheel_link = builder.add_link(
        xform=wp.transform(pos_world, parent_xform.q),
        label=name,
        mass=HuskyConfig.WHEEL_MASS,
        inertia=HuskyConfig.WHEEL_I,
        com=None,
    )

    cfg_kwargs = {
        "density": 0.0,
        "is_visible": is_visible,
        "collision_group": -1,
        "mu": mu,
        "mu_rolling": mu_rolling,
    }
    if ke is not None:
        cfg_kwargs["ke"] = ke
    if kd is not None:
        cfg_kwargs["kd"] = kd
    if kf is not None:
        cfg_kwargs["kf"] = kf

    shape_custom_attrs = {}
    if mu_long is not None:
        shape_custom_attrs["friction_axis_local"] = wp.vec3(0.0, 1.0, 0.0)
        shape_custom_attrs["mu_perp"] = mu_long

    # Cylinder, not capsule, for the same reason the Helhest Junior uses one: a
    # capsule's hemispherical caps bulge WHEEL_RADIUS past each rim and would
    # catch on curbs the real tread never touches. On the Husky that penalty is
    # smaller (r = 0.165 m, not 0.35) but the geometry argument is the same.
    builder.add_shape_cylinder(
        body=wheel_link,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), HuskyConfig.WHEEL_ROT),
        radius=HuskyConfig.WHEEL_RADIUS,
        half_height=HuskyConfig.WHEEL_WIDTH / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(**cfg_kwargs),
        custom_attributes=shape_custom_attrs if shape_custom_attrs else None,
    )
    return wheel_link


def create_husky_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
    control_mode: str = "velocity",
    k_p: float = 250.0,
    k_d: float = 0.0,
    friction: float = 0.5,
    friction_long: float = 0.9,
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu_rolling: float = 0.7,
):
    """Creates a Clearpath Husky A200 model: chassis + 4 driven wheels, skid-steer.

    Args:
        builder: The model builder to add the robot to.
        xform: The world transform of the robot base. Its origin is the
            wheel-axle plane (see HuskyConfig), so a spawn at
            `ground + HuskyConfig.WHEEL_RADIUS` rests the wheels on the ground.
        is_visible: Whether the shapes are drawn.
        control_mode: Actuation mode, either "velocity" or "position".
        k_p: Proportional gain (target_ke). In velocity mode this is the only
            gain the engine uses: the control constraint's compliance is
            1 / (dt * target_ke).
        k_d: Derivative gain (target_kd); used in position mode only.
        friction: Lateral (skid, along the axle) friction coefficient, all four
            wheels.
        friction_long: Longitudinal (rolling-direction) friction coefficient.
            None disables anisotropic friction (isotropic at `friction`).
        mu_rolling: Rolling resistance coefficient.

    Returns:
        (chassis link index, [front_left, front_right, rear_left, rear_right]).
    """

    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    # 2. Wheels, in the order that fixes the dof indices in HuskyConfig.
    wheels = []
    for name, pos in (
        ("front_left_wheel", HuskyConfig.FRONT_LEFT_WHEEL_POS),
        ("front_right_wheel", HuskyConfig.FRONT_RIGHT_WHEEL_POS),
        ("rear_left_wheel", HuskyConfig.REAR_LEFT_WHEEL_POS),
        ("rear_right_wheel", HuskyConfig.REAR_RIGHT_WHEEL_POS),
    ):
        wheels.append(
            _add_wheel(
                builder,
                xform,
                name,
                pos,
                friction,
                is_visible,
                ke=ke,
                kd=kd,
                kf=kf,
                mu_rolling=mu_rolling,
                mu_long=friction_long,
            )
        )

    # 3. Wheel joints
    Y_AXIS = (0.0, 1.0, 0.0)
    mode = JointMode.TARGET_VELOCITY if control_mode == "velocity" else JointMode.TARGET_POSITION

    joints = []
    for wheel, name, pos in zip(
        wheels,
        ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"),
        (
            HuskyConfig.FRONT_LEFT_WHEEL_POS,
            HuskyConfig.FRONT_RIGHT_WHEEL_POS,
            HuskyConfig.REAR_LEFT_WHEEL_POS,
            HuskyConfig.REAR_RIGHT_WHEEL_POS,
        ),
    ):
        joints.append(
            builder.add_joint_revolute(
                parent=chassis,
                child=wheel,
                parent_xform=wp.transform(pos, wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=Y_AXIS,
                target_ke=k_p,
                target_kd=k_d,
                label=f"{name}_j",
                custom_attributes={
                    "joint_dof_mode": [mode],
                },
            )
        )

    # 4. Articulation
    builder.add_articulation([j_base] + joints, label="husky")

    return chassis, wheels
