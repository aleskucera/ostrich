import pathlib

import newton
import numpy as np
import openmesh
import warp as wp
from axion import JointMode

# Reuses the shared examples/assets directory (same wheel mesh asset as helhest).
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class HelhestJuniorConfig:
    """Configuration for the Helhest Junior robot model.

    Smaller Helhest variant: same topology (chassis + 3 driven wheels) but a
    two-box chassis instead of helhest's box + fixed-component cluster.

    Axes: X = longitudinal (front = +X), Y = lateral (left wheel = +Y),
    Z = up. The front (left/right) wheel axis defines X = 0.
    """

    # Wheels
    WHEEL_RADIUS = 0.35  # 35 cm
    WHEEL_WIDTH = 0.10  # 10 cm
    WHEEL_MASS = 5.5  # kg, each
    # Solid-cylinder inertia for m=5.5, r=0.35, h=0.10:
    #   I_axial      = 1/2 · m · r²          = 0.336875
    #   I_transverse = 1/12 · m · (3r² + h²) = 0.173021
    WHEEL_I = wp.mat33(
        0.173021, 0.0, 0.0,
        0.0, 0.173021, 0.0,
        0.0, 0.0, 0.336875,
    )
    WHEEL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
    # Visual mesh scale ∝ radius (helhest used 0.78 for r=0.36 → 0.78·0.35/0.36).
    WHEEL_MESH_SCALE = 0.7583

    # Wheel Positions: front wheels at X = 0 (11 cm behind the front box's
    # front edge); track = 73 cm; rear wheel 75 cm behind the front wheels
    # (10 cm forward of the rear box's −0.85 back edge).
    LEFT_WHEEL_POS = wp.vec3(0.0, 0.365, 0.0)
    RIGHT_WHEEL_POS = wp.vec3(0.0, -0.365, 0.0)
    REAR_WHEEL_POS = wp.vec3(-0.75, 0.0, 0.0)

    # Joint Control
    TARGET_KE = 150.0
    TARGET_KD = 0.0

    # Chassis = two boxes rigidly fixed together (both shapes on the chassis
    # link). Total chassis mass = 89.7 kg (106.2 kg robot − 3 × 5.5 kg wheels).
    #
    # Box masses are NOT a guessed density ratio — they are solved from the
    # per-wheel scale measurement (front wheels 39.1 kg each, rear 28.0 kg)
    # with rear wheel at X = −0.75. Sizes are [world-X, world-Y, world-Z]:
    # the 0.48 longitudinal (X) side runs front↔back, the 0.56 / 0.24 side
    # is lateral (Y) — i.e. the boxes are oriented with hx/hy swapped vs the
    # raw "56 long / 48 wide" description.
    #   robot CoM  X = (28.0·−0.75) / 106.2          = −0.1977 m
    #   chassis CoM X = (−21.0 − 5.5·−0.75) / 89.7   = −0.1881 m
    #   −0.13·m1 − 0.61·m2 = 89.7·(−0.1881), m1+m2 = 89.7
    #   ⇒ m1 = 78.8375 kg, m2 = 10.8625 kg  (density ratio ≈ 3.11)
    # Format: name: (center_pos, size [x, y, z], mass_kg)
    CHASSIS_BOXES = {
        # X-extent 0.48; front edge 11 cm ahead of the front-wheel axis
        #   → center X = 0.11 − 0.48/2 = −0.13
        "front_box": (wp.vec3(-0.13, 0.0, 0.0), [0.48, 0.56, 0.20], 78.8375),
        # X-extent 0.48; back edge 10 cm behind the rear wheel (X = −0.85)
        #   → center X = −0.85 + 0.48/2 = −0.61 (flush with front box at −0.37)
        "rear_box": (wp.vec3(-0.61, 0.0, 0.0), [0.48, 0.24, 0.20], 10.8625),
    }


def _load_wheel_mesh():
    """Loads and prepares the wheel mesh."""
    wheel_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("helhest/wheel2.obj")))
    s = HelhestJuniorConfig.WHEEL_MESH_SCALE
    scale = np.array([s, s, s])
    mesh_points = np.array(wheel_m.points()) * scale
    mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
    return newton.Mesh(mesh_points, mesh_indices)


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    """Adds the chassis link as two rigidly-fixed boxes (front + rear)."""
    chassis = builder.add_link(
        xform=xform,
        label="chassis",
    )

    for name, (pos, size, mass) in HelhestJuniorConfig.CHASSIS_BOXES.items():
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
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu_rolling: float = 0.7,
) -> int:
    """Adds a wheel link, shapes, and returns the link index."""
    pos_world = wp.transform_point(parent_xform, pos_local)
    rot_world = parent_xform.q

    wheel_link = builder.add_link(
        xform=wp.transform(pos_world, rot_world),
        label=name,
        mass=HelhestJuniorConfig.WHEEL_MASS,
        inertia=HelhestJuniorConfig.WHEEL_I,
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
        "mu_rolling": mu_rolling,
    }
    if ke is not None:
        collision_cfg_kwargs["ke"] = ke
    if kd is not None:
        collision_cfg_kwargs["kd"] = kd
    if kf is not None:
        collision_cfg_kwargs["kf"] = kf

    builder.add_shape_cylinder(
        body=wheel_link,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), HelhestJuniorConfig.WHEEL_ROT),
        radius=HelhestJuniorConfig.WHEEL_RADIUS,
        half_height=HelhestJuniorConfig.WHEEL_WIDTH / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(**collision_cfg_kwargs),
    )
    return wheel_link


def create_helhest_junior_model(
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
    mu_rolling: float = 0.7,
):
    """
    Creates a Helhest Junior robot model — a smaller variant of the Helhest
    sharing the same topology (chassis + 3 driven wheels) but with its own
    dimensions / masses / motor gains (see HelhestJuniorConfig).

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
        builder,
        xform,
        "left_wheel",
        HelhestJuniorConfig.LEFT_WHEEL_POS,
        friction_left_right,
        wheel_mesh_render,
        is_visible,
        ke=ke,
        kd=kd,
        kf=kf,
        mu_rolling=mu_rolling,
    )
    right_wheel = _add_wheel(
        builder,
        xform,
        "right_wheel",
        HelhestJuniorConfig.RIGHT_WHEEL_POS,
        friction_left_right,
        wheel_mesh_render,
        is_visible,
        ke=ke,
        kd=kd,
        kf=kf,
        mu_rolling=mu_rolling,
    )
    rear_wheel = _add_wheel(
        builder,
        xform,
        "rear_wheel",
        HelhestJuniorConfig.REAR_WHEEL_POS,
        friction_rear,
        wheel_mesh_render,
        is_visible,
        ke=ke,
        kd=kd,
        kf=kf,
        mu_rolling=mu_rolling,
    )

    # 3. Wheel Joints
    Y_AXIS = (0.0, 1.0, 0.0)

    mode = JointMode.TARGET_VELOCITY if control_mode == "velocity" else JointMode.TARGET_POSITION

    j_left = builder.add_joint_revolute(
        parent=chassis,
        child=left_wheel,
        parent_xform=wp.transform(HelhestJuniorConfig.LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="left_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    j_right = builder.add_joint_revolute(
        parent=chassis,
        child=right_wheel,
        parent_xform=wp.transform(HelhestJuniorConfig.RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="right_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    j_rear = builder.add_joint_revolute(
        parent=chassis,
        child=rear_wheel,
        parent_xform=wp.transform(HelhestJuniorConfig.REAR_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=k_p,
        target_kd=k_d,
        label="rear_wheel_j",
        custom_attributes={
            "joint_dof_mode": [mode],
        },
    )

    # 4. Articulation
    builder.add_articulation([j_base, j_left, j_right, j_rear], label="helhest_junior")

    return chassis, [left_wheel]
