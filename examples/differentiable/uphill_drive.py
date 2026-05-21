"""
Test: Wheeled robot driving uphill on mesh-based terrain.

Creates a ramp using triangle mesh shapes and places the 4-wheel robot at the bottom.
The robot drives forward up the slope.
"""
import os
import pathlib
import sys

import newton
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.engine_config import AxionEngineConfig
from axion.core.types import JointMode
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

# 4-wheel robot config
CHASSIS_HX = 0.6
CHASSIS_HY = 0.4
CHASSIS_HZ = 0.08
CHASSIS_MASS = 80.0
WHEEL_RADIUS = 0.22
WHEEL_MASS = 10.0
NUM_WHEELS = 4

# Terrain parameters
SLOPE_ANGLE = 15.0  # degrees
SLIP_WIDTH = 1.5  # slippery strip width (meters)


def create_4wheel_robot(
    builder,
    xform=wp.transform_identity(),
    is_visible=True,
    k_p=200.0,
    k_d=0.0,
    wheel_mu=0.3,
):
    """Simple symmetric 4-wheel robot with sphere wheels."""
    chassis = builder.add_link(xform=xform, label="chassis")
    chassis_vol = CHASSIS_HX * CHASSIS_HY * CHASSIS_HZ * 8.0
    builder.add_shape_box(
        body=chassis,
        xform=wp.transform_identity(),
        hx=CHASSIS_HX,
        hy=CHASSIS_HY,
        hz=CHASSIS_HZ,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=CHASSIS_MASS / chassis_vol,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )
    j_base = builder.add_joint_free(parent=-1, child=chassis, label="base_joint")

    wheel_z = -CHASSIS_HZ - WHEEL_RADIUS * 0.3
    wheel_positions = {
        "front_left": wp.vec3(CHASSIS_HX, CHASSIS_HY, wheel_z),
        "front_right": wp.vec3(CHASSIS_HX, -CHASSIS_HY, wheel_z),
        "rear_left": wp.vec3(-CHASSIS_HX, CHASSIS_HY, wheel_z),
        "rear_right": wp.vec3(-CHASSIS_HX, -CHASSIS_HY, wheel_z),
    }

    joints = [j_base]
    for name, pos_local in wheel_positions.items():
        pos_world = wp.transform_point(xform, pos_local)
        I_spin = 0.5 * WHEEL_MASS * WHEEL_RADIUS**2
        I_other = (1.0 / 12.0) * WHEEL_MASS * (3.0 * WHEEL_RADIUS**2 + 0.1**2)
        wheel_I = wp.mat33(I_other, 0.0, 0.0, 0.0, I_spin, 0.0, 0.0, 0.0, I_other)

        is_rear = "rear" in name
        wheel_link = builder.add_link(
            xform=wp.transform(pos_world, xform.q),
            label=name,
            mass=WHEEL_MASS,
            inertia=wheel_I,
        )
        wheel_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        builder.add_shape_capsule(
            body=wheel_link,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wheel_rot),
            radius=WHEEL_RADIUS,
            half_height=0.06,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                is_visible=is_visible,
                collision_group=-1,
                mu=wheel_mu,
            ),
        )
        j = builder.add_joint_revolute(
            parent=chassis,
            child=wheel_link,
            parent_xform=wp.transform(pos_local, wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=(0.0, 1.0, 0.0),
            target_ke=k_p if is_rear else 0.0,
            target_kd=k_d,
            label=f"{name}_j",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
        )
        joints.append(j)

    builder.add_articulation(joints, label="robot")
    return chassis


def make_quad_mesh(y0, z0, y1, z1, half_width=3.0, ny=20, nx=4):
    """Create a subdivided mesh strip from (y0,z0) to (y1,z1), extruded in x by ±half_width."""
    w = half_width
    verts = []
    for iy in range(ny + 1):
        t = iy / ny
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)
        for ix in range(nx + 1):
            x = -w + ix * (2.0 * w / nx)
            verts.append([x, y, z])
    vertices = np.array(verts, dtype=np.float32)

    tris = []
    for iy in range(ny):
        for ix in range(nx):
            v00 = iy * (nx + 1) + ix
            v10 = v00 + 1
            v01 = v00 + (nx + 1)
            v11 = v01 + 1
            tris.extend([v00, v10, v11, v00, v11, v01])
    indices = np.array(tris, dtype=np.int32)
    return newton.Mesh(vertices, indices)


def build_mesh_terrain(builder, slope_angle, slip_width, half_width=3.0):
    """Build the uphill terrain from triangle meshes. Returns (slip_y_start, slip_y_end)."""
    slope_rad = np.radians(slope_angle)
    z_per_m = np.tan(slope_rad)

    flat_len = 3.0
    ramp1_len = 5.0
    ramp2_len = 5.0
    flat_top_len = 10.0

    z_ramp1_end = z_per_m * ramp1_len
    z_slip_end = z_ramp1_end + z_per_m * slip_width
    z_ramp2_end = z_slip_end + z_per_m * ramp2_len

    normal_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=2.0)
    slippery_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=0.0)

    slip_y_start = flat_len + ramp1_len
    ramp2_y_start = slip_y_start + slip_width
    flat_top_y_start = ramp2_y_start + ramp2_len

    # Part 1: Flat start [0, flat_len] at z=0
    mesh1 = make_quad_mesh(0.0, 0.0, flat_len, 0.0, half_width)
    builder.add_shape_mesh(body=-1, mesh=mesh1, cfg=normal_cfg, label="terrain_flat_start")

    # Part 2: Ramp up [flat_len, slip_y_start]
    mesh2 = make_quad_mesh(flat_len, 0.0, slip_y_start, z_ramp1_end, half_width)
    builder.add_shape_mesh(body=-1, mesh=mesh2, cfg=normal_cfg, label="terrain_ramp1")

    # Part 3: Slippery ramp [slip_y_start, ramp2_y_start]
    mesh3 = make_quad_mesh(slip_y_start, z_ramp1_end, ramp2_y_start, z_slip_end, half_width)
    builder.add_shape_mesh(body=-1, mesh=mesh3, cfg=slippery_cfg, label="terrain_slippery")

    # Part 4: Ramp up continues [ramp2_y_start, flat_top_y_start]
    mesh4 = make_quad_mesh(ramp2_y_start, z_slip_end, flat_top_y_start, z_ramp2_end, half_width)
    builder.add_shape_mesh(body=-1, mesh=mesh4, cfg=normal_cfg, label="terrain_ramp2")

    # Part 5: Flat top [flat_top_y_start, flat_top_y_start + flat_top_len]
    mesh5 = make_quad_mesh(
        flat_top_y_start, z_ramp2_end, flat_top_y_start + flat_top_len, z_ramp2_end, half_width
    )
    builder.add_shape_mesh(body=-1, mesh=mesh5, cfg=normal_cfg, label="terrain_flat_top")

    return slip_y_start, slip_y_start + slip_width


@wp.kernel
def drive_kernel(
    joint_target_vel: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    step_counter: wp.array(dtype=wp.int32),
    settle_steps: int,
    wheel_speed: float,
    dof_offset: int,
    steering_gain: float,
):
    """Drive with heading correction. Steers left/right wheels to stay on x=0."""
    current_step = step_counter[0]
    if current_step >= settle_steps:
        chassis_pos = wp.transform_get_translation(body_q[0])
        x_error = chassis_pos[0]

        chassis_rot = wp.transform_get_rotation(body_q[0])
        fwd = wp.quat_rotate(chassis_rot, wp.vec3(1.0, 0.0, 0.0))
        heading_error = fwd[0]

        correction = steering_gain * x_error + steering_gain * 2.0 * heading_error

        joint_target_vel[dof_offset + 0] = 0.0
        joint_target_vel[dof_offset + 1] = 0.0
        joint_target_vel[dof_offset + 2] = wheel_speed + correction
        joint_target_vel[dof_offset + 3] = wheel_speed - correction

    wp.atomic_add(step_counter, 0, 1)


class UphillDrive(InteractiveSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, engine_config, logging_config)

        self._step_counter = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._settle_steps = int(1.0 / self.clock.dt)

        if self.viewer:
            self.viewer.set_camera(
                pos=wp.vec3(-18.0, -14.0, 15.0),
                pitch=-25.0,
                yaw=55.0,
            )
            self.viewer.show_ui = False

    def build_model(self):
        self.builder.rigid_gap = 0.2

        build_mesh_terrain(self.builder, SLOPE_ANGLE, SLIP_WIDTH)

        rot_z90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        robot_xform = wp.transform(
            wp.vec3(0.0, 1.5, WHEEL_RADIUS + 0.2),
            rot_z90,
        )

        create_4wheel_robot(
            self.builder,
            xform=robot_xform,
            is_visible=True,
            k_p=300.0,
            k_d=0.0,
            wheel_mu=0.7,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            gravity=-9.81,
        )

    def control_policy(self, current_state):
        wp.launch(
            kernel=drive_kernel,
            dim=1,
            inputs=[
                self.control.joint_target_vel,
                current_state.body_q,
                self._step_counter,
                self._settle_steps,
                25.0,
                6,
                0.0,
            ],
            device=self.model.device,
        )


def main():
    sim_config = SimulationConfig(
        duration_seconds=6.0,
        target_timestep_seconds=6e-2,
        num_worlds=1,
    )
    render_config = RenderingConfig(vis_type="gl")
    engine_config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=0.001),
        linear=LinearSolverConfig(max_iters=16, tol=1e-05, atol=1e-05, regularization=1e-06),
        compliance=ComplianceConfig(joint=5e-06, contact=1.0, friction=1e-12),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=512),
    )
    logging_config = LoggingConfig()

    sim = UphillDrive(sim_config, render_config, engine_config, logging_config)
    sim.run()


if __name__ == "__main__":
    main()
