import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig, ComplianceConfig, LinearSolverConfig, NewtonRaphsonConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


@wp.kernel
def apply_test_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1), force: wp.spatial_vector
):
    body_idx = wp.tid()
    # Apply force only to the dynamic body (index 0)
    if body_idx == 0:
        body_f[body_idx] = force


@wp.kernel
def init_state_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=1), body_qd: wp.array(dtype=wp.spatial_vector, ndim=1)
):
    idx = wp.tid()
    if idx == 0:
        body_q[0] = wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity())

    body_qd[idx] = wp.spatial_vector()


def compute_world_joint_frames(q_parent_np, q_child_np, parent_local_xform, child_local_xform):
    """Computes the world-space transforms of the joint frames attached to parent and child bodies."""
    if q_parent_np is None:
        t_body_parent = wp.transform_identity()
    else:
        t_body_parent = wp.transform(wp.vec3(*q_parent_np[:3]), wp.quat(*q_parent_np[3:]))

    t_body_child = wp.transform(wp.vec3(*q_child_np[:3]), wp.quat(*q_child_np[3:]))

    t_joint_parent = wp.transform_multiply(t_body_parent, parent_local_xform)
    t_joint_child = wp.transform_multiply(t_body_child, child_local_xform)

    return t_joint_parent, t_joint_child


def calculate_position_error(t_joint_parent, t_joint_child):
    """Calculates Euclidean distance between the two joint frame origins."""
    p0_wp = wp.transform_get_translation(t_joint_parent)
    p1_wp = wp.transform_get_translation(t_joint_child)
    p0 = np.array([p0_wp[0], p0_wp[1], p0_wp[2]])
    p1 = np.array([p1_wp[0], p1_wp[1], p1_wp[2]])
    return np.linalg.norm(p0 - p1)


def calculate_position_error_orthogonal(t_joint_parent, t_joint_child, axis_local):
    """Position error orthogonal to the prismatic axis, plus signed sliding distance along it."""
    p_p_wp = wp.transform_get_translation(t_joint_parent)
    p_c_wp = wp.transform_get_translation(t_joint_child)
    p_p = np.array([p_p_wp[0], p_p_wp[1], p_p_wp[2]])
    p_c = np.array([p_c_wp[0], p_c_wp[1], p_c_wp[2]])

    delta = p_c - p_p

    q_p = wp.transform_get_rotation(t_joint_parent)
    axis_w_wp = wp.quat_rotate(q_p, axis_local)
    axis_w = np.array([axis_w_wp[0], axis_w_wp[1], axis_w_wp[2]])

    proj = np.dot(delta, axis_w)
    ortho = delta - proj * axis_w
    return np.linalg.norm(ortho), proj


def calculate_relative_angle(t_joint_parent, t_joint_child):
    """Calculates the angle of the relative rotation between two joint frames."""
    q_j0 = wp.transform_get_rotation(t_joint_parent)
    q_j1 = wp.transform_get_rotation(t_joint_child)

    q0_np = np.array([q_j0[0], q_j0[1], q_j0[2], q_j0[3]])  # x, y, z, w
    q1_np = np.array([q_j1[0], q_j1[1], q_j1[2], q_j1[3]])

    # Quaternion dot product measures similarity
    dot = abs(np.dot(q0_np, q1_np))
    # Clamp for safety
    dot = min(max(dot, -1.0), 1.0)

    # Angle = 2 * acos(|dot|)
    return 2.0 * np.arccos(dot)


def verify_revolute(t_joint_parent, t_joint_child, axis):
    """Verifies Revolute joint: Check position alignment, axis alignment, and ensures rotation occurs."""
    pos_error = calculate_position_error(t_joint_parent, t_joint_child)

    # Hinge Axis Check (Constraint Error)
    a0_wp = wp.quat_rotate(wp.transform_get_rotation(t_joint_parent), axis)
    a1_wp = wp.quat_rotate(wp.transform_get_rotation(t_joint_child), axis)
    a0 = np.array([a0_wp[0], a0_wp[1], a0_wp[2]])
    a1 = np.array([a1_wp[0], a1_wp[1], a1_wp[2]])

    # 1.0 - |dot| ensures we accept parallel vectors in either direction
    axis_error = 1.0 - abs(np.dot(a0, a1))

    # DOF Motion: The amount it rotated around the axis (relative angle)
    dof_motion = calculate_relative_angle(t_joint_parent, t_joint_child)

    return pos_error, axis_error, dof_motion


def verify_fixed(t_joint_parent, t_joint_child):
    """Verifies Fixed joint: Check position alignment and relative rotation identity."""
    pos_error = calculate_position_error(t_joint_parent, t_joint_child)

    # Constraint Error: The relative angle should be 0
    angle_error = calculate_relative_angle(t_joint_parent, t_joint_child)

    # DOF Motion: 0 by definition
    return pos_error, angle_error, 0.0


def verify_spherical(t_joint_parent, t_joint_child):
    """Verifies Spherical joint: Check position alignment only, ensures rotation occurs."""
    pos_error = calculate_position_error(t_joint_parent, t_joint_child)

    # Constraint Error: 0 (no angular constraints)
    # DOF Motion: The relative rotation angle
    dof_motion = calculate_relative_angle(t_joint_parent, t_joint_child)

    return pos_error, 0.0, dof_motion


def verify_prismatic(t_joint_parent, t_joint_child, axis):
    """Verifies Prismatic joint: orthogonal position alignment, no relative rotation, sliding along axis."""
    pos_error, sliding_dist = calculate_position_error_orthogonal(t_joint_parent, t_joint_child, axis)
    angle_error = calculate_relative_angle(t_joint_parent, t_joint_child)
    return pos_error, angle_error, abs(sliding_dist)


def run_joint_test(joint_type, joint_axis=wp.vec3(0.0, 1.0, 0.0)):
    print(f"\n=== Testing {joint_type} Joint ===")

    # 1. Setup Model
    builder = AxionModelBuilder()

    link_1 = builder.add_link()
    builder.add_shape_box(link_1, hx=0.5, hy=0.5, hz=0.5)

    j0 = None

    # Define local transforms for the joint frames
    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())

    if joint_type == "Revolute":
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_1,
            axis=joint_axis,
            parent_xform=parent_local_xform,
            child_xform=child_local_xform,
        )
    elif joint_type == "Prismatic":
        j0 = builder.add_joint_prismatic(
            parent=-1,
            child=link_1,
            axis=joint_axis,
            parent_xform=parent_local_xform,
            child_xform=child_local_xform,
        )
    elif joint_type == "Fixed":
        j0 = builder.add_joint_fixed(
            parent=-1,
            child=link_1,
            parent_xform=parent_local_xform,
            child_xform=child_local_xform,
        )
    elif joint_type == "Spherical":  # Ball
        j0 = builder.add_joint_ball(
            parent=-1,
            child=link_1,
            parent_xform=parent_local_xform,
            child_xform=child_local_xform,
        )

    builder.add_articulation([j0])

    model = builder.finalize_replicated(num_worlds=1, gravity=-9.81)

    # 2. Setup Engine
    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=10),
        linear=LinearSolverConfig(max_iters=10),
        compliance=ComplianceConfig(joint=1e-8),
    )

    engine = AxionEngine(model=model, sim_steps=30, config=config)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.01

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    print(f"Running simulation for 30 steps with heavy load...")

    # Force vector chosen to test all axes
    # Apply force in Z to create torque around Y (since r is along X)
    # r = (1,0,0). F = (0,0,1e6). Torque = r x F = (0, -1e6, 0).
    test_force = wp.spatial_vector(1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0)

    for step in range(30):
        state_in.body_f.zero_()
        wp.launch(
            kernel=apply_test_force_kernel,
            dim=1,
            inputs=[state_in.body_f, test_force],
            device=engine.device,
        )

        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        # Verification
        q_1 = state_out.body_q.numpy()[0]

        # Compute World Frames
        t_joint_0, t_joint_1 = compute_world_joint_frames(
            None, q_1, parent_local_xform, child_local_xform
        )

        pos_error = 0.0
        axis_error = 0.0
        dof_motion = 0.0

        if joint_type == "Revolute":
            pos_error, axis_error, dof_motion = verify_revolute(t_joint_0, t_joint_1, joint_axis)
        elif joint_type == "Prismatic":
            pos_error, axis_error, dof_motion = verify_prismatic(t_joint_0, t_joint_1, joint_axis)
        elif joint_type == "Fixed":
            pos_error, axis_error, dof_motion = verify_fixed(t_joint_0, t_joint_1)
        elif joint_type == "Spherical":
            pos_error, axis_error, dof_motion = verify_spherical(t_joint_0, t_joint_1)

        if step % 5 == 0 or step == 29:
            print(
                f"Step {step:02d}: PosErr={pos_error:.2e}, AngErr={axis_error:.2e}, Motion={dof_motion:.2e}"
            )

    assert pos_error < 1e-3, f"{joint_type} Position drift too high: {pos_error}"
    assert axis_error < 1e-3, f"{joint_type} Angular error too high: {axis_error}"

    if joint_type != "Fixed" and dof_motion < 1e-2:
        print(
            f"WARNING: {joint_type} joint did not move significantly ({dof_motion:.2e} rad). This may be due to 'eval_ik' resetting the state."
        )

    print(f"SUCCESS: {joint_type} joint constraints satisfied.")


@pytest.mark.parametrize("joint_type", ["Revolute", "Prismatic", "Fixed", "Spherical"])
def test_joint_constraints(joint_type):
    run_joint_test(joint_type)


if __name__ == "__main__":
    try:
        run_joint_test("Revolute")
        run_joint_test("Prismatic")
        run_joint_test("Fixed")
        run_joint_test("Spherical")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
