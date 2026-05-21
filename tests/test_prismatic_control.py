import newton
import numpy as np
import warp as wp
from axion import JointMode
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig, ComplianceConfig, LinearSolverConfig, NewtonRaphsonConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()

def setup_test_engine():
    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=20),
        linear=LinearSolverConfig(max_iters=50),
        compliance=ComplianceConfig(joint=0.0),
    )
    return config

def create_prismatic_model():
    builder = AxionModelBuilder()

    link_1 = builder.add_link()
    builder.add_shape_box(
        link_1, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(density=10.0)
    )

    # Prismatic axis along X
    axis = wp.vec3(1.0, 0.0, 0.0)

    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())

    j0 = builder.add_joint_prismatic(
        parent=-1,
        child=link_1,
        axis=axis,
        parent_xform=parent_local_xform,
        child_xform=child_local_xform,
    )

    builder.add_articulation([j0])

    model = builder.finalize_replicated(num_worlds=1, gravity=0.0)
    return model, j0

def test_prismatic_position_control():
    print("\n=== Testing Prismatic Position Control ===")
    model, joint_id = create_prismatic_model()
    config = setup_test_engine()

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166

    # Position Mode
    wp.copy(
        model.joint_dof_mode,
        wp.array(
            np.array([int(JointMode.TARGET_POSITION)], dtype=np.int32), 
            dtype=wp.int32,
            device=model.device,
        ),
    )
    # Gains (Kp, Kd)
    wp.copy(
        model.joint_target_ke,
        wp.array(np.array([500.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )
    wp.copy(
        model.joint_target_kd,
        wp.array(np.array([50.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )

    engine = AxionEngine(model=model, sim_steps=100, config=config)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    target_pos = 1.0 # 1 meter

    for step in range(100):
        state_in.body_f.zero_()
        wp.copy(
            control.joint_target_pos,
            wp.array(
                np.array([target_pos], dtype=np.float32), dtype=wp.float32, device=model.device
            ),
        )

        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)
        current_q = model.joint_q.numpy()[0]
        
        if step % 20 == 0:
            print(f"Step {step}: q={current_q:.4f}")

    final_error = abs(current_q - target_pos)
    print(f"Final Position Error: {final_error:.4f}")
    assert final_error < 0.05, f"Position control failed. Error: {final_error}"

def test_prismatic_velocity_control():
    print("\n=== Testing Prismatic Velocity Control ===")
    model, joint_id = create_prismatic_model()
    config = setup_test_engine()

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166

    # Velocity Mode
    wp.copy(
        model.joint_dof_mode,
        wp.array(
            np.array([int(JointMode.TARGET_VELOCITY)], dtype=np.int32),
            dtype=wp.int32,
            device=model.device,
        ),
    )
    # Gains (Kv) - mapped to 'ke' in implementation
    wp.copy(
        model.joint_target_ke,
        wp.array(np.array([100.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )

    engine = AxionEngine(model=model, sim_steps=100, config=config)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    target_vel = 0.5 # 0.5 m/s

    for step in range(100):
        state_in.body_f.zero_()
        wp.copy(
            control.joint_target_vel,
            wp.array(
                np.array([target_vel], dtype=np.float32), dtype=wp.float32, device=model.device
            ),
        )

        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)
        current_qd = model.joint_qd.numpy()[0]
        
        if step % 20 == 0:
            print(f"Step {step}: qd={current_qd:.4f}")

    final_error = abs(current_qd - target_vel)
    print(f"Final Velocity Error: {final_error:.4f}")
    assert final_error < 0.05, f"Velocity control failed. Error: {final_error}"

if __name__ == "__main__":
    test_prismatic_position_control()
    test_prismatic_velocity_control()
