import newton
import numpy as np
import pytest
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


def create_revolute_model():
    builder = AxionModelBuilder()

    link_1 = builder.add_link()
    builder.add_shape_box(
        link_1, hx=0.5, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=100.0)
    )

    axis = wp.vec3(0.0, 0.0, 1.0)
    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(-0.5, 0.0, 0.0), q=wp.quat_identity())

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_1,
        axis=axis,
        parent_xform=parent_local_xform,
        child_xform=child_local_xform,
    )

    builder.add_articulation([j0])
    model = builder.finalize_replicated(num_worlds=1, gravity=0.0)
    return model, j0


# Per-mode test parameters: control target field, model measurement field,
# target value, kd gain (None means leave at default 0), and convergence tolerance.
MODE_SPECS = {
    JointMode.TARGET_POSITION: dict(
        label="Position",
        control_field="joint_target_pos",
        measure_field="joint_q",
        target=np.pi / 2.0,
        kd=100.0,
        tolerance=0.05,
    ),
    JointMode.TARGET_VELOCITY: dict(
        label="Velocity",
        control_field="joint_target_vel",
        measure_field="joint_qd",
        target=1.0,
        kd=None,  # 'ke' acts as velocity gain alone
        tolerance=0.01,
    ),
}


def run_revolute_control_test(mode: JointMode):
    spec = MODE_SPECS[mode]
    print(f"\n=== Testing Revolute {spec['label']} Control ===")

    model, _ = create_revolute_model()
    config = setup_test_engine()

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166  # 60Hz

    # Set Mode and Gains BEFORE engine initialization.
    wp.copy(
        model.joint_dof_mode,
        wp.array(np.array([int(mode)], dtype=np.int32), dtype=wp.int32, device=model.device),
    )
    wp.copy(
        model.joint_target_ke,
        wp.array(np.array([1000.0], dtype=np.float32), dtype=wp.float32, device=model.device),
    )
    if spec["kd"] is not None:
        wp.copy(
            model.joint_target_kd,
            wp.array(
                np.array([spec["kd"]], dtype=np.float32), dtype=wp.float32, device=model.device
            ),
        )

    engine = AxionEngine(model=model, sim_steps=100, config=config)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    target = spec["target"]
    print(f"Target {spec['label']}: {target:.4f}")

    target_arr = wp.array(np.array([target], dtype=np.float32), dtype=wp.float32, device=model.device)
    control_target = getattr(control, spec["control_field"])

    for step in range(100):
        state_in.body_f.zero_()
        wp.copy(control_target, target_arr)

        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)
        current = getattr(model, spec["measure_field"]).numpy()[0]

        if step % 10 == 0 or step == 99:
            print(f"Step {step}: {spec['measure_field']}={current:.4f}")

    final_error = abs(current - target)
    print(f"Final {spec['label']} Error: {final_error:.4f}")

    assert final_error < spec["tolerance"], (
        f"{spec['label']} control failed to converge. Error: {final_error}"
    )


@pytest.mark.parametrize("mode", [JointMode.TARGET_POSITION, JointMode.TARGET_VELOCITY])
def test_revolute_control(mode):
    run_revolute_control_test(mode)


if __name__ == "__main__":
    run_revolute_control_test(JointMode.TARGET_POSITION)
    run_revolute_control_test(JointMode.TARGET_VELOCITY)
