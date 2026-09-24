"""Test: the backward pass does not depend on leftover friction_w state.

EngineData.friction_w (the damped friction weight, nr.w_relaxation) is read
and rewritten by every friction-kernel call, including the calls made inside
step_backward. The adjoint of step i must therefore linearize friction with the
weight the forward converged to at step i, not with whatever the previous
kernel call left in the buffer. Overwriting friction_w with garbage before each
backward step must not change the gradient.
"""

import numpy as np
import warp as wp
import newton

from ostrich.simulation.trajectory_buffer import TrajectoryBuffer
from test_helhest_gradient import build_helhest, make_engine, WHEEL_DOF_OFFSET, NUM_WHEEL_DOFS

NUM_STEPS = 5
SETTLE_STEPS = 60


def _backward(engine, buffer, w, poison):
    buffer.zero_grad()
    wp.copy(
        buffer.body_vel.grad[NUM_STEPS],
        wp.array(w.reshape(engine.data.body_vel_grad.numpy().shape),
                 dtype=wp.spatial_vector, device=engine.data.body_vel_grad.device),
    )
    rng = np.random.default_rng(0)
    grad = None
    for step in range(NUM_STEPS - 1, -1, -1):
        if poison:
            fw = engine.data.friction_w
            wp.copy(fw, wp.array(rng.uniform(0.0, 10.0, (*fw.shape, 2)).astype(np.float32),
                                 dtype=wp.vec2, device=fw.device))
        buffer.load_step(step, engine.data, engine.ostrich_contacts)
        engine.data.zero_gradients()
        engine.step_backward()
        step_grad = engine.data.joint_target_vel.grad.numpy().flatten().copy()
        grad = step_grad if grad is None else grad + step_grad
        if step > 0:
            buffer.save_gradients(step, engine.data)
    return grad[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]


def test_backward_independent_of_leftover_friction_w():
    model = build_helhest()
    engine = make_engine(model, sim_steps=NUM_STEPS)
    dims = engine.dims

    target_vel = np.zeros(dims.joint_dof_count, dtype=np.float32)
    target_vel[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = 3.0

    state_in = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    control = model.control()
    wp.copy(control.joint_target_vel,
            wp.array(target_vel.reshape(1, -1), dtype=wp.float32, device=model.device))

    buffer = TrajectoryBuffer(
        data=engine.data, contacts=engine.ostrich_contacts,
        dims=dims, num_steps=NUM_STEPS, device=model.device,
    )
    state_out = model.state()
    # The robot starts above the ground; settle it so the recorded steps have
    # active wheel-ground friction.
    for _ in range(SETTLE_STEPS):
        engine.step(state_in, state_out, control, model.collide(state_in), 0.01)
        state_in, state_out = state_out, state_in
    assert engine.ostrich_contacts.contact_count.numpy()[0] > 0
    for step in range(NUM_STEPS):
        contacts = model.collide(state_in)
        engine.step(state_in, state_out, control, contacts, 0.01)
        buffer.save_step(step, engine.data, engine.ostrich_contacts)
        state_in, state_out = state_out, state_in

    w = np.random.default_rng(77).standard_normal(model.body_count * 6).astype(np.float32)
    clean = _backward(engine, buffer, w, poison=False)
    poisoned = _backward(engine, buffer, w, poison=True)

    print(f"  clean:    {clean}")
    print(f"  poisoned: {poisoned}")
    rel = np.abs(clean - poisoned).max() / max(np.abs(clean).max(), 1e-10)
    print(f"  max rel diff: {rel:.2e}")
    assert rel < 1e-5, f"backward depends on leftover friction_w (max rel diff {rel:.2e})"


if __name__ == "__main__":
    test_backward_independent_of_leftover_friction_w()
