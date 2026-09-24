"""Test: single-step adjoint vs central FD with loaded wheel-ground contact.

The other wheeled-robot tests never load their contacts (Helhest's multi-step
test is airborne, Taros-4's normal forces are ~0.002 N). Here the Helhest is
settled onto the ground first, so every wheel carries its share of the weight.

One step, loss L = w . qd_next. The adjoint's d(L)/d(qd_prev) along random
directions over all bodies and axes, and d(L)/d(wheel_target_vel), are compared with central differences
of the same step re-solved from the stored state. The forward is run to full
Newton convergence so the referee grades the adjoint, not a truncated solve.
"""

import numpy as np
import warp as wp
import newton

from ostrich.core.engine import OstrichEngine
from ostrich.core.engine_config import (
    AdjointConfig, ComplianceConfig, OstrichEngineConfig, LinearSolverConfig, NewtonRaphsonConfig,
)
from ostrich.core.logging_config import LoggingConfig
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer
from test_helhest_gradient import build_helhest, make_engine, WHEEL_DOF_OFFSET, NUM_WHEEL_DOFS

SETTLE_STEPS = 60
DT = 0.01
# FD step. At 1e-3 two identical runs disagree by up to 30% on the control
# gradient (solver tolerance noise); 1e-2 is stable.
EPS = 1e-2
NUM_DIRS = 6


def _gradient_errors(adjoint=AdjointConfig()):
    model = build_helhest()
    engine = OstrichEngine(
        model=model,
        sim_steps=1,
        config=OstrichEngineConfig(
            nr=NewtonRaphsonConfig(max_iters=100),
            linear=LinearSolverConfig(max_iters=200, tol=1e-8, atol=1e-8),
            adjoint=adjoint,
            # The forward's contact compliance is pinned away from the adjoint's
            # (1e-6) so the test can tell them apart: an adjoint that reused the
            # forward's Newton damper would differentiate a 1e-4 spring.
            compliance=ComplianceConfig(contact=1e-4),
        ),
        logging_config=LoggingConfig(),
        differentiable_simulation=True,
    )
    dims, data, dev = engine.dims, engine.data, model.device
    nb = model.body_count

    # Settle with the default (fast) engine; only the graded step needs the
    # converged one.
    settle_engine = make_engine(model, sim_steps=1)
    s_in, s_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
    control = model.control()
    for _ in range(SETTLE_STEPS):
        settle_engine.step(s_in, s_out, control, model.collide(s_in), DT)
        s_in, s_out = s_out, s_in
    assert settle_engine.data.constr_force.n.numpy().max() > 10.0, "wheels are not loaded"

    q0, qd0 = s_in.body_q.numpy().copy(), s_in.body_qd.numpy().copy()
    tv = np.zeros(dims.joint_dof_count, np.float32)
    tv[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = [1.0, 0.8, 0.5]
    rng = np.random.default_rng(77)
    w = rng.standard_normal(nb * 6).astype(np.float32)
    dirs = rng.standard_normal((NUM_DIRS, nb * 6)).astype(np.float32)

    def one_step(tv, qd):
        si, so = model.state(), model.state()
        si.body_q.assign(q0)
        si.body_qd.assign(qd)
        wp.copy(control.joint_target_vel,
                wp.array(tv.reshape(1, -1), dtype=wp.float32, device=dev))
        engine.step(si, so, control, model.collide(si), DT)
        return float(w @ so.body_qd.numpy().ravel())

    # Adjoint
    buffer = TrajectoryBuffer(data=data, contacts=engine.ostrich_contacts,
                              dims=dims, num_steps=1, device=dev)
    one_step(tv, qd0)
    buffer.save_step(0, data, engine.ostrich_contacts)
    buffer.zero_grad()
    wp.copy(buffer.body_vel.grad[1],
            wp.array(w.reshape(1, nb, 6), dtype=wp.spatial_vector, device=dev))
    buffer.load_step(0, data, engine.ostrich_contacts)
    data.zero_gradients()
    engine.step_backward()
    g_vel = dirs @ data.body_vel_prev.grad.numpy().ravel()
    g_ctrl = data.joint_target_vel.grad.numpy().ravel()[
        WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS].copy()

    # Central FD of the same step
    fd_vel = np.zeros(NUM_DIRS)
    for k in range(NUM_DIRS):
        d = (EPS * dirs[k]).reshape(qd0.shape)
        fd_vel[k] = (one_step(tv, qd0 + d) - one_step(tv, qd0 - d)) / (2 * EPS)
    fd_ctrl = np.zeros(NUM_WHEEL_DOFS)
    for k in range(NUM_WHEEL_DOFS):
        tp, tm = tv.copy(), tv.copy()
        tp[WHEEL_DOF_OFFSET + k] += EPS
        tm[WHEEL_DOF_OFFSET + k] -= EPS
        fd_ctrl[k] = (one_step(tp, qd0) - one_step(tm, qd0)) / (2 * EPS)

    print("  d/d(qd_prev) along random dirs, adjoint:", np.round(g_vel, 3))
    print("  d/d(qd_prev) along random dirs, FD:     ", np.round(fd_vel, 3))
    print("  d/d(wheel target), adjoint:", np.round(g_ctrl, 4), " FD:", np.round(fd_ctrl, 4))
    err_vel = np.linalg.norm(g_vel - fd_vel) / np.linalg.norm(fd_vel)
    err_ctrl = np.linalg.norm(g_ctrl - fd_ctrl) / np.linalg.norm(fd_ctrl)
    print(f"  rel err: velocity {err_vel:.4f}, control {err_ctrl:.4f}")
    return err_vel, err_ctrl


def test_loaded_contact_single_step():
    print("\n=== Helhest: loaded contact, single-step adjoint vs FD ===")
    err_vel, err_ctrl = _gradient_errors()
    # Measured 2026-09-24: 4.8% / 2.0%. Reusing the forward's Newton-damper
    # compliance (1e-4) in the adjoint gave 67% on the velocity gradient.
    assert err_vel < 0.10, f"velocity gradient rel err {err_vel:.4f}"
    assert err_ctrl < 0.10, f"control gradient rel err {err_ctrl:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_loaded_contact_single_step()
