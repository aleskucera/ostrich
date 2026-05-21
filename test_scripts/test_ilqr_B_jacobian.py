"""
Phase 3.5 of the iLQR plan: assemble the per-step control Jacobian
B_t = ds+/da and verify against FD.

Scene: single-revolute pendulum hanging from world, with a TARGET_POSITION
control. State: 1 body, 12 tangent dims. Control: 1 joint dof (target_pos),
so B_t is 12 x 1.

The same batch-dim repurposing trick used for A_t works here without changes:
12 worlds, one-hot tangent-output cotangent in each world, single
step_backward, then read joint_target_pos.grad[w, 0] as the entry for row w
of B_t. (We also recover A_t simultaneously and sanity-check it.)
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "glx"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warp as wp
import newton

from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.core.types import JointMode
from axion import AdjointConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig

# Reuse all tangent-space machinery from the A_t test (committed earlier).
from test_ilqr_jacobian import (
    perturb_q_tangent,
    recover_phi_tangent,
    pushforward_rot_columns,
)

wp.init()

TANGENT_DIM = 12
DT = 0.05
NEWTON_ITERS = 32
LINEAR_ITERS = 256
LINEAR_TOL = 1e-10
NEWTON_ATOL = 1e-9

INIT_JOINT_Q = 0.3            # operating-point joint angle (rad)
INIT_JOINT_QD = 0.0           # operating-point joint velocity
TARGET_Q = 0.3                # operating-point target_pos -- equals INIT_JOINT_Q so
                              # the operating point is at control equilibrium and
                              # the inexact-Macklin gap (which scales with lambda_c)
                              # is small enough to read FD-floor accuracy.
TARGET_KE = 200.0             # P gain
TARGET_KD = 20.0              # D gain


# --------------------------------------------------------------------------
# Pendulum scene
# --------------------------------------------------------------------------

def build_pendulum(num_worlds: int):
    """Single revolute joint to world, with TARGET_POSITION control."""
    no_collision = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=10.0)
    builder = AxionModelBuilder()
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.4, cfg=no_collision)
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
        target_ke=TARGET_KE,
        target_kd=TARGET_KD,
        label="pendulum_joint",
        custom_attributes={"joint_dof_mode": [JointMode.TARGET_POSITION]},
    )
    builder.add_articulation([j], label="pendulum")
    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81, requires_grad=True)


def make_engine(model, differentiable: bool):
    config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=NEWTON_ITERS, atol=NEWTON_ATOL),
        linear=LinearSolverConfig(max_iters=LINEAR_ITERS, tol=LINEAR_TOL, atol=LINEAR_TOL),
        linesearch=LinesearchConfig(enabled=False),
        adjoint=AdjointConfig(soft_blending=False, regularization=0.0),
    )
    return AxionEngine(
        model=model,
        sim_steps=1,
        config=config,
        logging_config=LoggingConfig(),
        differentiable_simulation=differentiable,
    )


def init_state_via_fk(model, joint_q_val: float, joint_qd_val: float):
    """Set model.joint_q / joint_qd, run FK, return the resulting body state arrays."""
    jq_np = np.full(model.joint_q.numpy().shape, joint_q_val, dtype=np.float32)
    jqd_np = np.full(model.joint_qd.numpy().shape, joint_qd_val, dtype=np.float32)
    wp.copy(model.joint_q, wp.array(jq_np, dtype=wp.float32, device=model.device))
    wp.copy(model.joint_qd, wp.array(jqd_np, dtype=wp.float32, device=model.device))
    s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s)
    return s


def make_control(model, target_pos_val: float):
    ctrl = model.control()
    tp_np = np.full(ctrl.joint_target_pos.numpy().shape, target_pos_val, dtype=np.float32)
    wp.copy(ctrl.joint_target_pos, wp.array(tp_np, dtype=wp.float32, device=model.device))
    return ctrl


# --------------------------------------------------------------------------
# IFT-side assembly
# --------------------------------------------------------------------------

def assemble_AB_implicit_tangent(engine, model, num_worlds, target_pos_val):
    assert num_worlds == TANGENT_DIM

    s_template = init_state_via_fk(model, INIT_JOINT_Q, INIT_JOINT_QD)
    state_in = model.state(); wp.copy(state_in.body_q, s_template.body_q); wp.copy(state_in.body_qd, s_template.body_qd)
    state_out = model.state()
    control = make_control(model, target_pos_val)

    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, DT)

    # Operating-point pose for tangent charts.
    q_minus = np.frombuffer(state_in.body_q.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, 1, 7)[0, 0].astype(np.float64).copy()
    q_plus = np.frombuffer(state_out.body_q.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, 1, 7)[0, 0].astype(np.float64).copy()

    G_minus = pushforward_rot_columns(q_minus[3:])
    G_plus = pushforward_rot_columns(q_plus[3:])

    # Per-world raw cotangents (same setup as the A_t test).
    pose_grad_np = np.zeros((num_worlds, 1, 7), dtype=np.float32)
    vel_grad_np = np.zeros((num_worlds, 1, 6), dtype=np.float32)
    for k in range(TANGENT_DIM):
        if k < 3:
            pose_grad_np[k, 0, k] = 1.0
        elif k < 6:
            col = 4.0 * G_plus[:, k - 3]
            pose_grad_np[k, 0, 3:7] = col.astype(np.float32)
        elif k < 9:
            vel_grad_np[k, 0, k - 6] = 1.0
        else:
            vel_grad_np[k, 0, k - 9 + 3] = 1.0

    wp.copy(engine.data.body_pose_grad,
            wp.array(pose_grad_np, dtype=wp.transform, device=model.device))
    wp.copy(engine.data.body_vel_grad,
            wp.array(vel_grad_np, dtype=wp.spatial_vector, device=model.device))

    engine.data.body_pose_prev.grad.zero_()
    engine.data.body_vel_prev.grad.zero_()
    engine.data.zero_gradients()

    engine.step_backward()

    # --- A_t (12x12) ---
    pose_prev_grad = np.frombuffer(
        engine.data.body_pose_prev.grad.numpy().tobytes(), dtype=np.float32
    ).reshape(num_worlds, 1, 7).astype(np.float64)
    vel_prev_grad = np.frombuffer(
        engine.data.body_vel_prev.grad.numpy().tobytes(), dtype=np.float32
    ).reshape(num_worlds, 1, 6).astype(np.float64)

    A = np.zeros((TANGENT_DIM, TANGENT_DIM), dtype=np.float64)
    for k in range(TANGENT_DIM):
        raw_pose = pose_prev_grad[k, 0]
        raw_vel = vel_prev_grad[k, 0]
        A[k, 0:3] = raw_pose[0:3]
        A[k, 3:6] = G_minus.T @ raw_pose[3:7]
        A[k, 6:9] = raw_vel[0:3]
        A[k, 9:12] = raw_vel[3:6]

    # --- B_t (12 x 1) ---
    # joint_target_pos.grad[w, j] = dL/d(target_pos[j]) when world w drove
    # cotangent e_w in tangent output space. So row w of B_t = that row.
    target_pos_grad = engine.data.joint_target_pos.grad.numpy()  # shape (num_worlds, joint_dof_count)
    B = np.zeros((TANGENT_DIM, target_pos_grad.shape[1]), dtype=np.float64)
    for k in range(TANGENT_DIM):
        B[k, :] = target_pos_grad[k, :].astype(np.float64)

    return A, B, q_plus


# --------------------------------------------------------------------------
# FD-side assembly
# --------------------------------------------------------------------------

def assemble_B_fd_tangent(model, target_pos_val, eps=1e-3):
    """Central-difference B_t over joint_target_pos perturbations."""
    fd_engine = make_engine(model, differentiable=False)

    state_in = model.state()
    state_out = model.state()

    s_template = init_state_via_fk(model, INIT_JOINT_Q, INIT_JOINT_QD)
    body_q_template = s_template.body_q.numpy().copy()
    body_qd_template = s_template.body_qd.numpy().copy()

    def run_at(target_val):
        wp.copy(state_in.body_q, wp.array(body_q_template, dtype=wp.transform, device=model.device))
        wp.copy(state_in.body_qd, wp.array(body_qd_template, dtype=wp.spatial_vector, device=model.device))
        ctrl = make_control(model, target_val)
        contacts = model.collide(state_in)
        fd_engine.step(state_in, state_out, ctrl, contacts, DT)
        q_out = np.frombuffer(state_out.body_q.numpy().tobytes(), dtype=np.float32)\
            .reshape(1, 1, 7)[0, 0].astype(np.float64).copy()
        qd_out = np.frombuffer(state_out.body_qd.numpy().tobytes(), dtype=np.float32)\
            .reshape(1, 1, 6)[0, 0].astype(np.float64).copy()
        return q_out, qd_out

    q_plus_ref, qd_plus_ref = run_at(target_pos_val)
    qplus_pos_ref = q_plus_ref[:3]
    qplus_quat_ref = q_plus_ref[3:]
    v_plus_ref = qd_plus_ref[:3]
    w_plus_ref = qd_plus_ref[3:]

    def state_to_tangent(q_out, qd_out):
        delta_p = q_out[:3] - qplus_pos_ref
        delta_phi = recover_phi_tangent(q_out[3:], qplus_quat_ref)
        delta_v = qd_out[:3] - v_plus_ref
        delta_w = qd_out[3:] - w_plus_ref
        return np.concatenate([delta_p, delta_phi, delta_v, delta_w])

    sp = run_at(target_pos_val + eps)
    sm = run_at(target_pos_val - eps)
    return ((state_to_tangent(*sp) - state_to_tangent(*sm)) / (2.0 * eps)).reshape(-1, 1)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_at(target_ke, target_kd, init_joint_q, target_q):
    """Returns (B_imp, B_fd, max_rel_err, omega_x_imp, omega_x_fd)."""
    global TARGET_KE, TARGET_KD, INIT_JOINT_Q, TARGET_Q
    TARGET_KE = target_ke
    TARGET_KD = target_kd
    INIT_JOINT_Q = init_joint_q
    TARGET_Q = target_q

    model_batch = build_pendulum(num_worlds=TANGENT_DIM)
    engine_batch = make_engine(model_batch, differentiable=True)
    _, B_imp, _ = assemble_AB_implicit_tangent(engine_batch, model_batch, TANGENT_DIM, TARGET_Q)

    model_fd = build_pendulum(num_worlds=1)
    B_fd = assemble_B_fd_tangent(model_fd, TARGET_Q, eps=1e-3)
    diff = np.abs(B_imp - B_fd)
    rel = diff / (np.abs(B_fd) + 1e-9)
    return B_imp, B_fd, diff.max(), rel.max(), B_imp[9, 0], B_fd[9, 0]


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    # ------- Headline test: at-equilibrium operating point (init_q == target_q).
    # Inexact-Macklin gradient is exact here because lambda_c = 0 at the operating point.
    print(f"=== Headline: at-equilibrium operating point ===")
    print(f"init_q={INIT_JOINT_Q}, init_qd={INIT_JOINT_QD}, target_q={TARGET_Q}, ke={TARGET_KE}, kd={TARGET_KD}")

    B_imp, B_fd, max_abs, max_rel, b_imp_9, b_fd_9 = run_at(
        TARGET_KE, TARGET_KD, INIT_JOINT_Q, TARGET_Q
    )
    print("\nB_implicit:")
    print(B_imp)
    print("\nB_FD:")
    print(B_fd)
    print(f"\nmax abs err: {max_abs:.3e}")
    print(f"max rel err: {max_rel:.3e}")

    # Tolerance: relative, sized for the large B-entry magnitude (O(1/dt) scale).
    rel_tol = 1e-3
    headline_pass = max_rel < rel_tol
    print(f"\n{'PASS' if headline_pass else 'FAIL'}: max rel err {max_rel:.3e} {'<' if headline_pass else '>='} {rel_tol}")

    # ------- Diagnostic 1: stiffness sweep -------
    # If the gap is tied to the constraint operator approximation, error should
    # roughly track stiffness. If it's tied to "off-target" amount, error tracks
    # |INIT_JOINT_Q - TARGET_Q|.
    print("\n=== Diagnostic 1: control stiffness sweep (init_q=0.3, target_q=0.5) ===")
    print(f"{'ke':>8s}  {'kd':>8s}  {'rel err':>10s}  {'B_imp[9]':>10s}  {'B_fd[9]':>10s}")
    for ke, kd in [(20.0, 2.0), (50.0, 5.0), (200.0, 20.0), (1000.0, 100.0)]:
        _, _, _, rel, b_imp_9, b_fd_9 = run_at(ke, kd, 0.3, 0.5)
        print(f"{ke:8.1f}  {kd:8.1f}  {rel:10.3e}  {b_imp_9:10.4f}  {b_fd_9:10.4f}")

    # ------- Diagnostic 2: at-equilibrium operating point -------
    print("\n=== Diagnostic 2: equilibrium operating point (init_q == target_q) ===")
    print(f"{'init_q':>8s}  {'target_q':>10s}  {'rel err':>10s}  {'B_imp[9]':>10s}  {'B_fd[9]':>10s}")
    for q in (0.0, 0.3, 0.5, 1.0):
        _, _, _, rel, b_imp_9, b_fd_9 = run_at(200.0, 20.0, q, q)
        print(f"{q:8.3f}  {q:10.3f}  {rel:10.3e}  {b_imp_9:10.4f}  {b_fd_9:10.4f}")


if __name__ == "__main__":
    main()
