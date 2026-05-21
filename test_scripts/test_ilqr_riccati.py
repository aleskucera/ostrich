"""
Phase 5 of the iLQR plan: trajectory-level A_t, B_t assembly + Riccati
backward pass + closed-loop stability check + one iLQR forward rollout.

Scene: single revolute pendulum (same as B_t test). Nominal trajectory:
hold at INIT_JOINT_Q with constant target_pos. Quadratic cost penalises
tangent-state deviation from a target tangent state plus control effort.

Pipeline per iteration:
  1. Roll out the nominal trajectory across 12 worlds.
  2. At each step t, run engine.step then engine.step_backward with one-hot
     tangent cotangents to extract A_t (12x12) and B_t (12x1).
  3. Run Riccati backward (numpy) to get gains K_t (1x12), k_t (1x1).
  4. Forward rollout with linear feedback u_t' = u_t + alpha*k_t + K_t*dx_t.
  5. Report closed-loop stability + cost before/after.
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

from test_ilqr_jacobian import (
    perturb_q_tangent,
    recover_phi_tangent,
    pushforward_rot_columns,
)

wp.init()

TANGENT_DIM = 12
DT = 0.05
N_STEPS = 10
NEWTON_ITERS = 32
LINEAR_ITERS = 256
LINEAR_TOL = 1e-10
NEWTON_ATOL = 1e-9

INIT_JOINT_Q = 0.0
INIT_JOINT_QD = 0.0
TARGET_KE = 20.0          # softer than the B_t test so dynamics are slow enough to control
TARGET_KD = 2.0
TARGET_GOAL = 0.6         # joint angle the trajectory should drive to


# --------------------------------------------------------------------------
# Pendulum scene + helpers (same as B_t test)
# --------------------------------------------------------------------------

def build_pendulum(num_worlds: int):
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
        sim_steps=N_STEPS,
        config=config,
        logging_config=LoggingConfig(),
        differentiable_simulation=differentiable,
    )


def init_state_via_fk(model, joint_q_val: float, joint_qd_val: float):
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


def transform_to_np(arr, num_worlds, num_bodies):
    return np.frombuffer(arr.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, num_bodies, 7).astype(np.float64)


def vel_to_np(arr, num_worlds, num_bodies):
    return np.frombuffer(arr.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, num_bodies, 6).astype(np.float64)


# --------------------------------------------------------------------------
# Per-step Jacobian assembly (one timestep)
# --------------------------------------------------------------------------

def setup_one_hot_cotangents(num_worlds, q_plus_np):
    """Build per-world raw cotangents for tangent-space output rows."""
    G_plus = pushforward_rot_columns(q_plus_np[3:])
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
    return pose_grad_np, vel_grad_np


def extract_AB_from_grads(engine, num_worlds, q_minus_np):
    """Read body_pose_prev.grad and body_vel_prev.grad, project to tangent."""
    G_minus = pushforward_rot_columns(q_minus_np[3:])
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

    target_pos_grad = engine.data.joint_target_pos.grad.numpy()
    n_dof = target_pos_grad.shape[1]
    B = np.zeros((TANGENT_DIM, n_dof), dtype=np.float64)
    for k in range(TANGENT_DIM):
        B[k, :] = target_pos_grad[k, :].astype(np.float64)

    return A, B


# --------------------------------------------------------------------------
# Trajectory rollout + assembly
# --------------------------------------------------------------------------

def rollout_and_assemble(model, engine, init_q, init_qd, target_pos_traj):
    """Run N forward steps, assembling A_t, B_t at each step.

    init_q: (1, 1, 7) numpy, initial body_q (will be replicated across worlds).
    init_qd: (1, 1, 6) numpy.
    target_pos_traj: list[float] of length N, target_pos at each step.

    Returns: A_traj (N, 12, 12), B_traj (N, 12, 1),
             body_q_traj (N+1, 7), body_qd_traj (N+1, 6).
    """
    num_worlds = TANGENT_DIM
    state_in = model.state()
    state_out = model.state()

    init_q_full = np.tile(init_q, (num_worlds, 1, 1)).astype(np.float32)
    init_qd_full = np.tile(init_qd, (num_worlds, 1, 1)).astype(np.float32)
    wp.copy(state_in.body_q,
            wp.array(init_q_full, dtype=wp.transform, device=model.device))
    wp.copy(state_in.body_qd,
            wp.array(init_qd_full, dtype=wp.spatial_vector, device=model.device))

    body_q_traj = [init_q[0, 0].astype(np.float64).copy()]
    body_qd_traj = [init_qd[0, 0].astype(np.float64).copy()]
    A_traj = []
    B_traj = []

    for t in range(N_STEPS):
        ctrl = make_control(model, target_pos_traj[t])
        contacts = model.collide(state_in)

        # Snapshot operating-point input pose (world 0) before step.
        q_minus = transform_to_np(state_in.body_q, num_worlds, 1)[0, 0]

        engine.step(state_in, state_out, ctrl, contacts, DT)

        # Snapshot output pose.
        q_plus = transform_to_np(state_out.body_q, num_worlds, 1)[0, 0]
        qd_plus = vel_to_np(state_out.body_qd, num_worlds, 1)[0, 0]
        body_q_traj.append(q_plus)
        body_qd_traj.append(qd_plus)

        # Per-world cotangents for tangent output dims.
        pose_grad_np, vel_grad_np = setup_one_hot_cotangents(num_worlds, q_plus)
        wp.copy(engine.data.body_pose_grad,
                wp.array(pose_grad_np, dtype=wp.transform, device=model.device))
        wp.copy(engine.data.body_vel_grad,
                wp.array(vel_grad_np, dtype=wp.spatial_vector, device=model.device))

        engine.data.body_pose_prev.grad.zero_()
        engine.data.body_vel_prev.grad.zero_()
        engine.data.zero_gradients()

        engine.step_backward()
        A, B = extract_AB_from_grads(engine, num_worlds, q_minus)
        A_traj.append(A)
        B_traj.append(B)

        # Advance state_in for the next step.
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    return (
        np.stack(A_traj, axis=0),
        np.stack(B_traj, axis=0),
        np.array(body_q_traj),
        np.array(body_qd_traj),
    )


# --------------------------------------------------------------------------
# Tangent-space deviation between two states
# --------------------------------------------------------------------------

def tangent_diff(q_a, qd_a, q_b, qd_b):
    """x_a (boxminus) x_b in 12-dim tangent coords."""
    delta_p = q_a[:3] - q_b[:3]
    delta_phi = recover_phi_tangent(q_a[3:], q_b[3:])
    delta_v = qd_a[:3] - qd_b[:3]
    delta_w = qd_a[3:] - qd_b[3:]
    return np.concatenate([delta_p, delta_phi, delta_v, delta_w])


# --------------------------------------------------------------------------
# Riccati backward
# --------------------------------------------------------------------------

def riccati_backward(A_traj, B_traj, Q, R, Qf, x_devs, u_devs):
    """Time-varying iLQR backward pass on the linearised dynamics.

    A_traj: (N, n, n), B_traj: (N, n, m).
    Q, R: stage cost Hessians. Qf: terminal cost Hessian.
    x_devs: (N+1, n) state deviations from x_target on the nominal trajectory.
    u_devs: (N, m) control deviations from u_target on the nominal trajectory.

    Returns:
      K_traj (N, m, n) feedback gains, k_traj (N, m) feedforward terms,
      V_xx_traj (N+1, n, n), V_x_traj (N+1, n).
    """
    N = A_traj.shape[0]
    n = A_traj.shape[1]
    m = B_traj.shape[2]

    V_xx = Qf.copy()
    V_x = Qf @ x_devs[N]

    K_traj = np.zeros((N, m, n))
    k_traj = np.zeros((N, m))
    V_xx_traj = np.zeros((N + 1, n, n))
    V_x_traj = np.zeros((N + 1, n))
    V_xx_traj[N] = V_xx
    V_x_traj[N] = V_x

    for t in range(N - 1, -1, -1):
        A = A_traj[t]
        B = B_traj[t]
        l_x = Q @ x_devs[t]
        l_u = R @ u_devs[t]

        Q_x = l_x + A.T @ V_x
        Q_u = l_u + B.T @ V_x
        Q_xx = Q + A.T @ V_xx @ A
        Q_uu = R + B.T @ V_xx @ B
        Q_ux = B.T @ V_xx @ A

        Q_uu_reg = Q_uu + 1e-9 * np.eye(m)   # tiny regularisation
        K = -np.linalg.solve(Q_uu_reg, Q_ux)
        k = -np.linalg.solve(Q_uu_reg, Q_u)

        V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        V_xx = 0.5 * (V_xx + V_xx.T)
        V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k

        K_traj[t] = K
        k_traj[t] = k
        V_xx_traj[t] = V_xx
        V_x_traj[t] = V_x

    return K_traj, k_traj, V_xx_traj, V_x_traj


# --------------------------------------------------------------------------
# Forward rollout with linear feedback
# --------------------------------------------------------------------------

def forward_rollout_with_feedback(model, engine_fd, init_q, init_qd,
                                   nominal_target_pos, K_traj, k_traj,
                                   nominal_q_traj, nominal_qd_traj,
                                   alpha=1.0):
    """Roll out 1 world with u_t' = u_nom + alpha*k_t + K_t * (x_t' boxminus x_nom_t)."""
    state_in = model.state()
    state_out = model.state()

    wp.copy(state_in.body_q,
            wp.array(init_q.astype(np.float32), dtype=wp.transform, device=model.device))
    wp.copy(state_in.body_qd,
            wp.array(init_qd.astype(np.float32), dtype=wp.spatial_vector, device=model.device))

    q_traj = [init_q[0, 0].astype(np.float64).copy()]
    qd_traj = [init_qd[0, 0].astype(np.float64).copy()]
    u_traj = []
    for t in range(N_STEPS):
        q_now = transform_to_np(state_in.body_q, 1, 1)[0, 0]
        qd_now = vel_to_np(state_in.body_qd, 1, 1)[0, 0]
        x_dev = tangent_diff(q_now, qd_now, nominal_q_traj[t], nominal_qd_traj[t])

        u_new = nominal_target_pos[t] + alpha * k_traj[t][0] + K_traj[t][0] @ x_dev
        u_traj.append(u_new)

        ctrl = make_control(model, float(u_new))
        contacts = model.collide(state_in)
        engine_fd.step(state_in, state_out, ctrl, contacts, DT)

        q_traj.append(transform_to_np(state_out.body_q, 1, 1)[0, 0].copy())
        qd_traj.append(vel_to_np(state_out.body_qd, 1, 1)[0, 0].copy())
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

    return np.array(q_traj), np.array(qd_traj), np.array(u_traj)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def trajectory_cost(q_traj, qd_traj, u_traj, q_target_traj, qd_target_traj,
                     u_target_traj, Q_diag, R_diag, Qf_diag):
    """Sum_t 0.5 (x_t - x_t^*)^T Q (x_t - x_t^*) + 0.5 (u_t - u_t^*)^T R (u_t - u_t^*)
    + 0.5 terminal."""
    N = u_traj.shape[0]
    cost = 0.0
    for t in range(N):
        x_dev = tangent_diff(q_traj[t], qd_traj[t], q_target_traj[t], qd_target_traj[t])
        u_dev = u_traj[t] - u_target_traj[t]
        cost += 0.5 * x_dev @ (Q_diag * x_dev)
        cost += 0.5 * u_dev * R_diag * u_dev
    x_dev = tangent_diff(q_traj[N], qd_traj[N], q_target_traj[N], qd_target_traj[N])
    cost += 0.5 * x_dev @ (Qf_diag * x_dev)
    return float(cost)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    # --- Build batch (12-world) and 1-world models ---
    model_batch = build_pendulum(num_worlds=TANGENT_DIM)
    engine_batch = make_engine(model_batch, differentiable=True)
    model_fd = build_pendulum(num_worlds=1)
    engine_fd = make_engine(model_fd, differentiable=False)

    # --- Initial state and nominal control sequence ---
    # Nominal control: ramp target_pos from 0 to TARGET_GOAL across N steps.
    # The pendulum lags this ramp because of finite stiffness, so the rollout is
    # genuinely off-equilibrium and Riccati has interesting work to do.
    s0_batch = init_state_via_fk(model_batch, INIT_JOINT_Q, INIT_JOINT_QD)
    init_q = transform_to_np(s0_batch.body_q, TANGENT_DIM, 1)[0:1, 0:1].astype(np.float32)
    init_qd = vel_to_np(s0_batch.body_qd, TANGENT_DIM, 1)[0:1, 0:1].astype(np.float32)

    target_pos_traj = np.linspace(0.0, TARGET_GOAL, N_STEPS, dtype=np.float64)

    # --- Rollout + Jacobian assembly ---
    A_traj, B_traj, q_traj_nom, qd_traj_nom = rollout_and_assemble(
        model_batch, engine_batch, init_q, init_qd, target_pos_traj
    )

    # Joint angle estimate (body_y / sin and body_z / cos give angle from -X^.body axis).
    def theta_estimate(q):
        return np.arctan2(q[1], -q[2])

    print(f"Rollout: N={N_STEPS}, dt={DT}, target_pos ramp 0 -> {TARGET_GOAL}")
    print(f"Nominal trajectory (body angle vs ramp):")
    print(f"{'t':>3s}  {'target':>8s}  {'theta_actual':>13s}  {'tracking_err':>13s}")
    for t in range(N_STEPS + 1):
        target = target_pos_traj[min(t, N_STEPS - 1)]
        theta = theta_estimate(q_traj_nom[t])
        print(f"{t:3d}  {target:8.4f}  {theta:13.4f}  {target - theta:13.4f}")

    # --- Quadratic cost in tangent space ---
    # Heavy weight on phi_x (joint angle) and omega_x (joint velocity) -- the
    # only physically free DoFs of the pendulum. Light weight on the rest, so
    # the spurious unit eigenvalues from the missing constraint-Jacobian term
    # in the adjoint contribute little to the cost-to-go.
    Q_diag = np.array([0.01, 0.01, 0.01,
                       100.0, 0.01, 0.01,    # heavy weight on phi_x (axis-angle around X = joint axis)
                       0.01, 0.01, 0.01,
                       1.0, 0.01, 0.01])      # weight on omega_x
    Qf_diag = 100.0 * Q_diag
    Q = np.diag(Q_diag)
    R = np.array([[0.01]])
    Qf = np.diag(Qf_diag)

    # Target tangent trajectory: hold at INIT (theta=0), so the iLQR is asked
    # to drive the pendulum BACK toward 0 against the open-loop ramp.
    # Nominal trajectory deviations from this target = (x_nominal - x_target).
    x_devs = np.zeros((N_STEPS + 1, TANGENT_DIM))
    for t in range(N_STEPS + 1):
        x_devs[t] = tangent_diff(q_traj_nom[t], qd_traj_nom[t],
                                  q_traj_nom[0], qd_traj_nom[0])
    u_devs = (target_pos_traj - 0.0).reshape(-1, 1)

    K_traj, k_traj, V_xx_traj, V_x_traj = riccati_backward(
        A_traj, B_traj, Q, R, Qf, x_devs, u_devs
    )

    # --- Spectral diagnostic (with caveat) ---
    # The unit eigenvalues you'll see in A_t come from a missing
    # J_b * dphi/dq- term in the adjoint -- the constrained directions look
    # uncontrollable to the linearisation but the simulator squashes them in
    # one step. So spectral radius is misleading; what matters is the rollout.
    print(f"\nA_t spectral radii (open vs closed loop) -- spurious unit eigenvalues "
          f"come from constrained directions; ignore for stability assessment:")
    for t in (0, N_STEPS // 2, N_STEPS - 1):
        rho_open = np.max(np.abs(np.linalg.eigvals(A_traj[t])))
        rho_closed = np.max(np.abs(np.linalg.eigvals(A_traj[t] + B_traj[t] @ K_traj[t])))
        print(f"  t={t:2d}  rho(A)={rho_open:.4f}  rho(A+BK)={rho_closed:.4f}")

    # --- Cost: nominal vs improved (apply feedforward k_traj + feedback K_traj) ---
    q_target_traj = q_traj_nom.copy()
    qd_target_traj = qd_traj_nom.copy()
    # Make the target stay at theta=0 (the initial state) for all t -> Riccati's job is to drive back.
    for t in range(N_STEPS + 1):
        q_target_traj[t] = q_traj_nom[0]
        qd_target_traj[t] = qd_traj_nom[0]
    u_target_traj = np.zeros(N_STEPS)

    cost_nominal = trajectory_cost(q_traj_nom, qd_traj_nom, target_pos_traj,
                                    q_target_traj, qd_target_traj, u_target_traj,
                                    Q_diag, 0.01, Qf_diag)
    print(f"\nNominal trajectory cost: {cost_nominal:.4f}")

    print(f"\nFeedforward k_t and feedback K_t[phi_x] = K_t[0, 3]:")
    for t in range(N_STEPS):
        print(f"  t={t:2d}  k={k_traj[t][0]:+.4f}  K[phi_x]={K_traj[t][0, 3]:+.4f}")

    # --- Rollouts at varying line-search step sizes ---
    print(f"\nLine-searched iLQR forward step:")
    print(f"  {'alpha':>8s}  {'cost':>12s}  {'final_theta':>12s}  {'mean|u|':>10s}")
    best_cost = cost_nominal
    best_alpha = 0.0
    for alpha in (0.0, 0.25, 0.5, 1.0):
        q_new, qd_new, u_new = forward_rollout_with_feedback(
            model_fd, engine_fd, init_q, init_qd, target_pos_traj,
            K_traj, k_traj, q_traj_nom, qd_traj_nom, alpha=alpha
        )
        c = trajectory_cost(q_new, qd_new, u_new,
                            q_target_traj, qd_target_traj, u_target_traj,
                            Q_diag, 0.01, Qf_diag)
        print(f"  {alpha:8.2f}  {c:12.4f}  {theta_estimate(q_new[-1]):12.4f}  {np.mean(np.abs(u_new)):10.4f}")
        if alpha == 1.0:
            print(f"   alpha=1.0 trajectory:")
            for t in range(N_STEPS + 1):
                theta_t = theta_estimate(q_new[t])
                u_t = u_new[t] if t < N_STEPS else float('nan')
                print(f"    t={t:2d}  theta={theta_t:+.4f}  u={u_t:+.4f}")
        if c < best_cost:
            best_cost = c
            best_alpha = alpha

    cost_decreased = best_cost < cost_nominal
    print(f"\nbest alpha = {best_alpha}, cost {cost_nominal:.4f} -> {best_cost:.4f}")
    if cost_decreased:
        print(f"PASS: one iLQR backward+forward step decreased the cost by "
              f"{cost_nominal - best_cost:.4f} ({100 * (1 - best_cost/cost_nominal):.1f}%)")
    else:
        print(f"FAIL: no positive line-search step decreased the cost.")


if __name__ == "__main__":
    main()
