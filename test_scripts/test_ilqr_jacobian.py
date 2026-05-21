"""
Phase 3 of the iLQR plan: assemble the per-step state-transition Jacobian
A_t in tangent-space coordinates and verify against tangent-space FD.

Tangent state per body has 12 dims (vs 13 raw):
    indices 0..2  : delta-position
    indices 3..5  : delta-rotation as world-frame axis-angle
    indices 6..8  : delta linear velocity
    indices 9..11 : delta angular velocity (world frame)

Convention (matches kinematic_mapping.py G_matvec):
    q+ = (1/2) omega_world (x) q  =>  q(phi) = exp([phi/2; 0]) (x) q_0

i.e. left-multiplication, world-frame angular velocity.

For the IFT side we feed step_backward raw-quat cotangents that are pushforwards
of tangent unit vectors, then pull the resulting raw-quat input gradients back
to tangent space.

For the FD side we perturb the operating point in tangent-space directions
(via exp on the rotation block) and project the output back via log.
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "glx"

import numpy as np
import warp as wp
import newton

from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion import AdjointConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig

wp.init()

TANGENT_DIM = 12
DT = 0.05
NEWTON_ITERS = 32
LINEAR_ITERS = 256
LINEAR_TOL = 1e-10
NEWTON_ATOL = 1e-9
ANGULAR_VEL = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# Quaternion helpers (warp/newton convention: (x, y, z, w), w = scalar last)
# --------------------------------------------------------------------------

def quat_mul(a, b):
    """Hamilton product a (x) b, both in (x, y, z, w) layout."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_exp_pure(phi):
    """exp([phi; 0]) for axis-angle phi, returning unit quaternion (x,y,z,w)."""
    phi = np.asarray(phi, dtype=np.float64)
    norm = np.linalg.norm(phi)
    if norm < 1e-12:
        # First-order Taylor: exp([phi; 0]) ~ (phi/2... wait no, phi is whole arg)
        # exp(0.5 * phi as pure quat with phi as whole vector) -- but here phi
        # is already the quaternion-log argument, i.e. half-angle * axis.
        return np.array([phi[0], phi[1], phi[2], 1.0], dtype=np.float64)
    s = np.sin(norm) / norm
    return np.array([phi[0] * s, phi[1] * s, phi[2] * s, np.cos(norm)], dtype=np.float64)


def quat_log_to_vec(q):
    """log(q) for unit q, returning the imaginary part of the log (3-vector).
    For q = (sin(t)*n, cos(t)) returns t*n."""
    qv = q[:3]
    qw = q[3]
    norm_v = np.linalg.norm(qv)
    if norm_v < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = np.arctan2(norm_v, qw)
    return qv / norm_v * angle


def perturb_q_tangent(q0, phi):
    """q = exp([phi/2; 0]) (x) q0  -- world-frame, left-mult convention.

    With phi an axis-angle 3-vector, the rotation magnitude perturbation is |phi|."""
    return quat_mul(quat_exp_pure(0.5 * phi), q0)


def recover_phi_tangent(q, q0):
    """Inverse of perturb_q_tangent: phi = 2 * log(q (x) q0^-1)."""
    return 2.0 * quat_log_to_vec(quat_mul(q, quat_conj(q0)))


def pushforward_rot_columns(q0):
    """Columns of the 4x3 pushforward matrix M_push for the rotation block:
    M_push[:, k] = (1/2) (e_k_pure_imag (x) q0)
    where the leading 1/2 comes from d(exp(phi/2)) / d(phi)|_{phi=0}.

    This is the linearisation of  q(phi) = exp([phi/2; 0]) (x) q0  at phi=0."""
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    e1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    cols = [0.5 * quat_mul(e, q0) for e in (e0, e1, e2)]
    return np.stack(cols, axis=1)   # shape (4, 3)


# --------------------------------------------------------------------------
# Engine setup (unchanged from Phase 1/2)
# --------------------------------------------------------------------------

def build_box(num_worlds: int):
    no_collision = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=10.0)
    builder = AxionModelBuilder()
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.2, hy=0.3, hz=0.4, cfg=no_collision)
    j = builder.add_joint_free(child=link)
    builder.add_articulation([j], label="floating_box")
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


def operating_point_arrays(num_worlds: int, device, q_override=None, qd_override=None):
    """Same operating point in every world."""
    pos = np.array([0.1, -0.2, 1.5], dtype=np.float32)
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    angle = 0.4
    quat = np.array(
        [axis[0] * np.sin(angle / 2),
         axis[1] * np.sin(angle / 2),
         axis[2] * np.sin(angle / 2),
         np.cos(angle / 2)],
        dtype=np.float32,
    )

    body_q = np.zeros((num_worlds, 1, 7), dtype=np.float32)
    body_q[:, 0, :3] = pos
    body_q[:, 0, 3:] = quat

    body_qd = np.zeros((num_worlds, 1, 6), dtype=np.float32)
    body_qd[:, 0, :3] = [0.7, -0.3, 0.5]
    body_qd[:, 0, 3:] = ANGULAR_VEL

    if q_override is not None:
        body_q[:, 0, :] = q_override
    if qd_override is not None:
        body_qd[:, 0, :] = qd_override

    return (
        wp.array(body_q, dtype=wp.transform, device=device, requires_grad=True),
        wp.array(body_qd, dtype=wp.spatial_vector, device=device, requires_grad=True),
    )


def operating_point_np():
    """Same content as operating_point_arrays, returned as plain numpy."""
    pos = np.array([0.1, -0.2, 1.5], dtype=np.float64)
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    angle = 0.4
    quat = np.array(
        [axis[0] * np.sin(angle / 2),
         axis[1] * np.sin(angle / 2),
         axis[2] * np.sin(angle / 2),
         np.cos(angle / 2)],
        dtype=np.float64,
    )
    body_q = np.concatenate([pos, quat])           # 7
    body_qd = np.concatenate([[0.7, -0.3, 0.5], list(ANGULAR_VEL)])  # 6
    return body_q.astype(np.float64), body_qd.astype(np.float64)


# --------------------------------------------------------------------------
# IFT-side assembly (12 worlds)
# --------------------------------------------------------------------------

def assemble_A_implicit_tangent(engine, model, num_worlds):
    assert num_worlds == TANGENT_DIM

    state_in = model.state()
    state_out = model.state()
    control = model.control()

    body_q, body_qd = operating_point_arrays(num_worlds, model.device)
    state_in.body_q = body_q
    state_in.body_qd = body_qd

    contacts = model.collide(state_in)
    engine.step(state_in, state_out, control, contacts, DT)

    q_minus_np, _ = operating_point_np()
    q_plus = np.frombuffer(state_out.body_q.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, 1, 7)[0, 0].astype(np.float64).copy()
    qd_plus = np.frombuffer(state_out.body_qd.numpy().tobytes(), dtype=np.float32)\
        .reshape(num_worlds, 1, 6)[0, 0].astype(np.float64).copy()

    # Pushforward matrices for the rotation block at the input and output points.
    G_minus = pushforward_rot_columns(q_minus_np[3:])    # 4 x 3, evaluated at q^-_0
    G_plus = pushforward_rot_columns(q_plus[3:])         # 4 x 3, evaluated at q^+_0

    # Build per-world raw cotangents.
    # Tangent layout (12): [dp(3), dphi(3), dv(3), dw(3)]
    # Raw layout: body_pose_grad has 7 (px, py, pz, qx, qy, qz, qw),
    #             body_vel_grad has 6 (vx, vy, vz, wx, wy, wz).
    #
    # For tangent output cotangent e_k (unit in tangent space):
    #   - dp_k (k in 0..2): raw pose cotangent = (e_k^pos, 0_quat),  raw vel cotangent = 0.
    #   - dphi_k (k in 3..5): raw pose cotangent = (0_pos, 2 * G_plus[:, k-3]),  raw vel = 0.
    #     The "2 *" is because  M_pull = 4 G^T  =>  (M_pull)^T e_k = 4 G e_k = 2 * G_plus_col.
    #     Wait, columns of G are (1/2) e_q (x) q+, so 2 * column = e_q (x) q+. We feed that
    #     directly as the raw quaternion cotangent.
    #   - dv_k (k in 6..8): raw vel cotangent = (e_k-6^lin, 0_ang).
    #   - dw_k (k in 9..11): raw vel cotangent = (0_lin, e_k-9^ang).
    # Output cotangent for tangent-output direction k (rotation rows):
    #   nabla_q+ L = (M_pull_+)^T e_k.
    # Since columns of M_push_+ are m_k = (1/2) e_k_quat (x) q+_0 with squared norm 1/4,
    # the (Moore-Penrose) M_pull_+ = 4 (M_push_+)^T, so (M_pull_+)^T = 4 M_push_+.
    # Hence nabla_q+ L = 4 * G_plus[:, k-3].
    pose_grad_np = np.zeros((num_worlds, 1, 7), dtype=np.float32)
    vel_grad_np = np.zeros((num_worlds, 1, 6), dtype=np.float32)
    for k in range(TANGENT_DIM):
        if k < 3:                       # dp_k
            pose_grad_np[k, 0, k] = 1.0
        elif k < 6:                     # dphi_k
            col = 4.0 * G_plus[:, k - 3]
            pose_grad_np[k, 0, 3:7] = col.astype(np.float32)
        elif k < 9:                     # dv_k
            vel_grad_np[k, 0, k - 6] = 1.0
        else:                           # dw_k
            vel_grad_np[k, 0, k - 9 + 3] = 1.0

    wp.copy(
        engine.data.body_pose_grad,
        wp.array(pose_grad_np, dtype=wp.transform, device=model.device),
    )
    wp.copy(
        engine.data.body_vel_grad,
        wp.array(vel_grad_np, dtype=wp.spatial_vector, device=model.device),
    )

    engine.data.body_pose_prev.grad.zero_()
    engine.data.body_vel_prev.grad.zero_()
    engine.data.zero_gradients()

    engine.step_backward()

    pose_prev_grad = np.frombuffer(
        engine.data.body_pose_prev.grad.numpy().tobytes(), dtype=np.float32
    ).reshape(num_worlds, 1, 7).astype(np.float64)
    vel_prev_grad = np.frombuffer(
        engine.data.body_vel_prev.grad.numpy().tobytes(), dtype=np.float32
    ).reshape(num_worlds, 1, 6).astype(np.float64)

    # Pull raw input gradients to tangent space.
    # For input layout: tangent indices 0..2 = pos (identity), 3..5 = rot (use G_minus^T / 2),
    # 6..8 = lin vel (identity), 9..11 = ang vel (identity).
    A = np.zeros((TANGENT_DIM, TANGENT_DIM), dtype=np.float64)
    for k in range(TANGENT_DIM):
        raw_pose = pose_prev_grad[k, 0]   # 7-vec
        raw_vel = vel_prev_grad[k, 0]     # 6-vec
        # Position columns (input)
        A[k, 0:3] = raw_pose[0:3]
        # Rotation columns (input): chain rule says nabla_xi- L = (M_push_-)^T nabla_q- L,
        # i.e. just G_minus^T @ raw_quat_grad with no extra scaling.
        raw_quat = raw_pose[3:7]          # 4-vec
        A[k, 3:6] = G_minus.T @ raw_quat
        # Linear vel
        A[k, 6:9] = raw_vel[0:3]
        # Angular vel
        A[k, 9:12] = raw_vel[3:6]

    return A, q_plus, qd_plus


# --------------------------------------------------------------------------
# FD-side assembly (1 world)
# --------------------------------------------------------------------------

def assemble_A_fd_tangent(model, eps_pos=1e-4, eps_rot=1e-4, eps_vel=1e-4):
    fd_engine = make_engine(model, differentiable=False)

    state_in = model.state()
    state_out = model.state()
    control = model.control()

    q0_np, qd0_np = operating_point_np()
    q0_pos = q0_np[:3].copy()
    q0_quat = q0_np[3:].copy()

    def run_at(q_np, qd_np):
        q_arr = wp.array(q_np.reshape(1, 1, 7).astype(np.float32),
                         dtype=wp.transform, device=model.device)
        qd_arr = wp.array(qd_np.reshape(1, 1, 6).astype(np.float32),
                          dtype=wp.spatial_vector, device=model.device)
        wp.copy(state_in.body_q, q_arr)
        wp.copy(state_in.body_qd, qd_arr)
        contacts = model.collide(state_in)
        fd_engine.step(state_in, state_out, control, contacts, DT)
        q_out = np.frombuffer(state_out.body_q.numpy().tobytes(), dtype=np.float32)\
            .reshape(1, 1, 7)[0, 0].astype(np.float64).copy()
        qd_out = np.frombuffer(state_out.body_qd.numpy().tobytes(), dtype=np.float32)\
            .reshape(1, 1, 6)[0, 0].astype(np.float64).copy()
        return q_out, qd_out

    # Reference forward at the operating point.
    q_plus_ref, qd_plus_ref = run_at(q0_np, qd0_np)
    qplus_pos_ref = q_plus_ref[:3]
    qplus_quat_ref = q_plus_ref[3:]
    v_plus_ref = qd_plus_ref[:3]
    w_plus_ref = qd_plus_ref[3:]

    A_fd = np.zeros((TANGENT_DIM, TANGENT_DIM), dtype=np.float64)

    def state_to_tangent(q_out, qd_out):
        """Project a forward-step result to the 12-dim tangent space at the reference s+."""
        delta_p = q_out[:3] - qplus_pos_ref
        delta_phi = recover_phi_tangent(q_out[3:], qplus_quat_ref)
        delta_v = qd_out[:3] - v_plus_ref
        delta_w = qd_out[3:] - w_plus_ref
        return np.concatenate([delta_p, delta_phi, delta_v, delta_w])

    for j in range(TANGENT_DIM):
        if j < 3:                      # dp_j
            eps = eps_pos
            q_p = q0_np.copy(); q_p[j] += eps
            q_m = q0_np.copy(); q_m[j] -= eps
            qd_p = qd0_np.copy(); qd_m = qd0_np.copy()
        elif j < 6:                    # dphi_j (axis-angle)
            eps = eps_rot
            phi_p = np.zeros(3); phi_p[j - 3] = eps
            phi_m = np.zeros(3); phi_m[j - 3] = -eps
            q_p = q0_np.copy()
            q_m = q0_np.copy()
            q_p[3:] = perturb_q_tangent(q0_quat, phi_p)
            q_m[3:] = perturb_q_tangent(q0_quat, phi_m)
            qd_p = qd0_np.copy(); qd_m = qd0_np.copy()
        elif j < 9:                    # dv_j
            eps = eps_vel
            qd_p = qd0_np.copy(); qd_p[j - 6] += eps
            qd_m = qd0_np.copy(); qd_m[j - 6] -= eps
            q_p = q0_np.copy(); q_m = q0_np.copy()
        else:                          # dw_j
            eps = eps_vel
            qd_p = qd0_np.copy(); qd_p[(j - 9) + 3] += eps
            qd_m = qd0_np.copy(); qd_m[(j - 9) + 3] -= eps
            q_p = q0_np.copy(); q_m = q0_np.copy()

        sp = run_at(q_p, qd_p)
        sm = run_at(q_m, qd_m)
        delta_plus = state_to_tangent(*sp)
        delta_minus = state_to_tangent(*sm)
        A_fd[:, j] = (delta_plus - delta_minus) / (2.0 * eps)

    return A_fd, q_plus_ref, qd_plus_ref


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_one(angular_vel):
    global ANGULAR_VEL
    ANGULAR_VEL = tuple(angular_vel)

    model_batch = build_box(num_worlds=TANGENT_DIM)
    engine_batch = make_engine(model_batch, differentiable=True)
    A_imp, _, _ = assemble_A_implicit_tangent(engine_batch, model_batch, TANGENT_DIM)

    model_fd = build_box(num_worlds=1)
    A_fd, _, _ = assemble_A_fd_tangent(model_fd)

    diff = np.abs(A_imp - A_fd)

    # Block-wise breakdown: rotation (rows 3..5), angular-velocity (rows 9..11).
    rot_diff = diff[3:6, :].max()
    angvel_diff = diff[9:12, 9:12].max()

    return diff.max(), rot_diff, angvel_diff, A_imp, A_fd


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    print("=== Test 1: omega = 0  (gyroscopic Jacobian inactive) ===")
    err, rot_err, angvel_err, A_imp, A_fd = run_one((0.0, 0.0, 0.0))
    print(f"max abs err overall: {err:.3e}")
    print(f"max abs err rot rows : {rot_err:.3e}")
    print(f"max abs err angvel block: {angvel_err:.3e}")

    print("\nA_implicit (12x12 tangent):")
    print(A_imp)
    print("\nA_FD (12x12 tangent):")
    print(A_fd)
    print("\nA_implicit - A_FD:")
    print(A_imp - A_fd)

    tol = 5e-4
    headline_pass = err < tol
    print(f"\n{'PASS' if headline_pass else 'FAIL'}: max abs err {err:.3e} {'<' if headline_pass else '>='} {tol}\n")

    print("=== Test 2: gyroscopic-Jacobian error scaling (tangent space) ===")
    print(f"{'|w|':>8s}  {'max overall':>12s}  {'max rot-block':>14s}  {'max angvel-block':>17s}")
    for omega_mag in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        ax = np.array([0.4, 0.1, -0.2])
        ax = ax / np.linalg.norm(ax) * omega_mag if omega_mag > 0 else ax * 0.0
        full_err, rot_err, angvel_err, _, _ = run_one(tuple(ax.tolist()))
        print(f"{omega_mag:8.3f}  {full_err:12.3e}  {rot_err:14.3e}  {angvel_err:17.3e}")

    print()
    if headline_pass:
        print("PASS: tangent-space A_t agrees with FD on all 12 directions at omega=0.")


if __name__ == "__main__":
    main()
