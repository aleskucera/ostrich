"""Planar top-down grasp of a block with an uncertain pose.

The successor of grasp_pick.py: the command is the grasp itself, not a phase schedule. A parallel-jaw
gripper hangs above the nominal block position with the pads open (80 mm). During the approach it moves to
the commanded grasp pose (x, y, yaw) while descending to the block's mid-height, then squeezes, and after a
commanded delay lifts. The block's true pose (offset dx, dy and yaw) is the instance: it differs per world
and the controller does not see it, so a grasp has to be robust to it. The open pads clear the 6 cm block
only within about +-25 deg of its yaw (its diagonal is 85 mm), so the yaw instance matters.

Controls u in [0, 1]^6:
    u[0] grasp x      X_RANGE (2u - 1)        [m]   relative to the nominal block position
    u[1] grasp y      Y_RANGE (2u - 1)        [m]
    u[2] grasp yaw    YAW_RANGE (2u - 1)      [rad]
    u[3] squeeze      F = SQUEEZE_KE V_SQUEEZE_MAX u   (0 .. 40 N), from the end of the approach on
    u[4] lift delay   T_DELAY_MAX u           [s]   after the end of the approach (smooth switch, LIFT_TAU)
    u[5] lift speed   V_LIFT_MAX u            [m/s]

Instances p = (dx, dy, yaw) per world, set with `set_instances`; the nominal grasp (`nominal_u`) is the
block's own pose, 20 N, a 0.3 s delay and 0.16 m/s.

Loss per world (final block pose against lifted-in-place-upright, plus the squeeze cost):
    ((z - Z_TARGET) / Z_SCALE)^2 + ((x - x0)^2 + (y - y0)^2) / XY_SCALE^2 + W_TILT (1 - up_z) + W_F u[3]^2

Outcome labels (`outcome`): held, held_shifted, dropped, pushed, toppled (as grasp_pick).

Modes:
    python -m examples.grasp.grasp_planar check
    python -m examples.grasp.grasp_planar modes [--worlds 256 --batches 4]
    python -m examples.grasp.grasp_planar view --x 0.5 --y 0.5 --yaw 0.5 --squeeze 0.5 --delay 0.5 --lift 0.4 --dx 0 --dy 0 --byaw 0 --vis gl

Body indices: 0 slide_x, 1 slide_y, 2 turret, 3 base, 4 left pad, 5 right pad, 6 block.
Joint DOFs: 0 x, 1 y, 2 yaw, 3 lift, 4 left pad, 5 right pad, 6-11 block (free).
"""

import argparse
import os
import time

import newton
import numpy as np
import warp as wp
from ostrich import ComplianceConfig
from ostrich import ContactsConfig
from ostrich import LinearSolverConfig
from ostrich import LinesearchConfig
from ostrich import LoggingConfig
from ostrich import NewtonRaphsonConfig
from ostrich import OstrichDifferentiableSimulator
from ostrich import RenderingConfig
from ostrich import SimulationConfig
from ostrich.collision import ContactReductionConfig
from ostrich.core.engine_config import OstrichEngineConfig
from ostrich.core.types import JointMode
from newton import Model

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# --- scene ---------------------------------------------------------------
BLOCK_HX, BLOCK_HY, BLOCK_HZ = 0.03, 0.03, 0.06
BLOCK_MASS = 1.0
MU = 0.5
PAD_HX, PAD_HY, PAD_HZ = 0.01, 0.005, 0.01
PAD_MASS = 0.05
BASE_MASS = 0.5
SLIDE_MASS = 0.5
OPENING = 0.08  # Franka Hand max opening: pad inner faces 80 mm apart when open
Z_START = 0.20  # pad centre height at the start: pad bottoms 7 cm above the block top
Z_GRASP = BLOCK_HZ  # pad centre at the block's mid-height after the descent

SQUEEZE_KE = 500.0
V_SQUEEZE_MAX = 0.08  # -> 40 N
LIFT_KE = 2.0e4
V_LIFT_MAX = 0.4
LIFT_TAU = 0.05  # width of the smooth lift switch and of the approach lag [s]
SLIDE_KE = 2.0e4
YAW_KE = 2.0e3

T_APPROACH = 0.6  # [s] move to the grasp pose and descend
T_DELAY_MAX = 0.6  # [s] lift delay range after the approach
X_RANGE, Y_RANGE, YAW_RANGE = 0.04, 0.04, np.pi / 4  # +- command box
INSTANCE_RANGE = np.array([0.03, 0.03, np.deg2rad(30.0)])  # +- dx, dy, yaw of the block

Z_TARGET = BLOCK_HZ + 0.15
Z_SCALE = 0.05
XY_SCALE = 0.05
W_TILT = 4.0
W_F = 0.5
SHIFT_TOL = 0.015

BODY_BLOCK = 6
NUM_BODIES = 7
DOF_X, DOF_Y, DOF_YAW, DOF_LIFT, DOF_PAD_L, DOF_PAD_R = 0, 1, 2, 3, 4, 5
NUM_DOFS = 12
D = 6
# --- macro steps (state in, state out): one segment is SEG_T seconds of constant velocity targets ---------------
SEG_T = 0.25
D_STATE = 25  # gripper joint q (6), joint qd (6); block position (3), quaternion (4), free-joint velocity (6)
D_LOCAL = 5  # slide x, slide y, yaw, lift, squeeze (both pads) velocity targets
LOCAL_LO = np.array([-0.10, -0.10, -1.6, -0.30, 0.0])
LOCAL_HI = np.array([0.10, 0.10, 1.6, 0.40, V_SQUEEZE_MAX])
N_JQ, N_JQD = 13, 12  # joint coordinates and dofs per world: 6 gripper + 7 / 6 block


def local_to_unit(V):
    return (np.asarray(V, dtype=np.float64) - LOCAL_LO) / (LOCAL_HI - LOCAL_LO)


def unit_to_local(u):
    return LOCAL_LO + np.asarray(u, dtype=np.float64) * (LOCAL_HI - LOCAL_LO)


@wp.kernel
def pose_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    xy0: wp.array(dtype=wp.vec2),
    last_step: int,
    z_target: float,
    z_scale: float,
    xy_scale: float,
    w_tilt: float,
    world_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    X = body_pose[last_step, w, BODY_BLOCK]
    p = wp.transform_get_translation(X)
    e = (p[2] - z_target) / z_scale
    l = e * e
    ex = (p[0] - xy0[w][0]) / xy_scale
    ey = (p[1] - xy0[w][1]) / xy_scale
    l += ex * ex + ey * ey
    up = wp.quat_rotate(wp.transform_get_rotation(X), wp.vec3(0.0, 0.0, 1.0))
    l += w_tilt * (1.0 - up[2])
    world_loss[w] = l
    wp.atomic_add(loss, 0, l)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class GraspPlanar(OstrichDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.d = D
        self.W = sim_config.num_worlds
        self.world_loss = wp.zeros(self.W, dtype=wp.float32, requires_grad=True)
        self.xy0 = wp.zeros(self.W, dtype=wp.vec2)
        self.T = self.clock.total_sim_steps
        dt = self.clock.dt
        self.t = (np.arange(self.T) + 1) * dt
        # approach profile: first-order lag of the approach-phase indicator, normalised to unit travel
        a = dt / LIFT_TAU
        prof, prev = np.zeros(self.T), 0.0
        for i in range(self.T):
            prev = prev + a * (float(self.t[i] <= T_APPROACH) - prev)
            prof[i] = prev
        self.W_app = prof / (prof.sum() * dt)  # (T,) velocity per unit of travel
        self.squeeze_on = (self.t > T_APPROACH).astype(np.float64)  # (T,)
        self.instances = np.zeros((self.W, 3))
        self.body_q0 = self.states[0].body_q.numpy().copy()
        self.set_instances(np.zeros((self.W, 3)))
        self.track_body(body_idx=BODY_BLOCK, name="block", color=(0.9, 0.5, 0.1))
        if self.viewer:
            self.viewer.set_camera(pos=wp.vec3(0.45, -0.45, 0.4), pitch=-28.0, yaw=135.0)
            self.viewer.show_ui = False

    def build_model(self) -> Model:
        b = self.builder
        b.rigid_gap = 0.01
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
        no_col = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0.0, is_visible=True)
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=0.0)
        block_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=BLOCK_MASS / (8 * BLOCK_HX * BLOCK_HY * BLOCK_HZ))
        origin = wp.transform(wp.vec3(0.0, 0.0, Z_START), wp.quat_identity())
        vel = lambda: {"joint_dof_mode": [JointMode.TARGET_VELOCITY]}
        # explicit rotational inertias: the links carry no dense shapes, and the yaw joint is the first rotational
        # degree of freedom in the chain, so a zero inertia makes the solve singular at the first pad contact
        I_link, I_pad = wp.mat33(np.diag([1e-3, 1e-3, 1e-3]).astype(np.float32)), wp.mat33(np.diag([1e-5, 1e-5, 1e-5]).astype(np.float32))
        joints = []
        sx = b.add_link(xform=origin, label="slide_x", mass=SLIDE_MASS, inertia=I_link)
        joints.append(b.add_joint_prismatic(parent=-1, child=sx, axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=origin, child_xform=wp.transform_identity(),
                                            target_ke=SLIDE_KE, target_kd=0.0, label="slide_x_j", custom_attributes=vel()))
        sy = b.add_link(xform=origin, label="slide_y", mass=SLIDE_MASS, inertia=I_link)
        joints.append(b.add_joint_prismatic(parent=sx, child=sy, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                                            target_ke=SLIDE_KE, target_kd=0.0, label="slide_y_j", custom_attributes=vel()))
        turret = b.add_link(xform=origin, label="turret", mass=SLIDE_MASS, inertia=I_link)
        joints.append(b.add_joint_revolute(parent=sy, child=turret, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                                           target_ke=YAW_KE, target_kd=0.0, label="yaw_j", custom_attributes=vel()))
        base = b.add_link(xform=origin, label="base", mass=BASE_MASS, inertia=I_link)
        b.add_shape_box(base, hx=0.06, hy=0.08, hz=0.01, xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()), cfg=no_col)  # visual only
        joints.append(b.add_joint_prismatic(parent=turret, child=base, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                                            target_ke=LIFT_KE, target_kd=0.0, label="lift", custom_attributes=vel()))
        y0 = OPENING / 2 + PAD_HY
        for name, sign in (("pad_l", 1.0), ("pad_r", -1.0)):
            pad = b.add_link(xform=wp.transform(wp.vec3(0.0, sign * y0, Z_START), wp.quat_identity()), label=name, mass=PAD_MASS, inertia=I_pad)
            b.add_shape_box(pad, hx=PAD_HX, hy=PAD_HY, hz=PAD_HZ, cfg=pad_cfg)
            joints.append(b.add_joint_prismatic(parent=base, child=pad, axis=wp.vec3(0.0, -sign, 0.0),
                                                parent_xform=wp.transform(wp.vec3(0.0, sign * y0, 0.0), wp.quat_identity()), child_xform=wp.transform_identity(),
                                                target_ke=SQUEEZE_KE, target_kd=0.0, label=name + "_j", custom_attributes=vel()))
        b.add_articulation(joints, label="gripper")
        block = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, BLOCK_HZ), wp.quat_identity()), label="block")
        b.add_shape_box(block, hx=BLOCK_HX, hy=BLOCK_HY, hz=BLOCK_HZ, cfg=block_cfg)
        b.add_articulation([b.add_joint_free(parent=-1, child=block, label="block_free")], label="block")
        return b.finalize_replicated(num_worlds=self.simulation_config.num_worlds, requires_grad=True)

    # --- instances and controls ----------------------------------------

    def set_instances(self, P):
        """P (W, 3): block offset dx, dy [m] and yaw [rad] per world."""
        P = np.asarray(P, dtype=np.float64)
        self.instances = P.copy()
        q = self.body_q0.copy()
        for w in range(self.W):
            i = w * NUM_BODIES + BODY_BLOCK
            q[i, 0] = P[w, 0]
            q[i, 1] = P[w, 1]
            q[i, 3:7] = [0.0, 0.0, np.sin(P[w, 2] / 2), np.cos(P[w, 2] / 2)]
        self.states[0].body_q.assign(q)
        self.xy0.assign(np.ascontiguousarray(P[:, :2], dtype=np.float32))

    def random_instances(self, rng):
        return (2 * rng.random((self.W, 3)) - 1) * INSTANCE_RANGE

    def nominal_u(self, P=None):
        return nominal_commands(self.instances if P is None else P)

    def _lift_profile(self, U):
        """Lift velocity per step and world (T, W) and its derivatives wrt u[4] (delay) and u[5] (speed)."""
        t_on = T_APPROACH + T_DELAY_MAX * U[:, 4]  # (W,)
        s = _sigmoid((self.t[:, None] - t_on[None, :]) / LIFT_TAU)  # (T, W)
        v = V_LIFT_MAX * U[None, :, 5] * s
        dv_ddelay = -V_LIFT_MAX * U[None, :, 5] * s * (1 - s) * T_DELAY_MAX / LIFT_TAU
        dv_dspeed = V_LIFT_MAX * s
        return v, dv_ddelay, dv_dspeed

    def apply(self, U):
        self._copy_targets(self._targets(U))

    def _copy_targets(self, vel):
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel, dtype=wp.float32))
        for i in range(self.T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def _targets(self, U):
        """Velocity targets (T, W, NUM_DOFS) of the episode commands U (W, 6)."""
        U = np.asarray(U, dtype=np.float64)
        vel = np.zeros((self.T, self.W, NUM_DOFS), dtype=np.float32)
        vel[:, :, DOF_X] = self.W_app[:, None] * (X_RANGE * (2 * U[:, 0] - 1))[None, :]
        vel[:, :, DOF_Y] = self.W_app[:, None] * (Y_RANGE * (2 * U[:, 1] - 1))[None, :]
        vel[:, :, DOF_YAW] = self.W_app[:, None] * (YAW_RANGE * (2 * U[:, 2] - 1))[None, :]
        v_lift, _, _ = self._lift_profile(U)
        vel[:, :, DOF_LIFT] = -self.W_app[:, None] * (Z_START - Z_GRASP) + v_lift
        sq = self.squeeze_on[:, None] * (V_SQUEEZE_MAX * U[:, 3])[None, :]
        vel[:, :, DOF_PAD_L] = sq
        vel[:, :, DOF_PAD_R] = sq
        return vel

    # --- macro steps ---------------------------------------------------

    def local_targets(self, U, n_seg):
        """The episode commands as n_seg piecewise-constant local commands: segment means of the velocity targets,
        (n_seg, W, D_LOCAL) in physical units."""
        v = self._targets(U).reshape(n_seg, self.T // n_seg, self.W, NUM_DOFS).mean(1)
        return np.stack([v[..., DOF_X], v[..., DOF_Y], v[..., DOF_YAW], v[..., DOF_LIFT], v[..., DOF_PAD_L]], -1).astype(np.float64)

    def _vel_of_local(self, V):
        vel = np.zeros((self.W, NUM_DOFS), dtype=np.float32)
        vel[:, DOF_X], vel[:, DOF_Y], vel[:, DOF_YAW], vel[:, DOF_LIFT] = V[:, 0], V[:, 1], V[:, 2], V[:, 3]
        vel[:, DOF_PAD_L] = vel[:, DOF_PAD_R] = V[:, 4]
        return vel

    def apply_local(self, V):
        """Local commands V (W, D_LOCAL), physical units, held for every step of this simulation (one macro step)."""
        self._copy_targets(np.broadcast_to(self._vel_of_local(np.asarray(V, dtype=np.float64)), (self.T, self.W, NUM_DOFS)).copy())

    def apply_piecewise(self, VS):
        """VS (n_seg, W, D_LOCAL): piecewise-constant local commands over this simulation's horizon."""
        n_seg, L = len(VS), self.T // len(VS)
        vel = np.zeros((self.T, self.W, NUM_DOFS), dtype=np.float32)
        for k in range(n_seg):
            vel[k * L: (k + 1) * L] = self._vel_of_local(np.asarray(VS[k], dtype=np.float64))[None]
        self._copy_targets(vel)

    def _joint_scratch(self):
        if not hasattr(self, "_jq"):
            self._jq, self._jqd = wp.zeros_like(self.model.joint_q), wp.zeros_like(self.model.joint_qd)
            assert self._jq.shape[0] == self.W * N_JQ and self._jqd.shape[0] == self.W * N_JQD, (self._jq.shape, self._jqd.shape)
        return self._jq, self._jqd

    def set_state(self, S):
        """Reduced state S (W, D_STATE) -> body poses and twists of states[0] by forward kinematics."""
        S = np.asarray(S, dtype=np.float64)
        jq, jqd = np.zeros((self.W, N_JQ)), np.zeros((self.W, N_JQD))
        jq[:, :6], jqd[:, :6] = S[:, :6], S[:, 6:12]
        jq[:, 6:9] = S[:, 12:15]
        jq[:, 9:13] = S[:, 15:19] / np.linalg.norm(S[:, 15:19], axis=1, keepdims=True)
        jqd[:, 6:12] = S[:, 19:25]
        q_arr, qd_arr = self._joint_scratch()
        q_arr.assign(np.ascontiguousarray(jq.reshape(-1), dtype=np.float32))
        qd_arr.assign(np.ascontiguousarray(jqd.reshape(-1), dtype=np.float32))
        newton.eval_fk(self.model, q_arr, qd_arr, self.states[0])

    def get_state(self, step=None):
        """Reduced state (W, D_STATE) of states[step] (default: the last) by inverse kinematics."""
        q_arr, qd_arr = self._joint_scratch()
        newton.eval_ik(self.model, self.states[self.T if step is None else step], q_arr, qd_arr)
        jq = q_arr.numpy().reshape(self.W, N_JQ).astype(np.float64)
        jqd = qd_arr.numpy().reshape(self.W, N_JQD).astype(np.float64)
        return np.concatenate([jq[:, :6], jqd[:, :6], jq[:, 6:13], jqd[:, 6:12]], 1)

    def run_segment(self, S, V):
        """One macro step: start states S (W, D_STATE), local commands V (W, D_LOCAL) -> end states (W, D_STATE)."""
        self.set_state(S)
        self.apply_local(V)
        self.forward_step()
        wp.synchronize()
        self.loss.zero_()
        return self.get_state()

    def grad_u(self, U):
        """Adjoint gradient through the velocity targets plus the squeeze cost's direct part."""
        U = np.asarray(U, dtype=np.float64)
        g = self.trajectory.joint_target_vel.grad.numpy().astype(np.float64)  # (T, W, NUM_DOFS)
        G = np.zeros((self.W, self.d))
        G[:, 0] = 2 * X_RANGE * (self.W_app[:, None] * g[:, :, DOF_X]).sum(0)
        G[:, 1] = 2 * Y_RANGE * (self.W_app[:, None] * g[:, :, DOF_Y]).sum(0)
        G[:, 2] = 2 * YAW_RANGE * (self.W_app[:, None] * g[:, :, DOF_YAW]).sum(0)
        G[:, 3] = V_SQUEEZE_MAX * (self.squeeze_on[:, None] * (g[:, :, DOF_PAD_L] + g[:, :, DOF_PAD_R])).sum(0) + 2 * W_F * U[:, 3]
        _, dv_dd, dv_ds = self._lift_profile(U)
        G[:, 4] = (dv_dd * g[:, :, DOF_LIFT]).sum(0)
        G[:, 5] = (dv_ds * g[:, :, DOF_LIFT]).sum(0)
        return G

    def compute_loss(self):
        wp.launch(pose_loss_kernel, dim=self.W,
                  inputs=[self.trajectory.body_pose, self.xy0, self.T, Z_TARGET, Z_SCALE, XY_SCALE, W_TILT],
                  outputs=[self.world_loss, self.loss], device=self.model.device)

    def update(self):
        pass

    def _losses(self, U):
        return self.world_loss.numpy().astype(np.float64) + W_F * np.asarray(U, dtype=np.float64)[:, 3] ** 2

    def rollout(self, U):
        self.apply(U)
        self.diff_step()
        wp.synchronize()
        L = self._losses(U)
        G = self.grad_u(U)
        self.trajectory.joint_target_vel.grad.zero_()
        self.tape.zero()
        self.loss.zero_()
        return L, G

    def rollout_forward(self, U):
        """Losses (W,) without the backward pass (data generation; the final poses are in the trajectory as usual)."""
        self.apply(U)
        self.forward_step()
        wp.synchronize()
        L = self._losses(U)
        self.loss.zero_()
        return L

    def block_state(self):
        """Final block pose (W, 7)."""
        return self.trajectory.body_pose.numpy()[-1, :, BODY_BLOCK, :]

    def outcome(self):
        return outcome_of(self.block_state(), self.instances)


OUTCOMES = ("held", "held_shifted", "dropped", "pushed", "toppled")


def outcome_of(X, P):
    """Outcome label per world from the final block pose X (W, 7) and the instances P (W, 3)."""
    z = X[:, 2]
    shift = np.linalg.norm(X[:, :2] - P[:, :2], axis=1)
    qx, qy = X[:, 3], X[:, 4]
    up_z = 1 - 2 * (qx * qx + qy * qy)
    out = np.empty(len(X), dtype=object)
    held = z > BLOCK_HZ + 0.05
    out[held & (shift <= SHIFT_TOL)] = "held"
    out[held & (shift > SHIFT_TOL)] = "held_shifted"
    out[~held & (shift <= SHIFT_TOL)] = "dropped"
    out[~held & (shift > SHIFT_TOL)] = "pushed"
    out[up_z < 0.7] = "toppled"
    return out


def nominal_commands(P):
    """The block's own pose as the grasp, 20 N, 0.3 s delay, 0.16 m/s: (n, 6) in [0, 1]."""
    P = np.asarray(P, dtype=np.float64)
    u = np.tile([0.5, 0.5, 0.5, 0.5, 0.5, 0.4], (len(P), 1))
    u[:, 0] = (P[:, 0] / X_RANGE + 1) / 2
    u[:, 1] = (P[:, 1] / Y_RANGE + 1) / 2
    u[:, 2] = (P[:, 2] / YAW_RANGE + 1) / 2
    return np.clip(u, 0, 1)


def make_sim(num_worlds, duration=2.0, dt=5e-3, vis="null"):
    sim_config = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt, num_worlds=num_worlds)
    render_config = RenderingConfig(vis_type=vis)
    engine_config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=64, backtrack_min_iter=12, atol=1e-4),
        linear=LinearSolverConfig(max_iters=64, tol=1e-6, atol=1e-6, regularization=1e-6),
        compliance=ComplianceConfig(joint=1e-9, contact=1e-6, friction=1e-8),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=64, reduction=ContactReductionConfig(policy="cluster", max_per_pair=8)),
    )
    return GraspPlanar(sim_config, render_config, engine_config, LoggingConfig())


def check(args):
    W = 8
    sim = make_sim(W, vis=args.vis)
    print(f"bodies {sim.model.body_count} joint dofs {sim.model.joint_dof_count} steps {sim.T} dt {sim.clock.dt}")
    # rows 0-3: centred block, nominal grasp, four squeeze levels; rows 4-7: offset / yawed blocks with the nominal
    # grasp of the *centred* block (what a controller that cannot see the instance does)
    P = np.zeros((W, 3))
    P[4] = [0.02, 0.0, 0.0]
    P[5] = [0.0, 0.02, 0.0]
    P[6] = [0.0, 0.0, np.deg2rad(15)]
    P[7] = [0.0, 0.0, np.deg2rad(30)]
    sim.set_instances(P)
    U = sim.nominal_u(np.zeros((W, 3)))
    U[:4, 3] = [0.2, 0.35, 0.5, 0.8]
    t0 = time.perf_counter()
    L, G = sim.rollout(U)
    print(f"rollout + adjoint: {time.perf_counter() - t0:.2f} s")
    X, out = sim.block_state(), sim.outcome()
    for w in range(W):
        print(f"  inst dx {P[w, 0]:+.3f} dy {P[w, 1]:+.3f} yaw {np.rad2deg(P[w, 2]):+5.1f}  squeeze {SQUEEZE_KE * V_SQUEEZE_MAX * U[w, 3]:5.1f} N  "
              f"-> block x {X[w, 0]:+.3f} y {X[w, 1]:+.3f} z {X[w, 2]:.3f}  {out[w]:13s} loss {L[w]:.3f}  |g| {np.linalg.norm(G[w]):.3f}")
    h = 1e-3
    for j in range(D):
        Up, Um = U.copy(), U.copy()
        Up[:, j] += h
        Um[:, j] -= h
        Lp, _ = sim.rollout(Up)
        Lm, _ = sim.rollout(Um)
        print(f"  u[{j}]: adjoint {np.round(G[:, j], 3)}  FD {np.round((Lp - Lm) / (2 * h), 3)}")
    sim.close()


def modes(args):
    """Random commands and instances: outcome counts and the loss per outcome."""
    W = args.worlds
    sim = make_sim(W, vis=args.vis)
    rng = np.random.default_rng(args.seed)
    Ls, outs, Us, Ps = [], [], [], []
    for b in range(args.batches):
        P = sim.random_instances(rng)
        sim.set_instances(P)
        if args.around_nominal:
            U = np.clip(sim.nominal_u() + args.sigma * rng.standard_normal((W, sim.d)), 0, 1)
        else:
            U = rng.random((W, sim.d))
        t0 = time.perf_counter()
        L = sim.rollout_forward(U)
        Ls.append(L)
        outs.append(sim.outcome())
        Us.append(U)
        Ps.append(P)
        print(f"batch {b} ({time.perf_counter() - t0:.0f} s): " + "  ".join(f"{o} {int((outs[-1] == o).sum())}" for o in OUTCOMES), flush=True)
    L, out, U, P = np.concatenate(Ls), np.concatenate(outs), np.concatenate(Us), np.concatenate(Ps)
    print(f"\n{len(L)} rollouts, {'around the nominal grasp' if args.around_nominal else 'uniform commands'}:")
    for o in OUTCOMES:
        m = out == o
        if m.any():
            print(f"  {o:13s} {m.mean():5.1%}  loss mean {L[m].mean():6.2f}  sd {L[m].std():5.2f}  min {L[m].min():6.2f}  max {L[m].max():6.2f}")
    np.savez(args.out, L=L, outcome=out.astype(str), U=U, P=P)
    sim.close()


def view(args):
    sim = make_sim(1, vis=args.vis)
    sim.set_instances(np.array([[args.dx, args.dy, np.deg2rad(args.byaw)]]))
    U = sim.nominal_u()
    for j, v in enumerate((args.x, args.y, args.yaw, args.squeeze, args.delay, args.lift)):
        if v is not None:
            U[:, j] = v
    L, _ = sim.rollout(U)
    X = sim.block_state()[0]
    print(f"u {np.round(U[0], 2).tolist()}  block x {X[0]:+.3f} y {X[1]:+.3f} z {X[2]:.3f}  {sim.outcome()[0]}  loss {L[0]:.3f}")
    sim.render_episode(iteration=0, loop=True, loops_count=args.loops, playback_speed=args.speed)
    sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "modes", "view"])
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--around-nominal", action="store_true", help="modes: commands = nominal grasp + sigma noise instead of uniform")
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="examples/grasp/planar_modes.npz")
    p.add_argument("--vis", default="null", choices=["null", "gl", "usd"])
    for name in ("x", "y", "yaw", "squeeze", "delay", "lift"):
        p.add_argument(f"--{name}", type=float, default=None, help=f"view: command u for {name}; default = nominal")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--byaw", type=float, default=0.0, help="view: block yaw [deg]")
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--speed", type=float, default=0.5)
    args = p.parse_args()
    {"check": check, "modes": modes, "view": view}[args.mode](args)
