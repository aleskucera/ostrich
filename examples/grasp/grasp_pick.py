"""Pick with an uncertain object pose: approach, close, lift.

The realistic version of grasp_lift.py. The parallel-jaw gripper starts
behind the block (at x = X_START, pads open), slides toward it during the
first phase, squeezes and lifts. The block's initial pose is an *instance*
parameter, different in every world: offset (dx, dy) and yaw, drawn from
INSTANCE_RANGE. The loss is the final block pose against a target: lifted by
0.15 m, at its own initial xy, upright. Pushing the block, missing it,
pivoting it or dropping it all show up as different losses.

Controls u in [0, 1]^(2 + 2k):
    u[0] approach distance   base x travel = X_TRAVEL[0] + u[0] (X_TRAVEL[1] - X_TRAVEL[0])
    u[1] approach side       base y travel = Y_TRAVEL * (2 u[1] - 1)
    u[2 + 2j] squeeze force in phase j   F = SQUEEZE_KE V_SQUEEZE_MAX u
    u[3 + 2j] lift speed in phase j      v = V_LIFT_MAX u, through a LIFT_TAU lag
The approach travel is spread over phase 0 (through the same lag), so a
squeeze command in phase 0 closes the pads while the gripper still moves.

Instances p = (dx, dy, yaw) per world, set with `set_instances`; the nominal
grasp (`nominal_u`) approaches in phase 0 with the pads open and still, with
u[0] such that the base reaches x = dx and u[1] such that the pads straddle
the block, closes with 20 N in phase 1, then lifts at 0.16 m/s from phase 2.
Lifting while still closing loses the block over its top edge (the pads need
0.25 s to close the 1 cm gap per side).

Loss per world:
    ((z - Z_TARGET) / Z_SCALE)^2 + ((x - x0)^2 + (y - y0)^2) / XY_SCALE^2
    + W_TILT (1 - up_z) + W_F mean_j u_squeeze_j^2

Outcome labels (`outcome`): held, held_shifted (held, moved > SHIFT_TOL in xy),
dropped (on the table near its start), pushed (on the table, moved), toppled
(tilted more than 45 deg). These are the candidate modes; `modes` samples
random commands and instances and counts them.

Modes:
    python -m examples.grasp.grasp_pick check
    python -m examples.grasp.grasp_pick modes [--worlds 32 --batches 8]
    python -m examples.grasp.grasp_pick view --ax 0.5 --ay 0.5 --squeeze 0.5 --lift 0.35 --dx 0 --dy 0 --yaw 0 --vis gl

Body indices: 0 slide_x, 1 slide_y, 2 base, 3 left pad, 4 right pad, 5 block.
Joint DOFs: 0 x, 1 y, 2 lift, 3 left pad, 4 right pad, 5-10 block (free).
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
X_START = -0.12  # gripper start, behind the block

SQUEEZE_KE = 500.0
V_SQUEEZE_MAX = 0.08  # -> 40 N
LIFT_KE = 2.0e4
V_LIFT_MAX = 0.4
LIFT_TAU = 0.05
SLIDE_KE = 2.0e4

X_TRAVEL = (0.06, 0.18)  # approach travel range [m]; the block centre is reached at 0.12 + dx
Y_TRAVEL = 0.03  # +- side travel [m]
INSTANCE_RANGE = np.array([0.05, 0.015, np.deg2rad(15.0)])  # +- dx, dy, yaw

Z_TARGET = BLOCK_HZ + 0.15
Z_SCALE = 0.05
XY_SCALE = 0.05
W_TILT = 4.0
W_F = 0.5
SHIFT_TOL = 0.015

BODY_BLOCK = 5
NUM_BODIES = 6
DOF_X, DOF_Y, DOF_LIFT, DOF_PAD_L, DOF_PAD_R = 0, 1, 2, 3, 4
NUM_DOFS = 11


@wp.kernel
def pick_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    xy0: wp.array(dtype=wp.vec2),
    last_step: int,
    num_steps: int,
    z_target: float,
    z_scale: float,
    xy_scale: float,
    w_tilt: float,
    w_f: float,
    v_squeeze_max: float,
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
    s = float(0.0)
    for t in range(num_steps):
        u = target_vel[t, w, DOF_PAD_L] / v_squeeze_max
        s += u * u
    l += w_f * s / float(num_steps)
    world_loss[w] = l
    wp.atomic_add(loss, 0, l)


class GraspPick(OstrichDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config, phases=5):
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.k = phases
        self.d = 2 + 2 * phases
        self.W = sim_config.num_worlds
        self.world_loss = wp.zeros(self.W, dtype=wp.float32, requires_grad=True)
        self.xy0 = wp.zeros(self.W, dtype=wp.vec2)
        self.T = self.clock.total_sim_steps
        self.phase = np.arange(self.T) * self.k // self.T
        dt = self.clock.dt
        a = dt / LIFT_TAU
        # lift target: first-order lag of the phase indicator, (T, k)
        P = np.zeros((self.T, self.k))
        P[np.arange(self.T), self.phase] = 1.0
        self.W_lift = np.zeros((self.T, self.k))
        prev = np.zeros(self.k)
        for t in range(self.T):
            prev = prev + a * (P[t] - prev)
            self.W_lift[t] = prev
        # approach profile: lag of the phase-0 indicator, scaled so that the travel integrates to 1 m per unit command
        prof = np.zeros(self.T)
        prev = 0.0
        for t in range(self.T):
            prev = prev + a * (float(self.phase[t] == 0) - prev)
            prof[t] = prev
        self.W_app = prof / (prof.sum() * dt)  # (T,) velocity per metre of travel
        self.instances = np.zeros((self.W, 3))
        self.body_q0 = self.states[0].body_q.numpy().copy()
        self.set_instances(np.zeros((self.W, 3)))
        self.track_body(body_idx=BODY_BLOCK, name="block", color=(0.9, 0.5, 0.1))
        if self.viewer:
            self.viewer.set_camera(pos=wp.vec3(0.45, -0.45, 0.35), pitch=-20.0, yaw=135.0)
            self.viewer.show_ui = False

    def build_model(self) -> Model:
        b = self.builder
        b.rigid_gap = 0.01
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
        no_col = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0.0, is_visible=True)
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=0.0)
        block_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=BLOCK_MASS / (8 * BLOCK_HX * BLOCK_HY * BLOCK_HZ))
        z0 = BLOCK_HZ
        joints = []
        # xy slide: two massive links on prismatic joints (velocity servos), then the lift
        sx = b.add_link(xform=wp.transform(wp.vec3(X_START, 0.0, z0), wp.quat_identity()), label="slide_x", mass=SLIDE_MASS)
        joints.append(b.add_joint_prismatic(
            parent=-1, child=sx, axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(X_START, 0.0, z0), wp.quat_identity()), child_xform=wp.transform_identity(),
            target_ke=SLIDE_KE, target_kd=0.0, label="slide_x_j", custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        sy = b.add_link(xform=wp.transform(wp.vec3(X_START, 0.0, z0), wp.quat_identity()), label="slide_y", mass=SLIDE_MASS)
        joints.append(b.add_joint_prismatic(
            parent=sx, child=sy, axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
            target_ke=SLIDE_KE, target_kd=0.0, label="slide_y_j", custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        base = b.add_link(xform=wp.transform(wp.vec3(X_START, 0.0, z0), wp.quat_identity()), label="base", mass=BASE_MASS)
        b.add_shape_box(base, hx=0.06, hy=0.08, hz=0.01, xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()), cfg=no_col)  # visual only, clear of the block top
        joints.append(b.add_joint_prismatic(
            parent=sy, child=base, axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
            target_ke=LIFT_KE, target_kd=0.0, label="lift", custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        y0 = OPENING / 2 + PAD_HY
        for name, sign in (("pad_l", 1.0), ("pad_r", -1.0)):
            pad = b.add_link(xform=wp.transform(wp.vec3(X_START, sign * y0, z0), wp.quat_identity()), label=name, mass=PAD_MASS)
            b.add_shape_box(pad, hx=PAD_HX, hy=PAD_HY, hz=PAD_HZ, cfg=pad_cfg)
            joints.append(b.add_joint_prismatic(
                parent=base, child=pad, axis=wp.vec3(0.0, -sign, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, sign * y0, 0.0), wp.quat_identity()), child_xform=wp.transform_identity(),
                target_ke=SQUEEZE_KE, target_kd=0.0, label=name + "_j", custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]}))
        b.add_articulation(joints, label="gripper")
        block = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()), label="block")
        b.add_shape_box(block, hx=BLOCK_HX, hy=BLOCK_HY, hz=BLOCK_HZ, cfg=block_cfg)
        b.add_articulation([b.add_joint_free(parent=-1, child=block, label="block_free")], label="block")
        return b.finalize_replicated(num_worlds=self.simulation_config.num_worlds, requires_grad=True)

    # --- instances and controls ----------------------------------------

    def set_instances(self, P):
        """P (W, 3): block initial offset dx, dy [m] and yaw [rad] per world."""
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
        """Commands that centre the pads on each block: (W, d)."""
        return nominal_commands(self.instances if P is None else P, self.k)

    def apply(self, U):
        U = np.asarray(U, dtype=np.float64)
        vel = np.zeros((self.T, self.W, NUM_DOFS), dtype=np.float32)
        tx = X_TRAVEL[0] + U[:, 0] * (X_TRAVEL[1] - X_TRAVEL[0])
        ty = Y_TRAVEL * (2 * U[:, 1] - 1)
        vel[:, :, DOF_X] = self.W_app[:, None] * tx[None, :]
        vel[:, :, DOF_Y] = self.W_app[:, None] * ty[None, :]
        sq = V_SQUEEZE_MAX * U[:, 2::2]
        lift = V_LIFT_MAX * U[:, 3::2]
        vel[:, :, DOF_PAD_L] = sq.T[self.phase]
        vel[:, :, DOF_PAD_R] = sq.T[self.phase]
        vel[:, :, DOF_LIFT] = self.W_lift @ lift.T
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel, dtype=wp.float32))
        for i in range(self.T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def grad_u(self, U):
        """Adjoint gradient through the velocity targets plus the squeeze cost's direct part (see grasp_lift.grad_u)."""
        g = self.trajectory.joint_target_vel.grad.numpy()
        G = np.zeros((self.W, self.d))
        G[:, 0] = (X_TRAVEL[1] - X_TRAVEL[0]) * (self.W_app[:, None] * g[:, :, DOF_X]).sum(0)
        G[:, 1] = 2 * Y_TRAVEL * (self.W_app[:, None] * g[:, :, DOF_Y]).sum(0)
        U = np.asarray(U, dtype=np.float64)
        for j in range(self.k):
            steps = self.phase == j
            G[:, 2 + 2 * j] = V_SQUEEZE_MAX * (g[steps, :, DOF_PAD_L].sum(0) + g[steps, :, DOF_PAD_R].sum(0)) + W_F * 2.0 * U[:, 2 + 2 * j] * steps.sum() / self.T
            G[:, 3 + 2 * j] = V_LIFT_MAX * (self.W_lift[:, j][:, None] * g[:, :, DOF_LIFT]).sum(0)
        return G

    def compute_loss(self):
        wp.launch(
            pick_loss_kernel, dim=self.W,
            inputs=[self.trajectory.body_pose, self.trajectory.joint_target_vel, self.xy0, self.T, self.T,
                    Z_TARGET, Z_SCALE, XY_SCALE, W_TILT, W_F, V_SQUEEZE_MAX],
            outputs=[self.world_loss, self.loss],
            device=self.model.device,
        )

    def update(self):
        pass

    def rollout(self, U):
        self.apply(U)
        self.diff_step()
        wp.synchronize()
        L = self.world_loss.numpy().copy()
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
        L = self.world_loss.numpy().copy()
        self.loss.zero_()
        return L

    def block_state(self):
        """Final block pose (W, 7)."""
        return self.trajectory.body_pose.numpy()[-1, :, BODY_BLOCK, :]

    def outcome(self):
        """Outcome label per world from the final block pose."""
        X = self.block_state()
        z = X[:, 2]
        shift = np.linalg.norm(X[:, :2] - self.instances[:, :2], axis=1)
        qx, qy, qz, qw = X[:, 3], X[:, 4], X[:, 5], X[:, 6]
        up_z = 1 - 2 * (qx * qx + qy * qy)
        out = np.empty(self.W, dtype=object)
        held = z > BLOCK_HZ + 0.05
        out[held & (shift <= SHIFT_TOL)] = "held"
        out[held & (shift > SHIFT_TOL)] = "held_shifted"
        out[~held & (shift <= SHIFT_TOL)] = "dropped"
        out[~held & (shift > SHIFT_TOL)] = "pushed"
        out[up_z < 0.7] = "toppled"
        return out


OUTCOMES = ("held", "held_shifted", "dropped", "pushed", "toppled")


def nominal_commands(P, k):
    """Commands that centre the pads on blocks at poses P (n, 3), physical units: (n, 2k + 2)."""
    P = np.asarray(P)
    # phase 0: approach with the pads open and still; phase 1: close (20 N), no lift; then squeeze and lift
    u = np.tile([0.0, 0.0, 0.0, 0.0, 0.5, 0.0] + [0.5, 0.4] * (k - 2), (len(P), 1)).astype(np.float64)
    u[:, 0] = ((-X_START + P[:, 0]) - X_TRAVEL[0]) / (X_TRAVEL[1] - X_TRAVEL[0])
    u[:, 1] = (P[:, 1] / Y_TRAVEL + 1) / 2
    return np.clip(u, 0, 1)


def make_sim(num_worlds, phases, duration=1.6, dt=5e-3, vis="null"):
    sim_config = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt, num_worlds=num_worlds)
    render_config = RenderingConfig(vis_type=vis)
    engine_config = OstrichEngineConfig(
        # 64/64: with grasp_lift's 24/32 the six-body chain does not converge and the pad normal force sags
        # from 20 N to 6 N mid-lift, dropping the block (checked 20 Oct 2026 on dasenka)
        nr=NewtonRaphsonConfig(max_iters=64, backtrack_min_iter=12, atol=1e-4),
        linear=LinearSolverConfig(max_iters=64, tol=1e-6, atol=1e-6, regularization=1e-6),
        compliance=ComplianceConfig(joint=1e-9, contact=1e-6, friction=1e-8),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=64, reduction=ContactReductionConfig(policy="cluster", max_per_pair=8)),
    )
    return GraspPick(sim_config, render_config, engine_config, LoggingConfig(), phases=phases)


def check(args):
    W = 8
    sim = make_sim(W, args.phases, vis=args.vis)
    print(f"bodies {sim.model.body_count} joint dofs {sim.model.joint_dof_count} steps {sim.T} dt {sim.clock.dt}")
    # row 0-3: centred block, nominal approach, four squeeze levels; row 4-7: offsets with the nominal (corrected) approach
    P = np.zeros((W, 3))
    P[4] = [0.03, 0.0, 0.0]
    P[5] = [0.0, 0.012, 0.0]
    P[6] = [0.0, 0.0, np.deg2rad(12)]
    P[7] = [-0.04, -0.01, np.deg2rad(-10)]
    sim.set_instances(P)
    U = sim.nominal_u()
    U[:4, 4::2] = np.array([0.2, 0.35, 0.5, 0.8])[:, None]  # squeeze from phase 1 on
    # uncorrected approach for the offset rows (what a policy that ignores the instance would do)
    U[4:, 0:2] = sim.nominal_u(np.zeros((W, 3)))[4:, 0:2]
    t0 = time.perf_counter()
    L, G = sim.rollout(U)
    print(f"rollout + adjoint: {time.perf_counter() - t0:.2f} s")
    X, out = sim.block_state(), sim.outcome()
    for w in range(W):
        print(f"  inst dx {P[w, 0]:+.3f} dy {P[w, 1]:+.3f} yaw {np.rad2deg(P[w, 2]):+5.1f}  squeeze {SQUEEZE_KE * V_SQUEEZE_MAX * U[w, 4]:5.1f} N  "
              f"ax {U[w, 0]:.2f} ay {U[w, 1]:.2f} -> block x {X[w, 0]:+.3f} y {X[w, 1]:+.3f} z {X[w, 2]:.3f}  {out[w]:13s} loss {L[w]:.3f}  |g| {np.linalg.norm(G[w]):.3f}")
    h = 1e-3
    for j in (0, 1, 7):
        Up, Um = U.copy(), U.copy()
        Up[:, j] += h
        Um[:, j] -= h
        Lp, _ = sim.rollout(Up)
        Lm, _ = sim.rollout(Um)
        print(f"  coord {j}: adjoint {np.round(G[:, j], 3)}  FD {np.round((Lp - Lm) / (2 * h), 3)}")
    sim.close()


def modes(args):
    """Random commands and instances: outcome counts and the loss per outcome."""
    W = args.worlds
    sim = make_sim(W, args.phases, vis=args.vis)
    rng = np.random.default_rng(args.seed)
    Ls, outs, Us, Ps = [], [], [], []
    for b in range(args.batches):
        P = sim.random_instances(rng)
        sim.set_instances(P)
        if args.around_nominal:
            U = np.clip(sim.nominal_u() + args.sigma * rng.standard_normal((W, sim.d)), 0, 1)
        else:
            U = rng.random((W, sim.d))
        L, _ = sim.rollout(U)
        Ls.append(L)
        outs.append(sim.outcome())
        Us.append(U)
        Ps.append(P)
        print(f"batch {b}: " + "  ".join(f"{o} {int((outs[-1] == o).sum())}" for o in OUTCOMES), flush=True)
    L, out, U, P = np.concatenate(Ls), np.concatenate(outs), np.concatenate(Us), np.concatenate(Ps)
    print(f"\n{len(L)} rollouts, {'around the nominal grasp' if args.around_nominal else 'uniform commands'}:")
    for o in OUTCOMES:
        m = out == o
        if m.any():
            print(f"  {o:13s} {m.mean():5.1%}  loss mean {L[m].mean():6.2f}  sd {L[m].std():5.2f}  min {L[m].min():6.2f}  max {L[m].max():6.2f}")
    np.savez(args.out, L=L, outcome=out.astype(str), U=U, P=P)
    sim.close()


def view(args):
    sim = make_sim(1, args.phases, vis=args.vis)
    sim.set_instances(np.array([[args.dx, args.dy, np.deg2rad(args.yaw)]]))
    U = sim.nominal_u()
    if args.ax is not None:
        U[:, 0], U[:, 1] = args.ax, args.ay
    U[:, 4::2] = args.squeeze
    U[:, 7::2] = args.lift
    L, _ = sim.rollout(U)
    X = sim.block_state()[0]
    print(f"u {np.round(U[0], 2).tolist()}  block x {X[0]:+.3f} y {X[1]:+.3f} z {X[2]:.3f}  {sim.outcome()[0]}  loss {L[0]:.3f}")
    sim.render_episode(iteration=0, loop=True, loops_count=args.loops, playback_speed=args.speed)
    sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "modes", "view"])
    p.add_argument("--phases", type=int, default=5)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--around-nominal", action="store_true", help="modes: commands = nominal grasp + sigma noise instead of uniform")
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="examples/grasp/pick_modes.npz")
    p.add_argument("--vis", default="null", choices=["null", "gl", "usd"])
    p.add_argument("--ax", type=float, default=None, help="view: approach command; default = nominal for the instance")
    p.add_argument("--ay", type=float, default=0.5)
    p.add_argument("--squeeze", type=float, default=0.5)
    p.add_argument("--lift", type=float, default=0.4)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0, help="degrees")
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--speed", type=float, default=0.5)
    a = p.parse_args()
    {"check": check, "modes": modes, "view": view}[a.mode](a)
