"""Grasp and lift: a parallel-jaw gripper lifts a block with the least squeeze.

Scene (Ostrich, 3D): a block rests on the ground between two finger pads. The
gripper base slides vertically on a prismatic joint; each pad slides toward
the block on its own prismatic joint. All three joints are velocity servos
(JointMode.TARGET_VELOCITY): the drive force is ke * (v_target - v), so a
blocked pad pushes with ke * v_target. The squeeze command therefore sets a
force, the lift command a speed.

Controls: k phases over the run, each with a squeeze command and a lift
command, u in [0, 1]^(2k):

    squeeze force F_j = SQUEEZE_KE * V_SQUEEZE_MAX * u[2j]
    lift speed     v_j = V_LIFT_MAX * u[2j + 1], passed through a first-order
                   lag of LIFT_TAU (a linear map, so the gradient chains through it)

Loss per world: ((z_block(T) - Z_TARGET) / Z_SCALE)^2 + W_F mean_j u[2j]^2.
The block (1 kg) is held while 2 mu F covers its weight plus the lift's
acceleration, i.e. F above about 10 N; squeezing less drops it (a loss jump
of about (Z_TARGET - HZ)^2 / Z_SCALE^2 = 9). Dimensions follow a Franka
Hand: 2 x 2 cm fingertip pads, 68 mm opening, 0 to 40 N squeeze. The squeeze cost pulls the optimum to
that edge, as in experiments/12_smoothed_gradients/grasp.py of the axion repository.

Modes:
    python -m examples.grasp.grasp_lift view --squeeze 0.5 --lift 0.35 --vis gl
        one rollout with constant commands, replayed in the viewer (3 loops)
    python -m examples.grasp.grasp_lift check            one rollout per world at a
        few squeeze levels: final block height, held/dropped, and the adjoint
        gradient against central differences on two coordinates
    python -m examples.grasp.grasp_lift optimize [hard|zo]
        Adam on u with the exact adjoint gradient (hard, world 0) or an
        antithetic zeroth-order estimate across the batched worlds (zo)

Body indices: 0 base, 1 left pad, 2 right pad, 3 block. Joint DOFs: 0 lift,
1 left pad, 2 right pad, 3-8 block (free). Verified by `check`.

What `check` showed on 20 Oct 2026 (RTX A500, dt 5 ms, 240 steps, 6 worlds
in 9 s with the adjoint; 1 kg block, 2 x 2 cm pads, 0-40 N, lift lag 50 ms):
  * 13.2 N drops the block, 16.8 N lifts it with slip (z 0.15), 20.4 N and
    above hold cleanly; the loss jumps from about 0.13 to 9.0 across the
    edge near 15 N.
  * The lift gradient from the adjoint matches central differences in the
    firmly held regime to about 0.08 (0.12 vs 0.20 at 24 N).
  * The adjoint's squeeze gradient is exactly the squeeze cost's: the
    dynamics path squeeze -> normal force -> friction -> block height is
    zero in Ostrich's adjoint (both friction linearizations), while finite
    differences see a small positive part when firmly held. The exact
    gradient therefore always says "squeeze less".
  * Within a few newtons of the drop edge the forward is not reproducible
    run to run (partial slip amplifies float32 ordering) and the loss is
    rough at the 1e-3 scale in u; finite differences need h >= 1e-3 and are
    meaningless near the edge.
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
BLOCK_HX, BLOCK_HY, BLOCK_HZ = 0.03, 0.03, 0.06  # half sizes [m]
BLOCK_MASS = 1.0
MU = 0.5
PAD_HX, PAD_HY, PAD_HZ = 0.01, 0.005, 0.01  # pad half sizes: 2 cm wide, 1 cm thick, 2 cm tall (Franka-like fingertips)
PAD_GAP = 0.004  # initial clearance pad - block face [m]
PAD_MASS = 0.05
BASE_MASS = 0.5

SQUEEZE_KE = 500.0  # N per (m/s): blocked pad force = SQUEEZE_KE * v_target
V_SQUEEZE_MAX = 0.08  # m/s -> 40 N max squeeze per pad (Franka Hand: 20-70 N)
LIFT_KE = 2.0e4  # stiff velocity servo on the lift
V_LIFT_MAX = 0.4  # m/s
LIFT_TAU = 0.05  # s: first-order lag on the lift speed target (a real arm ramps; a step jerks the block)

Z_TARGET = BLOCK_HZ + 0.15
Z_SCALE = 0.05
W_F = 0.5

BODY_BLOCK = 3
DOF_LIFT, DOF_PAD_L, DOF_PAD_R = 0, 1, 2
NUM_DOFS = 9  # 3 gripper + 6 block


# --- kernels -------------------------------------------------------------


@wp.kernel
def grasp_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    last_step: int,
    num_steps: int,
    z_target: float,
    z_scale: float,
    w_f: float,
    v_squeeze_max: float,
    world_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    z = wp.transform_get_translation(body_pose[last_step, w, BODY_BLOCK])[2]
    e = (z - z_target) / z_scale
    l = e * e
    # squeeze cost: mean over steps of (v_target / v_max)^2 on the left pad (both pads carry the same command)
    s = float(0.0)
    for t in range(num_steps):
        u = target_vel[t, w, DOF_PAD_L] / v_squeeze_max
        s += u * u
    l += w_f * s / float(num_steps)
    world_loss[w] = l
    wp.atomic_add(loss, 0, l)


# --- simulator -----------------------------------------------------------


class GraspLift(OstrichDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config, phases=5):
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.k = phases
        self.d = 2 * phases
        self.W = sim_config.num_worlds
        self.world_loss = wp.zeros(self.W, dtype=wp.float32, requires_grad=True)
        self.T = self.clock.total_sim_steps
        self.phase = np.arange(self.T) * self.k // self.T
        # lift target = first-order lag of the phase command: a fixed linear map (T, k)
        P = np.zeros((self.T, self.k))
        P[np.arange(self.T), self.phase] = 1.0
        a = self.clock.dt / LIFT_TAU
        self.W_lift = np.zeros((self.T, self.k))
        prev = np.zeros(self.k)
        for t in range(self.T):
            prev = prev + a * (P[t] - prev)
            self.W_lift[t] = prev
        self.track_body(body_idx=BODY_BLOCK, name="block", color=(0.9, 0.5, 0.1))
        if self.viewer:
            # close-up on the gripper: 0.6 m away, slightly above, looking at the block
            self.viewer.set_camera(pos=wp.vec3(0.45, -0.45, 0.35), pitch=-20.0, yaw=135.0)
            self.viewer.show_ui = False

    def build_model(self) -> Model:
        b = self.builder
        b.rigid_gap = 0.01
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))

        no_col = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=0.0, is_visible=True)
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=0.0)
        block_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, density=BLOCK_MASS / (8 * BLOCK_HX * BLOCK_HY * BLOCK_HZ))

        # gripper base: slides along world z, starts at the block's mid height
        base = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, BLOCK_HZ), wp.quat_identity()), label="base", mass=BASE_MASS)
        b.add_shape_box(base, hx=0.06, hy=0.08, hz=0.01, xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity()), cfg=no_col)
        j_lift = b.add_joint_prismatic(
            parent=-1, child=base, axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, BLOCK_HZ), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            target_ke=LIFT_KE, target_kd=0.0, label="lift",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
        )

        # pads: at +-y of the block, each on a prismatic joint whose positive direction closes
        joints = [j_lift]
        y0 = BLOCK_HY + PAD_HY + PAD_GAP
        for name, sign in (("pad_l", 1.0), ("pad_r", -1.0)):
            pad = b.add_link(xform=wp.transform(wp.vec3(0.0, sign * y0, BLOCK_HZ), wp.quat_identity()), label=name, mass=PAD_MASS)
            b.add_shape_box(pad, hx=PAD_HX, hy=PAD_HY, hz=PAD_HZ, cfg=pad_cfg)
            j = b.add_joint_prismatic(
                parent=base, child=pad, axis=wp.vec3(0.0, -sign, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, sign * y0, 0.0), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                target_ke=SQUEEZE_KE, target_kd=0.0, label=name + "_j",
                custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
            )
            joints.append(j)
        b.add_articulation(joints, label="gripper")

        # block: free body resting on the ground
        block = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, BLOCK_HZ), wp.quat_identity()), label="block")
        b.add_shape_box(block, hx=BLOCK_HX, hy=BLOCK_HY, hz=BLOCK_HZ, cfg=block_cfg)
        j_block = b.add_joint_free(parent=-1, child=block, label="block_free")
        b.add_articulation([j_block], label="block")

        return b.finalize_replicated(num_worlds=self.simulation_config.num_worlds, requires_grad=True)

    # --- controls ----------------------------------------------------------

    def apply(self, U):
        """U (W, d) in [0, 1] -> joint velocity targets (T, W, dofs)."""
        U = np.asarray(U, dtype=np.float64)
        vel = np.zeros((self.T, self.W, NUM_DOFS), dtype=np.float32)
        sq = V_SQUEEZE_MAX * U[:, 0::2]  # (W, k)
        lift = V_LIFT_MAX * U[:, 1::2]
        vel[:, :, DOF_PAD_L] = sq.T[self.phase]
        vel[:, :, DOF_PAD_R] = sq.T[self.phase]
        vel[:, :, DOF_LIFT] = (self.W_lift @ lift.T)  # (T, W)
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel, dtype=wp.float32))
        for i in range(self.T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def grad_u(self, U):
        """d loss / d U (W, d): the dynamics part from the trajectory's
        velocity-target gradient (the adjoint), plus the squeeze cost's direct
        gradient in closed form. The direct part has to be added by hand:
        TrajectoryBuffer.save_gradients overwrites joint_target_vel.grad with
        the engine's adjoint, so the tape's gradient of any loss term that
        reads the targets is lost (the same happens to the regularization
        terms of the other differentiable examples)."""
        g = self.trajectory.joint_target_vel.grad.numpy()  # (T, W, dofs)
        G = np.zeros((self.W, self.d))
        for j in range(self.k):
            steps = self.phase == j
            G[:, 2 * j] = V_SQUEEZE_MAX * (g[steps, :, DOF_PAD_L].sum(0) + g[steps, :, DOF_PAD_R].sum(0))
            G[:, 2 * j + 1] = V_LIFT_MAX * (self.W_lift[:, j][:, None] * g[:, :, DOF_LIFT]).sum(0)
        U = np.asarray(U, dtype=np.float64)
        for j in range(self.k):
            G[:, 2 * j] += W_F * 2.0 * U[:, 2 * j] * (self.phase == j).sum() / self.T
        return G

    def compute_loss(self):
        wp.launch(
            grasp_loss_kernel, dim=self.W,
            inputs=[self.trajectory.body_pose, self.trajectory.joint_target_vel, self.T, self.T,
                    Z_TARGET, Z_SCALE, W_F, V_SQUEEZE_MAX],
            outputs=[self.world_loss, self.loss],
            device=self.model.device,
        )

    def update(self):
        pass

    def rollout(self, U):
        """Losses (W,) and adjoint gradients (W, d) for the controls U (W, d)."""
        self.apply(U)
        self.diff_step()
        wp.synchronize()
        L = self.world_loss.numpy().copy()
        G = self.grad_u(U)
        self.trajectory.joint_target_vel.grad.zero_()
        self.tape.zero()
        self.loss.zero_()
        return L, G

    def block_state(self):
        poses = self.trajectory.body_pose.numpy()  # (T+1, W, bodies, 7)
        return poses[-1, :, BODY_BLOCK, :3]


# --- drivers -------------------------------------------------------------


def make_sim(num_worlds, phases, duration=1.2, dt=5e-3, vis="null"):
    sim_config = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt, num_worlds=num_worlds)
    render_config = RenderingConfig(vis_type=vis)
    engine_config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=24, backtrack_min_iter=12, atol=1e-4),
        linear=LinearSolverConfig(max_iters=32, tol=1e-6, atol=1e-6, regularization=1e-6),
        compliance=ComplianceConfig(joint=1e-9, contact=1e-6, friction=1e-8),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=64, reduction=ContactReductionConfig(policy="cluster", max_per_pair=8)),
    )
    return GraspLift(sim_config, render_config, engine_config, LoggingConfig(), phases=phases)


def check(args):
    W = 6
    sim = make_sim(W, args.phases, vis=args.vis)
    print(f"bodies {sim.model.body_count} joint dofs {sim.model.joint_dof_count} steps {sim.T} dt {sim.clock.dt}")
    squeezes = np.linspace(0.15, 0.6, W)
    U = np.zeros((W, sim.d))
    U[:, 0::2] = squeezes[:, None]
    U[:, 1::2] = 0.35
    t0 = time.perf_counter()
    L, G = sim.rollout(U)
    print(f"rollout + adjoint: {time.perf_counter() - t0:.2f} s")
    z = sim.block_state()[:, 2]
    for w in range(W):
        print(f"  squeeze {squeezes[w]:.2f} ({SQUEEZE_KE * V_SQUEEZE_MAX * squeezes[w]:.2f} N): block z {z[w]:.4f} "
              f"{'held' if z[w] > BLOCK_HZ + 0.05 else 'dropped'}  loss {L[w]:.4f}  |g| {np.linalg.norm(G[w]):.4f}")
    # central differences on two coordinates of the held worlds, with the batch
    h = 1e-3
    for j in (0, 1):
        Up, Um = U.copy(), U.copy()
        Up[:, j] += h
        Um[:, j] -= h
        Lp, _ = sim.rollout(Up)
        Lm, _ = sim.rollout(Um)
        fd = (Lp - Lm) / (2 * h)
        print(f"  coord {j}: adjoint {np.round(G[:, j], 4)}  FD {np.round(fd, 4)}")
    sim.close()


def view(args):
    sim = make_sim(1, args.phases, vis=args.vis)
    U = np.zeros((1, sim.d))
    U[:, 0::2] = args.squeeze
    U[:, 1::2] = args.lift
    L, _ = sim.rollout(U)
    z = sim.block_state()[0, 2]
    print(f"squeeze {SQUEEZE_KE * V_SQUEEZE_MAX * args.squeeze:.2f} N  lift {V_LIFT_MAX * args.lift:.3f} m/s: "
          f"block z {z:.4f} {'held' if z > BLOCK_HZ + 0.05 else 'dropped'}  loss {L[0]:.4f}")
    sim.render_episode(iteration=0, loop=True, loops_count=args.loops, playback_speed=args.speed)
    sim.close()


def optimize(args):
    method = args.method
    W = 1 if method == "hard" else args.worlds
    sim = make_sim(W, args.phases, vis=args.vis)
    d = sim.d
    rng = np.random.default_rng(args.seed)
    u = np.tile([0.5, 0.35], args.phases).astype(np.float64)
    m, v = np.zeros(d), np.zeros(d)
    b1, b2 = 0.9, 0.999
    sigma = args.sigma
    print(f"{method}: d={d} worlds={W} steps={sim.T} iters={args.iters}")
    for it in range(1, args.iters + 1):
        if method == "hard":
            L, G = sim.rollout(u[None])
            g, loss = G[0], L[0]
        else:
            N = W // 2
            eps = rng.standard_normal((N, d))
            Ubatch = np.clip(np.concatenate([u + sigma * eps, u - sigma * eps]), 0, 1)
            L, _ = sim.rollout(Ubatch)
            g = ((L[:N] - L[N:])[:, None] * eps).mean(0) / (2 * sigma)
            loss = L.mean()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        u = np.clip(u - args.lr * (m / (1 - b1**it)) / (np.sqrt(v / (1 - b2**it)) + 1e-12), 0, 1)
        if it % args.print_every == 0 or it == 1:
            Lc, _ = sim.rollout(u[None].repeat(W, 0)) if method != "hard" else (np.array([loss]), None)
            z = sim.block_state()[0, 2]
            print(f"iter {it:4d}: batch loss {loss:.4f}  loss at u {Lc[0]:.4f}  block z {z:.4f} "
                  f"squeeze {np.round(SQUEEZE_KE * V_SQUEEZE_MAX * u[0::2], 2)} N  lift {np.round(V_LIFT_MAX * u[1::2], 3)} m/s")
    sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "optimize", "view"])
    p.add_argument("method", nargs="?", default="hard", choices=["hard", "zo"])
    p.add_argument("--phases", type=int, default=5)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--vis", default="null", choices=["null", "gl", "usd"])
    p.add_argument("--squeeze", type=float, default=0.5, help="view: squeeze command in [0, 1]")
    p.add_argument("--lift", type=float, default=0.35, help="view: lift command in [0, 1]")
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--speed", type=float, default=0.5)
    a = p.parse_args()
    {"check": check, "optimize": optimize, "view": view}[a.mode](a)
