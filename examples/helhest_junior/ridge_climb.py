"""Ridge climb: Helhest Junior drives over a climbable ridge on a heightfield.

Scene: flat ground with small random bumps and a Gaussian ridge of height
RIDGE_H (wheel radius 0.35 m) across the path at x = RIDGE_X; goal just
beyond it at GOAL, with an energy cost: the best policy is the slowest climb
that still gets over. Depending on the wheel speeds the robot climbs the
ridge or stalls against it: the stall-or-climb fork of a terrain robot,
decided over time at the ridge, and the optimum sits on it.

Controls: k phases, each with a left and a right wheel speed command,
u in [0, 1]^(2k) -> 0..V_MAX rad/s; the rear wheel follows their mean.
Loss per world: |xy(T) - GOAL|^2 + W_E mean(u^2).

Modes (as examples/grasp/grasp_lift.py):
    python -m examples.helhest_junior.ridge_climb view --speed 0.5 --vis gl
    python -m examples.helhest_junior.ridge_climb check
    python -m examples.helhest_junior.ridge_climb optimize [hard|zo]

Bodies per world: 0 chassis, 1 left wheel, 2 right wheel, 3 rear wheel.
Joint DOFs per world: 0-5 chassis (free), 6 left, 7 right, 8 rear.
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
from newton import Model

from examples.helhest_junior.common import HelhestJuniorConfig, create_helhest_junior_model

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# --- terrain ------------------------------------------------------------
CELL = 0.1
XLIM, YLIM = (-2.0, 9.0), (-3.0, 3.0)
RIDGE_X, RIDGE_H, RIDGE_W = 3.0, 0.22, 0.45  # centre [m], height [m], gaussian width [m]
BUMP_AMP, BUMP_SEED = 0.02, 0
TERRAIN_MU = 0.8
GOAL = (4.0, 0.0)  # just past the ridge: the optimum is the slowest climb
SPAWN = (0.0, 0.0)

V_MAX = 8.0  # rad/s (1.4 m/s... 2.8 m/s at the rim)
W_E = 0.3
BODY_CHASSIS = 0
DOF_WHEEL = 6  # left, right, rear
NUM_DOFS = 9


def make_heightmap():
    xs = np.arange(XLIM[0], XLIM[1] + 1e-9, CELL)
    ys = np.arange(YLIM[0], YLIM[1] + 1e-9, CELL)
    XX, YY = np.meshgrid(xs, ys)  # [ny, nx]
    rng = np.random.default_rng(BUMP_SEED)
    H = BUMP_AMP * rng.uniform(-1, 1, XX.shape)
    H += RIDGE_H * np.exp(-((XX - RIDGE_X) ** 2) / (2 * RIDGE_W**2))
    H[:, :8] = 0.0  # flat spawn area
    return xs, ys, H.astype(np.float32)


def terrain_height(xs, ys, H, x, y):
    ix = int(np.clip((x - xs[0]) / CELL, 0, len(xs) - 1))
    iy = int(np.clip((y - ys[0]) / CELL, 0, len(ys) - 1))
    return float(H[iy, ix])


# --- kernels -------------------------------------------------------------


@wp.kernel
def ridge_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    last_step: int,
    num_steps: int,
    goal_x: float,
    goal_y: float,
    w_e: float,
    v_max: float,
    world_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    p = wp.transform_get_translation(body_pose[last_step, w, BODY_CHASSIS])
    dx = p[0] - goal_x
    dy = p[1] - goal_y
    l = dx * dx + dy * dy
    s = float(0.0)
    for t in range(num_steps):
        a = target_vel[t, w, DOF_WHEEL] / v_max
        b = target_vel[t, w, DOF_WHEEL + 1] / v_max
        s += 0.5 * (a * a + b * b)
    l += w_e * s / float(num_steps)
    world_loss[w] = l
    wp.atomic_add(loss, 0, l)


# --- simulator -----------------------------------------------------------


class RidgeClimb(OstrichDifferentiableSimulator):
    def __init__(self, sim_config, render_config, engine_config, logging_config, phases=5):
        self.xs, self.ys, self.H = make_heightmap()
        super().__init__(sim_config, render_config, engine_config, logging_config)
        self.k = phases
        self.d = 2 * phases
        self.W = sim_config.num_worlds
        self.T = self.clock.total_sim_steps
        self.phase = np.arange(self.T) * self.k // self.T
        self.world_loss = wp.zeros(self.W, dtype=wp.float32, requires_grad=True)
        self.track_body(body_idx=BODY_CHASSIS, name="chassis", color=(0.0, 0.5, 1.0))
        if self.viewer:
            self.viewer.set_camera(pos=wp.vec3(RIDGE_X - 1.0, -7.0, 3.5), pitch=-18.0, yaw=90.0)
            self.viewer.show_ui = False

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.05
        ny, nx = self.H.shape
        gb = newton.ModelBuilder()
        gb.add_shape_heightfield(
            xform=wp.transform(wp.vec3((self.xs[0] + self.xs[-1]) / 2, (self.ys[0] + self.ys[-1]) / 2, 0.0), wp.quat_identity()),
            heightfield=newton.Heightfield(data=np.ascontiguousarray(self.H), nrow=ny, ncol=nx,
                                           hx=(nx - 1) * CELL / 2, hy=(ny - 1) * CELL / 2,
                                           min_z=float(self.H.min()), max_z=float(self.H.max())),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=True, mu=TERRAIN_MU),
        )
        z0 = terrain_height(self.xs, self.ys, self.H, *SPAWN) + HelhestJuniorConfig.WHEEL_RADIUS + 0.02
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(SPAWN[0], SPAWN[1], z0), wp.quat_identity()),
            control_mode="velocity",
            k_p=HelhestJuniorConfig.TARGET_KE,
            k_d=HelhestJuniorConfig.TARGET_KD,
            friction_left_right=0.7,
            friction_rear=0.4,
        )
        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds,
                                                 requires_grad=True, global_builder=gb)

    def apply(self, U):
        U = np.asarray(U, dtype=np.float64)
        vel = np.zeros((self.T, self.W, NUM_DOFS), dtype=np.float32)
        vl, vr = V_MAX * U[:, 0::2], V_MAX * U[:, 1::2]  # (W, k)
        vel[:, :, DOF_WHEEL] = vl.T[self.phase]
        vel[:, :, DOF_WHEEL + 1] = vr.T[self.phase]
        vel[:, :, DOF_WHEEL + 2] = 0.5 * (vl + vr).T[self.phase]
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel, dtype=wp.float32))
        for i in range(self.T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def grad_u(self, U):
        """Dynamics part from the adjoint plus the energy cost's direct gradient
        (the trajectory buffer overwrites the targets' tape gradient)."""
        g = self.trajectory.joint_target_vel.grad.numpy()  # (T, W, dofs)
        U = np.asarray(U, dtype=np.float64)
        G = np.zeros((self.W, self.d))
        for j in range(self.k):
            steps = self.phase == j
            gl, gr, gb = g[steps, :, DOF_WHEEL].sum(0), g[steps, :, DOF_WHEEL + 1].sum(0), g[steps, :, DOF_WHEEL + 2].sum(0)
            G[:, 2 * j] = V_MAX * (gl + 0.5 * gb)
            G[:, 2 * j + 1] = V_MAX * (gr + 0.5 * gb)
            frac = steps.sum() / self.T
            G[:, 2 * j] += W_E * U[:, 2 * j] * frac
            G[:, 2 * j + 1] += W_E * U[:, 2 * j + 1] * frac
        return G

    def compute_loss(self):
        wp.launch(ridge_loss_kernel, dim=self.W,
                  inputs=[self.trajectory.body_pose, self.trajectory.joint_target_vel, self.T, self.T,
                          GOAL[0], GOAL[1], W_E, V_MAX],
                  outputs=[self.world_loss, self.loss], device=self.model.device)

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

    def chassis_final(self):
        return self.trajectory.body_pose.numpy()[-1, :, BODY_CHASSIS, :]  # (W, 7)


# --- drivers -------------------------------------------------------------


def make_sim(num_worlds, phases, duration=5.0, dt=2e-2, vis="null"):
    sim_config = SimulationConfig(duration_seconds=duration, target_timestep_seconds=dt, num_worlds=num_worlds)
    render_config = RenderingConfig(vis_type=vis)
    engine_config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-10, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=128, reduction=ContactReductionConfig(policy="cluster", max_per_pair=8)),
    )
    return RidgeClimb(sim_config, render_config, engine_config, LoggingConfig(), phases=phases)


def check(args):
    W = 6
    sim = make_sim(W, args.phases, vis=args.vis)
    print(f"bodies {sim.model.body_count} joint dofs {sim.model.joint_dof_count} steps {sim.T} dt {sim.clock.dt}")
    speeds = np.linspace(0.2, 0.7, W)
    U = np.zeros((W, sim.d)) + speeds[:, None]
    t0 = time.perf_counter()
    L, G = sim.rollout(U)
    print(f"rollout + adjoint: {time.perf_counter() - t0:.2f} s")
    P = sim.chassis_final()
    for w in range(W):
        up = 1 - 2 * (P[w, 3] ** 2 + P[w, 4] ** 2)  # z-component of the rotated up vector
        print(f"  speed {speeds[w]:.2f} ({V_MAX * speeds[w]:.1f} rad/s): final x {P[w, 0]:.2f} y {P[w, 1]:.2f} z {P[w, 2]:.2f} "
              f"{'climbed' if P[w, 0] > RIDGE_X + 0.5 else 'stalled'} up {up:.2f}  loss {L[w]:.3f}  |g| {np.linalg.norm(G[w]):.3f}")
    h = 1e-2
    for j in (0, 2):
        Up, Um = U.copy(), U.copy()
        Up[:, j] += h
        Um[:, j] -= h
        Lp, _ = sim.rollout(Up)
        Lm, _ = sim.rollout(Um)
        print(f"  coord {j}: adjoint {np.round(G[:, j], 3)}  FD {np.round((Lp - Lm) / (2 * h), 3)}")
    sim.close()


def view(args):
    sim = make_sim(1, args.phases, vis=args.vis)
    U = np.full((1, sim.d), args.speed)
    L, _ = sim.rollout(U)
    P = sim.chassis_final()[0]
    print(f"speed {V_MAX * args.speed:.1f} rad/s: final x {P[0]:.2f} {'climbed' if P[0] > RIDGE_X + 0.5 else 'stalled'} loss {L[0]:.3f}")
    sim.render_episode(iteration=0, loop=True, loops_count=args.loops, playback_speed=args.playback)
    sim.close()


def optimize(args):
    method = args.method
    W = 1 if method == "hard" else args.worlds
    sim = make_sim(W, args.phases, vis=args.vis)
    d = sim.d
    rng = np.random.default_rng(args.seed)
    u = np.full(d, args.speed)
    m, v = np.zeros(d), np.zeros(d)
    for it in range(1, args.iters + 1):
        if method == "hard":
            L, G = sim.rollout(u[None])
            g, loss = G[0], L[0]
        else:
            N = W // 2
            eps = rng.standard_normal((N, d))
            L, _ = sim.rollout(np.clip(np.concatenate([u + args.sigma * eps, u - args.sigma * eps]), 0, 1))
            g = ((L[:N] - L[N:])[:, None] * eps).mean(0) / (2 * args.sigma)
            loss = L.mean()
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        u = np.clip(u - args.lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-12), 0, 1)
        if it % args.print_every == 0 or it == 1:
            Lc, _ = sim.rollout(np.repeat(u[None], W, 0))
            P = sim.chassis_final()[0]
            print(f"iter {it:4d}: batch loss {loss:.3f}  loss at u {Lc[0]:.3f}  final x {P[0]:.2f} {'climbed' if P[0] > RIDGE_X + 0.5 else 'stalled'}  "
                  f"left {np.round(V_MAX * u[0::2], 1)}  right {np.round(V_MAX * u[1::2], 1)} rad/s")
    sim.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "optimize", "view"])
    p.add_argument("method", nargs="?", default="hard", choices=["hard", "zo"])
    p.add_argument("--phases", type=int, default=5)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.4, help="initial / viewed command in [0, 1]")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--playback", type=float, default=1.0)
    p.add_argument("--vis", default="null", choices=["null", "gl", "usd"])
    a = p.parse_args()
    {"check": check, "optimize": optimize, "view": view}[a.mode](a)
