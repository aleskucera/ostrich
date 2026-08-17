"""Zeroth-order (OpenAI-ES) baseline for the box2 final-pose task, wall-clock
matched against the existing first-order (Ostrich implicit-adjoint) result in
experiments/3_gradient_quality_box2/results/ostrich_postfix_vjp.json.

SAME task as the first-order reference: trial 0 (seed 1042) of that JSON —
same IC, target, loss weights, K=10 wheel-velocity spline knots x 3 wheels =
30 params, dt=0.1, horizon=6s (T=60 steps).

Population-based evaluation: uses Ostrich's replicated multi-world sim
(finalize_replicated) so ONE generation = ONE batched forward-only rollout of
POP=64 candidate splines, all sharing the fixed IC. This is the strongest
fair zeroth-order setup on the same GPU (batched, no per-eval sim
construction -> no GPU-leak-per-construction).

`cma` (CMA-ES) is not installed in this venv, so this implements OpenAI-ES
(Salimans et al. 2017): antithetic Gaussian population sampling + centered
rank-based fitness shaping + a plain SGD mean update.

Usage:
    python experiments/3_gradient_quality_box2/zeroth_order_es.py

Output: experiments/3_gradient_quality_box2/results/zeroth_order_es.json
"""
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import warp as wp
from ostrich import (
    OstrichDifferentiableSimulator,
    OstrichEngineConfig,
    ComplianceConfig,
    ContactsConfig,
    LinearSolverConfig,
    LinesearchConfig,
    LoggingConfig,
    NewtonRaphsonConfig,
    RenderingConfig,
    SimulationConfig,
)
from ostrich.collision.config import ContactReductionConfig
import newton

from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.replay_real import BOX_CENTER, BOX_HALF_EXTENTS

from optimize_ostrich import (
    make_interp_matrix,
    initial_spline,
    make_configs,
    HelhestJuniorBoxFinalPoseOptimizer,
    WHEEL_DOF_OFFSET,
    NUM_WHEEL_DOFS,
)

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
REF_JSON = REPO / "experiments/3_gradient_quality_box2/results/ostrich_postfix_vjp.json"

# ----------------------------- SAME task as the FO reference -----------------
K = 10
HORIZON_S = 6.0
DT = 0.1
IC_XY = [0.05479120971119267, -0.012224312049589542]
IC_YAW = 0.06258714393256555
TARGET_XY = [3.1184208174356183, -0.24349359126741027]
TARGET_YAW = 0.24903528096418898
WEIGHTS = {"track": 1.0, "pos": 1.0, "yaw": 0.5, "vel": 0.3, "smooth": 1e-3, "reg": 1e-5}
INIT_TYPE = "constant"
INIT_NOISE_STD = 0.2
POP = 64  # = num_worlds; one generation == one batched forward rollout

# ----------------------------- multi-world per-population kernels ------------
@wp.kernel
def track_loss_mw_kernel(body_pose: wp.array(dtype=wp.transform, ndim=3),
                          ref_xy: wp.array(dtype=wp.vec2), weight: float,
                          loss: wp.array(dtype=wp.float32)):
    t, w = wp.tid()
    p = wp.transform_get_translation(body_pose[t, w, 0])
    r = ref_xy[t]
    dx = p[0] - r[0]
    dy = p[1] - r[1]
    wp.atomic_add(loss, w, weight * (dx * dx + dy * dy))


@wp.kernel
def final_pos_loss_mw_kernel(body_pose: wp.array(dtype=wp.transform, ndim=3),
                              target_x: float, target_y: float, final_idx: int,
                              weight: float, loss: wp.array(dtype=wp.float32)):
    w = wp.tid()
    p = wp.transform_get_translation(body_pose[final_idx, w, 0])
    dx = p[0] - target_x
    dy = p[1] - target_y
    wp.atomic_add(loss, w, weight * (dx * dx + dy * dy))


@wp.kernel
def final_yaw_loss_mw_kernel(body_pose: wp.array(dtype=wp.transform, ndim=3),
                              target_yaw: float, final_idx: int, weight: float,
                              loss: wp.array(dtype=wp.float32)):
    w = wp.tid()
    q = wp.transform_get_rotation(body_pose[final_idx, w, 0])
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    delta = yaw - target_yaw
    wp.atomic_add(loss, w, weight * (1.0 - wp.cos(delta)))


@wp.kernel
def terminal_vel_loss_mw_kernel(body_pose: wp.array(dtype=wp.transform, ndim=3),
                                 start_idx: int, end_idx: int, inv_dt: float,
                                 weight: float, loss: wp.array(dtype=wp.float32)):
    t_local, w = wp.tid()
    t = start_idx + t_local
    if t >= end_idx:
        return
    p0 = wp.transform_get_translation(body_pose[t, w, 0])
    p1 = wp.transform_get_translation(body_pose[t + 1, w, 0])
    vx = (p1[0] - p0[0]) * inv_dt
    vy = (p1[1] - p0[1]) * inv_dt
    wp.atomic_add(loss, w, weight * (vx * vx + vy * vy))


@wp.kernel
def smoothness_loss_mw_kernel(joint_target_vel: wp.array(dtype=wp.float32, ndim=3),
                               wheel_dof_offset: int, weight: float,
                               loss: wp.array(dtype=wp.float32)):
    t, w, k = wp.tid()
    dof = wheel_dof_offset + k
    u0 = joint_target_vel[t, w, dof]
    u1 = joint_target_vel[t + 1, w, dof]
    du = u1 - u0
    wp.atomic_add(loss, w, weight * du * du)


@wp.kernel
def magnitude_reg_mw_kernel(joint_target_vel: wp.array(dtype=wp.float32, ndim=3),
                             wheel_dof_offset: int, weight: float,
                             loss: wp.array(dtype=wp.float32)):
    t, w, k = wp.tid()
    dof = wheel_dof_offset + k
    u = joint_target_vel[t, w, dof]
    wp.atomic_add(loss, w, weight * u * u)


class PopulationBoxOptimizer(OstrichDifferentiableSimulator):
    """Replicated-worlds, forward-only (no backward) box2 final-pose sim.

    All `num_worlds` share the SAME IC. Each world gets its OWN spline
    (population member). `self.loss` is a per-world array — no reduction
    across worlds, so each generation's POP rollouts can be individually
    ranked by a zeroth-order optimizer.
    """

    def __init__(self, sim_config, render_config, engine_config, logging_config,
                 ic_xy, ic_yaw, target_xy, target_yaw, weights, K=10,
                 mu_front=0.8, mu_rear=1.2, mu_rolling=0.7, terminal_tail_frac=0.10):
        self.K = K
        self.num_worlds = sim_config.num_worlds
        self.ic_xy = tuple(ic_xy)
        self.ic_yaw = float(ic_yaw)
        self.target_xy = tuple(target_xy)
        self.target_yaw = float(target_yaw)
        self.weights = weights
        self.mu_front = mu_front
        self.mu_rear = mu_rear
        self.mu_rolling = mu_rolling
        self.terminal_tail_frac = float(terminal_tail_frac)
        super().__init__(sim_config, render_config, engine_config, logging_config)

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, K)
        self.loss = wp.zeros(self.num_worlds, dtype=wp.float32, requires_grad=False)

        T_states = T + 1
        t_norm = np.arange(T_states, dtype=np.float32) / max(T_states - 1, 1)
        ref_xy_np = np.zeros((T_states, 2), dtype=np.float32)
        ref_xy_np[:, 0] = self.ic_xy[0] + t_norm * (self.target_xy[0] - self.ic_xy[0])
        ref_xy_np[:, 1] = self.ic_xy[1] + t_norm * (self.target_xy[1] - self.ic_xy[1])
        self.ref_xy_buf = wp.array(ref_xy_np, dtype=wp.vec2, device=self.model.device,
                                    requires_grad=False)

    def build_model(self):
        self.builder.rigid_gap = 0.2
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0], hy=BOX_HALF_EXTENTS[1], hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, ke=150.0, kd=150.0, kf=500.0),
        )
        chassis_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self.ic_yaw)
        chassis_xform = wp.transform(
            wp.vec3(float(self.ic_xy[0]), float(self.ic_xy[1]), 0.5), chassis_q
        )
        create_helhest_junior_model(
            self.builder, xform=chassis_xform, control_mode="velocity",
            k_p=250.0, k_d=0.0,
            friction_left_right=self.mu_front, friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
        )
        # requires_grad=False: zeroth-order never backprops through the sim.
        return self.builder.finalize_replicated(num_worlds=self.num_worlds, requires_grad=False)

    def _forward_backward(self):
        """Forward-only rollout (overrides the base class's forward+backward).

        Zeroth-order needs no tape/backward -- skipping it halves per-generation
        cost vs. reusing the gradient-engine's full diff_step().
        """
        self.trajectory.zero_grad()
        for i in range(self.clock.total_sim_steps):
            self.collision_pipeline.collide(self.states[i], self.contacts)
            self.solver.step(state_in=self.states[i], state_out=self.states[i + 1],
                              control=self.controls[i], contacts=self.contacts, dt=self.clock.dt)
            self.trajectory.save_step(i, self.solver.data, self.solver.ostrich_contacts)
        self.compute_loss()

    def _expand(self, params):
        return self.W @ params  # [T,K]@[K,3] = [T,3]

    def apply_population(self, pop_params):
        """pop_params: [POP, K, 3] -> per-world joint_target_vel."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = np.einsum("tk,wkd->wtd", self.W, pop_params)  # [W,T,3]
        vel_np = np.zeros((T, self.num_worlds, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = (
            expanded.transpose(1, 0, 2)
        )
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        T_states = self.trajectory.body_pose.shape[0]
        T_steps = self.clock.total_sim_steps
        final_idx = T_states - 1
        tail_len = max(2, int(round(T_states * self.terminal_tail_frac)))
        tail_start = T_states - 1 - tail_len
        device = self.solver.model.device
        w = self.weights

        if w.get("track", 0.0) > 0.0:
            wp.launch(track_loss_mw_kernel, dim=(T_states, self.num_worlds),
                      inputs=[self.trajectory.body_pose, self.ref_xy_buf,
                              float(w["track"] / T_states)],
                      outputs=[self.loss], device=device)

        wp.launch(final_pos_loss_mw_kernel, dim=self.num_worlds,
                  inputs=[self.trajectory.body_pose, float(self.target_xy[0]),
                          float(self.target_xy[1]), final_idx, float(w["pos"])],
                  outputs=[self.loss], device=device)

        wp.launch(final_yaw_loss_mw_kernel, dim=self.num_worlds,
                  inputs=[self.trajectory.body_pose, float(self.target_yaw), final_idx,
                          float(w["yaw"])],
                  outputs=[self.loss], device=device)

        if w["vel"] > 0.0:
            wp.launch(terminal_vel_loss_mw_kernel, dim=(tail_len, self.num_worlds),
                      inputs=[self.trajectory.body_pose, tail_start, final_idx,
                              float(1.0 / self.clock.dt), float(w["vel"] / tail_len)],
                      outputs=[self.loss], device=device)

        if w["smooth"] > 0.0:
            wp.launch(smoothness_loss_mw_kernel, dim=(T_steps - 1, self.num_worlds, NUM_WHEEL_DOFS),
                      inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                              float(w["smooth"] / (T_steps - 1))],
                      outputs=[self.loss], device=device)

        if w["reg"] > 0.0:
            wp.launch(magnitude_reg_mw_kernel, dim=(T_steps, self.num_worlds, NUM_WHEEL_DOFS),
                      inputs=[self.trajectory.joint_target_vel, WHEEL_DOF_OFFSET,
                              float(w["reg"] / T_steps)],
                      outputs=[self.loss], device=device)

    def population_final_metrics(self):
        """Per-world (pos_error_m, terminal_speed_mps), vectorized version of
        HelhestJuniorBoxFinalPoseOptimizer.final_metrics() (same T-1/T-2 index
        convention), so success (pos<0.2 AND speed<0.3) is directly comparable
        to the FO reference without needing a separate single-world sim."""
        T = self.clock.total_sim_steps
        dt = self.clock.dt
        bp = self.trajectory.body_pose.numpy()  # [T_states, W, B, 7]
        chassis_T = bp[T - 1, :, 0, :]
        chassis_Tm1 = bp[max(T - 2, 0), :, 0, :]
        xy_T = chassis_T[:, :2]
        v_xy = (chassis_T[:, :2] - chassis_Tm1[:, :2]) / dt
        terminal_speed = np.linalg.norm(v_xy, axis=1)
        pos_error = np.linalg.norm(xy_T - np.asarray(self.target_xy)[None, :], axis=1)
        return pos_error, terminal_speed

    def eval_population(self, pop_params):
        """pop_params: [POP, K, 3] -> (losses, pos_error_m, terminal_speed_mps),
        each [POP] (one batched rollout)."""
        self.apply_population(pop_params)
        self.loss.zero_()
        self.diff_step()
        wp.synchronize()
        losses = self.loss.numpy().copy()
        pos_error, terminal_speed = self.population_final_metrics()
        return losses, pos_error, terminal_speed

    def update(self):
        pass


# ----------------------------- OpenAI-ES --------------------------------------
def centered_ranks(x):
    """Rank-transform x (lower is better here) to [-0.5, 0.5], best=+0.5."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))  # 0 = best (lowest loss)
    # invert so best (rank 0) gets the highest shaped fitness
    shaped = (len(x) - 1 - ranks) / (len(x) - 1) - 0.5
    return shaped


def _is_success(pos_error_m, terminal_speed_mps):
    return bool(pos_error_m < 0.2 and terminal_speed_mps < 0.3)


def run_openai_es(sim, mean0, seed, wall_budget_s, sigma=0.3, lr=0.2, log_every_gens=1):
    """Antithetic OpenAI-ES. Returns dict with history + best solution.

    Runs for the FULL wall_budget_s (intended to be the largest budget of
    interest, e.g. 10x); callers extract 1x/5x checkpoints from `history`.
    Since the RNG stream is deterministic given `seed` and generation index,
    the state at generation g is identical whether this run stops at g or
    continues past it -- so checkpointing a single long run is equivalent to
    (and more efficient than) three independent restarts.

    Tracks, per generation, the population member with the LOWEST loss
    ("generation-best candidate") and its (pos_error_m, terminal_speed_mps,
    success). `first_success_wall_s` / `first_success_gen` record the first
    generation whose generation-best candidate satisfies the success
    criterion (pos<0.2 AND speed<0.3) -- None if never met within the budget.
    `first_success_wall_s_any_member` is a looser variant checking ALL POP
    members each generation (a member can be "successful" without having the
    single lowest weighted loss, e.g. if it trades off smoothness/reg).
    """
    rng = np.random.default_rng(seed)
    theta = mean0.flatten().astype(np.float64).copy()  # [D], D = K*3
    D = theta.shape[0]
    half = sim.num_worlds // 2
    assert 2 * half == sim.num_worlds, "POP must be even for antithetic sampling"

    best_loss = float("inf")
    best_theta = theta.copy()
    best_pos_error = None
    best_terminal_speed = None
    first_success_wall_s = None
    first_success_gen = None
    first_success_wall_s_any_member = None
    first_success_gen_any_member = None
    history = []

    t0 = time.perf_counter()
    gen = 0
    n_rollouts = 0
    while True:
        wall = time.perf_counter() - t0
        if wall >= wall_budget_s:
            break
        eps = rng.standard_normal((half, D))
        pop_theta = np.concatenate([theta[None, :] + sigma * eps,
                                     theta[None, :] - sigma * eps], axis=0)  # [POP, D]
        pop_params = pop_theta.reshape(sim.num_worlds, sim.K, NUM_WHEEL_DOFS)
        losses, pos_err, term_speed = sim.eval_population(pop_params)
        losses = losses.astype(np.float64)
        n_rollouts += sim.num_worlds

        gen_best_idx = int(np.argmin(losses))
        gen_best_pos = float(pos_err[gen_best_idx])
        gen_best_speed = float(term_speed[gen_best_idx])
        gen_best_success = _is_success(gen_best_pos, gen_best_speed)
        if losses[gen_best_idx] < best_loss:
            best_loss = float(losses[gen_best_idx])
            best_theta = pop_theta[gen_best_idx].copy()
            best_pos_error = gen_best_pos
            best_terminal_speed = gen_best_speed

        wall_now = time.perf_counter() - t0
        if gen_best_success and first_success_wall_s is None:
            first_success_wall_s = wall_now
            first_success_gen = gen
        any_success_mask = (pos_err < 0.2) & (term_speed < 0.3)
        if any_success_mask.any() and first_success_wall_s_any_member is None:
            first_success_wall_s_any_member = wall_now
            first_success_gen_any_member = gen

        # OpenAI-ES update: mean += lr/(POP*sigma) * sum_i F_i * eps_i
        # (F_i is the centered-rank shaped fitness; eps_i is +eps for the first
        # half of the population and -eps for the antithetic mirror.)
        fit = centered_ranks(losses)  # [POP], best (lowest loss) -> +0.5
        fit_signed_eps = fit[:half, None] * eps - fit[half:, None] * eps
        grad_est = fit_signed_eps.sum(axis=0) / (sim.num_worlds * sigma)
        theta = theta + lr * grad_est  # + because fit is best=+0.5 (ascend fitness = descend loss)

        wall = time.perf_counter() - t0
        if gen % log_every_gens == 0:
            history.append({
                "gen": gen, "wall_s": wall, "n_rollouts": n_rollouts,
                "best_so_far_loss": best_loss,
                "best_so_far_pos_error_m": best_pos_error,
                "best_so_far_terminal_speed_mps": best_terminal_speed,
                "best_so_far_success": (_is_success(best_pos_error, best_terminal_speed)
                                         if best_pos_error is not None else False),
                "gen_min_loss": float(losses.min()), "gen_mean_loss": float(losses.mean()),
                "gen_best_pos_error_m": gen_best_pos,
                "gen_best_terminal_speed_mps": gen_best_speed,
                "gen_best_success": gen_best_success,
            })
        gen += 1

    return {
        "best_loss": best_loss,
        "best_theta": best_theta,
        "best_pos_error_m": best_pos_error,
        "best_terminal_speed_mps": best_terminal_speed,
        "best_success": (_is_success(best_pos_error, best_terminal_speed)
                          if best_pos_error is not None else False),
        "first_success_wall_s": first_success_wall_s,
        "first_success_gen": first_success_gen,
        "first_success_wall_s_any_member": first_success_wall_s_any_member,
        "first_success_gen_any_member": first_success_gen_any_member,
        "history": history,
        "n_generations": gen,
        "n_rollouts": n_rollouts,
        "wall_s": time.perf_counter() - t0,
    }


# ----------------------------- local FO timing ---------------------------------
def time_local_first_order(n_iters=8):
    """Locally measure per-iter wall time of the FO (Adam) optimizer on the
    SAME task/settings as the reference JSON, so the zeroth-order wall-clock
    budget is matched on THIS machine (the reference 0.664 s/iter was measured
    on a 3090; this box may have very different hardware)."""
    from optimize_ostrich import SplineAdam

    sim_cfg, rc, ec = make_configs(HORIZON_S, DT)
    sim = HelhestJuniorBoxFinalPoseOptimizer(
        sim_cfg, rc, ec, LoggingConfig(),
        ic_xy=IC_XY, ic_yaw=IC_YAW, target_xy=TARGET_XY, target_yaw=TARGET_YAW,
        weights=WEIGHTS, K=K,
    )
    sim.spline_params = initial_spline(K, NUM_WHEEL_DOFS, 1042, INIT_TYPE, TARGET_XY,
                                        HORIZON_S, noise_std=INIT_NOISE_STD)
    sim.spline_adam = SplineAdam(K=K, num_dofs=NUM_WHEEL_DOFS, lr=0.3, lr_min_ratio=0.2,
                                  total_steps=50)
    sim._apply_params(sim.spline_params)
    # warm-up (graph capture) iter, excluded from timing
    sim.opt_step()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sim.opt_step()
    wp.synchronize()
    per_iter = (time.perf_counter() - t0) / n_iters
    sim.close(); del sim
    return per_iter


def checkpoint_at(history, budget_s):
    """Last history entry with wall_s <= budget_s (the state of the run "as
    if" it had been budgeted to stop at budget_s). Falls back to the first
    entry if budget_s is shorter than the first generation's wall time."""
    candidates = [h for h in history if h["wall_s"] <= budget_s]
    return candidates[-1] if candidates else (history[0] if history else None)


def main():
    print("=" * 70)
    print("Measuring local per-iter FO wall time (for a hardware-matched budget)...")
    local_per_iter = time_local_first_order(n_iters=8)
    print(f"Local FO per-iter: {local_per_iter:.4f} s  (reference 3090: 0.664 s/iter)")
    local_budget_s = local_per_iter * 50  # reference used 50 iterations
    budget_multipliers = [1, 5, 10]
    budgets_s = [m * local_budget_s for m in budget_multipliers]
    print(f"Local wall-clock budgets (1x/5x/10x of 50 FO-iter-equivalent): "
          f"{[f'{b:.1f}s' for b in budgets_s]}")
    print("=" * 70)

    sim_cfg, rc, ec = make_configs(HORIZON_S, DT)
    sim_cfg.num_worlds = POP
    sim = PopulationBoxOptimizer(
        sim_cfg, rc, ec, LoggingConfig(),
        ic_xy=IC_XY, ic_yaw=IC_YAW, target_xy=TARGET_XY, target_yaw=TARGET_YAW,
        weights=WEIGHTS, K=K,
    )

    seeds = [0, 1, 2]
    max_budget_s = budgets_s[-1]
    all_results = []
    for seed in seeds:
        print(f"\n--- ES seed {seed} (running to max budget {max_budget_s:.1f}s) ---")
        mean0 = initial_spline(K, NUM_WHEEL_DOFS, 1042 + seed, INIT_TYPE, TARGET_XY,
                                HORIZON_S, noise_std=INIT_NOISE_STD)
        res = run_openai_es(sim, mean0, seed=seed, wall_budget_s=max_budget_s,
                             sigma=0.3, lr=0.2)
        print(f"  total generations={res['n_generations']}  rollouts={res['n_rollouts']}  "
              f"wall={res['wall_s']:.1f}s  best_loss={res['best_loss']:.4f}  "
              f"first_success_wall_s={res['first_success_wall_s']}")

        # Checkpoint the SAME run's history at each budget multiplier. The ES
        # RNG stream is deterministic given (seed, generation index), so the
        # state at generation g is identical whether the run was stopped at g
        # or continued past it -- this is equivalent to (and cheaper than)
        # three independent restarts at 1x/5x/10x.
        budget_checkpoints = {}
        for mult, budget_s in zip(budget_multipliers, budgets_s):
            cp = checkpoint_at(res["history"], budget_s)
            budget_checkpoints[f"{mult}x"] = {
                "budget_s": budget_s,
                "gen": cp["gen"] if cp else None,
                "wall_s": cp["wall_s"] if cp else None,
                "n_rollouts": cp["n_rollouts"] if cp else None,
                "best_loss": cp["best_so_far_loss"] if cp else None,
                "pos_error_m": cp["best_so_far_pos_error_m"] if cp else None,
                "terminal_speed_mps": cp["best_so_far_terminal_speed_mps"] if cp else None,
                "success": cp["best_so_far_success"] if cp else False,
                "first_success_wall_s": (
                    res["first_success_wall_s"]
                    if (res["first_success_wall_s"] is not None
                        and res["first_success_wall_s"] <= budget_s)
                    else None
                ),
            }
            print(f"    [{mult}x, {budget_s:.1f}s]: gen={cp['gen'] if cp else 'n/a'}  "
                  f"best_loss={budget_checkpoints[f'{mult}x']['best_loss']}  "
                  f"pos_err={budget_checkpoints[f'{mult}x']['pos_error_m']}  "
                  f"speed={budget_checkpoints[f'{mult}x']['terminal_speed_mps']}  "
                  f"success={budget_checkpoints[f'{mult}x']['success']}")

        all_results.append({
            "seed": seed,
            "n_generations": res["n_generations"],
            "n_rollouts": res["n_rollouts"],
            "wall_s": res["wall_s"],
            "best_loss": res["best_loss"],
            "best_pos_error_m": res["best_pos_error_m"],
            "best_terminal_speed_mps": res["best_terminal_speed_mps"],
            "best_success": res["best_success"],
            "first_success_wall_s": res["first_success_wall_s"],
            "first_success_gen": res["first_success_gen"],
            "first_success_wall_s_any_member": res["first_success_wall_s_any_member"],
            "first_success_gen_any_member": res["first_success_gen_any_member"],
            "budget_checkpoints": budget_checkpoints,
            "history": res["history"],
        })

    sim.close(); del sim

    out = {
        "task": {
            "K": K, "num_dofs": NUM_WHEEL_DOFS, "param_dim": K * NUM_WHEEL_DOFS,
            "dt": DT, "horizon_s": HORIZON_S, "ic_xy": IC_XY, "ic_yaw": IC_YAW,
            "target_xy": TARGET_XY, "target_yaw": TARGET_YAW, "weights": WEIGHTS,
            "init_type": INIT_TYPE, "init_noise_std": INIT_NOISE_STD,
            "source_trial": str(REF_JSON) + " trials[0] (seed=1042)",
        },
        "first_order_reference": {
            "source_file": str(REF_JSON),
            "per_iter_wall_s_3090": 0.664,
            "iterations": 50,
            "trial0_wall_s_3090": 33.56,
            "median_pos_error_m_25trials": 0.0653,
            "success_rate_25trials": 1.0,
            "local_per_iter_wall_s": local_per_iter,
            "local_budget_s": local_budget_s,
        },
        "zeroth_order": {
            "algorithm": "OpenAI-ES (antithetic sampling, centered-rank fitness shaping)",
            "population": POP,
            "sigma": 0.3,
            "lr": 0.2,
            "seeds": seeds,
            "budget_multipliers": budget_multipliers,
            "budgets_s": budgets_s,
            "note": "Each seed is run ONCE to the max (10x) budget; the 1x/5x/10x "
                    "rows are checkpoints of that single run's history (deterministic "
                    "RNG stream => equivalent to 3 independent restarts, cheaper).",
            "results": all_results,
        },
    }
    out_path = RESULTS_DIR / "zeroth_order_es.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
