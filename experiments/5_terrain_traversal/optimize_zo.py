"""Terrain traversal with the zeroth-order optimizer (ostrich.optim.zeroth_order).

Same task, start, target and loss as optimize.py (the task object from
optimize.py settles the robot and runs the target episode), but the gradient
is the zeroth-order estimate of the smoothed loss: 2 n_pairs + 1 perturbed
plans per iteration, all as parallel worlds in one CUDA-graph rollout.

On the paper-batch terrain task (roughness 2, freq 1.5) this reached a median
final RMSE of ~0.3 m over 8 seeds where the hard adjoint ended at ~2.2 m.

Rollouts on mesh terrain are GPU-nondeterministic (~13% in loss), so the plan
is graded as the mean RMSE over --eval-reps identical worlds.

Usage:
    python experiments/5_terrain_traversal/optimize_zo.py --seed 0
    python experiments/5_terrain_traversal/optimize_zo.py --seed 0 --n-pairs 8 --save results/zo_seed0.json
"""
import argparse
import json
import pathlib
import sys
import types

import newton
import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import optimize as TT  # noqa: E402
from ostrich import (ComplianceConfig, ContactsConfig, LinearSolverConfig, LinesearchConfig,  # noqa: E402
                     LoggingConfig, NewtonRaphsonConfig, OstrichEngineConfig, RenderingConfig, SimulationConfig)
from ostrich.core.model_builder import OstrichModelBuilder  # noqa: E402
from ostrich.optim import BatchedRollout, ZeroOrderConfig, ZeroOrderOptimizer, spline_first_steps  # noqa: E402
from ostrich.simulation.sim_config import SyncMode  # noqa: E402

WHEELS = slice(TT.WHEEL_DOF_OFFSET, TT.WHEEL_DOF_OFFSET + TT.NUM_WHEEL_DOFS)


def engine_config():
    """optimize.run_single's engine settings (BatchedRollout removes early stopping)."""
    return OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=14, backtrack_min_iter=10, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-6, friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256))


def make_task(args):
    """optimize.py's task object after its train() preamble: settled start, target episode."""
    target_spline, init_spline = TT.generate_splines(args.seed, K=args.K, sigma=args.perturbation_sigma,
                                                     wildness=args.curvature)
    sim_config = SimulationConfig(duration_seconds=args.duration, target_timestep_seconds=args.dt, num_worlds=1,
                                  sync_mode=SyncMode.ALIGN_FPS_TO_DT, use_cuda_graph=True)
    rc = RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False)
    sim = TT.TerrainTraversalOptimizer(sim_config, rc, engine_config(), LoggingConfig(), num_control_points=args.K,
                                       target_spline=target_spline, init_spline=init_spline, terrain_seed=args.seed,
                                       roughness=args.roughness, terrain_freq=args.terrain_freq, lr=args.lr,
                                       lr_min_ratio=0.1, check_grad=False, visualize=False)
    sim._settle()
    newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd, sim.states[0])
    newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd, sim.target_states[0])
    T = sim.clock.total_sim_steps
    sim.W, sim.W_col_sums = TT.make_interp_matrix(T, sim.K)
    sim._apply_spline_to_controls(sim._target_spline, sim.target_controls, T)
    sim.run_target_episode()
    return sim


def make_rollout(task, args, n_worlds):
    fake = types.SimpleNamespace(builder=OstrichModelBuilder(), _terrain_seed=args.seed, _roughness=args.roughness,
                                 _terrain_freq=args.terrain_freq,
                                 simulation_config=types.SimpleNamespace(num_worlds=n_worlds))
    model = TT.TerrainTraversalOptimizer.build_model(fake)
    T = task.clock.total_sim_steps
    target = task.trajectory.target_body_pose.numpy()[:, 0, 0].astype(np.float64)  # (T+1, 7) chassis
    w_traj, w_yaw, w_reg = task.trajectory_weight, task.yaw_weight, task.regularization_weight

    def apply_params(params, controls):
        u = np.einsum("tk,wkd->twd", task.W, params).astype(np.float32)
        jtv = np.zeros((len(params), controls[0].joint_target_vel.size // len(params)), np.float32)
        for t, c in enumerate(controls):
            jtv[:, WHEELS] = u[t]
            c.joint_target_vel.assign(jtv.reshape(c.joint_target_vel.shape))

    def forward_axis(q):
        x, y, z, w = (q[..., i] for i in range(4))
        return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)

    def step_losses(body_q, body_qd, params):
        """TerrainTraversalOptimizer.compute_loss split per state: 3D position +
        heading at state t, plus the regulariser of the control that first
        affects state t."""
        chassis = body_q[:, :, 0].astype(np.float64)
        Ts = chassis.shape[0]
        lt = w_traj / Ts * np.sum((chassis[:, :, :3] - target[:, None, :3]) ** 2, -1)
        dot = np.sum(forward_axis(chassis[:, :, 3:7]) * forward_axis(target[:, None, 3:7]), -1)
        lt += w_yaw / Ts * (1 - dot ** 2)
        u = np.einsum("tk,wkd->twd", task.W, params)
        lt[1:] += w_reg / Ts * np.sum(u ** 2, -1)
        return lt

    ro = BatchedRollout(model, engine_config(), T, task.clock.dt, apply_params, step_losses,
                        initial_body_q=task.states[0].body_q.numpy(), initial_body_qd=task.states[0].body_qd.numpy())
    ro.target = target
    return ro


def rmse(rollout):
    """Chassis 3D RMSE per world of the last rollout (as optimize.py)."""
    d = rollout.body_q[:, :, 0, :3] - rollout.target[:, None, :3]
    return np.sqrt(np.mean(np.sum(d ** 2, -1), 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--sigma-final", type=float, default=None)
    ap.add_argument("--directions", choices=["orth", "iid"], default="orth")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eval-reps", type=int, default=16)
    # task: paper-batch terrain settings (run_batch.sh)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=TT.DT)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--perturbation-sigma", type=float, default=1.5)
    ap.add_argument("--curvature", type=float, default=3.0)
    ap.add_argument("--roughness", type=float, default=2.0)
    ap.add_argument("--terrain-freq", type=float, default=1.5)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    cfg = ZeroOrderConfig(iterations=args.iterations, n_pairs=args.n_pairs, sigma=args.sigma,
                          sigma_final=args.sigma_final, directions=args.directions, causal=not args.no_causal,
                          lr=args.lr, seed=args.seed)
    task = make_task(args)
    rollout = make_rollout(task, args, cfg.num_worlds)
    opt = ZeroOrderOptimizer(rollout, cfg, first_step=spline_first_steps(task.W))
    history = []

    def log(it, theta, loss, grad):
        history.append(float(rmse(rollout)[0]))
        if it % 10 == 0 or it == cfg.iterations - 1:
            print(f"  iter {it:3d}: loss {loss:8.3f}  RMSE(theta) {history[-1]:.3f} m", flush=True)

    theta0 = task._init_spline.astype(np.float64)
    res = opt.optimize(theta0, callback=log)

    grader = make_rollout(task, args, args.eval_reps)
    graded = {}
    for name, p in [("init", theta0), ("best", res["best_params"]), ("final", res["final_params"])]:
        grader.rollout(np.stack([p] * args.eval_reps))
        r = rmse(grader)
        graded[name] = {"rmse_mean": float(r.mean()), "rmse_sd": float(r.std())}
    print(f"seed {args.seed}: RMSE init {graded['init']['rmse_mean']:.3f} -> best {graded['best']['rmse_mean']:.3f} "
          f"final {graded['final']['rmse_mean']:.3f} m (means over {args.eval_reps} rollouts); "
          f"{res['rollouts']} rollouts")
    if args.save:
        out = {"seed": args.seed, "config": vars(args), "rmse_history": history, "losses": res["losses"],
               "best_iter": res["best_iter"], "best_params": res["best_params"].tolist(),
               "final_params": res["final_params"].tolist(), "eval": graded}
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
