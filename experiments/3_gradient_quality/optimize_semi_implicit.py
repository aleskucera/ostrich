"""Helhest trajectory optimization using Newton Semi-Implicit + Warp tape BPTT.

Optimizes K spline control points to match a real robot trajectory.
Uses calibrated physics params from Experiment 1.
Gradients via backpropagation through time on Warp's computation tape.

Usage:
    python experiments/3_gradient_quality/optimize_semi_implicit.py
    python experiments/3_gradient_quality/optimize_semi_implicit.py \
        --ground-truth ../data/right_turn_b.json \
        --save results/semi_implicit.json
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig
from axion.simulation.differentiable_simulator import NewtonDifferentiableSimulator
from axion.simulation.sim_config import SyncMode
from newton import Model

from examples.helhest.common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# Calibrated params from Experiment 1 (sweep_semi_implicit.json)
DT = 0.0005
K_P = 0.0
K_D = 400.0
MU = 0.02
KE = 8000.0
KD_CONTACT = 2000.0
KF = 1500.0

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    col_sums = W.sum(axis=0)
    return W, col_sums


class SplineAdam:
    def __init__(self, K, num_dofs, lr, total_steps=200, lr_min_ratio=0.05,
                 betas=(0.2, 0.999), eps=1e-8, grad_clip=None):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.grad_clip = grad_clip
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self):
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params, grad):
        self.t += 1
        if self.grad_clip is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        lr = self._cosine_lr()
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


@wp.kernel
def loss_xy_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_xy: wp.array(dtype=wp.vec2),
    t_idx: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    pos = wp.transform_get_translation(body_q[0])
    target = target_xy[t_idx]
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def yaw_loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_xy: wp.array(dtype=wp.vec2),
    t_idx: int,
    num_steps: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    if t_idx >= num_steps - 1:
        return
    q = wp.transform_get_rotation(body_q[0])
    fwd = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    dx = target_xy[t_idx + 1][0] - target_xy[t_idx][0]
    dy = target_xy[t_idx + 1][1] - target_xy[t_idx][1]
    target_dir = wp.normalize(wp.vec3(dx, dy, 0.0))
    dot_fwd = wp.dot(fwd, target_dir)
    wp.atomic_add(loss, 0, weight * (1.0 - dot_fwd * dot_fwd))


class HelhestSemiImplicitOptimizer(NewtonDifferentiableSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config,
                 logging_config, num_control_points=10,
                 target_trajectory_xy=None, lr=0.01, iterations=200):
        self.K = num_control_points
        self._lr = lr
        self._iterations = iterations
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7

        # Resample target trajectory to match simulation steps
        T = self.clock.total_sim_steps
        real_t = np.linspace(0, 1, len(target_trajectory_xy))
        sim_t = np.linspace(0, 1, T + 1)
        xy_resampled = np.zeros((T + 1, 2), dtype=np.float32)
        xy_resampled[:, 0] = np.interp(sim_t, real_t, target_trajectory_xy[:, 0])
        xy_resampled[:, 1] = np.interp(sim_t, real_t, target_trajectory_xy[:, 1])
        self.target_xy = wp.array(xy_resampled, dtype=wp.vec2, requires_grad=False)
        self.target_xy_np = xy_resampled

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=MU, ke=KE, kd=KD_CONTACT, kf=KF,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=K_P, k_d=K_D,
            friction_left_right=MU, friction_rear=MU,
            ke=KE, kd=KD_CONTACT, kf=KF,
        )
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def _expand(self, params):
        return self.W @ params

    def _contract(self, grad_v):
        return self.W.T @ grad_v

    def _apply_params(self, params):
        T = self.clock.total_sim_steps
        expanded = self._expand(params)
        for i in range(T):
            ctrl_np = np.zeros(9, dtype=np.float32)
            ctrl_np[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(
                self.controls[i].joint_target_vel,
                wp.array(ctrl_np, dtype=wp.float32, device=self.model.device),
            )

    def compute_loss(self):
        T = self.clock.total_sim_steps
        for t in range(T):
            wp.launch(
                kernel=loss_xy_kernel, dim=1,
                inputs=[self.states[t + 1].body_q, self.target_xy,
                        t + 1, self.trajectory_weight / T],
                outputs=[self.loss], device=self.model.device,
            )
            wp.launch(
                kernel=yaw_loss_kernel, dim=1,
                inputs=[self.states[t + 1].body_q, self.target_xy,
                        t + 1, T + 1, self.yaw_weight / T],
                outputs=[self.loss], device=self.model.device,
            )

    def update(self):
        grad_v = np.zeros((self.clock.total_sim_steps, NUM_WHEEL_DOFS), dtype=np.float64)
        for i in range(self.clock.total_sim_steps):
            g = self.controls[i].joint_target_vel.grad.numpy()
            grad_v[i] = g[WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS]
        grad_params = self._contract(grad_v)
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def train(self, init_ctrl, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps

        # Spline setup
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array(
            [[init_ctrl[0], init_ctrl[1], init_ctrl[2]]] * self.K, dtype=np.float64
        )
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS,
            lr=self._lr, lr_min_ratio=0.1, total_steps=iterations,
        )
        self._apply_params(self.spline_params)

        print(f"Optimizing: T={T}, dt={self.clock.dt}, K={self.K}, lr={self._lr}")

        trial = {
            "init_ctrl": list(init_ctrl),
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "time_ms": [],
            "best_iters": [],
        }

        best_loss = float("inf")

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            # Compute RMSE from final state positions
            poses_xy = []
            for t in range(T + 1):
                bq = self.states[t].body_q.numpy()
                poses_xy.append([float(bq[0, 0]), float(bq[0, 1])])
            poses_xy = np.array(poses_xy)
            rmse_m = float(np.sqrt(np.mean(
                (poses_xy[:, 0] - self.target_xy_np[:, 0])**2 +
                (poses_xy[:, 1] - self.target_xy_np[:, 1])**2
            )))

            is_best = curr_loss < best_loss
            if is_best:
                best_loss = curr_loss
                trial["best_iters"].append(i)

            marker = " *" if is_best else ""
            print(f"  Iter {i:3d}: loss={curr_loss:.4f} | RMSE={rmse_m:.3f}m | "
                  f"best={best_loss:.4f} | t={t_iter * 1000:.0f}ms{marker}")

            trial["iterations"].append(i)
            trial["loss"].append(float(curr_loss))
            trial["rmse_m"].append(rmse_m)
            trial["time_ms"].append(t_iter * 1000)

            self.update()
            self.tape.zero()
            self.tape.reset()  # release recorded ops; otherwise tape grows each iter
            self.loss.zero_()

        trial["best_loss"] = float(best_loss)
        return trial


def load_ground_truth(path):
    with open(path) as f:
        gt = json.load(f)
    target_ctrl = gt["target_ctrl_rad_s"]
    duration = gt["trajectory"].get("constant_speed_duration_s", gt["trajectory"]["duration_s"])
    traj_xy = np.array(
        [[p["x"], p["y"]] for p in gt["trajectory"]["points"] if p["t"] <= duration],
        dtype=np.float32,
    )
    return target_ctrl, duration, traj_xy


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth", type=str,
                        default=str(DATA_DIR / "right_turn_b.json"))
    parser.add_argument("--save", metavar="PATH",
                        default=str(RESULTS_DIR / "semi_implicit.json"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--init", choices=["perturbed", "zeros", "forward"],
                        default="perturbed")
    parser.add_argument("--horizon-s", type=float, default=None,
                        help="Truncate trajectory to first N seconds (default: full duration)")
    parser.add_argument("--num-trials", type=int, default=3,
                        help="Number of independent runs (different perturbed init guesses)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="First seed; trial k uses seed_base + k")
    args = parser.parse_args()

    target_ctrl, duration, traj_xy = load_ground_truth(args.ground_truth)
    if args.horizon_s is not None and args.horizon_s < duration:
        keep = max(2, int(args.horizon_s / duration * len(traj_xy)))
        traj_xy = traj_xy[:keep]
        duration = args.horizon_s

    print(f"Target: real robot trajectory ({len(traj_xy)} points)")
    print(f"Real robot ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f}")
    print(f"Duration: {duration:.1f}s, dt={DT}, K={args.K}, lr={args.lr}, "
          f"num_trials={args.num_trials}")

    sim_config = SimulationConfig(
        duration_seconds=duration,
        target_timestep_seconds=DT,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="null", target_fps=30, usd_file=None, start_paused=False,
    )
    exec_config = ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1)
    engine_config = SemiImplicitEngineConfig(
        angular_damping=0.05, friction_smoothing=0.1,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    sim = HelhestSemiImplicitOptimizer(
        sim_config, render_config, exec_config, engine_config, logging_config,
        num_control_points=args.K,
        target_trajectory_xy=traj_xy, lr=args.lr, iterations=args.iterations,
    )

    trials = []
    for k in range(args.num_trials):
        seed = args.seed_base + k
        np.random.seed(seed)
        if args.init == "zeros":
            init_ctrl = [0.0, 0.0, 0.0]
        elif args.init == "forward":
            avg = float(np.mean(target_ctrl))
            init_ctrl = [avg, avg, avg]
        else:
            init_ctrl = [c + np.random.randn() * args.noise_std for c in target_ctrl]
        print(f"\n=== Trial {k + 1}/{args.num_trials} (seed={seed}) ===")
        print(f"Init ctrl ({args.init}): L={init_ctrl[0]:.3f} R={init_ctrl[1]:.3f} "
              f"Rear={init_ctrl[2]:.3f}")
        trial = sim.train(init_ctrl, iterations=args.iterations)
        trial["seed"] = seed
        trials.append(trial)

    aggregate = {
        "simulator": "Semi-Implicit",
        "gradient_method": "warp_tape",
        "dt": DT,
        "T": sim.clock.total_sim_steps,
        "K": args.K,
        "num_trials": args.num_trials,
        "trials": trials,
    }
    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(aggregate, indent=2))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
