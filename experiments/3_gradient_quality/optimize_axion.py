"""Helhest trajectory optimization using Axion adjoint gradients.

Optimizes K spline control points to match a real robot trajectory.
Uses calibrated physics params from Experiment 1.

Usage:
    python experiments/3_gradient_quality/optimize_axion.py \
        --ground-truth ../data/right_turn_b.json

    python experiments/3_gradient_quality/optimize_axion.py \
        --ground-truth ../data/right_turn_b.json \
        --save results/axion.json --iterations 200 --lr 0.1
"""
import argparse
import json
import os
import pathlib
import time

import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.simulation.sim_config import SyncMode
from newton import Model

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# Calibrated params from Experiment 1 (sweep_axion.json), using
# dt = largest value satisfying both stability (Exp 2: 0.2 s) and accuracy (0.125 s).
DT = 0.125
K_P = 4000.0
MU = 0.1
FRICTION_COMPLIANCE = 2e-2
CONTACT_COMPLIANCE = 1e-1

WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


def make_interp_matrix(T: int, K: int):
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
    """Adam with cosine LR decay and gradient norm clipping."""

    def __init__(
        self,
        K,
        num_dofs,
        lr,
        total_steps=200,
        lr_min_ratio=0.05,
        betas=(0.2, 0.999),
        eps=1e-8,
        grad_clip=None,
    ):
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
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_xy: wp.array(dtype=wp.vec2),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    target = target_xy[t]
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    wp.atomic_add(loss, 0, weight * (dx * dx + dy * dy))


@wp.kernel
def yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_xy: wp.array(dtype=wp.vec2),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize heading mismatch by comparing forward direction with direction to next target point."""
    t = wp.tid()
    n = target_xy.shape[0]
    if t >= n - 1:
        return
    # Robot forward direction from quaternion
    q = wp.transform_get_rotation(body_pose[t, 0, 0])
    fwd = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    # Target direction from trajectory
    dx = target_xy[t + 1][0] - target_xy[t][0]
    dy = target_xy[t + 1][1] - target_xy[t][1]
    target_dir = wp.normalize(wp.vec3(dx, dy, 0.0))
    # 1 - cos(angle)^2 penalty
    dot_fwd = wp.dot(fwd, target_dir)
    wp.atomic_add(loss, 0, weight * (1.0 - dot_fwd * dot_fwd))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def set_friction_coefficient_kernel(
    mu: float,
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, shape_idx = wp.tid()
    shape_material_mu[world_idx, shape_idx] = mu


class HelhestOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=10,
        target_trajectory_xy=None,
        lr=0.1,
        iterations=200,
        visualize=False,
    ):
        self.K = num_control_points
        self._lr = lr
        self._iterations = iterations
        self._visualize = visualize
        self._render_frame = 0
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

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        # Set calibrated friction
        wp.launch(
            kernel=set_friction_coefficient_kernel,
            dim=(self.solver.dims.num_worlds, self.solver.axion_model.shape_count),
            inputs=[MU],
            outputs=[self.solver.axion_model.shape_material_mu],
        )

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=MU, ke=50.0, kd=50.0, kf=50.0)
        self.builder.add_ground_plane(cfg=ground_cfg)
        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=K_P,
            k_d=HelhestConfig.TARGET_KD,
            friction_left_right=MU,
            friction_rear=MU,
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
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)
        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]
        wp.launch(
            kernel=loss_xy_kernel,
            dim=num_steps,
            inputs=[self.trajectory.body_pose, self.target_xy, self.trajectory_weight / num_steps],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=yaw_loss_kernel,
            dim=num_steps,
            inputs=[self.trajectory.body_pose, self.target_xy, self.yaw_weight / num_steps],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / self.clock.total_sim_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]
        grad_params = self._contract(grad_v)
        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if not self._visualize:
            return
        if self._render_frame > 0 and train_iter % 5 != 0:
            return

        target_xy_np = self.target_xy.numpy()
        n = len(target_xy_np) - 1
        if n > 0:
            z = 0.05
            starts_np = np.array(
                [[target_xy_np[i, 0], target_xy_np[i, 1], z] for i in range(n)], dtype=np.float32
            )
            ends_np = np.array(
                [[target_xy_np[i + 1, 0], target_xy_np[i + 1, 1], z] for i in range(n)],
                dtype=np.float32,
            )

            def draw_target(viewer, step_idx, state):
                viewer.log_lines(
                    "target",
                    wp.array(starts_np, dtype=wp.vec3),
                    wp.array(ends_np, dtype=wp.vec3),
                    (1.0, 0.0, 0.0),
                    width=0.02,
                )

            self.render_episode(
                iteration=train_iter,
                callback=draw_target,
                loop=True,
                loops_count=1,
                playback_speed=3.0,
            )
        self._render_frame += 1

    def train(self, init_ctrl, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps

        # Spline setup
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)
        self.spline_params = np.array(
            [[init_ctrl[0], init_ctrl[1], init_ctrl[2]]] * self.K, dtype=np.float64
        )
        self.spline_adam = SplineAdam(
            K=self.K,
            num_dofs=NUM_WHEEL_DOFS,
            lr=self._lr,
            lr_min_ratio=0.1,
            total_steps=iterations,
        )
        self._apply_params(self.spline_params)

        # Optimization
        print(f"Optimizing: T={T}, dt={self.clock.dt:.4f}, K={self.K}, lr={self._lr}")

        trial = {
            "init_ctrl": list(init_ctrl),
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "time_ms": [],
            "best_iters": [],
        }

        best_loss = float("inf")
        target_xy_np = self.target_xy.numpy()

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            # Compute RMSE
            poses = self.trajectory.body_pose.numpy()[:, 0, 0, :3]
            rmse_m = float(
                np.sqrt(
                    np.mean(
                        (poses[:, 0] - target_xy_np[:, 0]) ** 2
                        + (poses[:, 1] - target_xy_np[:, 1]) ** 2
                    )
                )
            )

            is_best = curr_loss < best_loss
            if is_best:
                best_loss = curr_loss
                trial["best_iters"].append(i)

            marker = " *" if is_best else ""
            print(
                f"  Iter {i:3d}: loss={curr_loss:.4f} | RMSE={rmse_m:.3f}m | "
                f"best={best_loss:.4f} | t={t_iter * 1000:.0f}ms{marker}"
            )

            trial["iterations"].append(i)
            trial["loss"].append(float(curr_loss))
            trial["rmse_m"].append(rmse_m)
            trial["time_ms"].append(t_iter * 1000)

            self.render(i)
            self.update()
            self.tape.zero()
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
    parser.add_argument("--ground-truth", type=str, default=str(DATA_DIR / "right_turn_b.json"))
    parser.add_argument("--save", metavar="PATH", default=str(RESULTS_DIR / "axion.json"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument(
        "--noise-std", type=float, default=0.2, help="Std of Gaussian noise for initial guess"
    )
    parser.add_argument("--init", choices=["perturbed", "zeros", "forward"], default="perturbed")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable OpenGL visualization of optimization progress",
    )
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
    print(
        f"Real robot ctrl: L={target_ctrl[0]:.3f} R={target_ctrl[1]:.3f} Rear={target_ctrl[2]:.3f}"
    )
    print(f"Duration: {duration:.1f}s, dt={DT}, K={args.K}, lr={args.lr}, "
          f"num_trials={args.num_trials}")

    sim_config = SimulationConfig(
        duration_seconds=duration,
        target_timestep_seconds=DT,
        num_worlds=1,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
    )
    render_config = RenderingConfig(
        vis_type="gl" if args.visualize else "null",
        target_fps=30,
        usd_file=None,
        world_offset_x=5.0,
        world_offset_y=5.0,
        start_paused=False,
    )
    exec_config = ExecutionConfig(use_cuda_graph=True, headless_steps_per_segment=1)
    engine_config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
        backtrack_min_iter=12,
        newton_atol=1e-5,
        linear_atol=1e-5,
        linear_tol=1e-5,
        enable_linesearch=False,
        joint_compliance=6e-8,
        contact_compliance=CONTACT_COMPLIANCE,
        friction_compliance=FRICTION_COMPLIANCE,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=8,
        differentiable_simulation=True,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    sim = HelhestOptimizer(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        num_control_points=args.K,
        target_trajectory_xy=traj_xy,
        lr=args.lr,
        iterations=args.iterations,
        visualize=args.visualize,
    )
    if args.visualize and hasattr(sim.viewer, "set_camera"):
        sim.viewer.set_camera(pos=wp.vec3(0.0, -1.5, 8.0), pitch=-90.0, yaw=90.0)

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
        "simulator": "Axion",
        "gradient_method": "adjoint",
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
