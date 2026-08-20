"""Terrain traversal with first-order bundled gradients.

Same task as optimize.py, same exact adjoint backward pass, but instead of
one gradient at the nominal spline, we run N parallel perturbed worlds and
average the N exact gradients. Each world i uses spline_params + sigma * w_i
(antithetic pairs by default), all worlds share one batched forward pass and
one batched adjoint via the differentiable simulator.

Distinct from optimize_rs.py, which is the *zero-order* SPSA-style estimator
(finite differences across worlds, no adjoint). This file is the first-order
bundled gradient from Suh, Pang, Tedrake (2021), made tractable by the
batched forward + adjoint that other diffsims can't afford at this N.

Usage:
    python -m examples.terrain_traversal.optimize_bg --seed 42
    python -m examples.terrain_traversal.optimize_bg --seed 42 --num-samples 32 --sigma 0.3
    python -m examples.terrain_traversal.optimize_bg --num-seeds 50 \
        --save results/terrain_bundled.json
"""
import argparse
import json
import os
# This experiment was tuned with contact FB alpha = 0.5. It is baked into a
# warp constant when ostrich.constraints is first imported, so it has to be
# set before any ostrich import below -- not at the config site. setdefault,
# so an explicit value from the environment still wins.
os.environ.setdefault("OSTRICH_CONTACT_FB_ALPHA", "0.5")

import pathlib
import time

import newton
import numpy as np
import warp as wp
from ostrich import OstrichDifferentiableSimulator
from ostrich import OstrichEngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig
from ostrich.simulation.sim_config import SyncMode
from newton import Model

from examples.terrain_traversal.helhest_model import create_helhest_model
from examples.terrain_traversal.helhest_model import HelhestConfig
from examples.terrain_traversal.terrain import generate_terrain_mesh
from examples.terrain_traversal.optimize import (
    generate_splines,
    make_interp_matrix,
    SplineAdam,
    WHEEL_DOF_OFFSET,
    NUM_WHEEL_DOFS,
    DT,
)
from ostrich.core.engine_config import ComplianceConfig
from ostrich.core.engine_config import ContactsConfig
from ostrich.core.engine_config import LinearSolverConfig
from ostrich.core.engine_config import LinesearchConfig
from ostrich.core.engine_config import NewtonRaphsonConfig
from ostrich.core.logging_config import HDF5LoggingConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"


# Loss kernels iterate over (timestep, world). Target poses are read from world 0
# (replicated across all worlds), each perturbed world contributes its own gradient.

@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    t, w = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, w, 0])
    target_pos = wp.transform_get_translation(target_body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))


@wp.kernel
def yaw_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    t, w = wp.tid()
    q = wp.transform_get_rotation(body_pose[t, w, 0])
    q_target = wp.transform_get_rotation(target_body_pose[t, 0, 0])
    fwd = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    fwd_target = wp.quat_rotate(q_target, wp.vec3(1.0, 0.0, 0.0))
    dot_fwd = wp.dot(fwd, fwd_target)
    wp.atomic_add(loss, 0, weight * (1.0 - dot_fwd * dot_fwd))


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    sim_step, w, wheel_idx = wp.tid()
    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, w, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


class TerrainTraversalBundledOptimizer(OstrichDifferentiableSimulator):
    """First-order bundled-gradient optimizer for terrain traversal.

    N = num_worlds parallel rollouts, each with spline_params + sigma * w_i.
    One batched forward pass + one batched adjoint produces N exact gradients;
    they are averaged into a single bundled gradient before the Adam step.
    """

    def __init__(
        self,
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=10,
        target_spline=None,
        init_spline=None,
        terrain_seed=None,
        roughness=1.0,
        terrain_freq=1.0,
        lr=0.1,
        sigma=0.3,
        sigma_min_ratio=0.1,
        antithetic=True,
        visualize=False,
    ):
        self.K = num_control_points
        self._target_spline = target_spline
        self._init_spline = init_spline
        self._terrain_seed = terrain_seed
        self._roughness = roughness
        self._terrain_freq = terrain_freq
        self._lr = lr
        self._sigma_init = sigma
        self._sigma_min_ratio = sigma_min_ratio
        self._antithetic = antithetic
        self._visualize = visualize
        self._render_frame = 0

        super().__init__(sim_config, render_config, engine_config, logging_config)

        self.N = sim_config.num_worlds  # bundled samples = parallel worlds
        if self._antithetic and self.N % 2 != 0:
            raise ValueError(f"antithetic=True requires even num_worlds, got {self.N}")

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7

        # Per-iter spline noise [N, K, 3]
        self.noise = np.zeros((self.N, self.K, NUM_WHEEL_DOFS), dtype=np.float64)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

        if self._visualize:
            self.viewer.set_camera(
                pos=wp.vec3(-15.0, -15.0, 18.0),
                pitch=-35.0,
                yaw=45.0,
            )

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.2

        if self._terrain_seed is None:
            raise ValueError("terrain_seed is required")
        surface_mesh, terrain_h = generate_terrain_mesh(
            seed=self._terrain_seed,
            roughness=self._roughness,
            terrain_freq=self._terrain_freq,
        )
        spawn_z = terrain_h + HelhestConfig.WHEEL_RADIUS + 0.05

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(-8.0, 0.0, spawn_z), wp.quat_identity()),
            control_mode="velocity",
            k_p=250.0,
            k_d=0.0,
            friction_left_right=0.8,
            friction_rear=0.35,
        )

        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.5,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def _current_sigma(self):
        progress = min(self.spline_adam.t / self.spline_adam.total_steps, 1.0)
        sigma_min = self._sigma_init * self._sigma_min_ratio
        return sigma_min + 0.5 * (self._sigma_init - sigma_min) * (1.0 + np.cos(np.pi * progress))

    def _sample_noise(self, sigma):
        """[N, K, 3] noise. Antithetic: second half mirrors the first."""
        if self._antithetic:
            half = self.N // 2
            base = self._rng.standard_normal((half, self.K, NUM_WHEEL_DOFS)).astype(np.float64) * sigma
            return np.concatenate([base, -base], axis=0)
        return self._rng.standard_normal((self.N, self.K, NUM_WHEEL_DOFS)).astype(np.float64) * sigma

    def _expand_per_world(self, params):
        """[K, 3] params + [N, K, 3] noise -> [T, N, 3] per-step per-world wheel velocities."""
        perturbed = params[None, :, :] + self.noise  # [N, K, 3]
        return np.einsum("tk,nkd->tnd", self.W, perturbed)  # [T, N, 3]

    def _apply_params(self, params):
        """Resample noise, expand per-world, write into joint_target_vel."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        self.noise = self._sample_noise(self._current_sigma())
        expanded = self._expand_per_world(params)  # [T, N, 3]

        vel_np = np.zeros((T, self.N, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded.astype(np.float32)

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def _apply_spline_to_controls(self, spline, controls, T):
        """Expand a (K, 3) spline and broadcast across N worlds — used for the target episode."""
        expanded = self.W @ spline  # (T, 3)
        num_dofs = controls[0].joint_target_vel.shape[-1]
        for i in range(T):
            ctrl = np.zeros((self.N, num_dofs), dtype=np.float32)
            ctrl[:, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded[i]
            wp.copy(
                controls[i].joint_target_vel,
                wp.array(ctrl, dtype=wp.float32, device=self.model.device),
            )

    def compute_loss(self):
        T = self.trajectory.body_pose.shape[0]
        wp.launch(
            kernel=loss_kernel,
            dim=(T, self.N),
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.trajectory_weight / T,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=yaw_loss_kernel,
            dim=(T, self.N),
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.yaw_weight / T,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(T, self.N, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / T,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # grad shape: [T, N, num_dofs] — independent per perturbed world
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]  # [T, N, 3]

        # Per-world spline contraction: [T, N, 3] -> [K, N, 3]
        grad_per_world = np.einsum("tk,tnd->knd", self.W, grad_v)  # [K, N, 3]

        # Bundled gradient: average across the N noisy samples
        grad_params = grad_per_world.mean(axis=1)  # [K, 3]

        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if not self._visualize:
            return
        if self._render_frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0] / self.N

        target_poses = self.trajectory.target_body_pose.numpy()
        num_steps = target_poses.shape[0]

        waypoint_stride = max(1, num_steps // 20)
        waypoint_indices = list(range(0, num_steps, waypoint_stride))

        waypoint_xforms = wp.array(
            [target_poses[i, 0, 0] for i in waypoint_indices],
            dtype=wp.transform,
        )
        waypoint_colors = wp.array(
            [wp.vec3(1.0, 0.2, 0.0)] * len(waypoint_indices),
            dtype=wp.vec3,
        )

        half = (
            HelhestConfig.CHASSIS_SIZE[0] / 8.0,
            HelhestConfig.CHASSIS_SIZE[1] / 8.0,
            HelhestConfig.CHASSIS_SIZE[2] / 8.0,
        )

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target_trajectory",
                newton.GeoType.BOX,
                half,
                waypoint_xforms,
                waypoint_colors,
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")
        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=3.0,
        )
        self._render_frame += 1

    def _settle(self, num_steps=10):
        """Run non-differentiable settling steps so the robot rests on the terrain."""
        print(f"Settling for {num_steps} steps...")
        settle_state_a = self.model.state()
        settle_state_b = self.model.state()
        settle_control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, settle_state_a)

        for i in range(num_steps):
            self.collision_pipeline.collide(settle_state_a, self.contacts)
            self.solver.step(
                state_in=settle_state_a,
                state_out=settle_state_b,
                control=settle_control,
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            settle_state_a, settle_state_b = settle_state_b, settle_state_a

        wp.copy(self.model.joint_q, settle_state_a.joint_q)
        wp.copy(self.model.joint_qd, settle_state_a.joint_qd)
        self.solver.reset_timestep_counter()

    def train(self, iterations=200, rng_seed=42):
        self._rng = np.random.default_rng(rng_seed)

        self._settle()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        T = self.clock.total_sim_steps
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        # Target episode — broadcast the target spline across all N worlds
        self._apply_spline_to_controls(self._target_spline, self.target_controls, T)
        self.run_target_episode()

        if self._visualize:
            print("Rendering target episode...")
            self.states, self.target_states = self.target_states, self.states
            self.render_episode(
                iteration=-1, loop=True, loops_count=1, playback_speed=1.0, start_paused=True
            )
            self.states, self.target_states = self.target_states, self.states

        # Initialize optimizer state
        self.spline_params = self._init_spline.copy().astype(np.float64)
        self.spline_adam = SplineAdam(
            K=self.K,
            num_dofs=NUM_WHEEL_DOFS,
            lr=self._lr,
            lr_min_ratio=0.1,
            total_steps=iterations,
        )
        self._apply_params(self.spline_params)

        print(
            f"Terrain traversal (BUNDLED): T={T}, dt={self.clock.dt:.4f}, "
            f"K={self.K}, N={self.N}, sigma={self._sigma_init} -> "
            f"{self._sigma_init * self._sigma_min_ratio:.3f}, "
            f"antithetic={self._antithetic}, seed={self._terrain_seed}"
        )

        results = {
            "simulator": "Ostrich",
            "problem": "terrain_traversal_bundled",
            "method": "first_order_bundled",
            "seed": self._terrain_seed,
            "target_spline": self._target_spline.tolist(),
            "init_spline": self._init_spline.tolist(),
            "dt": self.clock.dt,
            "T": T,
            "K": self.K,
            "N": self.N,
            "sigma_init": self._sigma_init,
            "sigma_min_ratio": self._sigma_min_ratio,
            "antithetic": self._antithetic,
            "duration": T * self.clock.dt,
            "roughness": self._roughness,
            "terrain_freq": self._terrain_freq,
            "iterations": [],
            "loss": [],
            "rmse_m": [],
            "rmse_world0_m": [],
            "time_ms": [],
            "trajectories": {},
            "best_iters": [],
        }

        # Target trajectory (world 0) — same across all worlds since target was broadcast
        target_pos = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]
        results["target_trajectory"] = {
            "x": target_pos[:, 0].tolist(),
            "y": target_pos[:, 1].tolist(),
        }

        best_rmse = float("inf")

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_iter = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0] / self.N  # per-world average

            # Per-world RMSE (poses across all N worlds vs shared target)
            poses = self.trajectory.body_pose.numpy()[:, :, 0, :3]  # [T, N, 3]
            target_xyz = self.trajectory.target_body_pose.numpy()[:, 0, 0, :3]  # [T, 3]
            err = poses - target_xyz[:, None, :]  # [T, N, 3]
            per_world_rmse = np.sqrt(np.mean(np.sum(err ** 2, axis=-1), axis=0))  # [N]
            rmse_m = float(per_world_rmse.mean())  # mean across worlds — best apples-to-apples
            rmse_world0 = float(per_world_rmse[0])  # for sanity comparison with optimize.py

            if rmse_m < best_rmse:
                best_rmse = rmse_m
                results["best_iters"].append(i)

            sigma_now = self._current_sigma()
            print(
                f"  Iter {i:3d}: loss={curr_loss:.4f} | "
                f"RMSE={rmse_m:.3f}m (w0={rmse_world0:.3f}m) | "
                f"best={best_rmse:.3f}m | "
                f"sigma={sigma_now:.3f} | "
                f"t={t_iter * 1000:.0f}ms"
            )

            results["iterations"].append(i)
            results["loss"].append(float(curr_loss))
            results["rmse_m"].append(rmse_m)
            results["rmse_world0_m"].append(rmse_world0)
            results["time_ms"].append(t_iter * 1000)

            results["trajectories"][str(i)] = {
                "x": poses[:, 0, 0].tolist(),  # world 0's x
                "y": poses[:, 0, 1].tolist(),
            }

            self.render(i)
            self.update()
            self.tape.zero()
            self.loss.zero_()

        return results


def run_single(args, seed):
    """Run bundled optimization for a single terrain seed. Returns results dict."""
    visualize = getattr(args, "visualize", False)

    target_spline, init_spline = generate_splines(
        seed,
        K=args.K,
        sigma=args.sigma_init_perturb,
        wildness=args.curvature,
    )
    print(
        f"  Target spline[0]: L={target_spline[0,0]:.2f} R={target_spline[0,1]:.2f} "
        f"Rear={target_spline[0,2]:.2f}"
    )
    print(
        f"  Init   spline[0]: L={init_spline[0,0]:.2f} R={init_spline[0,1]:.2f} "
        f"Rear={init_spline[0,2]:.2f}"
    )

    sim_config = SimulationConfig(
        duration_seconds=args.duration,
        target_timestep_seconds=args.dt,
        num_worlds=args.num_samples,
        sync_mode=SyncMode.ALIGN_FPS_TO_DT,
        use_cuda_graph=True,
    )
    render_config = RenderingConfig(
        vis_type="gl" if visualize else "null",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    engine_config = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(
            max_iters=14,
            backtrack_min_iter=10,
            atol=1e-3,
        ),
        linear=LinearSolverConfig(
            max_iters=16,
            atol=1e-3,
            tol=1e-3,
            regularization=1e-6,
        ),
        compliance=ComplianceConfig(
            joint=6e-8,
            contact=0.1,
            friction=1e-6,
        ),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    logging_config = LoggingConfig(
        hdf5=HDF5LoggingConfig(enabled=False),
    )

    sim = TerrainTraversalBundledOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=args.K,
        target_spline=target_spline,
        init_spline=init_spline,
        terrain_seed=seed,
        roughness=args.roughness,
        terrain_freq=args.terrain_freq,
        lr=args.lr,
        sigma=args.sigma,
        sigma_min_ratio=args.sigma_min_ratio,
        antithetic=not args.no_antithetic,
        visualize=visualize,
    )
    return sim.train(iterations=args.iterations, rng_seed=args.noise_seed)


def main():
    parser = argparse.ArgumentParser(description="Terrain traversal with first-order bundled gradients")
    parser.add_argument("--save", metavar="PATH", help="Save results to JSON")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="Terrain/control seed")
    parser.add_argument("--num-seeds", type=int, default=None, help="Run over N random seeds (0..N-1)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of bundled samples = parallel worlds (default: 32). Even if --antithetic.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="Initial bundling perturbation std dev in spline space (default: 0.3)",
    )
    parser.add_argument(
        "--sigma-min-ratio",
        type=float,
        default=0.1,
        help="Final sigma as fraction of initial (cosine annealed). Default 0.1.",
    )
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic sampling (default: enabled).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="RNG seed for the bundling noise (separate from terrain seed). Default 42.",
    )
    parser.add_argument(
        "--sigma-init-perturb",
        type=float,
        default=0.5,
        help="Std dev for the *initial guess* perturbation away from the target. Default 0.5.",
    )
    parser.add_argument("--curvature", type=float, default=0.8)
    parser.add_argument("--roughness", type=float, default=1.0)
    parser.add_argument("--terrain-freq", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.num_seeds is not None:
        all_results = []
        for seed in range(args.num_seeds):
            print(f"\n{'='*60}")
            print(f"  SEED {seed}/{args.num_seeds - 1}")
            print(f"{'='*60}")
            result = run_single(args, seed=seed)
            all_results.append(result)

        final_rmses = [r["rmse_m"][-1] for r in all_results]
        best_rmses = [min(r["rmse_m"]) for r in all_results]
        median_times = [float(np.median(r["time_ms"][1:])) for r in all_results]

        summary = {
            "num_seeds": args.num_seeds,
            "method": "first_order_bundled",
            "N": args.num_samples,
            "sigma": args.sigma,
            "sigma_min_ratio": args.sigma_min_ratio,
            "antithetic": not args.no_antithetic,
            "dt": args.dt,
            "duration": args.duration,
            "K": args.K,
            "iterations": args.iterations,
            "final_rmse_median": float(np.median(final_rmses)),
            "final_rmse_mean": float(np.mean(final_rmses)),
            "final_rmse_std": float(np.std(final_rmses)),
            "final_rmse_min": float(np.min(final_rmses)),
            "final_rmse_max": float(np.max(final_rmses)),
            "best_rmse_median": float(np.median(best_rmses)),
            "best_rmse_mean": float(np.mean(best_rmses)),
            "best_rmse_std": float(np.std(best_rmses)),
            "median_iter_time_ms": float(np.median(median_times)),
            "per_seed": all_results,
        }

        print(f"\n{'='*60}")
        print(f"  BATCH SUMMARY (BUNDLED, {args.num_seeds} seeds, N={args.num_samples}, sigma={args.sigma})")
        print(f"{'='*60}")
        print(
            f"  Final RMSE: {summary['final_rmse_median']:.3f}m median, "
            f"{summary['final_rmse_mean']:.3f} +/- {summary['final_rmse_std']:.3f}m"
        )
        print(
            f"  Best  RMSE: {summary['best_rmse_median']:.3f}m median, "
            f"{summary['best_rmse_mean']:.3f} +/- {summary['best_rmse_std']:.3f}m"
        )
        print(f"  Iter  time: {summary['median_iter_time_ms']:.0f}ms median")

        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(summary, indent=2))
            print(f"  Saved to {args.save}")
    else:
        result = run_single(args, seed=args.seed)
        if args.save:
            pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(args.save).write_text(json.dumps(result, indent=2))
            print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
