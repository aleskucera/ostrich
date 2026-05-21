"""Bundled-gradient trajectory optimization for the Helhest robot on a mesh terrain.

Sibling of trajectory_spline_surface_fast.py. Same problem, same loss, same Adam.
The only change is the gradient: instead of one exact gradient at the nominal
spline control points, we draw N noise samples around them, run N parallel rollouts,
and average the N exact gradients before stepping Adam.

Reference: Suh, Pang, Tedrake, "Bundled Gradients through Contact via Randomized
Smoothing" (2021).
"""
import argparse
import os
import pathlib
import time

import newton
import numpy as np
import openmesh
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import AxionEngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import ComplianceConfig
from axion import ContactsConfig
from axion import LinearSolverConfig
from axion import LinesearchConfig
from axion import NewtonRaphsonConfig
from newton import Model

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
ASSETS_DIR = pathlib.Path(__file__).parent.parent.parent.joinpath("assets")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# Ground-truth controls used to generate the target trajectory (constant over time,
# so the true spline has all K knots equal to this triple).
TARGET_WHEEL_VEL = (5.0, 3.0, 4.0)


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, K] linear interpolation weight matrix and per-column normalization."""
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
    """Adam optimizer for a [K, num_dofs] numpy parameter array."""

    def __init__(
        self,
        K: int,
        num_dofs: int,
        lr: float,
        total_steps: int = 200,
        lr_min_ratio: float = 0.05,
        betas=(0.1, 0.999),
        eps=1e-8,
    ):
        self.lr_init = lr
        self.lr_min = lr * lr_min_ratio
        self.total_steps = total_steps
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros((K, num_dofs), dtype=np.float64)
        self.v = np.zeros((K, num_dofs), dtype=np.float64)
        self.t = 0

    def _cosine_lr(self) -> float:
        progress = min(self.t / self.total_steps, 1.0)
        return self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1.0 + np.cos(np.pi * progress))

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        lr = self._cosine_lr()
        return params - lr * m_hat / (np.sqrt(v_hat) + self.eps)


# Loss kernels iterate over (timestep, world). The target trajectory is shared across
# all worlds (read from world 0); each perturbed world contributes its own gradient
# to the summed loss.


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


class HelhestTrajectorySplineSurfaceBundledOptimizer(AxionDifferentiableSimulator):
    """Bundled-gradient version of HelhestTrajectorySplineSurfaceOptimizer.

    Each iteration draws N noise samples on spline_params (in [K, 3] space), runs N
    parallel rollouts (one per world), computes N exact gradients via the batched
    adjoint, and averages them into a single bundled gradient before stepping Adam.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 10,
        sigma: float = 0.3,
        sigma_min_ratio: float = 0.1,
        antithetic: bool = True,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.K = num_control_points
        self.N = sim_config.num_worlds  # bundled samples = parallel worlds
        self.sigma_init = sigma
        self.sigma_min = sigma * sigma_min_ratio
        self.antithetic = antithetic

        if self.antithetic and self.N % 2 != 0:
            raise ValueError(f"antithetic=True requires even num_worlds, got {self.N}")

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.trajectory_weight = 10.0
        self.yaw_weight = 5.0
        self.regularization_weight = 1e-7

        self.frame = 0
        self.best_loss = float("inf")

        self.export_path: str | None = None
        self._iter_body_poses: list[np.ndarray] = []
        self._iter_indices: list[int] = []
        self._iter_losses: list[float] = []

        # Initial guess (Left, Right, Rear)
        self.init_wheel_vel = (4.0, 4.0, 3.0)

        # Per-iter noise on the spline control points: [N, K, 3]
        self.noise = np.zeros((self.N, self.K, NUM_WHEEL_DOFS), dtype=np.float64)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.2
        TARGET_KE = 250.0
        TARGET_KD = 0.0

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            control_mode="velocity",
            k_p=TARGET_KE,
            k_d=TARGET_KD,
            friction_left_right=0.8,
            friction_rear=0.35,
        )

        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
        scale = np.array([6.0, 6.0, 5.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])
        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        # Add the surface mesh to a separate builder so it gets shape_world=-1
        # (Newton's "global" sentinel). The mesh is then stored once and the
        # broadphase tests it against shapes from every world without duplication.
        globals_builder = newton.ModelBuilder()
        globals_builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.8,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            global_builder=globals_builder,
        )

    @staticmethod
    def _compose_xform(parent_xf: np.ndarray, local_xf: np.ndarray) -> np.ndarray:
        """Compose two 7-element transforms [tx, ty, tz, qx, qy, qz, qw]."""
        p1 = parent_xf[:3]
        qx1, qy1, qz1, qw1 = parent_xf[3:7]
        p2 = local_xf[:3]
        qx2, qy2, qz2, qw2 = local_xf[3:7]
        qxyz1 = np.array([qx1, qy1, qz1], dtype=np.float32)
        rot_p2 = p2 + 2.0 * np.cross(qxyz1, np.cross(qxyz1, p2) + qw1 * p2)
        out_p = p1 + rot_p2
        out_qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        out_qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        out_qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        out_qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        return np.array(
            [out_p[0], out_p[1], out_p[2], out_qx, out_qy, out_qz, out_qw], dtype=np.float32
        )

    def _collect_ghost_shapes(self):
        if hasattr(self, "_ghost_shapes_cache"):
            return self._ghost_shapes_cache
        m = self.model
        shape_body = m.shape_body.numpy()
        shape_transform = m.shape_transform.numpy()
        shape_type = m.shape_type.numpy()
        shape_scale = m.shape_scale.numpy()
        shape_thickness = m.shape_margin.numpy()
        shape_is_solid = m.shape_is_solid.numpy()
        shape_flags = m.shape_flags.numpy()
        shape_source = m.shape_source

        visible_mask = int(newton.ShapeFlags.VISIBLE)
        mesh_types = {
            int(newton.GeoType.MESH),
            int(newton.GeoType.CONVEX_MESH),
            int(newton.GeoType.HFIELD),
        }
        ghosts = []
        for s in range(len(shape_body)):
            if shape_body[s] == -1 or not (shape_flags[s] & visible_mask):
                continue
            gt = int(shape_type[s])
            ghosts.append(
                {
                    "shape_idx": s,
                    "body_idx": int(shape_body[s]),
                    "local_xform": shape_transform[s].astype(np.float32),
                    "geo_type": gt,
                    "geo_scale": tuple(float(v) for v in shape_scale[s]),
                    "geo_thickness": float(shape_thickness[s]),
                    "geo_is_solid": bool(shape_is_solid[s]),
                    "geo_src": shape_source[s] if gt in mesh_types else None,
                }
            )
        self._ghost_shapes_cache = ghosts
        return ghosts

    def _sample_noise(self, sigma: float) -> np.ndarray:
        """Draw [N, K, 3] noise. With antithetic, second half mirrors the first."""
        if self.antithetic:
            half = self.N // 2
            base = np.random.randn(half, self.K, NUM_WHEEL_DOFS).astype(np.float64) * sigma
            return np.concatenate([base, -base], axis=0)
        return np.random.randn(self.N, self.K, NUM_WHEEL_DOFS).astype(np.float64) * sigma

    def _current_sigma(self) -> float:
        progress = min(self.spline_adam.t / self.spline_adam.total_steps, 1.0)
        return self.sigma_min + 0.5 * (self.sigma_init - self.sigma_min) * (
            1.0 + np.cos(np.pi * progress)
        )

    def _expand_per_world(self, params: np.ndarray) -> np.ndarray:
        """Expand [K, 3] + [N, K, 3] noise -> [T, N, 3] per-step per-world wheel velocities."""
        # perturbed[i] = params + noise[i]; then expand each via W: [T, K] @ [K, 3] = [T, 3]
        perturbed = params[None, :, :] + self.noise  # [N, K, 3]
        # einsum: tk, nkd -> tnd
        return np.einsum("tk,nkd->tnd", self.W, perturbed)

    def _apply_params(self, params: np.ndarray):
        """Resample noise, expand per-world, write into joint_target_vel and per-step controls."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        self.noise = self._sample_noise(self._current_sigma())
        expanded = self._expand_per_world(params)  # [T, N, 3]

        vel_np = np.zeros((T, self.N, num_dofs), dtype=np.float32)
        vel_np[:, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded.astype(
            np.float32
        )

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]

        wp.launch(
            kernel=loss_kernel,
            dim=(num_steps, self.N),
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.trajectory_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=yaw_loss_kernel,
            dim=(num_steps, self.N),
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
                self.yaw_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, self.N, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # grad shape: [T, N, num_dofs] — independent gradient per perturbed world
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, :, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]  # [T, N, 3]

        # Per-world spline contraction: [T, N, 3] -> [K, N, 3]
        safe_sums = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        grad_per_world = np.einsum("tk,tnd->knd", self.W, grad_v) / safe_sums[:, None, None]

        # Bundled gradient: average across the N noisy samples
        grad_params = grad_per_world.mean(axis=1)  # [K, 3]

        self.trajectory.joint_target_vel.grad.zero_()
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)
        self._apply_params(self.spline_params)  # also resamples noise for next iter

    def render(self, train_iter):
        if train_iter % 2 != 0:
            return

        # self.loss has been summed across N worlds; divide for per-world comparison
        loss_val = self.loss.numpy()[0] / self.N
        if loss_val >= self.best_loss:
            return
        self.best_loss = loss_val

        # World 0's trajectory (with its share of noise[0]). Approximate but cheap.
        self._iter_body_poses.append(
            self.trajectory.body_pose.numpy()[:, 0].copy().astype(np.float32)
        )
        self._iter_indices.append(int(train_iter))
        self._iter_losses.append(float(loss_val))

        target_poses = self.trajectory.target_body_pose.numpy()
        num_steps = target_poses.shape[0]

        ghost_shapes = self._collect_ghost_shapes()
        ghost_color = wp.array([wp.vec3(1.0, 0.2, 0.0)], dtype=wp.vec3)
        GHOST_OPACITY = 0.3

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            idx = min(step_idx, num_steps - 1)
            target_step = target_poses[idx, 0]
            bodies_per_world = target_step.shape[0]
            for g in ghost_shapes:
                if g["body_idx"] >= bodies_per_world:
                    continue  # shape belongs to a replicated world (N > 1) — skip
                world_xf = self._compose_xform(target_step[g["body_idx"]], g["local_xform"])
                name = f"/target_ghost/shape_{g['shape_idx']}"
                viewer.log_shapes(
                    name,
                    g["geo_type"],
                    g["geo_scale"],
                    wp.array([world_xf], dtype=wp.transform),
                    ghost_color,
                    geo_thickness=g["geo_thickness"],
                    geo_is_solid=g["geo_is_solid"],
                    geo_src=g["geo_src"],
                )
                viewer.set_opacity(name, GHOST_OPACITY)

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=3.0,
        )

        self.frame += 1

    def train(self, iterations=200):
        try:
            self._train_impl(iterations)
        finally:
            self.close()
            if self.export_path:
                self._export_blender_npz(self.export_path)

    def _export_blender_npz(self, path: str):
        m = self.model
        shape_body = m.shape_body.numpy()
        shape_transform = m.shape_transform.numpy()
        shape_type = m.shape_type.numpy()
        shape_scale = m.shape_scale.numpy()
        shape_thickness = m.shape_margin.numpy()
        shape_is_solid = m.shape_is_solid.numpy()
        shape_flags = m.shape_flags.numpy()
        shape_source = m.shape_source

        visible_mask = int(newton.ShapeFlags.VISIBLE)
        mesh_types = {int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)}
        shapes = []
        for s in range(len(shape_body)):
            if not (shape_flags[s] & visible_mask):
                continue
            gt = int(shape_type[s])
            entry = {
                "body_idx": int(shape_body[s]),
                "geo_type": gt,
                "geo_scale": np.array(shape_scale[s], dtype=np.float32),
                "geo_thickness": float(shape_thickness[s]),
                "geo_is_solid": bool(shape_is_solid[s]),
                "local_xform": shape_transform[s].astype(np.float32),
            }
            if gt in mesh_types and shape_source[s] is not None:
                mesh = shape_source[s]
                entry["mesh_verts"] = np.asarray(mesh.vertices, dtype=np.float32)
                entry["mesh_faces"] = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
            shapes.append(entry)

        target_body_pose = self.trajectory.target_body_pose.numpy()[:, 0]
        body_pose_iters = (
            np.stack(self._iter_body_poses, axis=0)
            if self._iter_body_poses
            else np.empty(
                (0, target_body_pose.shape[0], target_body_pose.shape[1], 7), dtype=np.float32
            )
        )
        np.savez_compressed(
            path,
            dt=np.float32(self.clock.dt),
            fps=np.float32(1.0 / self.clock.dt),
            target_body_pose=target_body_pose.astype(np.float32),
            body_pose_iters=body_pose_iters.astype(np.float32),
            iter_indices=np.array(self._iter_indices, dtype=np.int32),
            iter_losses=np.array(self._iter_losses, dtype=np.float32),
            shapes=np.array(shapes, dtype=object),
        )
        print(
            f"Blender export saved to {path} ({len(self._iter_body_poses)} iterations, {len(shapes)} shapes)"
        )

    def _train_impl(self, iterations):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        # --- Target episode (replicated across all N worlds, identical controls) ---
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        T = self.clock.total_sim_steps

        for i in range(T):
            target_ctrl = np.zeros((self.N, num_dofs), dtype=np.float32)
            target_ctrl[:, WHEEL_DOF_OFFSET + 0] = TARGET_WHEEL_VEL[0]
            target_ctrl[:, WHEEL_DOF_OFFSET + 1] = TARGET_WHEEL_VEL[1]
            target_ctrl[:, WHEEL_DOF_OFFSET + 2] = TARGET_WHEEL_VEL[2]

            target_ctrl_wp = wp.array(target_ctrl, dtype=wp.float32, device=self.model.device)
            wp.copy(self.target_controls[i].joint_target_vel, target_ctrl_wp)

        # Constant target spline: every knot at TARGET_WHEEL_VEL — used for recovery diagnostics.
        target_spline = np.tile(np.asarray(TARGET_WHEEL_VEL, dtype=np.float64), (self.K, 1))

        self.run_target_episode()

        # --- Spline setup ---
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        self.spline_params = np.array(
            [[self.init_wheel_vel[0], self.init_wheel_vel[1], self.init_wheel_vel[2]]] * self.K,
            dtype=np.float64,
        )

        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.11, lr_min_ratio=0.2, total_steps=50
        )

        self._apply_params(self.spline_params)  # also samples first noise batch

        # --- Optimization ---
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0] / self.N  # per-world average
            sigma_now = self._current_sigma()
            err = np.abs(self.spline_params - target_spline)  # [K, 3]
            err_max = err.max()
            err_mean = err.mean()
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | err_max={err_max:.3f} err_mean={err_mean:.3f} | "
                f"K={self.K} N={self.N} sigma={sigma_now:.3f} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()

        self._print_final_recovery(target_spline)

    def _print_final_recovery(self, target_spline: np.ndarray):
        """Show every knot vs target, so we don't have to trust cp[0] alone."""
        print("\n=== Final spline recovery (knot vs target) ===")
        print(f"Target (constant):  L={TARGET_WHEEL_VEL[0]:.3f} R={TARGET_WHEEL_VEL[1]:.3f} Rear={TARGET_WHEEL_VEL[2]:.3f}")
        print("knot |   L     R    Rear |  errL   errR  errRear")
        for k in range(self.K):
            p = self.spline_params[k]
            e = p - target_spline[k]
            print(f"  {k:2d} | {p[0]:5.3f} {p[1]:5.3f} {p[2]:5.3f} | {e[0]:+6.3f} {e[1]:+6.3f} {e[2]:+6.3f}")
        err = np.abs(self.spline_params - target_spline)
        print(f"|err|_max = {err.max():.4f}   |err|_mean = {err.mean():.4f}   ||err||_F = {np.linalg.norm(err):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--vis",
        choices=["gl", "headless"],
        default="gl",
        help="Viewer backend (default: gl)",
    )
    output_group.add_argument(
        "--usd",
        type=str,
        default=None,
        metavar="PATH",
        help="Render to a USD file at PATH instead of opening the GL viewer",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump per-iteration trajectory + shape metadata to a .npz for the Blender importer",
    )
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=32,
        help="Number of bundled samples (parallel worlds). Even number required if --antithetic.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="Initial perturbation scale on spline control points (rad/s).",
    )
    parser.add_argument(
        "--sigma-min-ratio",
        type=float,
        default=0.1,
        help="Final sigma as a fraction of initial (cosine annealed).",
    )
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="Disable antithetic sampling (default: enabled, halves variance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for noise sampling.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.usd:
        vis_type = "usd"
    elif args.vis == "headless":
        vis_type = "null"
    else:
        vis_type = "gl"

    sim_config = SimulationConfig(
        duration_seconds=5.0,
        target_timestep_seconds=5e-2,
        num_worlds=args.num_worlds,
    )
    render_config = RenderingConfig(
        vis_type=vis_type,
        target_fps=30,
        usd_file=args.usd,
        world_offset_x=20.0,
        world_offset_y=20.0,
    )
    engine_config = AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=0.001),
        linear=LinearSolverConfig(max_iters=16, atol=0.001, tol=0.001, regularization=1e-06),
        compliance=ComplianceConfig(joint=6e-08, contact=1e-10, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    logging_config = LoggingConfig()

    sim = HelhestTrajectorySplineSurfaceBundledOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=10,
        sigma=args.sigma,
        sigma_min_ratio=args.sigma_min_ratio,
        antithetic=not args.no_antithetic,
    )
    sim.export_path = args.export
    sim.train(iterations=24)


if __name__ == "__main__":
    main()
