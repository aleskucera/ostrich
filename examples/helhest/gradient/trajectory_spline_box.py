import argparse
import os
import pathlib
import time

import newton
import numpy as np
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

os.environ["PYOPENGL_PLATFORM"] = "glx"

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3

# Static box obstacle the robot must drive over.
BOX_CENTER = (1.5, 0.0, 0.05)
BOX_HALF_EXTENTS = (0.3, 1.5, 0.05)  # 0.6 m long (X), 3.0 m wide (Y), 0.10 m tall


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Build [T, K] linear interpolation weight matrix and per-column normalization.

    W[t, k] is the weight of control point k at timestep t.
    Expanding:   v = W @ params                          ([T,3] = [T,K] @ [K,3])
    Contracting: grad_params = (W.T @ grad_v) / col_sums  (average, not sum)
    """
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        k_float = t * (K - 1) / max(T - 1, 1)
        k_low = int(k_float)
        k_high = min(k_low + 1, K - 1)
        alpha = k_float - k_low
        W[t, k_low] += 1.0 - alpha
        W[t, k_high] += alpha
    col_sums = W.sum(axis=0)  # [K] — total weight each control point receives
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


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 distance between chassis position summed over all timesteps."""
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
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
    """Yaw heading error via forward vector dot product, summed over all timesteps."""
    t = wp.tid()
    q = wp.transform_get_rotation(body_pose[t, 0, 0])
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
    """L2 magnitude regularization:  weight * Σ_t ||v_t||² over wheel DOFs."""
    sim_step, wheel_idx = wp.tid()

    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


class HelhestTrajectorySplineBoxOptimizer(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 10,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.K = num_control_points

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
        self.init_wheel_vel = (2.0, 2.0, 2.0)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.2

        TARGET_KE = 250.0
        TARGET_KD = 0.0

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=TARGET_KE,
            k_d=TARGET_KD,
            friction_left_right=0.8,
            friction_rear=0.35,
        )

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.8,
            ke=150.0,
            kd=150.0,
            kf=500.0,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*BOX_CENTER), wp.quat_identity()),
            hx=BOX_HALF_EXTENTS[0],
            hy=BOX_HALF_EXTENTS[1],
            hz=BOX_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=0.8,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
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
        """Per-shape data needed to mirror the live robot at the target pose."""
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

    def _expand(self, params: np.ndarray) -> np.ndarray:
        """Expand [K, 3] control points → [T, 3] per-step wheel velocities."""
        return self.W @ params  # [T, K] @ [K, 3] = [T, 3]

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        """Contract [T, 3] velocity gradients → [K, 3] control point gradients (normalized average)."""
        safe_sums = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        """Write expanded spline params into joint_target_vel and per-step controls."""
        T = self.clock.total_sim_steps
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        expanded = self._expand(params)  # [T, 3]

        vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
        vel_np[:, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]

        wp.launch(
            kernel=loss_kernel,
            dim=num_steps,
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
            dim=num_steps,
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
            dim=(self.clock.total_sim_steps, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight / num_steps,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        # Contract: [T, 3] gradients → [K, 3] control point gradients
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[
            :, 0, WHEEL_DOF_OFFSET : WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS
        ]  # [T, 3]
        grad_params = self._contract(grad_v)  # [K, 3]

        self.trajectory.joint_target_vel.grad.zero_()

        # Adam step on control points
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)

        # Expand and sync back
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if train_iter % 2 != 0:
            return

        loss_val = self.loss.numpy()[0]
        if loss_val >= self.best_loss:
            return
        self.best_loss = loss_val

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
            target_step = target_poses[idx, 0]  # [num_bodies, 7]
            for g in ghost_shapes:
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
        """Snapshot model + accumulated trajectories to a single npz for Blender."""
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

        # --- Target episode ---
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        T = self.clock.total_sim_steps

        # Slightly asymmetric drive: forward + gentle left curve, so the
        # optimizer must learn both forward speed and steering — and the
        # trajectory still passes over the box.
        for i in range(T):
            target_ctrl = np.zeros(num_dofs, dtype=np.float32)
            target_ctrl[WHEEL_DOF_OFFSET + 0] = 3.0  # left wheel
            target_ctrl[WHEEL_DOF_OFFSET + 1] = 2.0  # right wheel
            target_ctrl[WHEEL_DOF_OFFSET + 2] = 2.5  # rear wheel

            target_ctrl_wp = wp.array(target_ctrl, dtype=wp.float32, device=self.model.device)
            wp.copy(self.target_controls[i].joint_target_vel, target_ctrl_wp)

        self.run_target_episode()

        # --- Spline setup ---
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)  # [T, K], [K]

        self.spline_params = np.array(
            [[self.init_wheel_vel[0], self.init_wheel_vel[1], self.init_wheel_vel[2]]] * self.K,
            dtype=np.float64,
        )  # [K, 3]

        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=NUM_WHEEL_DOFS, lr=0.1, lr_min_ratio=0.2, total_steps=50
        )

        self._apply_params(self.spline_params)

        # --- Optimization ---
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            p0 = self.spline_params[0]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | cp[0] L={p0[0]:.3f} R={p0[1]:.3f} | K={self.K} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


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
    args = parser.parse_args()

    if args.usd:
        vis_type = "usd"
    elif args.vis == "headless":
        vis_type = "null"
    else:
        vis_type = "gl"

    sim_config = SimulationConfig(
        duration_seconds=4.0,
        target_timestep_seconds=5e-2,
        num_worlds=1,
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
        compliance=ComplianceConfig(joint=6e-08, contact=1e-06, friction=1e-06),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    logging_config = LoggingConfig()

    sim = HelhestTrajectorySplineBoxOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=10,
    )
    sim.export_path = args.export
    sim.train(iterations=30)


if __name__ == "__main__":
    main()
