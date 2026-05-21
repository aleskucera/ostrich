"""
Cart Pole Swing-Up via Spline Trajectory Optimization.

Goal: optimize a spline of target velocities for the cart (actuated)
so that it pumps energy into the passive pole (starting from pointing straight down),
swings it up, and actively balances it at the top without spinning or driving away.
"""
import os
import pathlib
import time

import hydra
import newton
import numpy as np
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.types import JointMode
from newton import Model
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


def bourke_color_map(v_min, v_max, v):
    c = wp.vec3(1.0, 1.0, 1.0)
    v = np.clip(v, v_min, v_max)
    dv = v_max - v_min

    if v < (v_min + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - v_min) / dv
    elif v < (v_min + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (v_min + 0.25 * dv - v) / dv
    elif v < (v_min + 0.75 * dv):
        c[0] = 4.0 * (v - v_min - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (v_min + 0.75 * dv - v) / dv
        c[2] = 0.0
    return c


# --- SPLINE OPTIMIZER UTILITIES ---


def make_interp_matrix(T: int, K: int) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(
        self,
        K: int,
        num_dofs: int,
        lr: float,
        total_steps: int = 200,
        lr_min_ratio: float = 0.05,
        betas=(0.9, 0.999),
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


# --- LOSS KERNELS ---


@wp.kernel
def cartpole_swingup_loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=3),
    num_steps: int,
    weight_pole_pos: float,
    weight_pole_vel: float,
    weight_cart_pos: float,
    weight_cart_vel: float,
    loss: wp.array(dtype=wp.float32),
):
    t = wp.tid()

    # TIME WEIGHTING: Linearly increases from 0.0 to 1.0.
    # Gives the cart a "grace period" to swing up early on,
    # but demands strict balancing at the end of the trajectory.
    time_weight = float(t) / float(num_steps)

    # Body 0 = Cart, Body 1 = Pole
    cart_pos = wp.transform_get_translation(body_pose[t, 0, 0])
    pole_pose = body_pose[t, 0, 1]

    cart_vel = body_vel[t, 0, 0]
    pole_vel = body_vel[t, 0, 1]

    # 1. Cart Bounds Penalty (keep it near Y=0)
    cart_y = cart_pos[1]
    wp.atomic_add(loss, 0, weight_cart_pos * cart_y * cart_y * time_weight)

    # 2. Cart Velocity Penalty (Prevents jittering/driving crazy)
    cart_v = wp.spatial_top(cart_vel)
    wp.atomic_add(loss, 0, weight_cart_vel * wp.dot(cart_v, cart_v) * time_weight)

    # 3. Pole Upright Penalty (align local Z with world Z)
    pole_rot = wp.transform_get_rotation(pole_pose)
    pole_up_vector = wp.quat_rotate(pole_rot, wp.vec3(0.0, 0.0, 1.0))
    delta = pole_up_vector - wp.vec3(0.0, 0.0, 1.0)
    wp.atomic_add(loss, 0, weight_pole_pos * wp.dot(delta, delta) * time_weight)

    # 4. Pole Angular Velocity Penalty (Stops the flipping!)
    pole_w = wp.spatial_bottom(pole_vel)
    wp.atomic_add(loss, 0, weight_pole_vel * wp.dot(pole_w, pole_w) * time_weight)


@wp.kernel
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """Penalize high control target velocities overall."""
    t = wp.tid()
    v = target_vel[t, 0, 0]  # Only penalize the cart's velocity
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def init_cartpole_state_kernel(
    joint_q: wp.array(dtype=wp.float32),
    cart_init_pos: float,
    pole_init_angle: float,
):
    world_idx = wp.tid()

    # Each world has 2 DOFs (Cart=0, Pole=1)
    cart_idx = world_idx * 2 + 0
    pole_idx = world_idx * 2 + 1

    joint_q[cart_idx] = cart_init_pos
    joint_q[pole_idx] = pole_init_angle


class CartPoleSwingUp(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        num_control_points: int = 50,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Spline parameters
        self.K = num_control_points
        self.actuated_dofs = 1  # We only actuate the cart

        # --- LOSS WEIGHTS ---
        self.weight_pole_pos = 50.0  # High priority: get to the top
        self.weight_pole_vel = 1.0  # Medium priority: stop spinning!
        self.weight_cart_pos = 0.5  # Low priority: stay near origin
        self.weight_cart_vel = 0.1  # Low priority: don't drive too fast
        self.regularization_weight = 1e-4

        self.frame = 0
        self.track_body(body_idx=0, name="cart", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        # Create a shape config that disables collision so the Cart and Pole don't explode
        no_collision_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

        # --- 1. The Cart ---
        link_cart = self.builder.add_link()
        self.builder.add_shape_box(link_cart, hx=0.3, hy=0.5, hz=0.2, cfg=no_collision_cfg)

        # We want the cart to slide along the World Y-axis.
        # We hardcode the joint axis to X (1,0,0) so Axion/Newton agree perfectly.
        # Then we rotate the joint frame 90 degrees around Z, pointing local X down World Y.
        rot_z_90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)

        j_cart = self.builder.add_joint_prismatic(
            parent=-1,
            child=link_cart,
            axis=wp.vec3(1.0, 0.0, 0.0),  # <--- HARDCODED TO X
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=rot_z_90),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=rot_z_90),
            target_ke=1000.0,
            target_kd=100.0,
            label="cart_joint",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
        )

        # --- 2. The Pole ---
        link_pole = self.builder.add_link()
        self.builder.add_shape_box(link_pole, hx=0.1, hy=0.1, hz=1.0, cfg=no_collision_cfg)

        # We want the pole to swing in the Y-Z plane (rotating around World X).
        # The local X-axis is already aligned with World X, so no rotation is needed!
        j_pole = self.builder.add_joint_revolute(
            parent=link_cart,
            child=link_pole,
            axis=wp.vec3(1.0, 0.0, 0.0),  # <--- HARDCODED TO X
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.2), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.0), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.0,
            label="pole_joint",
        )

        self.builder.add_articulation([j_cart, j_pole], label="cartpole")

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    # --- SPLINE MAPPING ---

    def _expand(self, params: np.ndarray) -> np.ndarray:
        return self.W @ params  # [T, K] @ [K, 1] = [T, 1]

    def _contract(self, grad_v: np.ndarray) -> np.ndarray:
        safe_sums = np.where(self.W_col_sums > 0, self.W_col_sums, 1.0)
        return (self.W.T @ grad_v) / safe_sums[:, None]

    def _apply_params(self, params: np.ndarray):
        T = self.clock.total_sim_steps
        expanded = self._expand(params)  # [T, 1]

        # The system has 2 DOFs (Cart, Pole)
        vel_np = np.zeros((T, 1, 2), dtype=np.float32)

        # Map the spline specifically to the Cart (DOF 0)
        vel_np[:, 0, 0] = expanded[:, 0]
        # Pole (DOF 1) remains 0.0 implicitly

        wp.copy(self.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
        for i in range(T):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def compute_loss(self):
        num_steps = self.clock.total_sim_steps

        wp.launch(
            kernel=cartpole_swingup_loss_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.body_vel,
                num_steps,
                self.weight_pole_pos / num_steps,
                self.weight_pole_vel / num_steps,
                self.weight_cart_pos / num_steps,
                self.weight_cart_vel / num_steps,
                self.loss,
            ],
            device=self.solver.model.device,
        )

        wp.launch(
            kernel=regularization_kernel,
            dim=num_steps,
            inputs=[
                self.trajectory.joint_target_vel,
                self.regularization_weight / num_steps,
                self.loss,
            ],
            device=self.solver.model.device,
        )

    def update(self):
        # 1. Get raw gradients from the trajectory buffer (only for the cart!)
        grad_v = self.trajectory.joint_target_vel.grad.numpy()[:, 0, :1]  # [T, 1]

        # 2. Map gradients down to the K spline control points
        grad_params = self._contract(grad_v)  # [K, 1]
        self.trajectory.joint_target_vel.grad.zero_()

        # 3. Adam step on the control points
        self.spline_params = self.spline_adam.step(self.spline_params, grad_params)

        # 4. Map the updated control points back to the full simulation trajectory
        self._apply_params(self.spline_params)

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 20 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 20.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,
        )

        self.frame += 1

    def train(self, iterations=400):
        # --- THE HARD TASK: SWING UP FROM THE BOTTOM ---
        # Set cart to origin, pole hanging straight down (pi radians)
        wp.launch(
            kernel=init_cartpole_state_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[
                self.model.joint_q,
                0.0,  # Cart position
                wp.pi,  # Pole angle (hanging down)
            ],
            device=self.model.device,
        )

        # Re-evaluate forward kinematics to sync the bodies with the new joints
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        T = self.clock.total_sim_steps

        # Initialize Spline Matrices
        self.W, self.W_col_sums = make_interp_matrix(T, self.K)

        # Initialize parameters with zeros (do nothing initially)
        self.spline_params = np.zeros((self.K, self.actuated_dofs), dtype=np.float64)
        self.spline_adam = SplineAdam(
            K=self.K, num_dofs=self.actuated_dofs, lr=1.0, total_steps=iterations
        )

        self._apply_params(self.spline_params)

        # Optimization loop
        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]

            p_start = self.spline_params[0, 0]
            p_mid = self.spline_params[self.K // 2, 0]
            p_end = self.spline_params[-1, 0]

            print(
                f"Iter {i:3d}: Loss={curr_loss:.4f} | Spline(start={p_start:.2f}, mid={p_mid:.2f}, end={p_end:.2f}) | eps={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = CartPoleSwingUp(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        num_control_points=100,  # 50 points gives it high flexibility to swing
    )
    sim.train(iterations=1000)


if __name__ == "__main__":
    main()
