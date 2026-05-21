"""
Pendulum swing-up via trajectory optimization.

Goal: optimize per-timestep target velocities so the pendulum trajectory
matches a reference (target) trajectory. The target trajectory is generated
by running the simulation with a known target velocity that swings the
pendulum upward. The optimizer starts from a different initial velocity and
uses implicit gradients to recover the controls that reproduce the target
motion.

Scene: single pendulum, pivot fixed at (0, 0, 5), link swings in the XZ plane
       (revolute joint around world Y axis).
Optimization parameters: target velocity at each timestep.
Loss: L2 trajectory matching (position error summed across all timesteps).
"""
import os
import pathlib

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

HX = 1.0
HY = 0.2
HZ = 0.2


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


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    loss: wp.array(dtype=wp.float32),
):
    """L2 trajectory loss: sum of squared position errors across all timesteps.
    Launched with dim=total_sim_steps+1 to include the final state (index 0 is
    the shared initial state and contributes zero)."""
    tid = wp.tid()

    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])

    delta = pos - target_pos
    wp.atomic_add(loss, 0, wp.dot(delta, delta))


@wp.kernel
def sgd_update_kernel(
    grad: wp.array(dtype=wp.float32, ndim=3),
    lr: float,
    param: wp.array(dtype=wp.float32, ndim=3),
):
    sim_step, world_idx, dof_idx = wp.tid()

    g = grad[sim_step, world_idx, dof_idx]
    max_grad = 100.0
    g = wp.clamp(g, -max_grad, max_grad)

    param[sim_step, world_idx, dof_idx] = param[sim_step, world_idx, dof_idx] - lr * g


class PendulumSwingUp(AxionDifferentiableSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 1e-1

        self.frame = 0

        # The target trajectory is generated with this velocity.
        # The optimizer starts from a different value and must recover it.
        self.true_target_vel = 3.0
        self.init_target_vel = 1.0

        self.track_body(body_idx=0, name="pendulum", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        link_0 = self.builder.add_link()
        self.builder.add_shape_box(link_0, hx=HX, hy=HY, hz=HZ)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-HX, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=1000.0,
            target_kd=1000.0,
            label="pendulum_joint",
            custom_attributes={
                "joint_dof_mode": [JointMode.TARGET_VELOCITY],
            },
        )

        self.builder.add_articulation([j0], label="pendulum")

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self):
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps + 1,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=sgd_update_kernel,
            dim=(self.clock.total_sim_steps, self.simulation_config.num_worlds, 1),
            inputs=[
                self.trajectory.joint_target_vel.grad,
                self.learning_rate,
                self.trajectory.joint_target_vel,
            ],
            device=self.solver.model.device,
        )
        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 5 != 0:
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

    def _run_forward_loss(self):
        """Run forward pass and compute loss, return scalar loss value."""
        for i in range(self.clock.total_sim_steps):
            self.collision_pipeline.collide(self.states[i], self.contacts)
            self.solver.step(
                state_in=self.states[i],
                state_out=self.states[i + 1],
                control=self.controls[i],
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            self.trajectory.save_step(i, self.solver.data, self.solver.axion_contacts)

        self.loss.zero_()
        self.tape.zero()
        with self.tape:
            self.compute_loss()
        return self.loss.numpy()[0]

    def verify_gradients(self, steps_to_check, eps=1e-3):
        """Finite-difference gradient check for joint_target_vel."""
        # Set up initial state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        self.trajectory.joint_target_vel.fill_(self.init_target_vel)
        for i in range(self.clock.total_sim_steps):
            self.target_controls[i].joint_target_vel.fill_(self.true_target_vel)
            self.controls[i].joint_target_vel.fill_(self.init_target_vel)

        self.run_target_episode()

        # Run one full diff_step to get analytical gradients
        self.diff_step()
        analytical_grads = self.trajectory.joint_target_vel.grad.numpy()[:, 0, 0].copy()
        base_loss = self.loss.numpy()[0]

        print(f"Base loss: {base_loss:.6f}")
        print(f"{'Step':>6} | {'Analytical':>12} | {'Finite Diff':>12} | {'Ratio':>8}")
        print("-" * 50)

        for step_idx in steps_to_check:
            # Save original value
            orig_val = self.controls[step_idx].joint_target_vel.numpy().flat[0]

            # Perturb +eps
            self.controls[step_idx].joint_target_vel.fill_(orig_val + eps)
            loss_plus = self._run_forward_loss()

            # Perturb -eps
            self.controls[step_idx].joint_target_vel.fill_(orig_val - eps)
            loss_minus = self._run_forward_loss()

            print(
                f"DEBUG: Step {step_idx} loss_plus: {loss_plus:.6f}, loss_minus: {loss_minus:.6f}"
            )

            # Restore
            self.controls[step_idx].joint_target_vel.fill_(orig_val)

            fd_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ag = analytical_grads[step_idx]
            ratio = ag / fd_grad if abs(fd_grad) > 1e-10 else float("nan")

            print(f"{step_idx:>6} | {ag:>12.6f} | {fd_grad:>12.6f} | {ratio:>8.4f}")

        self.tape.zero()
        self.loss.zero_()

    def train(self, iterations=200):
        # Set up initial state via forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        # Fill the trajectory buffer with initial guess
        self.trajectory.joint_target_vel.fill_(self.init_target_vel)

        # Set up controls: target uses true velocity, optimizer starts from init
        for i in range(self.clock.total_sim_steps):
            self.target_controls[i].joint_target_vel.fill_(self.true_target_vel)
            self.controls[i].joint_target_vel.fill_(self.init_target_vel)

        # Generate the target trajectory (the "ground truth" we want to match)
        self.run_target_episode()

        # Optimization loop
        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            all_vels = self.trajectory.joint_target_vel.numpy()[:, 0, 0]
            all_grads = self.trajectory.joint_target_vel.grad.numpy()[:, 0, 0]
            print(
                f"Iter {i:3d}: Loss={curr_loss:.6f} | "
                f"vel[0]={all_vels[0]:.4f} vel[mid]={all_vels[len(all_vels)//2]:.4f} "
                f"vel[-1]={all_vels[-1]:.4f} | "
                f"grad[0]={all_grads[0]:.6f} grad[-1]={all_grads[-1]:.6f} "
                f"(true={self.true_target_vel})"
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

    sim = PendulumSwingUp(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    # sim.verify_gradients(steps_to_check=[0, 10, 25, 49], eps=1e-3)
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
