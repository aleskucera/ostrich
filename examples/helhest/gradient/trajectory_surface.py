import os
import pathlib
import time

import hydra
import newton
import numpy as np
import openmesh
import warp as wp
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.optim.trajectory_adam import TrajectoryAdam
from newton import Model
from omegaconf import DictConfig

from examples.helhest.common import create_helhest_model
from examples.helhest.common import HelhestConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.parent.parent.joinpath("assets")

# DOF layout: [0..5] = free base joint, [6] = left wheel, [7] = right wheel, [8] = rear wheel
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3


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
def regularization_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 magnitude regularization: weight * Σ_t ||v_t||² over wheel DOFs."""
    sim_step, wheel_idx = wp.tid()

    dof_idx = wheel_dof_offset + wheel_idx
    v = target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * v * v)


@wp.kernel
def smoothness_kernel(
    target_vel: wp.array(dtype=wp.float32, ndim=3),
    wheel_dof_offset: int,
    weight: float,
    loss: wp.array(dtype=wp.float32),
):
    """L2 finite-difference penalty: weight * Σ_t ||v_{t+1} - v_t||² over wheel DOFs."""
    sim_step, wheel_idx = wp.tid()

    dof_idx = wheel_dof_offset + wheel_idx
    diff = target_vel[sim_step + 1, 0, dof_idx] - target_vel[sim_step, 0, dof_idx]
    wp.atomic_add(loss, 0, weight * diff * diff)


class HelhestTrajectorySurfaceOptimizer(AxionDifferentiableSimulator):
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
        self.trajectory_weight = 10.0
        self.smoothness_weight = 1e-2
        self.regularization_weight = 1e-7

        self.frame = 0

        # Initial guess (Left, Right, Rear)
        self.init_wheel_vel = (2.0, 3.0, 0.0)

        self.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 0.1

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.4), wp.quat_identity()),
            control_mode="velocity",
            k_p=HelhestConfig.TARGET_KE,
            k_d=HelhestConfig.TARGET_KD,
            friction_left_right=0.7,
            friction_rear=0.35,
        )

        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
        scale = np.array([6.0, 6.0, 3.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])
        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=1.0,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

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
            kernel=regularization_kernel,
            dim=(self.clock.total_sim_steps, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.regularization_weight,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )
        wp.launch(
            kernel=smoothness_kernel,
            dim=(self.clock.total_sim_steps - 1, NUM_WHEEL_DOFS),
            inputs=[
                self.trajectory.joint_target_vel,
                WHEEL_DOF_OFFSET,
                self.smoothness_weight,
            ],
            outputs=[self.loss],
            device=self.solver.model.device,
        )

    def update(self):
        self.optimizer.step(self.trajectory.joint_target_vel.grad)
        self.trajectory.joint_target_vel.grad.zero_()

        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, self.trajectory.joint_target_vel[i])

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]

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

        self.frame += 1

    def _make_wheel_vel_array(self, vels: tuple) -> np.ndarray:
        num_dofs = self.trajectory.joint_target_vel.shape[-1]
        arr = np.zeros((self.clock.total_sim_steps, 1, num_dofs), dtype=np.float32)
        arr[:, :, WHEEL_DOF_OFFSET + 0] = vels[0]  # left
        arr[:, :, WHEEL_DOF_OFFSET + 1] = vels[1]  # right
        arr[:, :, WHEEL_DOF_OFFSET + 2] = vels[2]  # rear
        return arr

    def train(self, iterations=200):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.target_states[0])

        # --- Target episode ---
        num_dofs = self.trajectory.joint_target_vel.shape[-1]

        for i in range(self.clock.total_sim_steps):
            target_ctrl = np.zeros(num_dofs, dtype=np.float32)
            target_ctrl[WHEEL_DOF_OFFSET + 0] = 6.0
            target_ctrl[WHEEL_DOF_OFFSET + 1] = 1.0
            target_ctrl[WHEEL_DOF_OFFSET + 2] = 2.0

            target_ctrl_wp = wp.array(target_ctrl, dtype=wp.float32, device=self.model.device)
            wp.copy(self.target_controls[i].joint_target_vel, target_ctrl_wp)

        self.run_target_episode()

        print("Rendering target episode...")
        self.states, self.target_states = self.target_states, self.states
        self.render_episode(iteration=-1, loop=True, loops_count=2, playback_speed=1.0)
        self.states, self.target_states = self.target_states, self.states

        # --- Optimization ---
        init_arr = self._make_wheel_vel_array(self.init_wheel_vel)
        wp.copy(self.trajectory.joint_target_vel, wp.array(init_arr, dtype=wp.float32))

        init_ctrl = np.zeros(num_dofs, dtype=np.float32)
        init_ctrl[WHEEL_DOF_OFFSET + 0] = self.init_wheel_vel[0]
        init_ctrl[WHEEL_DOF_OFFSET + 1] = self.init_wheel_vel[1]
        init_ctrl[WHEEL_DOF_OFFSET + 2] = self.init_wheel_vel[2]
        init_ctrl_wp = wp.array(init_ctrl, dtype=wp.float32, device=self.model.device)

        self.optimizer = TrajectoryAdam(
            params=self.trajectory.joint_target_vel,
            lr=5.0,
            betas=(0.9, 0.999),
            dof_offset=WHEEL_DOF_OFFSET,
            num_dofs=NUM_WHEEL_DOFS,
        )

        for i in range(self.clock.total_sim_steps):
            wp.copy(self.controls[i].joint_target_vel, init_ctrl_wp)

        for i in range(iterations):
            t0 = time.perf_counter()
            self.diff_step()
            wp.synchronize()
            t_episode = time.perf_counter() - t0

            curr_loss = self.loss.numpy()[0]
            vels = self.trajectory.joint_target_vel.numpy()[0, 0]
            vel_l = vels[WHEEL_DOF_OFFSET + 0]
            vel_r = vels[WHEEL_DOF_OFFSET + 1]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | L={vel_l:.3f} R={vel_r:.3f} | episode={t_episode:.3f}s"
            )

            self.render(i)
            self.update()

            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="helhest_diff")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = HelhestTrajectorySurfaceOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=200)


if __name__ == "__main__":
    main()
