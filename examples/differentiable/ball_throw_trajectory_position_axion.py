import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
import warp.optim
from axion import AxionDifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
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


@wp.kernel
def loss_kernel(
    body_pose: wp.array(dtype=wp.transform, ndim=3),
    target_body_pose: wp.array(dtype=wp.transform, ndim=3),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    pos = wp.transform_get_translation(body_pose[tid, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[tid, 0, 0])

    delta = pos - target_pos
    l2_loss = wp.dot(delta, delta)
    wp.atomic_add(loss, 0, l2_loss)


@wp.kernel
def update_kernel(
    body_pose_grad: wp.array(dtype=wp.transform, ndim=2),
    alpha: float,
    body_q: wp.array(dtype=wp.transform, ndim=1),
):
    tid = wp.tid()
    if tid > 0:
        return

    grad_pos = wp.transform_get_translation(body_pose_grad[0, 0])

    max_grad = 5.0
    grad_pos_clamped = wp.vec3(
        wp.clamp(grad_pos[0], -max_grad, max_grad),
        wp.clamp(grad_pos[1], -max_grad, max_grad),
        wp.clamp(grad_pos[2], -max_grad, max_grad),
    )

    wp.printf("Gradient (pos): [%f %f %f]\n", grad_pos[0], grad_pos[1], grad_pos[2])

    current = body_q[0]
    new_pos = wp.transform_get_translation(current) - alpha * grad_pos_clamped
    body_q[0] = wp.transform(new_pos, wp.transform_get_rotation(current))


class BallThrowPositionOptimizer(AxionDifferentiableSimulator):
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
        self.learning_rate = 5e-2

        self.frame = 0

        # Both runs use the same fixed velocity
        self.fixed_vel = wp.spatial_vector(0.0, 4.0, 7.0, 0.0, 0.0, 0.0)

        # Initial position guess and target position to recover
        self.init_pos = wp.vec3(0.0, 0.0, 1.0)
        self.target_init_pos = wp.vec3(2.0, 0.0, 3.0)

        self.track_body(body_idx=0, name="ball", color=(0.0, 1.0, 0.0))

    def build_model(self) -> Model:
        self.builder.rigid_gap = 1.0

        ball = self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        self.builder.add_shape_sphere(body=ball, radius=0.2)

        self.builder.add_ground_plane()
        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=self.clock.total_sim_steps,
            inputs=[
                self.trajectory.body_pose,
                self.trajectory.target_body_pose,
            ],
            outputs=[
                self.loss,
            ],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_pose.grad[0],
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_q,
            ],
        )

    def render(self, train_iter):
        if self.frame > 0 and train_iter % 3 != 0:
            return

        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 10.0, loss_val)
        self._tracked_bodies[0]["color"] = tuple(color)

        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)
            viewer.log_shapes(
                "/target",
                newton.GeoType.SPHERE,
                0.18,
                wp.array(self.trajectory.target_body_pose.numpy()[-1, 0], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=2.0,
        )

        self.frame += 1

    def train(self, iterations=50):
        vel_array = wp.array([self.fixed_vel], dtype=wp.spatial_vector)

        # Run target episode with target initial position
        self.target_states[0].body_q.assign(
            wp.array(
                [wp.transform(self.target_init_pos, wp.quat_identity())],
                dtype=wp.transform,
            )
        )
        wp.copy(self.target_states[0].body_qd, vel_array)
        self.run_target_episode()

        # Set initial guess position and the same fixed velocity
        self.states[0].body_q.assign(
            wp.array(
                [wp.transform(self.init_pos, wp.quat_identity())],
                dtype=wp.transform,
            )
        )
        self.states[0].body_q.requires_grad = True
        wp.copy(self.states[0].body_qd, vel_array)

        for i in range(iterations):
            self.diff_step()

            curr_loss = self.loss.numpy()[0]
            pos = wp.transform_get_translation(
                wp.transform(*self.states[0].body_q.numpy()[0])
            )
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | Init Pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
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

    sim = BallThrowPositionOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=50)


if __name__ == "__main__":
    main()
