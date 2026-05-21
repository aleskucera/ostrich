import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
import warp.optim
from axion import DifferentiableSimulator
from axion import EngineConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from newton import Model
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


def bourke_color_map(v_min, v_max, v):
    """Maps a scalar value to a color gradient (White -> Blue -> Red)."""
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
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    loss: wp.array(dtype=wp.float32),
):
    """
    Calculates loss based on distance to target AND remaining velocity.
    We want the stone to stop AT the target.
    """
    tid = wp.tid()
    if tid > 0:
        return

    # 1. Position Error
    pos = wp.transform_get_translation(body_q[0])
    diff = pos - target_pos
    dist_sq = wp.dot(diff, diff)

    # 2. Velocity Penalty (We want it to stop)
    # Extract linear velocity from spatial vector (indices 3,4,5)
    qd_y = body_qd[0][1]

    # Combine: distance + penalty for still moving
    loss[0] = dist_sq + 1.0 * wp.pow(qd_y, 2.0)


@wp.kernel
def update_kernel(
    qd_grad: wp.array(dtype=wp.spatial_vector),
    alpha: float,
    qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    if tid > 0:
        return

    # Gradient Descent Step
    # We only update the initial velocity to minimize loss
    qd_y = qd[0][1]
    qd_y_grad = qd_grad[0][1]
    qd_y_new = qd_y - alpha * qd_y_grad
    qd[0] = wp.spatial_vector(0.0, qd_y_new, 0.0, 0.0, 0.0, 0.0)


class CurlingOptimizer(DifferentiableSimulator):
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

        # --- Optimization Setup ---
        # Target: 4 meters along X-axis
        self.target_pos = wp.vec3(0.0, 3.0, 0.1)

        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 0.3

        self.frame = 0

        # Initial velocity guessing
        self.init_vel = wp.spatial_vector(0.0, 0.5, 0.0, 0.0, 0.0, 0.0)

        # --- Tracking Setup ---
        # Auto-track the stone (Body 0)
        self.track_body(body_idx=0, name="stone", color=(0.0, 0.5, 1.0))

        # Compile graph
        # self.capture()

    def build_model(self) -> Model:
        # Use moderate friction (mu=0.1) to allow sliding
        shape_config = newton.ModelBuilder.ShapeConfig(
            ke=1e5,
            kd=1e2,
            kf=1e3,
            mu=1.0,
            contact_margin=0.3,
        )

        # 1. The Stone (Box)
        self.builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
            mass=5.0,  # Heavy stone
        )
        # self.builder.add_shape_box(body=0, hx=0.1, hy=0.1, hz=0.1, cfg=shape_config)
        self.builder.add_shape_cylinder(body=0, radius=0.3, half_height=0.1, cfg=shape_config)

        # 2. The Ice/Floor
        self.builder.add_ground_plane(cfg=shape_config)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=True,
        )

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.states[-1].body_q,  # Final Position
                self.states[-1].body_qd,  # Final Velocity
                self.target_pos,
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
                self.states[0].body_qd.grad,
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_qd,
            ],
        )

    def render(self, train_iter):
        # Render every 5 iterations to see progress
        if self.frame > 0 and train_iter % 5 != 0:
            return

        loss_val = self.loss.numpy()[0]

        # Define callback for drawing the target
        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

            # Draw Target Marker (Red Cylinder)
            viewer.log_shapes(
                "/target",
                newton.GeoType.CYLINDER,
                (0.3, 0.01, 0.3),  # radius, null, half_height
                wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),  # Red
            )

        print(f"Rendering iteration {train_iter} (Loss: {loss_val:.4f})...")

        # Playback Loop
        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,
            loops_count=1,
            playback_speed=1.0,  # Real-time speed
        )

        self.frame += 1

    def train(self, iterations=50):
        # Set initial velocity state
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        for i in range(iterations):
            # 1. Run Simulation (Forward + Backward)
            self.diff_step()

            # 2. Visualize
            self.render(i)

            # 3. Print Progress
            curr_loss = self.loss.numpy()[0]
            vel = self.states[0].body_qd.numpy()[0][0:3]
            print(
                f"Iter {i}: Loss={curr_loss:.4f} | Init Vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"
            )

            # 4. Update
            self.update()

            # 5. Clear Gradients
            self.tape.zero()
            self.loss.zero_()


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = CurlingOptimizer(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )
    sim.train(iterations=30)


if __name__ == "__main__":
    main()
