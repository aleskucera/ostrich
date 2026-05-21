"""
Interactive Cart Pole environment.

Goal: Manually swing up and balance the cart pole.
"""
import os
import pathlib

import hydra
import newton
import numpy as np
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.types import JointMode
from newton import Model
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"
CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class CartPoleInteractive(InteractiveSimulator):
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
        # self.track_body(body_idx=0, name="cart", color=(0.0, 0.5, 1.0))

        # User input velocity accumulator
        self.target_vel = 0.0
        self.acceleration_step = 2.0

    def build_model(self) -> Model:
        # Disable collisions so the Cart and Pole don't explode at the hinge
        no_collision_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)

        # --- 1. The Cart ---
        link_cart = self.builder.add_link()
        self.builder.add_shape_box(link_cart, hx=0.3, hy=0.5, hz=0.2, cfg=no_collision_cfg)

        # Y-axis sliding cart
        rot_z_90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        j_cart = self.builder.add_joint_prismatic(
            parent=-1,
            child=link_cart,
            axis=wp.vec3(1.0, 0.0, 0.0),  # Hardcoded X (Axion/Newton sync)
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=rot_z_90),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=rot_z_90),
            target_ke=1000.0,
            target_kd=0.0,
            label="cart_joint",
            custom_attributes={"joint_dof_mode": [JointMode.TARGET_VELOCITY]},
        )

        # --- 2. The Pole ---
        link_pole = self.builder.add_link()
        self.builder.add_shape_box(link_pole, hx=0.05, hy=0.05, hz=1.0, cfg=no_collision_cfg)

        # X-axis swinging pole
        j_pole = self.builder.add_joint_revolute(
            parent=link_cart,
            child=link_pole,
            axis=wp.vec3(1.0, 0.0, 0.0),  # Hardcoded X
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.2), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.0), q=wp.quat_identity()),
            target_ke=0.0,
            target_kd=0.0,
            label="pole_joint",
            custom_attributes={"joint_dof_mode": [JointMode.NONE]},
        )

        self.builder.add_articulation([j_cart, j_pole], label="cartpole")

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            requires_grad=False,  # CRITICAL: Turn off autograd for interactive speed!
        )

    def on_start(self):
        """Called once before the interactive loop starts."""
        # Initialize the pole hanging straight down
        joint_q = self.model.joint_q.numpy()
        for w in range(self.simulation_config.num_worlds):
            joint_q[w * 2 + 0] = 0.0  # Cart position
            joint_q[w * 2 + 1] = wp.pi  # Pole angle (hanging down)

        self.model.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.model.device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

    def before_step(self):
        """
        Called every single frame right before the solver evaluates.
        This is where we inject our manual controls!
        """
        # If your specific viewer (e.g., PyVista/OpenGL) exposes keyboard events,
        # you would capture them here. For example:
        # if self.viewer.key_pressed("left"): self.target_vel -= 0.5
        # if self.viewer.key_pressed("right"): self.target_vel += 0.5

        # Apply the target velocity to the Cart DOF (index 0)
        vel_np = self.solver.control.joint_target_vel.numpy()
        for w in range(self.simulation_config.num_worlds):
            vel_np[w, 0] = self.target_vel

        self.solver.control.joint_target_vel = wp.array(
            vel_np, dtype=wp.float32, device=self.model.device
        )


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(cfg: DictConfig):
    # Ensure num_worlds is strictly 1 for interactive play
    cfg.simulation.num_worlds = 1

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    sim = CartPoleInteractive(
        sim_config,
        render_config,
        engine_config,
        logging_config,
    )

    # Start the blocking interactive window loop
    sim.run()


if __name__ == "__main__":
    main()
