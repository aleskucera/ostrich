"""Interactive Axion obstacle traversal — for visual tuning before dt sweep.

Robot drives straight into a box obstacle. Use this to verify traversal
works at various dt before running the headless stability sweep.

Usage:
    python examples/comparison_gradient/helhest/dt_sweep/tune_obstacle_axion.py
    python examples/comparison_gradient/helhest/dt_sweep/tune_obstacle_axion.py --dt 0.05
    python examples/comparison_gradient/helhest/dt_sweep/tune_obstacle_axion.py --obstacle-height 0.15
"""
import argparse
import os
from typing import override

import newton
import numpy as np
import warp as wp
from axion import AxionEngineConfig
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig

from examples.helhest.common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Calibrated params from parameter_sweep
K_P = 4000.0
MU = 0.1
FRICTION_COMPLIANCE = 1.2e-5
CONTACT_COMPLIANCE = 1e-10

# Drive straight: all wheels at same speed
WHEEL_VEL = 2.0


@wp.kernel
def set_friction_coefficient_kernel(
    mu: float,
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, shape_idx = wp.tid()
    shape_material_mu[world_idx, shape_idx] = mu


@wp.kernel
def set_shape_friction_kernel(
    mu: float,
    shape_idx: int,
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx = wp.tid()
    shape_material_mu[world_idx, shape_idx] = mu


class HelhestObstacleSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: AxionEngineConfig,
        logging_config: LoggingConfig,
        k_p: float = K_P,
        k_d: float = 0.0,
        mu: float = MU,
        wheel_vel: float = WHEEL_VEL,
        obstacle_x: float = 2.0,
        obstacle_height: float = 0.1,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._obstacle_x = obstacle_x
        self._obstacle_height = obstacle_height
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.set_friction_coefficient(mu)

        # Constant forward velocity: all wheels same speed
        robot_joint_target = np.zeros(9, dtype=np.float32)
        robot_joint_target[6] = wheel_vel  # left
        robot_joint_target[7] = wheel_vel  # right
        robot_joint_target[8] = wheel_vel  # rear
        joint_target = np.tile(robot_joint_target, sim_config.num_worlds)
        self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)

    @override
    def init_state_fn(self, current_state, next_state, contacts, dt):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state):
        wp.copy(self.control.joint_target_vel, self.joint_target)

    def build_model(self) -> newton.Model:
        DUMMY_FRICTION = 0.0

        self.builder.rigid_gap = 0.1
        self.builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=DUMMY_FRICTION))

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=self._k_p,
            k_d=self._k_d,
            friction_left_right=DUMMY_FRICTION,
            friction_rear=DUMMY_FRICTION,
        )

        # Static box obstacle
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self._obstacle_x, 0.0, self._obstacle_height),
                wp.quat_identity(),
            ),
            hx=0.5,  # 1m long
            hy=1.0,  # 2m wide
            hz=self._obstacle_height,
            cfg=newton.ModelBuilder.ShapeConfig(mu=DUMMY_FRICTION),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )

    def set_friction_coefficient(self, mu: float, obstacle_mu: float = 1.0):
        # Set mu on all shapes (ground + wheels)
        wp.launch(
            kernel=set_friction_coefficient_kernel,
            dim=(self.solver.dims.num_worlds, self.solver.axion_model.shape_count),
            inputs=[mu],
            outputs=[self.solver.axion_model.shape_material_mu],
        )
        # Override obstacle box with higher friction
        # The box is the last shape added in build_model
        obstacle_idx = self.solver.axion_model.shape_count - 1
        wp.launch(
            kernel=set_shape_friction_kernel,
            dim=self.solver.dims.num_worlds,
            inputs=[obstacle_mu, obstacle_idx],
            outputs=[self.solver.axion_model.shape_material_mu],
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dt", type=float, default=0.2, help="Timestep (default: 0.15)")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration (default: 5s)")
    parser.add_argument("--k-p", type=float, default=K_P, help=f"Servo gain (default: {K_P})")
    parser.add_argument("--mu", type=float, default=MU, help=f"Friction (default: {MU})")
    parser.add_argument(
        "--wheel-vel",
        type=float,
        default=WHEEL_VEL,
        help=f"Wheel velocity rad/s (default: {WHEEL_VEL})",
    )
    parser.add_argument(
        "--obstacle-x", type=float, default=2.0, help="Obstacle x position (default: 2.0)"
    )
    parser.add_argument(
        "--obstacle-height",
        type=float,
        default=0.1,
        help="Obstacle half-height in meters (default: 0.1)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    args = parser.parse_args()

    print(f"  dt={args.dt}, k_p={args.k_p}, mu={args.mu}, wheel_vel={args.wheel_vel}")
    print(f"  obstacle: x={args.obstacle_x}, height={args.obstacle_height*2:.2f}m")

    sim_config = SimulationConfig(
        duration_seconds=args.duration,
        target_timestep_seconds=args.dt,
        num_worlds=1,
    )
    render_config = RenderingConfig(
        vis_type="null" if args.headless else "gl",
        target_fps=30,
        usd_file=None,
        start_paused=False,
    )
    exec_config = ExecutionConfig(
        use_cuda_graph=True,
        headless_steps_per_segment=1,
    )
    engine_config = AxionEngineConfig(
        max_newton_iters=16,
        max_linear_iters=16,
        backtrack_min_iter=12,
        newton_atol=1e-5,
        linear_atol=1e-5,
        linear_tol=1e-5,
        enable_linesearch=False,
        joint_compliance=6e-10,
        contact_compliance=CONTACT_COMPLIANCE,
        friction_compliance=FRICTION_COMPLIANCE,
        regularization=1e-6,
        contact_fb_alpha=0.5,
        contact_fb_beta=1.0,
        friction_fb_alpha=1.0,
        friction_fb_beta=1.0,
        max_contacts_per_world=16,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = HelhestObstacleSim(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        k_p=args.k_p,
        mu=args.mu,
        wheel_vel=args.wheel_vel,
        obstacle_x=args.obstacle_x,
        obstacle_height=args.obstacle_height,
    )

    if hasattr(sim.viewer, "set_camera"):
        sim.viewer.set_camera(pos=wp.vec3(8.0, -5.0, 6.0), pitch=-40.0, yaw=150.0)

    sim.run()


if __name__ == "__main__":
    main()
