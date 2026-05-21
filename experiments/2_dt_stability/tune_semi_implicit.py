"""Interactive semi-implicit obstacle traversal — showcase bad dt behaviour.

Robot drives straight into a box obstacle using the semi-implicit engine with
calibrated params from 1_sim_to_real. Use this to visually inspect what happens
at dt values near / above the stability limit (explosion, penetration, jitter).

Usage:
    python experiments/2_dt_stability/tune_semi_implicit.py
    python experiments/2_dt_stability/tune_semi_implicit.py --dt 0.005
    python experiments/2_dt_stability/tune_semi_implicit.py --dt 0.01 --obstacle-height 0.15
"""
import argparse
import os
from typing import override

import newton
import numpy as np
import warp as wp
from axion import ExecutionConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig

from examples.helhest.common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

# Calibrated params from 1_sim_to_real sweep (match sweep_semi_implicit.py)
K_P = 0.0
K_D = 400.0
MU = 0.02
KE = 6000.0
KD_CONTACT = 3000.0
KF = 1500.0

# Obstacle + drive config (match sweep_semi_implicit.py)
OBSTACLE_X = 2.0
OBSTACLE_HEIGHT = 0.1
WHEEL_VEL = 4.0
RAMP_TIME = 1.0
DURATION = 8.0


@wp.kernel
def set_control_from_sequence_kernel(
    step_idx: wp.array(dtype=wp.int32),
    ctrl_sequence: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel: wp.array(dtype=wp.float32),
    num_steps: int,
):
    dof = wp.tid()
    s = step_idx[0]
    if s >= num_steps:
        s = num_steps - 1
    joint_target_vel[dof] = ctrl_sequence[s, dof]


@wp.kernel
def increment_step_kernel(
    step_idx: wp.array(dtype=wp.int32),
):
    step_idx[0] = step_idx[0] + 1


class HelhestSemiImplicitObstacleSim(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: SemiImplicitEngineConfig,
        logging_config: LoggingConfig,
        k_p: float = K_P,
        k_d: float = K_D,
        mu: float = MU,
        ke: float = KE,
        kd_contact: float = KD_CONTACT,
        kf: float = KF,
        wheel_vel: float = WHEEL_VEL,
        ramp_time: float = RAMP_TIME,
        obstacle_x: float = OBSTACLE_X,
        obstacle_height: float = OBSTACLE_HEIGHT,
    ):
        self._k_p = k_p
        self._k_d = k_d
        self._mu = mu
        self._ke = ke
        self._kd_contact = kd_contact
        self._kf = kf
        self._obstacle_x = obstacle_x
        self._obstacle_height = obstacle_height
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        num_dofs = 9
        total_steps = self.clock.total_sim_steps
        ctrl_np = np.zeros((total_steps, num_dofs), dtype=np.float32)
        for i in range(total_steps):
            t = (i + 1) * self.clock.dt
            ramp = min(t / ramp_time, 1.0) if ramp_time > 0 else 1.0
            wv = wheel_vel * ramp
            ctrl_np[i, 6] = wv
            ctrl_np[i, 7] = wv
            ctrl_np[i, 8] = wv

        self._ctrl_sequence = wp.array(ctrl_np, dtype=wp.float32)
        self._ctrl_step_idx = wp.zeros(1, dtype=wp.int32)
        self._ctrl_total_steps = total_steps

    @override
    def init_state_fn(self, current_state, next_state, contacts, dt):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state):
        wp.launch(
            kernel=set_control_from_sequence_kernel,
            dim=9,
            inputs=[
                self._ctrl_step_idx,
                self._ctrl_sequence,
                self.control.joint_target_vel,
                self._ctrl_total_steps,
            ],
        )
        wp.launch(
            kernel=increment_step_kernel,
            dim=1,
            inputs=[self._ctrl_step_idx],
        )

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.1

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=self._mu,
            ke=self._ke,
            kd=self._kd_contact,
            kf=self._kf,
        )
        self.builder.add_ground_plane(cfg=ground_cfg)

        create_helhest_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p=self._k_p,
            k_d=self._k_d,
            friction_left_right=self._mu,
            friction_rear=self._mu,
            ke=self._ke,
            kd=self._kd_contact,
            kf=self._kf,
        )

        # Static box obstacle (high friction to match sweep_semi_implicit.py)
        self.builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                (self._obstacle_x, 0.0, self._obstacle_height),
                wp.quat_identity(),
            ),
            hx=0.5,
            hy=1.0,
            hz=self._obstacle_height,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=1.0,
                ke=self._ke,
                kd=self._kd_contact,
                kf=self._kf,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep (default: 0.002)")
    parser.add_argument(
        "--duration", type=float, default=DURATION, help=f"Duration (default: {DURATION}s)"
    )
    parser.add_argument(
        "--k-p", type=float, default=K_P, help=f"Velocity servo gain (default: {K_P})"
    )
    parser.add_argument(
        "--k-d", type=float, default=K_D, help=f"Velocity servo damping (default: {K_D})"
    )
    parser.add_argument(
        "--mu", type=float, default=MU, help=f"Friction coefficient (default: {MU})"
    )
    parser.add_argument(
        "--ke", type=float, default=KE, help=f"Contact elastic stiffness (default: {KE})"
    )
    parser.add_argument(
        "--kd", type=float, default=KD_CONTACT, help=f"Contact damping (default: {KD_CONTACT})"
    )
    parser.add_argument("--kf", type=float, default=KF, help=f"Friction damping (default: {KF})")
    parser.add_argument(
        "--wheel-vel",
        type=float,
        default=WHEEL_VEL,
        help=f"Wheel velocity rad/s (default: {WHEEL_VEL})",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=RAMP_TIME,
        help=f"Velocity ramp time (default: {RAMP_TIME}s)",
    )
    parser.add_argument(
        "--obstacle-x",
        type=float,
        default=OBSTACLE_X,
        help=f"Obstacle x position (default: {OBSTACLE_X})",
    )
    parser.add_argument(
        "--obstacle-height",
        type=float,
        default=OBSTACLE_HEIGHT,
        help=f"Obstacle half-height in meters (default: {OBSTACLE_HEIGHT})",
    )
    parser.add_argument("--friction-smoothing", type=float, default=0.1)
    parser.add_argument("--angular-damping", type=float, default=0.05)
    parser.add_argument(
        "--joint-attach-ke",
        type=float,
        default=1.0e5,
        help="Joint attachment spring stiffness (default: 1e4)",
    )
    parser.add_argument(
        "--joint-attach-kd",
        type=float,
        default=1.0e3,
        help="Joint attachment damping (default: 1e2)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    args = parser.parse_args()

    print(f"  dt={args.dt}, k_p={args.k_p}, k_d={args.k_d}, mu={args.mu}")
    print(f"  ke={args.ke}, kd={args.kd}, kf={args.kf}")
    print(f"  joint_attach_ke={args.joint_attach_ke}, joint_attach_kd={args.joint_attach_kd}")
    print(f"  wheel_vel={args.wheel_vel} rad/s, ramp={args.ramp_time}s")
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
    engine_config = SemiImplicitEngineConfig(
        angular_damping=args.angular_damping,
        friction_smoothing=args.friction_smoothing,
        joint_attach_ke=args.joint_attach_ke,
        joint_attach_kd=args.joint_attach_kd,
    )
    logging_config = LoggingConfig(
        enable_timing=False,
        enable_hdf5_logging=False,
    )

    sim = HelhestSemiImplicitObstacleSim(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
        k_p=args.k_p,
        k_d=args.k_d,
        mu=args.mu,
        ke=args.ke,
        kd_contact=args.kd,
        kf=args.kf,
        wheel_vel=args.wheel_vel,
        ramp_time=args.ramp_time,
        obstacle_x=args.obstacle_x,
        obstacle_height=args.obstacle_height,
    )

    if hasattr(sim.viewer, "show_ui"):
        sim.viewer.show_ui = False

    if hasattr(sim.viewer, "set_camera"):
        sim.viewer.set_camera(pos=wp.vec3(8.0, -5.0, 6.0), pitch=-40.0, yaw=150.0)

    sim.run()


if __name__ == "__main__":
    main()
