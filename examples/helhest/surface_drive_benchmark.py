"""Headless, deterministic Helhest-on-surface benchmark scene.

Mirrors `surface_drive.py` but with constant velocity targets and no viewer
input, so the workload is reproducible across optimization branches. Intended
to be run with `engine=axion_profile_per_iter` so the per-component profiler
prints `linear_system / preconditioner / cr_solve / step_or_linesearch /
convergence_check` timings at the end of the run.
"""
import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import openmesh
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class HelhestSurfaceBenchmark(InteractiveSimulator):
    """Constant-velocity drive of Helhest across the surface mesh.

    All three wheels are commanded at the same forward velocity, so the robot
    drives roughly straight. The simulation is fully headless (ViewerNull) and
    terminates after `simulation.duration_seconds`.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        control_mode: str = "velocity",
        k_p: float = 150.0,
        k_d: float = 0.0,
        friction: float = 0.5,
        drive_velocity: float = 5.0,
    ):
        if control_mode != "velocity":
            raise ValueError(
                "surface_drive_benchmark only supports control_mode='velocity'; "
                f"got {control_mode!r}"
            )

        self.control_mode = control_mode
        self.k_p = k_p
        self.k_d = k_d
        self.friction = friction
        self.drive_velocity = drive_velocity

        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )

        # 9 joint DOFs: 6 free-joint base + 3 wheels (left, right, rear).
        targets = np.zeros(9, dtype=np.float32)
        targets[6] = drive_velocity  # left wheel
        targets[7] = drive_velocity  # right wheel
        targets[8] = drive_velocity  # rear wheel
        self.joint_target = wp.array(targets, dtype=wp.float32, device=self.model.device)

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        wp.copy(self.control.joint_target_vel, self.joint_target)

    def build_model(self) -> newton.Model:
        """Helhest on the same surface mesh used by `surface_drive.py`."""
        self.builder.rigid_gap = 0.5

        robot_x = -1.5
        robot_y = 0.0
        robot_z = 1.7

        create_helhest_model(
            self.builder,
            xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.friction,
            friction_rear=self.friction * 0.5,
        )

        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()

        scale = np.array([6.0, 6.0, 4.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])

        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        globals_builder = newton.ModelBuilder()
        globals_builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=True,
                mu=0.5,
                ke=150.0,
                kd=150.0,
                kf=500.0,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            global_builder=globals_builder,
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest_benchmark", version_base=None)
def helhest_surface_benchmark(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = HelhestSurfaceBenchmark(
        sim_config,
        render_config,
        engine_config,
        logging_config,
        control_mode=cfg.control.mode,
        k_p=cfg.control.k_p,
        k_d=cfg.control.k_d,
        friction=cfg.friction_coeff,
        drive_velocity=cfg.drive_velocity,
    )
    simulator.run()


if __name__ == "__main__":
    helhest_surface_benchmark()
