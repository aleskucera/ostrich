import os
import pathlib
from importlib.resources import files
from typing import override
import pathlib

import hydra
import numpy as np
import newton
import warp as wp
from axion import EngineConfig
from axion import InteractiveSimulator
from axion import JointMode
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from pendulum_articulation_definition import PENDULUM_HEIGHT
from pendulum_articulation_definition import build_pendulum_model
from pendulum_utils import generalized_to_maximal
from pendulum_utils import set_tilted_plane_from_coefficients

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

ENABLE_STATE_LOGGING = False  # set True to write pendulum-state HDF5
if ENABLE_STATE_LOGGING:
    from axion.neural_solver.logging.state_logger_for_examples import PendulumStateLogger

ENABLE_CONTROL = False  # True=controller active, False=controller off
TARGET_POS_Q0 = np.pi / 2
TARGET_POS_Q1 = np.pi / 6

class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        plane_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0),
        initial_state: tuple[float, float, float, float] | None = None,
        state_logger=None,
    ):
        self.plane_coefficients = plane_coefficients
        self.state_logger = state_logger
        super().__init__(
            sim_config,
            render_config,
            engine_config,
            logging_config,
        )
        if initial_state is not None:
            q0, q1, qd0, qd1 = initial_state
            generalized_to_maximal(
                self.model, self.current_state,
                q0=q0, q1=q1, qd0=qd0, qd1=qd1,
            )

        # Preallocate the target buffer once so it stays alive across CUDA graph replays.
        self.q_target = wp.array(
            [TARGET_POS_Q0, TARGET_POS_Q1],
            dtype=wp.float32,
            device=self.model.device,
        )

    @override
    def control_policy(self, state: newton.State):
        if not ENABLE_CONTROL:
            return
        wp.copy(self.control.joint_target_pos, self.q_target)

    @override
    def _render(self, segment_num: int):
        """Renders the current state to the appropriate viewers, including world XYZ axes."""
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        
        # Draw world axes at origin
        axis_length = 1.0  # Length of each axis
        origin = wp.vec3(0.0, 0.0, 0.0)
        
        # Define axis endpoints
        x_end = wp.vec3(axis_length, 0.0, 0.0)  # X axis (red)
        y_end = wp.vec3(0.0, axis_length, 0.0)  # Y axis (green)
        z_end = wp.vec3(0.0, 0.0, axis_length)  # Z axis (blue)
        
        # Create arrays for line starts and ends
        device = wp.get_device()
        starts = wp.array([origin, origin, origin], dtype=wp.vec3, device=device)
        ends = wp.array([x_end, y_end, z_end], dtype=wp.vec3, device=device)
        
        # Colors: red for X, green for Y, blue for Z
        colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0),  # Red for X
             wp.vec3(0.0, 1.0, 0.0),  # Green for Y
             wp.vec3(0.0, 0.0, 1.0)], # Blue for Z
            dtype=wp.vec3,
            device=device
        )
        
        # Draw the axes
        self.viewer.log_lines("world_axes", starts, ends, colors, width=0.08)
        
        # Draw reference frame at the first pendulum link anchor point
        anchor_x = 0.0
        anchor_y = 0.0
        anchor_z = PENDULUM_HEIGHT  # Position from parent_xform
        anchor_axis_length = 0.5  # Slightly shorter than world axes
        
        # Define axis endpoints (absolute positions)
        anchor_point = wp.vec3(anchor_x, anchor_y, anchor_z)
        anchor_x_end = wp.vec3(anchor_x + anchor_axis_length, anchor_y, anchor_z)
        anchor_y_end = wp.vec3(anchor_x, anchor_y + anchor_axis_length, anchor_z)
        anchor_z_end = wp.vec3(anchor_x, anchor_y, anchor_z + anchor_axis_length)
        
        # Create arrays for anchor frame lines
        anchor_starts = wp.array(
            [anchor_point, anchor_point, anchor_point],
            dtype=wp.vec3,
            device=device
        )
        anchor_ends = wp.array(
            [anchor_x_end, anchor_y_end, anchor_z_end],
            dtype=wp.vec3,
            device=device
        )
        
        # Same colors as world axes
        anchor_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0),  # Red for X
             wp.vec3(0.0, 1.0, 0.0),  # Green for Y
             wp.vec3(0.0, 0.0, 1.0)], # Blue for Z
            dtype=wp.vec3,
            device=device
        )
        
        # Draw the anchor reference frame
        self.viewer.log_lines("anchor_frame", anchor_starts, anchor_ends, anchor_colors, width=0.08)
        
        self.viewer.end_frame()

        if self.state_logger is not None and self.use_cuda_graph:
            self.state_logger.log_step(self.current_state, sim_time)

    @override
    def _run_segment_without_graph(self, segment_num: int):
        if segment_num == 0:
            prewarm_fn = getattr(self.solver, "prewarm", None)
            if prewarm_fn is not None:
                print("INFO: Pre-warming neural solver history buffer (no-graph path)...")
                prewarm_fn(self.current_state, self.contacts, self.clock.dt)
        for step in range(self.steps_per_segment):
            self._single_physics_step(step)
            if self.state_logger is not None:
                global_step = segment_num * self.steps_per_segment + step + 1
                self.state_logger.log_step(self.current_state, global_step * self.clock.dt)

    def build_model(self,) -> newton.Model:
        """
        Use the same pendulum articulation as AxionEngineWrapper.py
        """
        model = build_pendulum_model(num_worlds=1, device="cuda:0")
        mode = JointMode.TARGET_POSITION if ENABLE_CONTROL else JointMode.NONE
        model.joint_dof_mode.assign(wp.array([mode, mode], dtype=wp.int32, device=model.device))
        a, b, c, d = self.plane_coefficients
        set_tilted_plane_from_coefficients(model, a, b, c, d, world_idx=0)
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="gpt_pendulum", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    # Plane equation: nx*x + ny*y + nz*z + d = 0 (default: horizontal z=0)
    plane_coefficients = [0.0, 0.0, 1.0, 0.0]
    #plane_coefficients = [-0.2354, -0.0000, 0.9719, -2.3318]

    # Custom initial conditions: (q0, q1, qd0, qd1)
    # Set to None to start from the default rest position.
    INITIAL_STATE = (-0.5704, 2.8907, -3.6530, -7.6918)  # e.g. (0.5, -0.3, 1.0, -2.0)
    INITIAL_STATE = (0, 0., 0, 0,)
    #INITIAL_STATE = (0.5, -0.3, 1.0, -2.0)
    #INITIAL_STATE = (np.pi/6, 0, 1.0, 1.0)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        engine_config=engine_config,
        logging_config=logging_config,
        plane_coefficients=plane_coefficients,
        initial_state=INITIAL_STATE,
    )

    if ENABLE_STATE_LOGGING:
        log_dt = (
            simulator.clock.dt
            if not simulator.use_cuda_graph
            else simulator.steps_per_segment * simulator.clock.dt
        )
        simulator.state_logger = PendulumStateLogger(
            script_name="GPTEngine_example",
            dt=log_dt,
            duration_seconds=sim_config.duration_seconds,
        )

    simulator.run()

    if ENABLE_STATE_LOGGING:
        simulator.state_logger.save()

if __name__ == "__main__":
    basic_pendulum_example()