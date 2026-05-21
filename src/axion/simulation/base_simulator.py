from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal
from typing import Optional

import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import EngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from newton import Model

from .sim_config import RenderingConfig
from .sim_config import SimulationConfig
from .simulation_clock import SimulationClock


class BaseSimulator(ABC):
    """
    The foundational class for Axion simulations.
    Handles configuration, model building, solver initialization, and timing resolution.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: Optional[LoggingConfig] = None,
    ):
        self.simulation_config = simulation_config
        self.rendering_config = rendering_config
        self.engine_config = engine_config
        self.logging_config = logging_config

        # --- Time Management ---
        # Delegated to the external clock class
        self.clock = SimulationClock(simulation_config, rendering_config)

        self.builder = AxionModelBuilder()
        self.model = self.build_model()

        self.current_state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.current_state)

        # --- Solver Initialization ---
        # Uses the factory method on the config object.
        # We explicitly pass logging_config; the factory decides whether to use it.
        self.solver = self.engine_config.create_engine(
            model=self.model,
            sim_steps=self.clock.total_sim_steps,
            logging_config=self.logging_config,
            differentiable_simulation=hasattr(self, "differentiable_simulation"),
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.current_state)

        # Viewer initialization is deferred to subclasses (Interactive vs Headless/Diff)
        self.viewer = None

    @property
    def use_cuda_graph(self) -> bool:
        return self.simulation_config.use_cuda_graph and wp.get_device().is_cuda

    @abstractmethod
    def build_model(self) -> Model:
        """
        Builds the physics model for the simulation.
        This method MUST be implemented by any subclass.
        """
        pass

    def control_policy(self, current_state: newton.State):
        """
        Implements the control policy for the simulation.
        This method may be optionally overridden by any subclass.
        """
        pass

    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        """
        Initialization hook for Axion engine.
        """
        wp.copy(dest=next_state.body_q, src=current_state.body_q)
        wp.copy(dest=next_state.body_qd, src=current_state.body_qd)
        # self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    def _copy_state(self, dest: newton.State, src: newton.State):
        """Copies the physics state data from src to dest."""
        wp.copy(dest.body_q, src.body_q)
        wp.copy(dest.body_qd, src.body_qd)
        wp.copy(dest.joint_q, src.joint_q)
        wp.copy(dest.joint_qd, src.joint_qd)

    def _single_physics_step(self, step_num: int):
        """Performs one fundamental integration step of the simulation."""
        self.current_state.clear_forces()

        # End_to_end profiling: mark the start of the "collide" phase. The
        # engine closes the phase by recording boundary 1 at engine.step
        # entry, so the elapsed window covers collide + the trivial Python
        # gap (control_policy, viewer.apply_forces) before engine.step.
        prof = self.solver.profiler if isinstance(self.solver, AxionEngine) else None
        if prof is not None and prof.enabled and prof.mode == "end_to_end":
            prof.record_boundary(0)
        self.contacts = self.model.collide(self.current_state)

        self.control_policy(self.current_state)

        if self.viewer:
            self.viewer.apply_forces(self.current_state)

        # Compute simulation step
        # Note: We use the effective timestep calculated by the clock
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.clock.dt,
        )

        # Explicitly copy next_state back to current_state
        # This ensures the data flow is recorded in Tape and CUDA Graphs
        self._copy_state(self.current_state, self.next_state)

        # Advance the simulation clock
        self.clock.advance()

    # --- Backward Compatibility / Convenience Properties ---

    @property
    def steps_per_segment(self) -> int:
        return self.clock.steps_per_segment

    @property
    def num_segments(self) -> int:
        return self.clock.num_segments
