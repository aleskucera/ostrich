from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import newton
import numpy as np
import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_config import EngineConfig
from axion.core.logging_config import LoggingConfig

from .base_simulator import BaseSimulator
from .sim_config import RenderingConfig
from .sim_config import SimulationConfig
from .trajectory import NewtonTargetTrajectory
from .trajectory_buffer import TrajectoryBuffer


class DifferentiableSimulator(BaseSimulator, ABC):
    """
    Base class for differentiable physics simulators.

    Manages the full trajectory of states, gradient tape, and visualization.
    Subclasses implement `_forward_backward` for their specific differentiation scheme.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: Optional[LoggingConfig] = None,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            engine_config,
            logging_config,
        )

        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.clock.total_sim_steps + 1)
        ]
        self.controls = [
            self.model.control(requires_grad=True) for _ in range(self.clock.total_sim_steps)
        ]

        self.target_states = [self.model.state() for _ in range(self.clock.total_sim_steps + 1)]
        self.target_controls = [self.model.control() for _ in range(self.clock.total_sim_steps)]

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            requires_grad=False,
        )
        self.collision_pipeline.collide(self.states[0], self.contacts)

        self.viewer = self.rendering_config.create_viewer(
            self.model, num_segments=self.clock.total_sim_steps
        )
        if self.viewer:
            self.viewer.set_model(self.model)

        self.cuda_graph: Optional[wp.Graph] = None
        self.tape = wp.Tape()
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        self._tracked_bodies = {}
        self._render_time_offset = 0.0

    def close(self):
        """Flush the viewer to disk. ViewerUSD only writes the file on close()."""
        if self.viewer is not None and self.rendering_config.vis_type == "usd":
            self.viewer.close()
            self.viewer = None

    # ============== ABSTRACT METHODS ==============
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def _forward_backward(self):
        pass

    # ============== DIFFERENTIABLE SIMULATION METHODS ==============
    def diff_step(self):
        if self.use_cuda_graph and self.cuda_graph is None:
            with wp.ScopedCapture() as capture:
                self._forward_backward()
            self.cuda_graph = capture.graph

        if self.use_cuda_graph and self.cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self._forward_backward()

    # ============== VISUALIZATION METHODS ==============
    def track_body(self, body_idx: int, name: str = None, color: tuple = (1.0, 0.0, 0.0)):
        """
        Register a body to have its trajectory visualized automatically.
        """
        if name is None:
            name = f"body_{body_idx}"
        self._tracked_bodies[body_idx] = {"name": name, "color": color}

    def render_episode(
        self,
        iteration: int = 0,
        callback: Optional[Callable[[Any, int, newton.State], None]] = None,
        loop: bool = False,
        loops_count: int = 1,
        playback_speed: float = 1.0,
        start_paused: bool = False,
    ):
        """
        Replays the simulation episode in the viewer, rendering tracked trajectories.

        Args:
            iteration: Current training iteration (used for unique names).
            callback: Optional function(viewer, step_idx, state) for extra rendering.
            loop: Whether to replay the episode multiple times.
            loops_count: Number of times to replay if loop is True.
            playback_speed: Speed multiplier (1.0 = real time, 0.5 = slow motion).
        """
        if not self.viewer:
            return

        import time

        if start_paused and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True

        # 1. Pre-calculate trajectories for all tracked bodies
        trajectories = {}
        for body_idx, metadata in self._tracked_bodies.items():
            path = []
            for state in self.states:
                q = state.body_q.numpy()[body_idx]
                path.append(q[:3])
            trajectories[body_idx] = np.array(path, dtype=np.float32)

        # 2. Setup Timing
        dt = self.clock.dt
        total_sim_time = len(self.states) * dt
        plays = loops_count if loop else 1

        # 3. Replay Loop
        for play_idx in range(plays):
            start_wall_time = time.time()
            paused_elapsed = 0.0

            while True:
                current_wall_time = time.time()

                if self.viewer.is_paused():
                    start_wall_time = current_wall_time - paused_elapsed / playback_speed
                    elapsed_sim_time = paused_elapsed
                else:
                    elapsed_sim_time = (current_wall_time - start_wall_time) * playback_speed
                    paused_elapsed = elapsed_sim_time

                if elapsed_sim_time > total_sim_time:
                    break

                step_idx = int(elapsed_sim_time / dt)
                step_idx = min(step_idx, len(self.states) - 1)
                state = self.states[step_idx]

                self.viewer.begin_frame(self._render_time_offset + elapsed_sim_time)

                # A. Log Physics State
                self.viewer.log_state(state)

                # B. Log Trajectories
                iter_tag = str(iteration) if iteration >= 0 else f"n{-iteration}"
                for body_idx, metadata in self._tracked_bodies.items():
                    pts = trajectories[body_idx]
                    if len(pts) > 1:
                        line_name = f"/traj_{iter_tag}/{metadata['name']}"
                        self.viewer.log_lines(
                            line_name,
                            wp.array(pts[:-1], dtype=wp.vec3),
                            wp.array(pts[1:], dtype=wp.vec3),
                            metadata["color"],
                        )

                # C. Custom Callback
                if callback:
                    callback(self.viewer, step_idx, state)

                self.viewer.end_frame()

            self._render_time_offset += total_sim_time


class AxionDifferentiableSimulator(DifferentiableSimulator, ABC):
    """
    Differentiable simulator using the Axion implicit integration engine.

    Uses implicit differentiation via a TrajectoryBuffer and a custom
    backward pass through the Axion solver.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        engine_config: AxionEngineConfig,
        logging_config: Optional[LoggingConfig] = None,
    ):
        self.differentiable_simulation = True

        super().__init__(
            simulation_config,
            rendering_config,
            engine_config,
            logging_config,
        )

        self.trajectory = TrajectoryBuffer(
            data=self.solver.data,
            contacts=self.solver.axion_contacts,
            dims=self.solver.dims,
            num_steps=self.clock.total_sim_steps,
            device=self.model.device,
        )

    def _forward_backward(self):
        self.trajectory.zero_grad()

        # --- FORWARD PASS ---
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

        self.tape.zero()
        with self.tape:
            self.compute_loss()

        # --- BACKWARD PASS ---
        # Explicit gradients come from the TrajectoryBuffer
        self.tape.backward(self.loss)

        # # DEBUG: Check initial gradients from tape
        # print(f"DEBUG: Loss gradient norm: {np.linalg.norm(self.trajectory.body_pose.grad.numpy())}")
        #
        for i in range(self.clock.total_sim_steps - 1, -1, -1):
            self.trajectory.load_step(i, self.solver.data, self.solver.axion_contacts)
            self.solver.data.zero_gradients()
            self.solver.step_backward()
            self.trajectory.save_gradients(i, self.solver.data)
            self.trajectory.save_pose_gradients(i, self.solver.data)
            if self.solver.config.adjoint.gradient_normalization and i > 0:
                self.trajectory.normalize_gradients(i)

    def run_target_episode(self):
        for i in range(self.clock.total_sim_steps):
            self.collision_pipeline.collide(self.target_states[i], self.contacts)
            self.solver.step(
                state_in=self.target_states[i],
                state_out=self.target_states[i + 1],
                control=self.target_controls[i],
                contacts=self.contacts,
                dt=self.clock.dt,
            )
            self.trajectory.save_target_step(i, self.solver.data)
        self.solver.reset_timestep_counter()


class NewtonDifferentiableSimulator(DifferentiableSimulator, ABC):
    """
    Differentiable simulator using Newton-based explicit integration (e.g. SemiImplicit).

    Uses standard backpropagation through time (BPTT) by unrolling the
    simulation loop on a Warp tape.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: Optional[LoggingConfig] = None,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            engine_config,
            logging_config,
        )

        self.episode_trajectory = NewtonTargetTrajectory(
            model=self.model,
            num_steps=self.clock.total_sim_steps,
            device=self.model.device,
        )

    def _forward_backward(self):
        """
        Standard explicit differentiation (unrolling loop on tape).
        """
        self.tape.zero()

        # --- FORWARD PASS ---
        for i in range(self.clock.total_sim_steps):
            with self.tape:
                self.states[i].clear_forces()

            self.collision_pipeline.collide(self.states[i], self.contacts)

            with self.tape:
                self.solver.step(
                    state_in=self.states[i],
                    state_out=self.states[i + 1],
                    control=self.controls[i],
                    contacts=self.contacts,
                    dt=self.clock.dt,
                )

        # Compute loss on the final state (or trajectory)
        with self.tape:
            self.compute_loss()

        # --- BACKWARD PASS ---
        self.tape.backward(self.loss)
