import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .sim_config import RenderingConfig
from .sim_config import SimulationConfig
from .sim_config import SyncMode


class SimulationClock:
    """
    Manages the flow of time in the simulation.
    Handles the synchronization between physics time steps and rendering frames.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
    ):
        self.sim_config = simulation_config
        self.render_config = rendering_config

        # --- State Tracking ---
        self._current_step: int = 0
        self._current_time: float = 0.0

        # --- Calculated Parameters ---
        self.dt: float = 0.0
        self.steps_per_segment: int = 0
        self.num_segments: int = 0
        self.total_sim_steps: int = 0
        self.sim_duration: float = 0.0

        self._resolve_timing_parameters()

        # print(f"Num segments {self.num_segments}")
        # print(f"Total sim steps {self.total_sim_steps}")

    @property
    def time(self) -> float:
        return self._current_time

    @property
    def step(self) -> int:
        return self._current_step

    def advance(self):
        """Advances the clock by one physics step."""
        self._current_step += 1
        self._current_time += self.dt

    def get_segment_time(self, segment_idx: int) -> float:
        """Returns the simulation time at the start of a specific segment (frame)."""
        return segment_idx * self.steps_per_segment * self.dt

    def _resolve_timing_parameters(self):
        """Calculates timing parameters based on configuration."""
        mode = self.sim_config.sync_mode
        target_dt = self.sim_config.target_timestep_seconds

        # 1. Determine Timestep and Steps-per-Segment
        if self.render_config.vis_type in ["usd", "gl"]:
            target_fps = self.render_config.target_fps

            if mode == SyncMode.ALIGN_DT_TO_FPS:
                self.dt, self.steps_per_segment = self._calculate_render_aligned_timestep(
                    target_dt, target_fps, force_even=False
                )
            elif mode == SyncMode.ALIGN_FPS_TO_DT:
                self.dt = target_dt
                target_frame_duration = 1.0 / target_fps
                ideal_steps = round(target_frame_duration / target_dt)
                self.steps_per_segment = max(1, ideal_steps)

                # Adjust FPS to match exact multiple of dt
                actual_frame_duration = self.steps_per_segment * self.dt
                new_fps = 1.0 / actual_frame_duration

                if abs(new_fps - target_fps) > 0.1:
                    print(f"INFO: Rendering FPS adjusted from {target_fps} to {new_fps:.2f}")
                self.render_config.target_fps = new_fps
        else:
            self.dt = target_dt
            # Headless: one physics step per segment. There used to be an
            # ``exec_config.headless_steps_per_segment`` knob here that
            # unrolled multiple steps inside the captured CUDA graph;
            # measurement showed no speed-up (graph-launch overhead is
            # noise relative to per-step compute at this codebase's
            # scale), so the knob was dropped.
            self.steps_per_segment = 1

        # 2. Determine Total Duration
        self.sim_duration, self.num_segments = self._align_duration_to_segment(
            self.sim_config.duration_seconds
        )
        self.total_sim_steps = self.steps_per_segment * self.num_segments

        # GL handles infinite loops differently, but calculation remains valid
        if self.render_config.vis_type == "gl":
            self.num_segments = None

    def _calculate_render_aligned_timestep(
        self, target_dt: float, fps: int, force_even: bool = True
    ) -> Tuple[float, int]:
        frame_duration = 1.0 / fps
        ideal_steps_per_frame = frame_duration / target_dt
        steps_per_frame = round(ideal_steps_per_frame) or 1

        if force_even and steps_per_frame % 2 != 0:
            steps_per_frame += 1

        effective_timestep = frame_duration / steps_per_frame

        adj_ratio = abs(effective_timestep - target_dt) / target_dt
        if adj_ratio > 0.01:
            print(f"INFO: Target timestep adjusted to {effective_timestep*1000:.3f}ms")

        return effective_timestep, steps_per_frame

    def _align_duration_to_segment(self, target_duration: float) -> Tuple[float, int]:
        segment_duration = self.dt * self.steps_per_segment
        num_segments = math.ceil(target_duration / segment_duration)
        effective_duration = num_segments * segment_duration

        adj_ratio = abs(effective_duration - target_duration) / target_duration
        if adj_ratio > 0.01:
            print(f"INFO: Simulation duration adjusted to {effective_duration:.4f}s")

        return effective_duration, num_segments
