from dataclasses import dataclass
from enum import Enum
from typing import Literal

import newton


class SyncMode(Enum):
    ALIGN_FPS_TO_DT = 1
    ALIGN_DT_TO_FPS = 2


@dataclass
class SimulationConfig:
    """Parameters defining the simulation's timeline and execution
    strategy.

    ``use_cuda_graph`` controls whether the inner physics step is
    captured into a CUDA graph at run time. It used to live on a
    separate ``ExecutionConfig`` alongside a ``headless_steps_per_segment``
    knob; both were collapsed here once measurement showed the
    per-segment unroll knob bought no measurable speed-up at this
    codebase's scale.
    """

    duration_seconds: float = 3.0
    target_timestep_seconds: float = 1e-3
    num_worlds: int = 1
    sync_mode: SyncMode = SyncMode.ALIGN_FPS_TO_DT
    use_cuda_graph: bool = True


@dataclass
class RenderingConfig:
    """Parameters for rendering the simulation to a USD file."""

    vis_type: Literal["gl", "usd", "null", None] = "gl"
    target_fps: int | None = 30
    usd_file: str | None = "sim.usd"
    usd_scaling: float | None = 100.0
    start_paused: bool = True
    world_offset_x: float = 20.0
    world_offset_y: float = 20.0

    def create_viewer(self, model: newton.Model, num_segments: int | None):
        """
        Factory method to create the appropriate viewer instance.
        """
        if self.vis_type == "usd":
            return newton.viewer.ViewerUSD(
                output_path=self.usd_file,
                fps=self.target_fps,
                up_axis="Z",
                num_frames=num_segments,
            )
        elif self.vis_type == "gl":
            return newton.viewer.ViewerGL()
        elif self.vis_type == "null" or self.vis_type is None:
            return newton.viewer.ViewerNull(num_segments)
        else:
            raise ValueError(f"Unsupported rendering type: {self.vis_type}")


