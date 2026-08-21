import os
import pathlib
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
    """Parameters for rendering the simulation to a USD file.

    ``real_time`` only affects the ``gl`` viewer: it paces the interactive
    loop so one rendered frame takes as long in wall-clock as it covers in
    sim time. Off, the loop runs as fast as the solver does, which for a
    faster-than-real-time solver means the window plays back sped up.
    """

    vis_type: Literal["gl", "usd", "null", None] = "gl"
    target_fps: int | None = 30
    usd_file: str | None = "sim.usd"
    usd_scaling: float | None = 100.0
    start_paused: bool = True
    real_time: bool = True
    world_offset_x: float = 20.0
    world_offset_y: float = 20.0

    def create_viewer(self, model: newton.Model, num_segments: int | None):
        """
        Factory method to create the appropriate viewer instance.
        """
        if self.vis_type == "gl":
            _keep_glx_off_the_compute_gpu()

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




def _keep_glx_off_the_compute_gpu():
    """Stop OpenGL being pinned to the same NVIDIA GPU warp computes on.

    With ``__GLX_VENDOR_LIBRARY_NAME=nvidia`` (a common setting on hybrid
    laptops, and a Hyprland/omarchy default), GL rendering and warp's CUDA
    graph launches contend for one GPU. The NVIDIA scheduler eventually fails
    to invalidate an active compute QMD and kills the context::

        NVRM: Xid 13, Graphics Exception: SKEDCHECK22_INVALIDATE_ACTIVE_QMD failed

    The process only notices one readback later, as **CUDA error 719** -- which
    reads as a physics divergence, not a driver fault, and has cost real
    debugging time. Any GL example dies within ~60 s of starting; headless runs
    never do.

    Clearing the variable sends GL to the integrated GPU and leaves the discrete
    one for compute. CUDA is unaffected: it never goes through libglvnd, and
    ViewerGL needs no GL/CUDA interop.

    On a machine whose *only* GPU is the NVIDIA one there is no second vendor to
    fall back to, so set ``OSTRICH_ALLOW_NVIDIA_GLX=1`` to keep the pin. See
    docs/gl_viewer_gpu_contention.md.
    """
    if os.environ.get("__GLX_VENDOR_LIBRARY_NAME") != "nvidia":
        return
    if not _has_non_nvidia_gl_vendor():
        # Single-GPU NVIDIA box: clearing the pin would leave libglvnd with no
        # usable vendor, so the contention is the lesser problem. Say so and
        # leave it alone.
        print(
            "WARNING: __GLX_VENDOR_LIBRARY_NAME=nvidia and no other GL vendor is "
            "installed, so OpenGL and CUDA must share the GPU. Watch for NVIDIA "
            "Xid 13 surfacing as a misleading 'CUDA error 719'. "
            "See docs/gl_viewer_gpu_contention.md."
        )
        return
    if os.environ.get("OSTRICH_ALLOW_NVIDIA_GLX") == "1":
        print(
            "WARNING: __GLX_VENDOR_LIBRARY_NAME=nvidia with OSTRICH_ALLOW_NVIDIA_GLX=1. "
            "GL and CUDA share the discrete GPU; expect Xid 13 surfacing as "
            "'CUDA error 719' within ~60s. See docs/gl_viewer_gpu_contention.md."
        )
        return
    del os.environ["__GLX_VENDOR_LIBRARY_NAME"]
    print(
        "INFO: cleared __GLX_VENDOR_LIBRARY_NAME=nvidia so OpenGL renders on the "
        "integrated GPU and leaves the discrete one for CUDA. Sharing them trips "
        "NVIDIA Xid 13, which surfaces as a misleading 'CUDA error 719'. "
        "Set OSTRICH_ALLOW_NVIDIA_GLX=1 to keep the pin (single-GPU machines)."
    )


def _has_non_nvidia_gl_vendor() -> bool:
    """Whether libglvnd has a vendor other than NVIDIA to fall back on.

    Clearing the GLX pin only helps if something else can drive the display.
    Each installed vendor drops a JSON into glvnd's vendor directory, so a
    non-NVIDIA entry there means there is an integrated or second GPU to render
    on. Absent the directory we assume there is, since the pin is normally only
    set on hybrid machines in the first place.
    """
    vendor_dir = pathlib.Path("/usr/share/glvnd/egl_vendor.d")
    if not vendor_dir.is_dir():
        return True
    entries = [f.name for f in vendor_dir.glob("*.json")]
    return any("nvidia" not in name.lower() for name in entries) if entries else True
