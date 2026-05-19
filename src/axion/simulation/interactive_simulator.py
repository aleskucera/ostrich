import time
from abc import ABC
from typing import Optional

import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import EngineConfig
from axion.core.logging_config import LoggingConfig
from tqdm import tqdm

from .base_simulator import BaseSimulator
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


class InteractiveSimulator(BaseSimulator, ABC):
    """
    Simulator designed for real-time visualization and interactive sessions.
    Supports GL/USD rendering, FPS synchronization, and CUDA graphs.
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        rendering_config: RenderingConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            simulation_config,
            rendering_config,
            engine_config,
            logging_config,
        )

        self.viewer = self.rendering_config.create_viewer(
            model=self.model,
            num_segments=self.num_segments,
        )

        self.viewer.set_model(self.model)
        world_offsets = getattr(
            self.rendering_config, "world_offsets", None
        ) or (20.0, 20.0, 0.0)
        self.viewer.set_world_offsets(world_offsets)

        # Optional initial camera position and heading (GL viewer only)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            r = self.rendering_config
            if getattr(r, "camera_pos", None) is not None:
                pos = wp.vec3(*r.camera_pos)
                pitch = getattr(r, "camera_pitch", None)
                yaw = getattr(r, "camera_yaw", None)
                if pitch is None:
                    pitch = 0.0
                if yaw is None:
                    yaw = -180.0
                self.viewer.set_camera(pos, float(pitch), float(yaw))

        # CUDA Graph Storage
        self.cuda_graph: Optional[wp.Graph] = None

    def run(self):
        """Main entry point to start the simulation."""
        pbar = tqdm(
            total=self.num_segments,
            desc="Simulating",
        )

        if self.rendering_config.start_paused and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True

        try:
            segment_num = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    self._run_simulation_segment(segment_num)
                    segment_num += 1
                    pbar.update(1)
                self._render(segment_num)

                if self.rendering_config.vis_type == "gl":
                    wp.synchronize()
        finally:
            pbar.close()

            # Flush logs for any solver that supports save_logs()
            save_logs_fn = getattr(self.solver, "save_logs", None)
            if callable(save_logs_fn):
                # self.solver.events.print_timings()
                self.solver.save_logs()

            prof = getattr(self.solver, "profiler", None)
            if prof is not None and prof.enabled:
                if self.steps_per_segment != 1:
                    # Only fires in render mode where steps_per_segment
                    # is sized by render fps vs dt; in headless mode it
                    # is always 1.
                    print(
                        f"WARNING: profiler enabled but steps_per_segment="
                        f"{self.steps_per_segment}; only the LAST step in each "
                        "segment is timed. For accurate stats, match render "
                        "fps to dt or run headless."
                    )
                prof.print_summary()

            if self.rendering_config.vis_type == "usd":
                self.viewer.close()
                print(f"Rendering complete. Output saved to {self.rendering_config.usd_file}")

    def run_visualization(self):
        """Runs the visualization loop without advancing the physics simulation."""
        if not isinstance(self.viewer, newton.viewer.ViewerGL):
            print(
                "Error: run_visualization() only supports ViewerGL. Please set rendering.vis_type='gl'."
            )
            return

        print("Starting visualization mode (Physics paused)...")
        self.contacts = self.model.collide(self.current_state)

        while self.viewer.is_running():
            self.viewer.begin_frame(0.0)
            self.viewer.log_state(self.current_state)
            self.viewer.log_contacts(self.contacts, self.current_state)
            self.viewer.end_frame()

    def _render(self, segment_num: int):
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        self.viewer.end_frame()

    def _run_simulation_segment(self, segment_num: int):
        if self.use_cuda_graph:
            self._run_segment_with_graph(segment_num)
        else:
            self._run_segment_without_graph(segment_num)

    def _run_segment_without_graph(self, segment_num: int):
        if segment_num == 0:
            prewarm_fn = getattr(self.solver, "prewarm", None)
            if prewarm_fn is not None:
                print("INFO: Pre-warming neural solver history buffer (no-graph path)...")
                prewarm_fn(self.current_state, self.contacts, self.clock.dt)
        n_steps = self.steps_per_segment
        for step in range(n_steps):
            self._single_physics_step(step)

        # if isinstance(self.solver, AxionEngine):
        #     self.solver.events.record_timings()

    def _run_segment_with_graph(self, segment_num: int):
        if self.cuda_graph is None:
            self._capture_cuda_graphs()

        # Coarse segment timer lives on engine.profiling.segment_timing
        # for AxionEngine; for non-Axion solvers there's no profiling
        # config, so the timer is just disabled.
        segment_timing = bool(
            getattr(getattr(self.solver, "config", None), "profiling", None)
            and self.solver.config.profiling.segment_timing
        )
        if segment_timing:
            wp.synchronize()
            t0 = time.perf_counter()
            wp.capture_launch(self.cuda_graph)
            wp.synchronize()
            t1 = time.perf_counter()
            ms_per_step = (t1 - t0) * 1000 / self.steps_per_segment
            print(f"segment: {(t1 - t0) * 1000:.2f} ms total, ~{ms_per_step:.2f} ms/step")
        else:
            wp.capture_launch(self.cuda_graph)

        # Profiler hook: read back per-replay event timings. Only valid
        # when the captured graph contains exactly one engine.step (i.e.
        # steps_per_segment == 1); otherwise events get overwritten by
        # the unrolled copies and only the last copy's times survive.
        prof = getattr(self.solver, "profiler", None)
        if prof is not None and prof.enabled:
            prof.collect()

    def _capture_cuda_graphs(self):
        n_steps = self.steps_per_segment
        # Some solvers (GPTEngine / HybridGPTEngine with the TensorRT path)
        # need an eager prewarm pass to seed their history ring buffer
        # *before* capture so that the first captured step has a stable,
        # fully-populated input. AxionEngine and similar classic solvers
        # don't define `prewarm`, in which case we skip it.
        prewarm_fn = getattr(self.solver, "prewarm", None)
        if prewarm_fn is not None:
            print("INFO: Pre-warming neural solver history buffer...")
            prewarm_fn(self.current_state, self.contacts, self.clock.dt)
        print(f"INFO: Capturing CUDA Graph (steps={n_steps})...")
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph = capture.graph
