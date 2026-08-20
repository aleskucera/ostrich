import time
from abc import ABC
from typing import Optional

import newton
import warp as wp
from ostrich.core.engine import OstrichEngine
from ostrich.core.engine_config import EngineConfig
from ostrich.core.logging_config import LoggingConfig
from tqdm import tqdm

from .base_simulator import BaseSimulator
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@wp.kernel
def _blend_body_q_kernel(
    prev: wp.array(dtype=wp.transform),
    curr: wp.array(dtype=wp.transform),
    alpha: float,
    out: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    out[i] = wp.transform(
        wp.lerp(
            wp.transform_get_translation(prev[i]),
            wp.transform_get_translation(curr[i]),
            alpha,
        ),
        wp.quat_slerp(
            wp.transform_get_rotation(prev[i]),
            wp.transform_get_rotation(curr[i]),
            alpha,
        ),
    )


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
        self.viewer.set_world_offsets((20.0, 20.0, 0.0))

        # CUDA Graph Storage
        self.cuda_graph: Optional[wp.Graph] = None

        # Wall-clock deadline for the next rendered frame (real-time pacing)
        self._frame_due: Optional[float] = None

        # Pose interpolation: with dt above the frame duration one physics
        # step is too coarse to render directly, so each step is drawn as
        # several frames blended between its start and end pose.
        self._display_frames = self.clock.display_frames
        if self._display_frames > 1 and self.current_state.body_q is not None:
            self._body_q_true = self.current_state.body_q
            self._body_q_prev = wp.clone(self._body_q_true)
            self._body_q_blend = wp.clone(self._body_q_true)
        else:
            self._display_frames = 1

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
                if self.viewer.is_paused():
                    # Nothing moved, so there is no span to interpolate
                    # across; redraw the current pose once.
                    self._render_frame(segment_num)
                    continue

                if self._display_frames > 1:
                    self.current_state.body_q = self._body_q_true
                    wp.copy(self._body_q_prev, self._body_q_true)
                self._run_simulation_segment(segment_num)
                segment_num += 1
                pbar.update(1)

                for i in range(self._display_frames):
                    self._blend_pose((i + 1) / self._display_frames)
                    self._render_frame(segment_num)
        finally:
            pbar.close()

            if isinstance(self.solver, OstrichEngine):
                # self.solver.events.print_timings()
                self.solver.save_logs()
                if self.solver.profiler.enabled:
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
                    self.solver.profiler.print_summary()

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

    def _render_frame(self, segment_num: int):
        """Draws one display frame and holds it for its wall-clock share."""
        self._render(segment_num)
        if self.rendering_config.vis_type == "gl":
            wp.synchronize()
            self._pace_real_time()

    def _blend_pose(self, alpha: float):
        """Points ``current_state.body_q`` at the blended pose for this frame.

        ``alpha`` runs 0 (segment start pose) to 1 (the pose physics
        actually produced), so the last frame of every segment shows the
        true state and the buffer never feeds back into the solver.
        """
        if self._display_frames == 1:
            return
        if alpha >= 1.0:
            self.current_state.body_q = self._body_q_true
            return

        wp.launch(
            _blend_body_q_kernel,
            dim=len(self._body_q_prev),
            inputs=[self._body_q_prev, self._body_q_true, alpha],
            outputs=[self._body_q_blend],
            device=self.model.device,
        )
        self.current_state.body_q = self._body_q_blend

    def _pace_real_time(self):
        """Sleeps out whatever is left of the frame's sim-time budget.

        The loop otherwise runs as fast as the solver does, so a solver
        that beats real time plays the window back sped up. Called after
        ``wp.synchronize()`` so the GPU work is already accounted for.
        An overrun (solver slower than real time) resyncs instead of
        accumulating debt the loop would then try to sprint off.
        """
        if not self.rendering_config.real_time:
            return

        # sim seconds covered by one display frame
        budget = self.steps_per_segment * self.clock.dt / self._display_frames
        now = time.perf_counter()
        if self._frame_due is None or now > self._frame_due + budget:
            self._frame_due = now
        else:
            time.sleep(max(0.0, self._frame_due - now))
        self._frame_due += budget

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
        n_steps = self.steps_per_segment
        for step in range(n_steps):
            self._single_physics_step(step)

        # if isinstance(self.solver, OstrichEngine):
        #     self.solver.events.record_timings()

    def _run_segment_with_graph(self, segment_num: int):
        if self.cuda_graph is None:
            self._capture_cuda_graphs()

        # Coarse segment timer lives on engine.profiling.segment_timing
        # for OstrichEngine; for non-Ostrich solvers there's no profiling
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
        if isinstance(self.solver, OstrichEngine) and self.solver.profiler.enabled:
            self.solver.profiler.collect()

    def _capture_cuda_graphs(self):
        n_steps = self.steps_per_segment
        print(f"INFO: Capturing CUDA Graph (steps={n_steps})...")
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph = capture.graph
