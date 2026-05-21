from abc import ABC
from enum import auto
from enum import Enum
from typing import Optional

import newton
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import EngineConfig
from axion.core.logging_config import LoggingConfig
from axion.core.types import JointMode
from tqdm import tqdm

from .base_simulator import BaseSimulator
from .base_simulator import RenderingConfig
from .base_simulator import SimulationConfig


@wp.kernel
def random_coords_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    pos_bound_min: wp.vec3,
    pos_bound_max: wp.vec3,
    seed: int,
):
    joint_idx = wp.tid()
    state = wp.rand_init(seed, joint_idx)

    j_type = joint_type[joint_idx]
    j_q_start = joint_q_start[joint_idx]

    # Get the DOF index to look up limits
    j_q_start = joint_q_start[joint_idx]

    if j_type == 0:  # Prismatic (1 DOF)
        lower = joint_limit_lower[j_q_start]
        upper = joint_limit_upper[j_q_start]
        joint_q[j_q_start] = wp.randf(state) * (upper - lower) + lower
        return

    elif j_type == 1:  # Revolute (1 DOF)
        lower = joint_limit_lower[j_q_start]
        upper = joint_limit_upper[j_q_start]
        joint_q[j_q_start] = wp.randf(state) * (upper - lower) + lower
        return

    elif j_type == 2:  # Ball
        axis_x = wp.randf(state) * 2.0 - 1.0
        axis_y = wp.randf(state) * 2.0 - 1.0
        axis_z = wp.randf(state) * 2.0 - 1.0
        axis = wp.normalize(wp.vec3(axis_x, axis_y, axis_z))
        angle = wp.randf(state) * 6.28318530718
        rot = wp.quat_from_axis_angle(axis, angle)

        # 3. Save Rotation
        joint_q[j_q_start + 0] = rot[0]
        joint_q[j_q_start + 1] = rot[1]
        joint_q[j_q_start + 2] = rot[2]
        joint_q[j_q_start + 3] = rot[3]
        return

    elif j_type == 3:  # Fixed
        return

    elif j_type == 4:  # Free (6 DOF, 7 floats)
        # 1. Randomize Translation
        joint_q[j_q_start + 0] = (
            wp.randf(state) * (pos_bound_max[0] - pos_bound_min[0]) + pos_bound_min[0]
        )
        joint_q[j_q_start + 1] = (
            wp.randf(state) * (pos_bound_max[1] - pos_bound_min[1]) + pos_bound_min[1]
        )
        joint_q[j_q_start + 2] = (
            wp.randf(state) * (pos_bound_max[2] - pos_bound_min[2]) + pos_bound_min[2]
        )

        # 2. Randomize Rotation
        axis_x = wp.randf(state) * 2.0 - 1.0
        axis_y = wp.randf(state) * 2.0 - 1.0
        axis_z = wp.randf(state) * 2.0 - 1.0
        axis = wp.normalize(wp.vec3(axis_x, axis_y, axis_z))
        angle = wp.randf(state) * 6.28318530718
        rot = wp.quat_from_axis_angle(axis, angle)

        # 3. Save Rotation
        joint_q[j_q_start + 3] = rot[0]
        joint_q[j_q_start + 4] = rot[1]
        joint_q[j_q_start + 5] = rot[2]
        joint_q[j_q_start + 6] = rot[3]


@wp.kernel
def random_velocities_kernel(
    body_qd: wp.array(dtype=wp.spatial_vector),
    lin_bound_min: float,
    lin_bound_max: float,
    ang_bound_min: float,
    ang_bound_max: float,
    seed: int,
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)

    # Generate random angular velocity (wx, wy, wz)
    wx = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min
    wy = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min
    wz = wp.randf(state) * (ang_bound_max - ang_bound_min) + ang_bound_min

    # Generate random linear velocity (vx, vy, vz)
    vx = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min
    vy = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min
    vz = wp.randf(state) * (lin_bound_max - lin_bound_min) + lin_bound_min

    rand_vel = wp.spatial_vector(wx, wy, wz, vx, vy, vz)

    # Forcefully overwrite both state arrays
    body_qd[tid] = rand_vel


@wp.kernel
def advance_step_kernel(step_buffer: wp.array(dtype=wp.int32)):
    # Safely increment the global step counter on the GPU
    step_buffer[0] = step_buffer[0] + 1


@wp.kernel
def random_joint_target_kernel(
    joint_target: wp.array(dtype=wp.float32),
    joint_type: wp.array(dtype=wp.int32),
    joint_dof_mode: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    target_lower_bound: wp.float32,
    target_upper_bound: wp.float32,
    seed: int,
    step_buffer: wp.array(dtype=wp.int32),
):
    joint_idx = wp.tid()
    step = step_buffer[0]  # <--- READ THE CURRENT STEP

    # Offset the random seed by the step counter so it's fresh every time
    state = wp.rand_init(seed, joint_idx + step * 65536)

    j_type = joint_type[joint_idx]
    j_qd_start = joint_qd_start[joint_idx]

    if j_type == 0:  # Prismatic (1 DOF)
        j_mode = joint_dof_mode[j_qd_start]
        if j_mode == JointMode.NONE:
            return
        joint_target[j_qd_start] = (
            wp.randf(state) * (target_upper_bound - target_lower_bound) + target_lower_bound
        )
        return

    elif j_type == 1:  # Revolute (1 DOF)
        j_mode = joint_dof_mode[j_qd_start]
        if j_mode == JointMode.NONE:
            return
        joint_target[j_qd_start] = (
            wp.randf(state) * (target_upper_bound - target_lower_bound) + target_lower_bound
        )
        return

    elif j_type == 2:  # Ball
        return

    elif j_type == 3:  # Fixed
        return

    elif j_type == 4:  # Free (6 DOF, 7 floats)
        return


@wp.kernel
def sync_prev_poses_kernel(
    pose: wp.array(dtype=wp.transform), pose_prev: wp.array(dtype=wp.transform)
):
    tid = wp.tid()
    pose_prev[tid] = pose[tid]


class StartupState(Enum):
    PRE_RESOLVE = auto()
    POST_RESOLVE = auto()
    RUNNING = auto()


class DatasetSimulator(BaseSimulator, ABC):
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
        self.viewer.set_world_offsets(
            (
                self.rendering_config.world_offset_x,
                self.rendering_config.world_offset_y,
                0.0,
            )
        )

        # CUDA Graph Storage
        self.cuda_graph: Optional[wp.Graph] = None
        self.global_step_buffer = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._startup_state = StartupState.PRE_RESOLVE
        if self.rendering_config.start_paused and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True
        else:
            self._startup_state = StartupState.RUNNING

    def _resolve_constraints(self):
        print("Resolving constraints...")
        for i in range(10):
            self.contacts = self.model.collide(self.current_state)
            self.solver.step(
                state_in=self.current_state,
                state_out=self.next_state,
                control=self.control,
                contacts=self.contacts,
                dt=self.clock.dt,
            )

            # Copy only positions
            wp.copy(self.current_state.body_q, self.next_state.body_q)
            self.current_state.body_qd.zero_()
            # self.next_state.body_qd.zero_()
            # wp.copy(self.next_state.body_qd, self.current_state.body_qd)

    def control_policy(self, current_state: newton.State):
        """
        Implements the control policy for the simulation.
        This method may be optionally overridden by any subclass.
        """
        # 1. Increment the GPU step counter
        wp.launch(
            kernel=advance_step_kernel,
            dim=1,
            inputs=[self.global_step_buffer],
            device=self.model.device,
        )

        # 2. Generate random targets using the new step counter
        wp.launch(
            kernel=random_joint_target_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.control.joint_target_vel,
                self.model.joint_type,
                self.model.joint_dof_mode,
                self.model.joint_qd_start,
                self.joint_target_lower_bound,
                self.joint_target_upper_bound,
                self.seed,
                self.global_step_buffer,
            ],
            device=self.model.device,
        )

    def run(self):
        """Main entry point to start the simulation."""

        pbar = tqdm(
            total=self.num_segments,
            desc="Simulating",
        )

        wp.launch(
            kernel=random_coords_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.current_state.joint_q,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.pos_min,
                self.pos_max,
                self.seed,
            ],
            device=self.model.device,
        )
        newton.eval_fk(
            self.model, self.current_state.joint_q, self.current_state.joint_qd, self.current_state
        )
        try:
            segment_num = 0
            while self.viewer.is_running():
                if not self.viewer.is_paused():
                    if self._startup_state == StartupState.PRE_RESOLVE:
                        self._resolve_constraints()
                        self._startup_state = StartupState.POST_RESOLVE
                        self.viewer._paused = True
                    elif self._startup_state == StartupState.POST_RESOLVE:
                        self._startup_state = StartupState.RUNNING
                        # Fall through to the simulation run.
                        self._run_simulation_segment(segment_num)
                        segment_num += 1
                        pbar.update(1)
                    elif self._startup_state == StartupState.RUNNING:
                        self._run_simulation_segment(segment_num)
                        segment_num += 1
                        pbar.update(1)

                self._render(segment_num)

                if self.rendering_config.vis_type == "gl":
                    wp.synchronize()
        finally:
            pbar.close()

            if isinstance(self.solver, AxionEngine):
                # self.solver.events.print_timings()
                self.solver.save_logs()
                if self.solver.profiler.enabled:
                    if self.steps_per_segment != 1:
                        # Only fires in render mode; headless is always 1.
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

    def _run_segment_with_graph(self, segment_num: int):
        if self.cuda_graph is None:
            self._capture_cuda_graphs()

        wp.capture_launch(self.cuda_graph)

        # See InteractiveSimulator for the profiler-hook rationale.
        if isinstance(self.solver, AxionEngine) and self.solver.profiler.enabled:
            self.solver.profiler.collect()

    def _capture_cuda_graphs(self):
        n_steps = self.steps_per_segment
        print(f"INFO: Capturing CUDA Graph (steps={n_steps})...")
        with wp.ScopedCapture() as capture:
            for i in range(n_steps):
                self._single_physics_step(i)
        self.cuda_graph = capture.graph
