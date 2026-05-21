import numpy as np
import warp as wp
from axion.constraints import fill_control_constraint_body_idx_kernel
from axion.constraints import fill_joint_constraint_body_idx_kernel
from axion.tiled import TiledSqNorm

from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_dims import EngineDimensions
from .history_group import HistoryGroup
from .model import AxionModel


def _compute_linesearch_step_size_array(config: EngineConfig) -> wp.array:
    # --- 1. Conservative Steps (Logarithmic) ---
    # "I don't trust the solver, let's try tiny steps."
    steps_conservative = np.logspace(
        np.log10(config.linesearch.min_step),
        np.log10(config.linesearch.conservative_upper_bound),
        config.linesearch.conservative_step_count,
    )

    # --- 2. Optimistic Steps (Linear) ---
    # "I trust the Newton direction, let's check around 1.0."
    half_window = config.linesearch.optimistic_window / 2.0
    steps_optimistic = np.linspace(
        1.0 - half_window, 1.0 + half_window, config.linesearch.optimistic_step_count
    )

    # --- 3. Combine & Sort ---
    ls_steps_np = np.concatenate([steps_conservative, steps_optimistic])
    ls_steps_np.sort()

    # Force exact 1.0 to ensure the standard Newton step is tested
    closest_idx = np.argmin(np.abs(ls_steps_np - 1.0))
    ls_steps_np[closest_idx] = 1.0

    return wp.from_numpy(ls_steps_np, dtype=wp.float32)


class EngineData:
    def __init__(
        self,
        model: AxionModel,
        dims: EngineDimensions,
        config: EngineConfig,
        device: wp.Device,
        alloc_history_arrays: bool = False,
        alloc_grad_arrays: bool = True,
    ):
        self.device = device

        self.dt: float = None

        # --- Helper for concise allocation ---
        def _alloc(shape, dtype, requires_grad=False):
            assert isinstance(shape, tuple)

            batched_shape = (dims.num_worlds,) + shape
            return wp.zeros(batched_shape, dtype=dtype, device=device, requires_grad=requires_grad)

        def _alloc_buffer(buffer_size, array):
            return wp.zeros((buffer_size,) + array.shape, dtype=array.dtype, device=device)

        # =========================================================================
        # Body State Arrays
        # =========================================================================
        # External force
        self.ext_force = _alloc((dims.body_count,), wp.spatial_vector, alloc_grad_arrays)

        # State of bodies (q - position, u - velocity)
        self.body_pose = _alloc((dims.body_count,), wp.transform)
        self.body_vel = _alloc((dims.body_count,), wp.spatial_vector)

        # State at previous timestep
        self.body_pose_prev = _alloc((dims.body_count,), wp.transform, alloc_grad_arrays)
        self.body_vel_prev = _alloc((dims.body_count,), wp.spatial_vector, alloc_grad_arrays)

        # Actuation
        self.joint_target_pos = _alloc((dims.joint_dof_count,), wp.float32, alloc_grad_arrays)
        self.joint_target_vel = _alloc((dims.joint_dof_count,), wp.float32, alloc_grad_arrays)

        # =========================================================================
        # Constraint Arrays
        # =========================================================================
        self._constr_force = _alloc((dims.num_constraints,), wp.float32)
        self._constr_force_prev_iter = _alloc((dims.num_constraints,), wp.float32)
        self._constr_body_idx = _alloc((dims.num_constraints, 2), wp.int32)
        self._constr_active_mask = _alloc((dims.num_constraints,), wp.float32)

        self.constr_force = ConstraintView(self._constr_force, dims)
        self.constr_force_prev_iter = ConstraintView(self._constr_force_prev_iter, dims)
        self.constr_body_idx = ConstraintView(self._constr_body_idx, dims, axis=-2)
        self.constr_active_mask = ConstraintView(self._constr_active_mask, dims)

        # =========================================================================
        # Linear System Arrays
        # =========================================================================
        # Residual
        self._res = _alloc((dims.N_u + dims.num_constraints,), wp.float32, alloc_grad_arrays)
        self._res_spatial = _alloc((dims.body_count,), wp.spatial_vector, alloc_grad_arrays)

        self.res = SystemView(self._res, dims, self._res_spatial)

        # Efficiently stored values of sparse system matrix
        self.world_inv_inertia = _alloc((dims.body_count,), wp.mat33)
        # self.world_M = _alloc((dims.body_count,), SpatialInertia)
        # self.world_M_inv = _alloc((dims.body_count,), SpatialInertia)
        self._J_values = _alloc((dims.num_constraints, 2), wp.spatial_vector)
        self._C_values = _alloc((dims.num_constraints,), wp.float32)

        self.J_values = ConstraintView(self._J_values, dims, axis=-2)
        self.C_values = ConstraintView(self._C_values, dims)

        # Intermediate array for linearization
        self.JT_dconstr_force = _alloc((dims.body_count,), wp.spatial_vector)

        # The unknown arrays for the linear solve
        self.dbody_vel = _alloc((dims.body_count,), wp.spatial_vector)
        self._dconstr_force = _alloc((dims.num_constraints,), wp.float32)

        self.dconstr_force = ConstraintView(self._dconstr_force, dims)

        # The right-hand side of the Schur-Complement
        self.rhs = _alloc((dims.num_constraints,), wp.float32)

        # =========================================================================
        # Newton-Raphson (NR) Arrays
        # =========================================================================
        with wp.ScopedDevice(device):
            self.keep_running = wp.zeros(shape=(1,), dtype=wp.int32)
            self.iter_count = wp.zeros(shape=(1,), dtype=wp.int32)
            self.res_norm_sq = wp.zeros(shape=(dims.num_worlds,), dtype=wp.float32)

        self.tiled_sq_norm = TiledSqNorm(
            shape=self._res.shape,
            dtype=wp.float32,
            device=device,
        )

        # =========================================================================
        # Adjoint Arrays
        # =========================================================================
        if alloc_grad_arrays:
            # The adjoint vector
            self._w = _alloc((dims.N_u + dims.num_constraints,), wp.float32)
            self._w_spatial = _alloc((dims.body_count,), wp.spatial_vector)

            self.w = SystemView(self._w, dims, self._w_spatial)

            # The right-hand side of the Schur-Complement of the adjoint linear system
            self.adjoint_rhs = _alloc((dims.num_constraints,), wp.float32)

            self.body_pose_grad = _alloc((dims.body_count,), wp.transform)
            self.body_vel_grad = _alloc((dims.body_count,), wp.spatial_vector)

        # =========================================================================
        # Linesearch Arrays
        # =========================================================================
        if config.linesearch.enabled:
            step_count = config.linesearch.conservative_step_count
            step_count += config.linesearch.optimistic_step_count

            self.linesearch_step_size = _compute_linesearch_step_size_array(config)

            self.linesearch_body_pose = _alloc_buffer(step_count, self.body_pose)
            self.linesearch_body_vel = _alloc_buffer(step_count, self.body_vel)

            self._linesearch_constr_force = _alloc_buffer(step_count, self._constr_force)
            self.linesearch_constr_force = ConstraintView(self._linesearch_constr_force, dims)

            self._linesearch_res = _alloc_buffer(step_count, self._res)
            self._linesearch_res_spatial = _alloc_buffer(step_count, self._res_spatial)

            self.linesearch_res = SystemView(
                self._linesearch_res, dims, self._linesearch_res_spatial
            )

            with wp.ScopedDevice(device):
                self.linesearch_res_norm_sq = wp.zeros((step_count, dims.num_worlds), wp.float32)
                self.linesearch_minimal_index = wp.zeros((dims.num_worlds,), wp.int32)

            # Class for computing squared norm efficiently
            self.linesearch_tiled_res_sq_norm = TiledSqNorm(
                shape=self._linesearch_res.shape,
                dtype=wp.float32,
                device=device,
            )

        self.candidates = HistoryGroup(
            capacity=config.nr.max_iters,
            index_array=self.iter_count,
            device=device,
        )
        self.candidates_body_pose = self.candidates.register("body_pose", self.body_pose)
        self.candidates_body_vel = self.candidates.register("body_vel", self.body_vel)
        self._candidates_constr_force = self.candidates.register("constr_force", self._constr_force)
        self.candidates_constr_force = ConstraintView(self._candidates_constr_force, dims)
        self._candidates_res = self.candidates.register("candidates_res", self._res)
        self.candidates_res = SystemView(self._candidates_res, dims)
        self.candidates_res_norm_sq = self.candidates.register(
            "candidates_res_norm_sq", self.res_norm_sq
        )
        self.candidates_best_idx = wp.zeros((dims.num_worlds,), dtype=wp.int32)

        # =========================================================================
        # LOGGING: History Group
        # =========================================================================
        # We use a single HistoryGroup to capture all data at the end of each Newton iteration.
        if alloc_history_arrays:
            self.history = HistoryGroup(
                capacity=config.nr.max_iters, index_array=self.iter_count, device=device
            )

            # 2. PCR History (Snapshots of the inner solver state)
            with wp.ScopedDevice(device):
                self.pcr_iter_count = wp.zeros((1,), wp.int32)  # Placeholder source
                self.pcr_final_res_norm_sq = wp.zeros((dims.num_worlds,), wp.float32)
                self.pcr_res_norm_sq_history = wp.zeros(
                    (config.linear.max_iters + 1, dims.num_worlds), wp.float32
                )

            self.pcr_history_iter_count = self.history.register(
                "pcr_iter_count", self.pcr_iter_count
            )
            self.pcr_history_final_res_norm_sq = self.history.register(
                "pcr_final_res_norm_sq", self.pcr_final_res_norm_sq
            )
            self.pcr_history_res_norm_sq_history = self.history.register(
                "pcr_res_norm_sq_history", self.pcr_res_norm_sq_history
            )

            # 3. Linesearch (LS) History
            if config.linesearch.enabled:
                self.ls_history_step_size = self.history.register(
                    "ls_step_size", self.linesearch_step_size
                )
                self.ls_history_res_norm_sq = self.history.register(
                    "ls_res_norm", self.linesearch_res_norm_sq
                )
                self.ls_history_minimal_index = self.history.register(
                    "ls_min_index", self.linesearch_minimal_index
                )

        # =========================================================================
        # Init Kernels
        # =========================================================================
        wp.launch(
            kernel=fill_joint_constraint_body_idx_kernel,
            dim=(dims.num_worlds, dims.joint_count),
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_constraint_offsets,
            ],
            outputs=[
                self.constr_body_idx.j,
            ],
            device=device,
        )

        wp.launch(
            kernel=fill_control_constraint_body_idx_kernel,
            dim=(dims.num_worlds, dims.joint_count),
            inputs=[
                model.joint_parent,
                model.joint_child,
                model.joint_type,
                model.joint_dof_mode,
                model.joint_qd_start,
                model.control_constraint_offsets,
            ],
            outputs=[
                self.constr_body_idx.ctrl,
            ],
            device=device,
        )

    def zero_gradients(self):
        # self.ext_force.grad.zero_()
        # self.body_pose_prev.grad.zero_()
        # self.body_vel_prev.grad.zero_()
        #
        # self._res.grad.zero_()
        # self._res_spatial.grad.zero_()

        self.joint_target_pos.grad.zero_()
        self.joint_target_vel.grad.zero_()

    def save_iter_to_history(self):
        if self.history:
            self.history.capture()

    def save_state_to_candidates(self):
        self.candidates.capture()
