import h5py
import numpy as np
import warp as wp
from axion.core.data_views import ConstraintView
from axion.core.data_views import SystemView
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.history_group import create_save_to_buffer_kernel
from axion.core.history_group import get_or_create_save_kernel


class SimulationHDF5Logger:
    def __init__(
        self,
        num_steps: int,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        device: wp.Device,
    ):
        self.device = device
        self.dims = dims
        self.max_num_steps = num_steps
        self.dt = data.dt if data.dt is not None else 0.0
        self.config = config

        # List of operations to perform during capture: (source, dest, kernel)
        self._capture_ops = []

        # --- Helper for allocation and kernel registration ---
        def _register_log(name, source_array):
            if not isinstance(source_array, wp.array):
                return None

            # 1. Allocate buffer: (num_steps, ...) + source_shape
            dest_shape = (num_steps,) + source_array.shape
            dest_array = wp.zeros(dest_shape, dtype=source_array.dtype, device=device)

            # 2. Get the specific kernel for this array's dimensionality
            # Note: source_array can be 1D, 2D, or 3D.
            # get_or_create_save_kernel handles dim -> dim+1 mapping.
            kernel = get_or_create_save_kernel(source_array)

            # 3. Store operation
            self._capture_ops.append((source_array, dest_array, kernel))

            return dest_array

        # =========================================================================
        # 1. Body State
        # =========================================================================
        self.ext_force = _register_log("ext_force", data.ext_force)
        self.body_pose = _register_log("body_pose", data.body_pose)
        self.body_vel = _register_log("body_vel", data.body_vel)
        self.body_pose_prev = _register_log("body_pose_prev", data.body_pose_prev)
        self.body_vel_prev = _register_log("body_vel_prev", data.body_vel_prev)
        self.joint_target_pos = _register_log("joint_target_pos", data.joint_target_pos)
        self.joint_target_vel = _register_log("joint_target_vel", data.joint_target_vel)

        # =========================================================================
        # 2. Constraints
        # =========================================================================
        self._constr_force = _register_log("_constr_force", data._constr_force)
        self._constr_body_idx = _register_log("_constr_body_idx", data._constr_body_idx)
        self._constr_active_mask = _register_log("_constr_active_mask", data._constr_active_mask)

        self.constr_force = ConstraintView(self._constr_force, dims)
        self.constr_body_idx = ConstraintView(self._constr_body_idx, dims, axis=-2)
        self.constr_active_mask = ConstraintView(self._constr_active_mask, dims)

        # # =========================================================================
        # # 3. Contact Data
        # # =========================================================================
        # self.contact_body_a = _register_log("contact_body_a", data.contact_body_a)
        # self.contact_body_b = _register_log("contact_body_b", data.contact_body_b)
        # self.contact_point_a = _register_log("contact_point_a", data.contact_point_a)
        # self.contact_point_b = _register_log("contact_point_b", data.contact_point_b)
        # self.contact_thickness_a = _register_log("contact_thickness_a", data.contact_thickness_a)
        # self.contact_thickness_b = _register_log("contact_thickness_b", data.contact_thickness_b)
        # self.contact_dist = _register_log("contact_dist", data.contact_dist)
        # self.contact_friction_coeff = _register_log("contact_friction_coeff", data.contact_friction_coeff)
        # self.contact_restitution_coeff = _register_log("contact_restitution_coeff", data.contact_restitution_coeff)
        #
        # self.contact_basis_n_a = _register_log("contact_basis_n_a", data.contact_basis_n_a)
        # self.contact_basis_t1_a = _register_log("contact_basis_t1_a", data.contact_basis_t1_a)
        # self.contact_basis_t2_a = _register_log("contact_basis_t2_a", data.contact_basis_t2_a)
        # self.contact_basis_n_b = _register_log("contact_basis_n_b", data.contact_basis_n_b)
        # self.contact_basis_t1_b = _register_log("contact_basis_t1_b", data.contact_basis_t1_b)
        # self.contact_basis_t2_b = _register_log("contact_basis_t2_b", data.contact_basis_t2_b)

        # # =========================================================================
        # # 4. Linear System
        # # =========================================================================
        # self._res = _register_log("_res", data._res)
        # self._res_spatial = _register_log("_res_spatial", data._res_spatial)
        # self.res = SystemView(self._res, dims, self._res_spatial)
        #
        # self.world_M = _register_log("world_M", data.world_M)
        # self.world_M_inv = _register_log("world_M_inv", data.world_M_inv)
        # self._J_values = _register_log("_J_values", data._J_values)
        # self._C_values = _register_log("_C_values", data._C_values)
        #
        # self.J_values = ConstraintView(self._J_values, dims, axis=-2)
        # self.C_values = ConstraintView(self._C_values, dims)
        #
        # self.JT_dconstr_force = _register_log("JT_dconstr_force", data.JT_dconstr_force)
        # self.dbody_vel = _register_log("dbody_vel", data.dbody_vel)
        # self._dconstr_force = _register_log("_dconstr_force", data._dconstr_force)
        # self.dconstr_force = ConstraintView(self._dconstr_force, dims)
        # self.rhs = _register_log("rhs", data.rhs)

        # =========================================================================
        # 5. NR Stats
        # =========================================================================
        self.iter_count = _register_log("iter_count", data.iter_count)
        self.res_norm_sq = _register_log("res_norm_sq", data.res_norm_sq)
        self._res = _register_log("_res", data._res)

        # # =========================================================================
        # # 6. Adjoint (Conditional)
        # # =========================================================================
        # if hasattr(data, "_w"):
        #     self._w = _register_log("_w", data._w)
        #     self._w_spatial = _register_log("_w_spatial", data._w_spatial)
        #     self.w = SystemView(self._w, dims, self._w_spatial)
        #     self.adjoint_rhs = _register_log("adjoint_rhs", data.adjoint_rhs)
        #     self.body_pose_grad = _register_log("body_pose_grad", data.body_pose_grad)
        #     self.body_vel_grad = _register_log("body_vel_grad", data.body_vel_grad)

        # =========================================================================
        # 8. Detailed History (From EngineData HistoryGroup)
        # =========================================================================

        self.candidates_best_idx = _register_log("candidates_best_idx", data.candidates_best_idx)
        self.candidates_body_pose = _register_log("candidates_body_pose", data.candidates_body_pose)
        self.candidates_body_vel = _register_log("candidates_body_vel", data.candidates_body_vel)
        self._candidates_constr_force = _register_log(
            "_candidates_constr_force", data._candidates_constr_force
        )
        self._candidates_res = _register_log("_candidates_res", data._candidates_res)
        self.candidates_res_norm_sq = _register_log(
            "candidates_res_norm_sq", data.candidates_res_norm_sq
        )

        self.candidates_constr_force = ConstraintView(self._candidates_constr_force, dims)
        self.candidates_res = SystemView(self._candidates_res, dims)

        self.pcr_history_iter_count = _register_log(
            "pcr_history_iter_count", data.pcr_history_iter_count
        )
        self.pcr_history_final_res_norm_sq = _register_log(
            "pcr_history_final_res_norm_sq", data.pcr_history_final_res_norm_sq
        )
        self.pcr_history_res_norm_sq_history = _register_log(
            "pcr_history_res_norm_sq_history", data.pcr_history_res_norm_sq_history
        )

        if config.linesearch.enabled:
            self.ls_history_step_size = _register_log(
                "ls_history_step_size", data.ls_history_step_size
            )
            self.ls_history_res_norm_sq = _register_log(
                "ls_history_res_norm_sq", data.ls_history_res_norm_sq
            )
            self.ls_history_minimal_index = _register_log(
                "ls_history_minimal_index", data.ls_history_minimal_index
            )

    def capture_step(self, step_idx_array: wp.array, data: EngineData):
        """
        Launches kernels to copy current state into the history buffer.

        Args:
            step_idx_array: A Warp array (size 1) containing the current step index.
                            This allows the index to be computed/incremented on GPU.
        """
        for source, dest, kernel in self._capture_ops:
            wp.launch(
                kernel=kernel,
                dim=source.shape,
                inputs=[
                    source,
                    step_idx_array,
                    self.max_num_steps,
                ],
                outputs=[dest],
                device=self.device,
            )

    def save_to_hdf5(self, filepath: str):
        """
        Syncs data to CPU and saves to HDF5.
        """
        print(f"Saving simulation log to {filepath}...")
        wp.synchronize()

        with h5py.File(filepath, "w") as f:
            f.attrs["num_steps"] = self.max_num_steps
            f.attrs["num_worlds"] = self.dims.num_worlds
            f.attrs["dt"] = self.dt

            grp_dims = f.create_group("dims")
            grp_dims.attrs["N_u"] = self.dims.N_u
            grp_dims.attrs["N_j"] = self.dims.N_j
            grp_dims.attrs["N_ctrl"] = self.dims.N_ctrl
            grp_dims.attrs["N_n"] = self.dims.N_n
            grp_dims.attrs["N_f"] = self.dims.N_f

            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, wp.array):
                    print(f"  - Writing {attr_name} {attr_value.shape}...")
                    data_np = attr_value.numpy()
                    f.create_dataset(
                        attr_name, data=data_np, compression="gzip", compression_opts=4
                    )

        print("Save complete.")
