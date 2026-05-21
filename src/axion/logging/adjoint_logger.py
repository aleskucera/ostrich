import h5py
import warp as wp

from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.history_group import create_save_to_buffer_kernel

# Avoid a circular import — PCRSolver is only used for type hints here.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axion.optim import PCRSolver


class AdjointHDF5Logger:
    """
    Logs the backward (implicit adjoint) pass per reverse-timestep.

    Captures:
      - Adjoint vector _w and adjoint RHS (the adjoint linear system)
      - Body pose / velocity gradients produced by the adjoint solve
      - Accumulated .grad arrays on all differentiable inputs

    Usage mirrors SimulationHDF5Logger:
        logger = AdjointHDF5Logger(num_steps, data, dims, config, device)
        # inside the reverse loop (step_idx counts DOWN):
        logger.capture_step(step_idx_array, data)
        logger.save_to_hdf5("adjoint_log.h5")
    """

    def __init__(
        self,
        num_steps: int,
        data: EngineData,
        dims: EngineDimensions,
        config: EngineConfig,
        device: wp.Device,
    ):
        self.device = device
        self.dims = dims
        self.max_num_steps = num_steps

        self._capture_ops = []

        def _register(source_array):
            if not isinstance(source_array, wp.array):
                return None
            dest_shape = (num_steps,) + source_array.shape
            dest = wp.zeros(dest_shape, dtype=source_array.dtype, device=device)
            kernel = create_save_to_buffer_kernel(source_array)
            self._capture_ops.append((source_array, dest, kernel))
            return dest

        # =====================================================================
        # 1. Adjoint linear system
        # =====================================================================
        # Full adjoint vector: same DOF layout as _res  (num_worlds, N_u + N_c)
        self._w = _register(data._w)

        # RHS of the Schur-complement adjoint solve  (num_worlds, N_c)
        self.adjoint_rhs = _register(data.adjoint_rhs)

        # =====================================================================
        # 2. Gradients produced by the adjoint solve (per step)
        # =====================================================================
        # These are overwritten each backward step, so we snapshot them here.
        self.body_pose_grad = _register(data.body_pose_grad)
        self.body_vel_grad  = _register(data.body_vel_grad)

        # =====================================================================
        # 3. PCR solver history for the adjoint linear solve
        # =====================================================================
        self.pcr_iter_count              = _register(data.pcr_iter_count)
        self.pcr_final_res_norm_sq       = _register(data.pcr_final_res_norm_sq)
        self.pcr_res_norm_sq_history     = _register(data.pcr_res_norm_sq_history)

        # =====================================================================
        # 4. Accumulated input gradients (grow monotonically during backward)
        # =====================================================================
        # Snapshotting these each step reveals how gradients flow back in time.
        self.body_pose_prev_grad    = _register(data.body_pose_prev.grad)
        self.body_vel_prev_grad     = _register(data.body_vel_prev.grad)
        self.joint_target_pos_grad  = _register(data.joint_target_pos.grad)
        self.joint_target_vel_grad  = _register(data.joint_target_vel.grad)
        self.ext_force_grad         = _register(data.ext_force.grad)

    def capture_step(self, step_idx_array: wp.array, data: EngineData):
        """
        Snapshot the current adjoint state into the history buffer.

        Args:
            step_idx_array: Warp array of shape (1,) holding the current
                            (reverse) step index, so captures land in the
                            correct slot without CPU synchronisation.
        """
        for source, dest, kernel in self._capture_ops:
            wp.launch(
                kernel=kernel,
                dim=source.shape,
                inputs=[source, step_idx_array, self.max_num_steps],
                outputs=[dest],
                device=self.device,
            )

    def save_to_hdf5(self, filepath: str):
        """Sync GPU data to CPU and write to HDF5."""
        print(f"Saving adjoint log to {filepath}...")
        wp.synchronize()

        with h5py.File(filepath, "w") as f:
            f.attrs["num_steps"] = self.max_num_steps
            f.attrs["num_worlds"] = self.dims.num_worlds

            # Dims needed to slice _w by constraint type (same layout as _res)
            grp_dims = f.create_group("dims")
            grp_dims.attrs["N_u"]    = self.dims.N_u
            grp_dims.attrs["N_j"]    = self.dims.N_j
            grp_dims.attrs["N_ctrl"] = self.dims.N_ctrl
            grp_dims.attrs["N_n"]    = self.dims.N_n
            grp_dims.attrs["N_f"]    = self.dims.N_f

            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, wp.array):
                    print(f"  - Writing {attr_name} {attr_value.shape}...")
                    f.create_dataset(
                        attr_name,
                        data=attr_value.numpy(),
                        compression="gzip",
                        compression_opts=4,
                    )

        print("Adjoint log saved.")
