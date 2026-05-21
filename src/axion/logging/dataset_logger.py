import h5py
import numpy as np
import warp as wp

from axion.core.contacts import AxionContacts
from axion.core.data_views import ConstraintView
from axion.core.data_views import SystemView
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.history_group import create_save_to_buffer_kernel
from axion.core.history_group import get_or_create_save_kernel
from axion.core.model import AxionModel


class DatasetHDF5Logger:
    def __init__(
        self,
        num_steps: int,
        model: AxionModel,
        data: EngineData,
        contacts: AxionContacts,
        config: EngineConfig,
        dims: EngineDimensions,
        device: wp.Device,
    ):
        self.device = device
        self.dims = dims
        self.model = model  # Store reference for zero-copy saving later
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
            kernel = create_save_to_buffer_kernel(source_array)

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

        # =========================================================================
        # 3. Contact Data
        # =========================================================================
        self.contact_count = _register_log("contact_count", contacts.contact_count)
        self.contact_point0 = _register_log("contact_point0", contacts.contact_point0)
        self.contact_point1 = _register_log("contact_point1", contacts.contact_point1)
        self.contact_normal = _register_log("contact_normal", contacts.contact_normal)
        self.contact_shape0 = _register_log("contact_shape0", contacts.contact_shape0)
        self.contact_shape1 = _register_log("contact_shape1", contacts.contact_shape1)
        self.contact_thickness0 = _register_log("contact_thickness0", contacts.contact_thickness0)
        self.contact_thickness1 = _register_log("contact_thickness1", contacts.contact_thickness1)

    def capture_step(self, step_idx_array: wp.array, data: EngineData):
        """Launches kernels to copy current dynamic state into the history buffer."""
        for source, dest, kernel in self._capture_ops:
            wp.launch(
                kernel=kernel,
                dim=source.shape,
                inputs=[source, step_idx_array, self.max_num_steps],
                outputs=[dest],
                device=self.device,
            )

    def save_to_hdf5(self, filepath: str):
        """Syncs data to CPU and saves to HDF5 using groups."""
        print(f"Saving simulation log to {filepath}...")
        wp.synchronize()

        with h5py.File(filepath, "w") as f:
            # Top-level metadata
            f.attrs["num_steps"] = self.max_num_steps
            f.attrs["dt"] = self.dt

            # Create HDF5 groups (folders)
            grp_data = f.create_group("data")
            grp_model = f.create_group("model")
            grp_dims = f.create_group("dims")

            # --- 1. Log Dynamic Data (to data/ folder) ---
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, wp.array):
                    data_np = attr_value.numpy()
                    grp_data.create_dataset(
                        attr_name, data=data_np, compression="gzip", compression_opts=4
                    )

            # --- 2. Log Static Model Data (to model/ folder) ---
            # Extract directly from the stored model reference
            for attr_name, attr_value in vars(self.model).items():
                if isinstance(attr_value, wp.array):
                    data_np = attr_value.numpy()
                    grp_model.create_dataset(
                        attr_name, data=data_np, compression="gzip", compression_opts=4
                    )

            # --- 3. Log Dimensions (to dims/ folder) ---
            # dir() + getattr() forces evaluation of @cached_property methods
            for attr_name in dir(self.dims):
                if not attr_name.startswith("_"):  # Skip private attributes/dunders
                    attr_value = getattr(self.dims, attr_name)

                    # Log only integers (ignoring slices, booleans, or methods)
                    if isinstance(attr_value, int) and not isinstance(attr_value, bool):
                        grp_dims.create_dataset(attr_name, data=attr_value)

        print("Save complete.")
