import numpy as np
import warp as wp
from newton import Model


@wp.kernel
def batch_joint_q_start_kernel(
    joint_count_per_world: wp.int32,
    joint_coord_count_per_world: wp.int32,
    joint_q_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_q_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_q_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    if joint_coord_count_per_world > 0:
        batched_joint_q_start[world_idx, slot] = (
            joint_q_start[joint_idx] % joint_coord_count_per_world
        )
    else:
        batched_joint_q_start[world_idx, slot] = 0


@wp.kernel
def batch_joint_qd_start_kernel(
    joint_count_per_world: wp.int32,
    joint_dof_count_per_world: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_qd_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    if joint_dof_count_per_world > 0:
        batched_joint_qd_start[world_idx, slot] = (
            joint_qd_start[joint_idx] % joint_dof_count_per_world
        )
    else:
        batched_joint_qd_start[world_idx, slot] = 0


@wp.kernel
def batch_joint_bodies_kernel(
    joint_count_per_world: wp.int32,
    body_count_per_world: wp.int32,
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_parent: wp.array(dtype=wp.int32, ndim=2),
    batched_joint_child: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()

    if joint_idx >= joint_parent.shape[0]:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    parent_idx = joint_parent[joint_idx]
    if parent_idx >= 0:
        batched_joint_parent[world_idx, slot] = parent_idx % body_count_per_world
    else:
        batched_joint_parent[world_idx, slot] = parent_idx

    child_idx = joint_child[joint_idx]
    if child_idx >= 0:
        batched_joint_child[world_idx, slot] = child_idx % body_count_per_world
    else:
        batched_joint_child[world_idx, slot] = child_idx


def compute_joint_constraint_offsets(joint_types: wp.array):
    """
    joint_types: numpy array of shape (num_worlds, num_joints)
    """

    constraint_count_map = np.array(
        [
            5,  # PRISMATIC = 0
            5,  # REVOLUTE  = 1
            3,  # BALL      = 2
            6,  # FIXED     = 3
            0,  # FREE      = 4
            1,  # DISTANCE  = 5
            6,  # D6        = 6
            0,  # CABLE     = 7
        ],
        dtype=np.int32,
    )

    joint_types_np = joint_types.numpy()  # (num_worlds, num_joints)
    # Map joint types → constraint counts
    constraint_counts = constraint_count_map[joint_types_np]  # (num_worlds, num_joints)

    # Total constraints for each batch
    total_constraints = constraint_counts.sum(axis=1)  # (num_worlds,)

    # Compute offsets per batch
    # For each batch: offsets[i, :] = cumsum(counts[i, :]) - counts[i, 0]
    constraint_offsets = np.zeros_like(constraint_counts)  # (num_worlds, num_joints)
    constraint_offsets[:, 1:] = np.cumsum(constraint_counts[:, :-1], axis=1)

    # Convert to wp.array (must flatten or provide device explicitly)
    constraint_offsets_wp = wp.array(
        constraint_offsets,
        dtype=wp.int32,
        device=joint_types.device,
    )

    return constraint_offsets_wp, total_constraints[0]


@wp.kernel
def count_control_constraints_kernel(
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    counts: wp.array(dtype=wp.int32, ndim=2),
):
    world_idx, joint_idx = wp.tid()
    j_type = joint_type[world_idx, joint_idx]
    count = 0
    if j_type == 1 or j_type == 0:
        qd_start = joint_qd_start[world_idx, joint_idx]
        mode = joint_dof_mode[world_idx, qd_start]
        if mode != 0:
            count = 1
    counts[world_idx, joint_idx] = count


def compute_control_constraint_offsets(
    joint_type: wp.array,
    joint_dof_mode: wp.array,
    joint_qd_start: wp.array,
):
    num_worlds = joint_type.shape[0]
    num_joints = joint_type.shape[1]
    counts = wp.zeros((num_worlds, num_joints), dtype=wp.int32, device=joint_type.device)

    wp.launch(
        kernel=count_control_constraints_kernel,
        dim=(num_worlds, num_joints),
        inputs=[joint_type, joint_dof_mode, joint_qd_start],
        outputs=[counts],
        device=joint_type.device,
    )

    counts_np = counts.numpy()

    offsets_np = np.zeros_like(counts_np)
    row_counts = counts_np[0]
    row_offsets = np.cumsum(np.concatenate(([0], row_counts[:-1])))
    total = np.sum(row_counts)
    offsets_np[:] = row_offsets
    offsets = wp.from_numpy(offsets_np, dtype=wp.int32, device=joint_type.device)

    return offsets, int(total)


class AxionModel:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.device = model.device
        self.num_worlds = model.world_count

        self.g_accel = model.gravity

        # Newton's flat layout (since v1.x): [globals | world_0 | world_1 | ...].
        # *_world_start[0] is where world 0 begins, i.e. the count of globals.
        # *_world_start[-1] is the total entity count.
        shape_starts_np = model.shape_world_start.numpy()
        body_starts_np = model.body_world_start.numpy()
        joint_starts_np = model.joint_world_start.numpy()

        self.num_global_shapes = int(shape_starts_np[0])
        self.num_global_bodies = int(body_starts_np[0])
        self.num_global_joints = int(joint_starts_np[0])

        # Globals are only wired up for shapes (e.g., a shared heightmap with no
        # body). Bodies and joints with shape_world=-1 would need the same
        # broadcast plumbing; not implemented yet.
        assert self.num_global_bodies == 0, (
            f"AxionModel does not yet support global bodies (got {self.num_global_bodies})."
        )
        assert self.num_global_joints == 0, (
            f"AxionModel does not yet support global joints (got {self.num_global_joints})."
        )

        W = model.world_count
        self.shape_count_per_world = (int(shape_starts_np[-1]) - self.num_global_shapes) // W
        self.body_count = (int(body_starts_np[-1]) - self.num_global_bodies) // W
        self.joint_count = (int(joint_starts_np[-1]) - self.num_global_joints) // W
        self.shape_count = self.shape_count_per_world + self.num_global_shapes
        self.joint_dof_count = model.joint_dof_count // W
        self.joint_coord_count = model.joint_coord_count // W

        with wp.ScopedDevice(self.device):
            self.body_com = model.body_com.reshape((W, -1))
            self.body_inertia = model.body_inertia.reshape((W, -1))
            self.body_inv_inertia = model.body_inv_inertia.reshape((W, -1))
            self.body_mass = model.body_mass.reshape((W, -1))
            self.body_inv_mass = model.body_inv_mass.reshape((W, -1))

            self.joint_X_c = model.joint_X_c.reshape((model.world_count, -1))
            self.joint_X_p = model.joint_X_p.reshape((model.world_count, -1))
            self.joint_axis = model.joint_axis.reshape((model.world_count, -1))
            self.joint_dof_dim = model.joint_dof_dim.reshape((model.world_count, -1))
            self.joint_dof_mode = model.joint_dof_mode.reshape((model.world_count, -1))
            self.joint_enabled = model.joint_enabled.reshape((model.world_count, -1))
            self.joint_target_ke = model.joint_target_ke.reshape((model.world_count, -1))
            self.joint_target_kd = model.joint_target_kd.reshape((model.world_count, -1))
            self.joint_target_ke.requires_grad = True
            self.joint_target_kd.requires_grad = True
            self.joint_compliance = model.joint_compliance.reshape((model.world_count, -1))
            self.joint_type = model.joint_type.reshape((model.world_count, -1))
            self.joint_q_start = wp.zeros((model.world_count, self.joint_count), dtype=wp.int32)
            self.joint_qd_start = wp.zeros((model.world_count, self.joint_count), dtype=wp.int32)
            self.joint_parent = wp.zeros((model.world_count, self.joint_count), dtype=wp.int32)
            self.joint_child = wp.zeros((model.world_count, self.joint_count), dtype=wp.int32)

        wp.launch(
            kernel=batch_joint_q_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_coord_count,
                model.joint_q_start,
            ],
            outputs=[
                self.joint_q_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_qd_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_dof_count,
                model.joint_qd_start,
            ],
            outputs=[
                self.joint_qd_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_bodies_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.body_count,
                model.joint_parent,
                model.joint_child,
            ],
            outputs=[
                self.joint_parent,
                self.joint_child,
            ],
            device=self.device,
        )

        # Shape arrays are (W, S+G) with the last G columns holding globals
        # broadcast across every world's row. Per-world contact lookups in the
        # downstream kernels then index uniformly into [0, S+G) without needing
        # to know which slot is global.
        self.shape_material_mu = self._broadcast_shape_array(model.shape_material_mu)
        self.shape_material_restitution = self._broadcast_shape_array(
            model.shape_material_restitution
        )
        self.shape_friction_axis_local = self._broadcast_shape_array(model.friction_axis_local)
        self.shape_mu_perp = self._broadcast_shape_array(model.mu_perp)
        self.shape_body = self._build_shape_body_array(model.shape_body)

        self.joint_constraint_offsets, self.num_joint_constraints = (
            compute_joint_constraint_offsets(
                self.joint_type,
            )
        )

        self.control_constraint_offsets, self.num_control_constraints = (
            compute_control_constraint_offsets(
                self.joint_type,
                self.joint_dof_mode,
                self.joint_qd_start,
            )
        )

    def _broadcast_shape_array(self, flat_arr: wp.array) -> wp.array:
        """Reshape a flat (G + W*S,) Newton shape array into (W, S+G), with the
        last G columns being the global section broadcast across every row."""
        W = self.num_worlds
        G = self.num_global_shapes
        S = self.shape_count_per_world
        if G == 0:
            return flat_arr.reshape((W, S))
        flat_np = flat_arr.numpy()
        per_world = flat_np[G:].reshape((W, S) + flat_np.shape[1:])
        globals_section = flat_np[:G]
        broadcast_shape = (W,) + globals_section.shape
        globals_broadcast = np.broadcast_to(globals_section, broadcast_shape)
        combined = np.concatenate([per_world, globals_broadcast], axis=1)
        return wp.array(combined, dtype=flat_arr.dtype, device=self.device)

    def _build_shape_body_array(self, shape_body_flat: wp.array) -> wp.array:
        """Build the (W, S+G) shape_body array. Per-world body indices are
        remapped from Newton's flat indexing to world-local; global shapes
        always have body=-1 (we assert num_global_bodies == 0 above, so a
        global shape attached to a body is rejected upstream)."""
        W = self.num_worlds
        G = self.num_global_shapes
        S = self.shape_count_per_world
        Bp = self.body_count

        sb_np = shape_body_flat.numpy()
        out = np.empty((W, S + G), dtype=np.int32)
        per_world = sb_np[G:].reshape(W, S)
        # Newton uses global body indices; remap into world-local [0, Bp).
        out[:, :S] = np.where(per_world >= 0, (per_world - self.num_global_bodies) % Bp, per_world)
        if G > 0:
            globals_ = sb_np[:G]
            assert np.all(globals_ < 0), (
                "Global shape attached to a body is not supported; expected shape_body=-1."
            )
            out[:, S:] = globals_  # all -1
        return wp.array(out, dtype=wp.int32, device=self.device)
