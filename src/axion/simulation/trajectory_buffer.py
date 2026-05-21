import warp as wp
from axion.core.contacts import AxionContacts
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


@wp.kernel
def vel_grad_norm_sq_kernel(
    vel_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    norm_sq: wp.array(dtype=wp.float32),
):
    """Accumulate squared norm of velocity gradient into a scalar."""
    world_idx, body_idx = wp.tid()
    v = vel_grad[world_idx, body_idx]
    s = wp.dot(wp.spatial_top(v), wp.spatial_top(v)) + wp.dot(
        wp.spatial_bottom(v), wp.spatial_bottom(v)
    )
    wp.atomic_add(norm_sq, 0, s)


@wp.kernel
def pose_grad_norm_sq_kernel(
    pose_grad: wp.array(dtype=wp.transform, ndim=2),
    norm_sq: wp.array(dtype=wp.float32),
):
    """Accumulate squared norm of pose gradient into a scalar."""
    world_idx, body_idx = wp.tid()
    g = pose_grad[world_idx, body_idx]
    p = wp.transform_get_translation(g)
    r = wp.transform_get_rotation(g)
    s = wp.dot(p, p) + r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]
    wp.atomic_add(norm_sq, 0, s)


@wp.kernel
def compute_grad_scale_kernel(
    norm_sq: wp.array(dtype=wp.float32),
    target_norm: wp.float32,
    scale: wp.array(dtype=wp.float32),
):
    """Compute scale = target_norm / sqrt(norm_sq), or 1.0 if norm is tiny."""
    n = wp.sqrt(norm_sq[0])
    if n < 1.0e-15:
        scale[0] = 1.0
    else:
        scale[0] = target_norm / n


@wp.kernel
def scale_vel_grad_kernel(
    vel_grad: wp.array(dtype=wp.spatial_vector, ndim=2),
    scale: wp.array(dtype=wp.float32),
):
    """Scale velocity gradient in-place."""
    world_idx, body_idx = wp.tid()
    s = scale[0]
    v = vel_grad[world_idx, body_idx]
    vel_grad[world_idx, body_idx] = wp.spatial_vector(
        wp.spatial_top(v) * s, wp.spatial_bottom(v) * s
    )


@wp.kernel
def scale_pose_grad_kernel(
    pose_grad: wp.array(dtype=wp.transform, ndim=2),
    scale: wp.array(dtype=wp.float32),
):
    """Scale pose gradient in-place."""
    world_idx, body_idx = wp.tid()
    s = scale[0]
    g = pose_grad[world_idx, body_idx]
    p = wp.transform_get_translation(g)
    r = wp.transform_get_rotation(g)
    pose_grad[world_idx, body_idx] = wp.transform(
        p * s, wp.quat(r[0] * s, r[1] * s, r[2] * s, r[3] * s)
    )


@wp.kernel
def accumulate_pose_grad_kernel(
    src: wp.array(dtype=wp.transform, ndim=2),
    dest: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()
    val = src[world_idx, body_idx]
    
    # Explicitly add components since atomic_add on structs might be limited
    p_src = wp.transform_get_translation(val)
    r_src = wp.transform_get_rotation(val)
    
    # We can't easily atomic_add to a transform's components directly if it's an array of transforms.
    # But we can read, add, and write. Since this is a single-threaded write per body_idx, 
    # it doesn't need to be atomic if we ensure no other thread writes to the same (world, body).
    # In TrajectoryBuffer, each (world, body) is indeed handled by one thread in this kernel.
    
    curr = dest[world_idx, body_idx]
    p_curr = wp.transform_get_translation(curr)
    r_curr = wp.transform_get_rotation(curr)
    
    dest[world_idx, body_idx] = wp.transform(
        p_curr + p_src,
        wp.quat(r_curr[0] + r_src[0], r_curr[1] + r_src[1], r_curr[2] + r_src[2], r_curr[3] + r_src[3])
    )


class TrajectoryBuffer:
    def __init__(
        self,
        data: EngineData,
        contacts: AxionContacts,
        dims: EngineDimensions,
        num_steps: int,
        device,
    ):
        self.data = data
        self.contacts = contacts
        self.num_steps = num_steps
        self.dims = dims
        self.device = device

        def _alloc_buffer(
            source_array: wp.array,
            requires_grad: bool = False,
            add_one_slot: bool = False,
        ):
            if not isinstance(source_array, wp.array):
                return None

            if add_one_slot:
                dest_shape = (num_steps + 1,) + source_array.shape
            else:
                dest_shape = (num_steps,) + source_array.shape

            dest_array = wp.zeros(
                dest_shape,
                dtype=source_array.dtype,
                device=device,
                requires_grad=requires_grad,
            )
            return dest_array

        # =========================================================================
        # 1. Contact Data
        # =========================================================================
        self.target_body_pose = _alloc_buffer(data.body_pose, add_one_slot=True)
        self.target_body_vel = _alloc_buffer(data.body_vel, add_one_slot=True)

        # =========================================================================
        # 2. Body State
        # =========================================================================
        self.ext_force = _alloc_buffer(data.ext_force, True)
        self.body_pose = _alloc_buffer(data.body_pose, True, add_one_slot=True)
        self.body_vel = _alloc_buffer(data.body_vel, True, add_one_slot=True)
        self.joint_target_pos = _alloc_buffer(data.joint_target_pos, True)
        self.joint_target_vel = _alloc_buffer(data.joint_target_vel, True)

        # =========================================================================
        # 3. Constraints
        # =========================================================================
        self._constr_force = _alloc_buffer(data._constr_force)
        self._constr_force_prev_iter = _alloc_buffer(data._constr_force)

        # =========================================================================
        # 4. Contact Data
        # =========================================================================
        self.contact_count = _alloc_buffer(contacts.contact_count)
        self.contact_point0 = _alloc_buffer(contacts.contact_point0)
        self.contact_point1 = _alloc_buffer(contacts.contact_point1)
        self.contact_normal = _alloc_buffer(contacts.contact_normal)
        self.contact_shape0 = _alloc_buffer(contacts.contact_shape0)
        self.contact_shape1 = _alloc_buffer(contacts.contact_shape1)
        self.contact_thickness0 = _alloc_buffer(contacts.contact_thickness0)
        self.contact_thickness1 = _alloc_buffer(contacts.contact_thickness1)

        # =========================================================================
        # 5. Scratch buffers for gradient normalization
        # =========================================================================
        self._grad_norm_sq = wp.zeros(1, dtype=wp.float32, device=device)
        self._grad_scale = wp.zeros(1, dtype=wp.float32, device=device)

    def zero_grad(self):
        if self.body_pose.requires_grad:
            self.body_pose.grad.zero_()
        if self.body_vel.requires_grad:
            self.body_vel.grad.zero_()
        if self.ext_force.requires_grad:
            self.ext_force.grad.zero_()
        if self.joint_target_pos.requires_grad:
            self.joint_target_pos.grad.zero_()
        if self.joint_target_vel.requires_grad:
            self.joint_target_vel.grad.zero_()

    def save_target_step(self, step_idx: int, data: EngineData):
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        # 1. Handle Initial Conditions (Only on first step)
        if step_idx == 0:
            wp.copy(self.target_body_pose[0], data.body_pose_prev)
            wp.copy(self.target_body_vel[0], data.body_vel_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.target_body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.target_body_vel[step_idx + 1], data.body_vel)

    def save_step(self, step_idx: int, data: EngineData, contacts: AxionContacts):
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        if step_idx == 0:
            wp.copy(self.body_pose[0], data.body_pose_prev)
            wp.copy(self.body_vel[0], data.body_vel_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.body_vel[step_idx + 1], data.body_vel)

        # --- Inputs ---
        wp.copy(self.ext_force[step_idx], data.ext_force)
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

        # --- Lambdas ---
        wp.copy(self._constr_force[step_idx], data._constr_force)
        wp.copy(self._constr_force_prev_iter[step_idx], data._constr_force_prev_iter)

        # --- Contacts ---
        wp.copy(self.contact_count[step_idx], contacts.contact_count)
        wp.copy(self.contact_point0[step_idx], contacts.contact_point0)
        wp.copy(self.contact_point1[step_idx], contacts.contact_point1)
        wp.copy(self.contact_normal[step_idx], contacts.contact_normal)
        wp.copy(self.contact_shape0[step_idx], contacts.contact_shape0)
        wp.copy(self.contact_shape1[step_idx], contacts.contact_shape1)
        wp.copy(self.contact_thickness0[step_idx], contacts.contact_thickness0)
        wp.copy(self.contact_thickness1[step_idx], contacts.contact_thickness1)

    def load_step(self, step_idx: int, data: EngineData, contacts: AxionContacts):
        """
        Restores the state from the buffer into EngineData.
        Start State <- Index [step_idx]
        Result State <- Index [step_idx + 1]
        """
        # --- Body State ---
        wp.copy(data.body_pose, self.body_pose[step_idx + 1])  # Load Result
        wp.copy(data.body_pose_prev, self.body_pose[step_idx])  # Load Start

        wp.copy(data.body_vel, self.body_vel[step_idx + 1])
        wp.copy(data.body_vel_prev, self.body_vel[step_idx])

        wp.copy(data.body_pose_grad, self.body_pose.grad[step_idx + 1])
        wp.copy(data.body_vel_grad, self.body_vel.grad[step_idx + 1])

        wp.copy(data.ext_force, self.ext_force[step_idx])

        # --- Inputs ---
        wp.copy(data.joint_target_pos, self.joint_target_pos[step_idx])
        wp.copy(data.joint_target_vel, self.joint_target_vel[step_idx])

        # --- Lambdas ---
        wp.copy(data._constr_force, self._constr_force[step_idx])
        wp.copy(data._constr_force_prev_iter, self._constr_force_prev_iter[step_idx])

        # --- Contacts ---
        wp.copy(contacts.contact_count, self.contact_count[step_idx])
        wp.copy(contacts.contact_point0, self.contact_point0[step_idx])
        wp.copy(contacts.contact_point1, self.contact_point1[step_idx])
        wp.copy(contacts.contact_normal, self.contact_normal[step_idx])
        wp.copy(contacts.contact_shape0, self.contact_shape0[step_idx])
        wp.copy(contacts.contact_shape1, self.contact_shape1[step_idx])
        wp.copy(contacts.contact_thickness0, self.contact_thickness0[step_idx])
        wp.copy(contacts.contact_thickness1, self.contact_thickness1[step_idx])

    def save_gradients(self, step_idx: int, data: EngineData):
        """
        Saves gradients computed during a backward pass from EngineData into the buffer.

        Note: body_pose.grad is NOT overwritten here. The implicit backward pass
        absorbs pose gradients into the velocity adjoint (via dt * G^T * grad_q in
        compute_body_adjoint_init_kernel), so body_pose_prev.grad is not computed.
        The tape.backward values in body_pose.grad (direct loss sensitivities) must
        be preserved for earlier backward steps to use.
        """
        # --- Body Velocity ---
        wp.copy(self.body_vel.grad[step_idx], data.body_vel_prev.grad)

        # Forces and inputs are interval-based (index T)
        wp.copy(self.ext_force.grad[step_idx], data.ext_force.grad)

        # --- Inputs ---
        wp.copy(self.joint_target_pos.grad[step_idx], data.joint_target_pos.grad)
        wp.copy(self.joint_target_vel.grad[step_idx], data.joint_target_vel.grad)

    def save_pose_gradients(self, step_idx: int, data: EngineData):
        """
        Accumulates the propagated pose gradient from EngineData into the buffer.
        """
        wp.launch(
            kernel=accumulate_pose_grad_kernel,
            dim=(self.dims.num_worlds, self.dims.body_count),
            inputs=[data.body_pose_prev.grad],
            outputs=[self.body_pose.grad[step_idx]],
            device=self.device,
        )

    def normalize_gradients(self, step_idx: int, target_norm: float = 1.0):
        """Normalize velocity+pose gradients at step_idx to target_norm.

        Preserves gradient direction but prevents exponential magnitude decay.
        Fully GPU-resident (CUDA graph safe).
        """
        dim = (self.dims.num_worlds, self.dims.body_count)

        # 1. Compute ||grad||^2
        self._grad_norm_sq.zero_()
        wp.launch(
            kernel=vel_grad_norm_sq_kernel,
            dim=dim,
            inputs=[self.body_vel.grad[step_idx]],
            outputs=[self._grad_norm_sq],
            device=self.device,
        )
        wp.launch(
            kernel=pose_grad_norm_sq_kernel,
            dim=dim,
            inputs=[self.body_pose.grad[step_idx]],
            outputs=[self._grad_norm_sq],
            device=self.device,
        )

        # 2. Compute scale = target_norm / ||grad||
        wp.launch(
            kernel=compute_grad_scale_kernel,
            dim=(1,),
            inputs=[self._grad_norm_sq, target_norm],
            outputs=[self._grad_scale],
            device=self.device,
        )

        # 3. Scale both arrays in-place
        wp.launch(
            kernel=scale_vel_grad_kernel,
            dim=dim,
            inputs=[self.body_vel.grad[step_idx], self._grad_scale],
            outputs=[],
            device=self.device,
        )
        wp.launch(
            kernel=scale_pose_grad_kernel,
            dim=dim,
            inputs=[self.body_pose.grad[step_idx], self._grad_scale],
            outputs=[],
            device=self.device,
        )
