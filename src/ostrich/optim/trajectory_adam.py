# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warp as wp


@wp.kernel
def adam_step_kernel_3d(
    g: wp.array(dtype=wp.float32, ndim=3),
    m: wp.array(dtype=wp.float32, ndim=3),
    v: wp.array(dtype=wp.float32, ndim=3),
    params: wp.array(dtype=wp.float32, ndim=3),
    dof_offset: int,
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
    clip_grad: float,
):
    """
    Applies the Adam optimization step to a specific slice of a 3D trajectory array.
    Grid dimensions should be: (total_sim_steps, num_worlds, num_optimized_dofs)
    """
    sim_step, world_idx, local_dof_idx = wp.tid()

    # Map the local thread index to the actual DOF index in the array
    dof_idx = dof_offset + local_dof_idx

    # Fetch current gradient
    grad = g[sim_step, world_idx, dof_idx]

    # Clamp raw gradient to prevent extreme spikes before they enter momentum
    if clip_grad > 0.0:
        grad = wp.clamp(grad, -clip_grad, clip_grad)

    # Fetch previous moments
    m_prev = m[sim_step, world_idx, dof_idx]
    v_prev = v[sim_step, world_idx, dof_idx]

    # Update biased moments
    m_new = beta1 * m_prev + (1.0 - beta1) * grad
    v_new = beta2 * v_prev + (1.0 - beta2) * grad * grad

    # Store updated moments
    m[sim_step, world_idx, dof_idx] = m_new
    v[sim_step, world_idx, dof_idx] = v_new

    # Bias correction (t is 0-indexed in our class, so we use t + 1.0)
    mhat = m_new / (1.0 - wp.pow(beta1, t + 1.0))
    vhat = v_new / (1.0 - wp.pow(beta2, t + 1.0))

    # Apply Adam update
    param_prev = params[sim_step, world_idx, dof_idx]
    params[sim_step, world_idx, dof_idx] = param_prev - lr * mhat / (wp.sqrt(vhat) + eps)


class TrajectoryAdam:
    """
    Adam Optimizer tailored for 3D Warp arrays (e.g., Trajectory buffers).
    Allows targeting specific DOF offsets so you don't update unactuated joints.
    """

    def __init__(
        self,
        params: wp.array,
        lr: float = 0.05,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-7,
        dof_offset: int = 0,
        num_dofs: int = None,
        clip_grad: float = 100.0,
    ):
        """
        Args:
            params: 3D warp array (steps, worlds, dofs) to optimize.
            lr: Learning rate.
            betas: (beta1, beta2) for momentum decay.
            eps: Numerical stability term.
            dof_offset: Starting index of the DOFs to optimize.
            num_dofs: Number of DOFs to optimize. If None, optimizes to the end of the array.
            clip_grad: Clamps raw gradients to [-clip_grad, clip_grad] before processing.
        """
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0
        self.dof_offset = dof_offset
        self.clip_grad = clip_grad

        # Allocate momentum buffers matching the exact shape of the parameters
        self.m = wp.zeros_like(params)
        self.v = wp.zeros_like(params)

        # Determine how many DOFs we are actually updating per world/step
        total_dofs_in_array = params.shape[2]
        if num_dofs is None:
            self.num_dofs = total_dofs_in_array - dof_offset
        else:
            self.num_dofs = num_dofs

    def step(self, grad: wp.array):
        """
        Applies a single optimization step.
        Args:
            grad: 3D warp array containing the gradients (e.g. self.trajectory.joint_target_vel.grad)
        """
        # Launch dimensions: (time_steps, worlds, optimized_dofs)
        dim = (self.params.shape[0], self.params.shape[1], self.num_dofs)

        wp.launch(
            kernel=adam_step_kernel_3d,
            dim=dim,
            inputs=[
                grad,
                self.m,
                self.v,
                self.params,
                self.dof_offset,
                self.lr,
                self.beta1,
                self.beta2,
                float(self.t),
                self.eps,
                self.clip_grad,
            ],
            device=self.params.device,
        )

        self.t += 1

    def reset_internal_state(self):
        """Resets the moment buffers and timestep to zero (useful between complete retrains)."""
        self.m.zero_()
        self.v.zero_()
        self.t = 0
