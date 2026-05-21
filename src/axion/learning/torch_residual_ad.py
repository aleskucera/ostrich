"""AxionResidual with exact gradients via warp autodiff (wp.Tape)."""

import torch
import warp as wp
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.mechanics import integrate_body_pose_kernel

from axion.core.contacts import AxionContacts
from axion.core.residual import compute_residual


class AxionResidualAD(torch.autograd.Function):
    """Differentiable residual evaluation using warp's tape-based autodiff.

    Unlike AxionResidual which approximates the backward with A_full^T,
    this uses wp.Tape to get exact gradients through all nonlinear
    constraint kernels (contacts, friction, joints, etc.).
    """

    @staticmethod
    def forward(
        ctx,
        model: AxionModel,
        contacts: AxionContacts,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        body_vel: torch.Tensor,
        constr_force: torch.Tensor,
    ):
        ctx.axion_args = (model, contacts, data, config, dims)

        # Convert body_vel to warp spatial_vector with gradient tracking
        body_vel_reshaped = body_vel.reshape(dims.num_worlds, dims.body_count, 6).contiguous()
        vel_wp = wp.from_torch(body_vel_reshaped, dtype=wp.spatial_vector, requires_grad=True)
        data.body_vel = vel_wp

        # For constr_force: write directly into the engine's _constr_force buffer
        # and enable gradient tracking on it so the tape captures it
        cf_src = wp.from_torch(constr_force.contiguous(), requires_grad=False)
        wp.copy(data._constr_force, cf_src)
        wp.copy(data._constr_force_prev_iter, cf_src)
        data._constr_force.requires_grad = True
        data._constr_force_prev_iter.requires_grad = True

        # Enable grad on outputs
        data._res.requires_grad = True
        if data.res._d_spatial is not None:
            data.res._d_spatial.requires_grad = True

        # Record forward pass on tape
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=integrate_body_pose_kernel,
                dim=(dims.num_worlds, dims.body_count),
                inputs=[data.body_vel, data.body_pose_prev, model.body_com, data.dt],
                outputs=[data.body_pose],
                device=data.device,
            )
            compute_residual(model, contacts, data, config, dims)

        residual_out = wp.to_torch(data.res.full).clone()

        # Save tape and tracked arrays for backward
        ctx.tape = tape
        ctx.vel_wp = vel_wp

        return residual_out

    @staticmethod
    def backward(ctx, grad_output):
        model, contacts, data, config, dims = ctx.axion_args
        tape = ctx.tape
        vel_wp = ctx.vel_wp

        # Inject grad_output as adjoint seed on residual outputs
        grad_out_wp = wp.from_torch(grad_output.contiguous())
        wp.copy(data._res.grad, grad_out_wp)

        # Also inject into the spatial dynamics buffer
        if data.res._d_spatial is not None:
            grad_dyn = grad_output[:, :dims.N_u].contiguous()
            grad_dyn_reshaped = grad_dyn.reshape(dims.num_worlds, dims.body_count, 6)
            wp.copy(
                data.res._d_spatial.grad,
                wp.from_torch(grad_dyn_reshaped, dtype=wp.spatial_vector),
            )

        # Backpropagate through the tape
        tape.backward()

        # Extract gradients — both _constr_force and _constr_force_prev_iter
        # come from the same input, so sum their gradients
        grad_vel = wp.to_torch(vel_wp.grad).reshape(dims.num_worlds, dims.N_u).clone()
        grad_cf = wp.to_torch(data._constr_force.grad).clone()
        if data._constr_force_prev_iter.grad is not None:
            grad_cf = grad_cf + wp.to_torch(data._constr_force_prev_iter.grad).clone()

        # Zero the tape to avoid accumulation on next call
        tape.zero()

        return (
            None,  # grad for model
            None,  # grad for contacts
            None,  # grad for data
            None,  # grad for config
            None,  # grad for dims
            grad_vel,  # grad for body_vel
            grad_cf,  # grad for constr_force
        )
