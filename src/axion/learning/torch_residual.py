import torch
import torch.nn as nn
import warp as wp
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.mechanics import integrate_body_pose_kernel
from axion.optim import FullSystemOperator

from axion.core.contacts import AxionContacts
from axion.core.linear_system import compute_linear_system


class AxionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        model: AxionModel,
        contacts: AxionContacts,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        full_A_op: FullSystemOperator,
        body_vel: torch.Tensor,
        constr_force: torch.Tensor,
    ):
        ctx.axion_args = (model, contacts, data, config, dims, full_A_op)

        # Reshape body_vel from (num_worlds, N_u) to (num_worlds, body_count, 6)
        # so wp.from_torch interprets it as spatial_vector dtype
        body_vel_reshaped = body_vel.reshape(dims.num_worlds, dims.body_count, 6)
        data.body_vel = wp.from_torch(body_vel_reshaped, dtype=wp.spatial_vector, requires_grad=False)

        # Copy into the underlying arrays so ConstraintView wrappers stay intact
        wp.copy(data._constr_force, wp.from_torch(constr_force, requires_grad=False))
        wp.copy(data._constr_force_prev_iter, wp.from_torch(constr_force.detach(), requires_grad=False))

        wp.launch(
            kernel=integrate_body_pose_kernel,
            dim=(dims.num_worlds, dims.body_count),
            inputs=[
                data.body_vel,
                data.body_pose_prev,
                model.body_com,
                data.dt,
            ],
            outputs=[
                data.body_pose,
            ],
            device=data.device,
        )

        compute_linear_system(model, contacts, data, config, dims)

        residual_out = wp.to_torch(data.res.full).clone()
        return residual_out

    @staticmethod
    def backward(ctx, grad_output):
        model, contacts, data, config, dims, full_A_op = ctx.axion_args

        # Split grad_output into dynamics (N_u) and constraint (N_c) parts
        grad_out_vel = grad_output[:, :dims.N_u].contiguous()
        grad_out_lam = grad_output[:, dims.N_u:].contiguous()

        # Convert to warp spatial_vector / float arrays
        grad_out_vel_wp = wp.from_torch(
            grad_out_vel.reshape(dims.num_worlds, dims.body_count, 6),
            dtype=wp.spatial_vector,
        )
        grad_out_lam_wp = wp.from_torch(grad_out_lam)

        # Allocate outputs
        out_vel_wp = wp.zeros((dims.num_worlds, dims.body_count), dtype=wp.spatial_vector)
        out_lam_wp = wp.zeros((dims.num_worlds, dims.num_constraints), dtype=wp.float32)

        # grad = A_full^T @ grad_output
        full_A_op.matvec_transpose(grad_out_vel_wp, grad_out_lam_wp, out_vel_wp, out_lam_wp)

        # Convert back to flat torch tensors
        grad_vel = wp.to_torch(out_vel_wp).reshape(dims.num_worlds, dims.N_u)
        grad_lam = wp.to_torch(out_lam_wp)

        return (
            None,  # grad for model
            None,  # grad for contacts
            None,  # grad for data
            None,  # grad for config
            None,  # grad for dims
            None,  # grad for full_A_op
            grad_vel,  # grad for body_vel
            grad_lam,  # grad for constr_force
        )


class AxionResidualLoss(nn.Module):
    def __init__(
        self,
        model: AxionModel,
        contacts: AxionContacts,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        full_A_op: FullSystemOperator,
    ):
        super().__init__()
        self.model = model
        self.contacts = contacts
        self.data = data
        self.config = config
        self.dims = dims
        self.full_A_op = full_A_op

    def forward(self, body_vel: torch.Tensor, constr_force: torch.Tensor):
        residual = AxionResidual.apply(
            self.model,
            self.contacts,
            self.data,
            self.config,
            self.dims,
            self.full_A_op,
            body_vel,
            constr_force,
        )

        loss = torch.sum(residual**2)
        return loss
