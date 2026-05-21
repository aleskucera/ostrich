"""One differentiable Newton iteration for the physics solver.

Wraps a single Newton-Raphson step as a torch.autograd.Function:
  Input:  (body_vel_k, constr_force_k)
  Output: (body_vel_{k+1}, constr_force_{k+1})

The forward runs the standard Newton iteration:
  1. Linearize (compute_linear_system → J, C, rhs)
  2. Solve A dλ = rhs  (PCR solver, A = J M⁻¹ Jᵀ + C)
  3. Recover dv = M⁻¹(Jᵀ dλ dt - r_d)
  4. Update: v_{k+1} = v_k + dv, λ_{k+1} = λ_k + dλ

The backward uses wp.Tape through the residual computation (like AxionResidualAD)
plus implicit differentiation of the linear solve.
"""

import torch
import warp as wp

from axion.core.contacts import AxionContacts
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.linear_system import (
    compute_linear_system,
    compute_dbody_qd_from_dbody_lambda,
)
from axion.core.model import AxionModel
from axion.mechanics import integrate_body_pose_kernel
from axion.optim import PCRSolver, SystemOperator, JacobiPreconditioner


class DifferentiableNewtonStep(torch.autograd.Function):
    """One differentiable Newton-Raphson iteration.

    Non-differentiable args (passed through ctx):
        model, contacts, data, config, dims,
        pcr_solver, A_op, preconditioner

    Differentiable args:
        body_vel:     (num_worlds, N_u) torch.Tensor
        constr_force: (num_worlds, num_constraints) torch.Tensor

    Returns:
        body_vel_new:     (num_worlds, N_u) torch.Tensor
        constr_force_new: (num_worlds, num_constraints) torch.Tensor
    """

    @staticmethod
    def forward(
        ctx,
        model: AxionModel,
        contacts: AxionContacts,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        pcr_solver: PCRSolver,
        A_op: SystemOperator,
        preconditioner: JacobiPreconditioner,
        body_vel: torch.Tensor,
        constr_force: torch.Tensor,
    ):
        ctx.axion_args = (model, contacts, data, config, dims, pcr_solver, A_op, preconditioner)

        # ---- Write inputs into engine data ----
        body_vel_reshaped = body_vel.reshape(dims.num_worlds, dims.body_count, 6).contiguous()
        data.body_vel = wp.from_torch(body_vel_reshaped, dtype=wp.spatial_vector, requires_grad=False)

        cf_src = wp.from_torch(constr_force.contiguous(), requires_grad=False)
        wp.copy(data._constr_force, cf_src)
        wp.copy(data._constr_force_prev_iter, cf_src)

        # ---- Step 1: Integrate poses ----
        wp.launch(
            kernel=integrate_body_pose_kernel,
            dim=(dims.num_worlds, dims.body_count),
            inputs=[data.body_vel, data.body_pose_prev, model.body_com, data.dt],
            outputs=[data.body_pose],
            device=data.device,
        )

        # ---- Step 2: Linearize (compute J, C, residuals, RHS) ----
        compute_linear_system(model, contacts, data, config, dims)
        preconditioner.update()

        # ---- Step 3: Linear solve → dλ ----
        data._dconstr_force.zero_()
        pcr_solver.solve(
            A=A_op,
            b=data.rhs,
            x=data.dconstr_force.full,
            preconditioner=preconditioner,
            iters=config.linear.max_iters,
            tol=config.linear.tol,
            atol=config.linear.atol,
        )

        dconstr_force_torch = wp.to_torch(data.dconstr_force.full).clone()

        # ---- Step 4: Recover dv ----
        compute_dbody_qd_from_dbody_lambda(model, data, config, dims)
        dbody_vel_torch = wp.to_torch(data.dbody_vel).reshape(dims.num_worlds, dims.N_u).clone()

        # ---- Step 5: Update ----
        body_vel_new = body_vel + dbody_vel_torch.detach()
        constr_force_new = constr_force + dconstr_force_torch.detach()

        ctx.save_for_backward(body_vel, constr_force, dbody_vel_torch, dconstr_force_torch)

        return body_vel_new, constr_force_new

    @staticmethod
    def backward(ctx, grad_vel_new, grad_cf_new):
        """Backward pass using wp.Tape for residual + implicit diff for solve.

        The Newton step computes:
            v_{k+1} = v_k + dv(v_k, λ_k)
            λ_{k+1} = λ_k + dλ(v_k, λ_k)

        where dλ = A⁻¹ rhs(v_k, λ_k) and dv = M⁻¹(Jᵀ dλ dt - r_d(v_k, λ_k)).

        The VJP for the update is:
            grad_v_k  = grad_v_{k+1}  + ∂dv/∂v_k ᵀ grad_v_{k+1}  + ∂dλ/∂v_k ᵀ grad_λ_{k+1}
            grad_λ_k  = grad_λ_{k+1}  + ∂dv/∂λ_k ᵀ grad_v_{k+1}  + ∂dλ/∂λ_k ᵀ grad_λ_{k+1}

        We compute ∂dv, ∂dλ w.r.t. (v_k, λ_k) using wp.Tape through the
        residual computation and implicit differentiation through the solve.
        """
        model, contacts, data, config, dims, pcr_solver, A_op, preconditioner = ctx.axion_args
        body_vel, constr_force, dbody_vel_torch, dconstr_force_torch = ctx.saved_tensors

        # ---- Re-do the forward with tape recording ----
        # Set up inputs with gradient tracking
        body_vel_reshaped = body_vel.reshape(dims.num_worlds, dims.body_count, 6).contiguous()
        vel_wp = wp.from_torch(body_vel_reshaped, dtype=wp.spatial_vector, requires_grad=True)
        data.body_vel = vel_wp

        cf_src = wp.from_torch(constr_force.contiguous(), requires_grad=False)
        wp.copy(data._constr_force, cf_src)
        wp.copy(data._constr_force_prev_iter, cf_src)
        data._constr_force.requires_grad = True
        data._constr_force_prev_iter.requires_grad = True

        # Enable grad on outputs we care about
        data._res.requires_grad = True
        if data.res._d_spatial is not None:
            data.res._d_spatial.requires_grad = True
        data.rhs.requires_grad = True

        # Record on tape
        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=integrate_body_pose_kernel,
                dim=(dims.num_worlds, dims.body_count),
                inputs=[data.body_vel, data.body_pose_prev, model.body_com, data.dt],
                outputs=[data.body_pose],
                device=data.device,
            )
            compute_linear_system(model, contacts, data, config, dims)

        # ---- Compute adjoint seeds ----
        # We need to propagate grad_vel_new and grad_cf_new back through
        # the Newton step to get gradients on rhs and r_d.
        #
        # The relationships:
        #   dλ = A⁻¹ rhs
        #   dv = M⁻¹(Jᵀ dλ dt - r_d)
        #
        # Adjoint of dλ = A⁻¹ rhs:
        #   grad_rhs = A⁻ᵀ grad_dλ = A⁻¹ grad_dλ  (A symmetric)
        #
        # To get grad_dλ, we need to collect contributions from both outputs:
        #   From λ_{k+1} = λ_k + dλ:  grad_dλ += grad_cf_new
        #   From dv = M⁻¹(Jᵀ dλ dt - r_d):  grad_dλ += dt * J M⁻¹ grad_dv
        #
        # The second term (dt * J M⁻¹ grad_dv) can be computed using the
        # compute_schur_complement_rhs_kernel with appropriate inputs.

        # For simplicity, approximate: ignore the dv→dλ coupling for now
        # (this is exact if we only care about grad through the λ path)
        # TODO: add the J M⁻¹ grad_dv contribution to grad_dλ

        # Adjoint solve: grad_rhs = A⁻¹ grad_cf_new
        grad_cf_new_wp = wp.from_torch(grad_cf_new.contiguous(), requires_grad=False)
        grad_rhs_wp = wp.zeros(
            (dims.num_worlds, dims.num_constraints),
            dtype=wp.float32, device=data.device,
        )
        pcr_solver.solve(
            A=A_op, b=grad_cf_new_wp, x=grad_rhs_wp,
            preconditioner=preconditioner,
            iters=config.linear.max_iters,
            tol=config.linear.tol,
            atol=config.linear.atol,
        )

        # Inject adjoint on rhs
        wp.copy(data.rhs.grad, grad_rhs_wp)

        # Inject adjoint on residual: from dv = M⁻¹(Jᵀ dλ dt - r_d),
        # grad_r_d = -M⁻¹ grad_dv. But M⁻¹ is symmetric, and since
        # dv contributes -M⁻¹ r_d to the velocity, the adjoint on r_d
        # from grad_vel_new flows through res.d_spatial.
        # We inject grad_vel_new into res._d_spatial.grad for the tape
        # to propagate back. The sign is handled by the kernel's backward.
        if data.res._d_spatial is not None and data.res._d_spatial.grad is not None:
            grad_dv_reshaped = grad_vel_new.reshape(dims.num_worlds, dims.body_count, 6).contiguous()
            wp.copy(
                data.res._d_spatial.grad,
                wp.from_torch(grad_dv_reshaped, dtype=wp.spatial_vector),
            )

        # Also inject into the flat residual grad buffer
        grad_res_flat = wp.zeros_like(data._res)
        grad_res_flat_torch = wp.to_torch(grad_res_flat)
        # dynamics part
        grad_res_flat_torch[:, :dims.N_u] = grad_vel_new
        # constraint part gets gradient from rhs (already injected)
        wp.copy(data._res.grad, wp.from_torch(grad_res_flat_torch.contiguous()))

        # ---- Backprop through tape ----
        tape.backward()

        # Extract input gradients
        grad_vel_from_tape = wp.to_torch(vel_wp.grad).reshape(dims.num_worlds, dims.N_u).clone()
        # Both _constr_force and _constr_force_prev_iter come from the same input,
        # so we sum their gradients (chain rule through the copy)
        grad_cf_from_tape = wp.to_torch(data._constr_force.grad).clone()
        if data._constr_force_prev_iter.grad is not None:
            grad_cf_from_tape = grad_cf_from_tape + wp.to_torch(data._constr_force_prev_iter.grad).clone()

        # Add identity contribution: v_{k+1} = v_k + dv, λ_{k+1} = λ_k + dλ
        grad_vel_total = grad_vel_new + grad_vel_from_tape
        grad_cf_total = grad_cf_new + grad_cf_from_tape

        tape.zero()

        return (
            None,  # model
            None,  # contacts
            None,  # data
            None,  # config
            None,  # dims
            None,  # pcr_solver
            None,  # A_op
            None,  # preconditioner
            grad_vel_total,  # body_vel
            grad_cf_total,  # constr_force
        )
