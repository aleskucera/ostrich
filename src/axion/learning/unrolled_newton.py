"""Unrolled Newton solver for differentiable simulation.

Chains K differentiable Newton steps, allowing gradient-based training
of neural networks that predict initial (body_vel, constr_force) guesses.
The loss is computed after K Newton iterations, so the network learns what
starting points lead to fast solver convergence.
"""

import torch
import warp as wp

from axion.core.contacts import AxionContacts
from axion.core.engine_config import EngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.core.model import AxionModel
from axion.core.residual import compute_residual
from axion.mechanics import integrate_body_pose_kernel
from axion.optim import PCRSolver, SystemOperator, JacobiPreconditioner

from axion.core.linear_system import (
    compute_linear_system,
    compute_dbody_qd_from_dbody_lambda,
)

from .differentiable_newton_step import DifferentiableNewtonStep
from .torch_residual_ad import AxionResidualAD


def unrolled_newton_loss(
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
    num_newton_steps: int = 2,
) -> torch.Tensor:
    """Run K differentiable Newton steps and return ||residual||^2 loss.

    Args:
        model, contacts, data, config, dims: Engine state (must have load_data called)
        pcr_solver, A_op, preconditioner: Solver infrastructure
        body_vel: (num_worlds, N_u) initial body velocities (from NN)
        constr_force: (num_worlds, num_constraints) initial constraint forces (from NN)
        num_newton_steps: Number of Newton iterations to unroll

    Returns:
        loss: scalar torch.Tensor (differentiable w.r.t. body_vel and constr_force)
    """
    v, lam = body_vel, constr_force

    for k in range(num_newton_steps):
        v, lam = DifferentiableNewtonStep.apply(
            model, contacts, data, config, dims,
            pcr_solver, A_op, preconditioner,
            v, lam,
        )

    # Compute residual at the final state
    residual = AxionResidualAD.apply(
        model, contacts, data, config, dims, v, lam,
    )

    loss = torch.sum(residual ** 2)
    return loss


def unrolled_newton_residual(
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
    num_newton_steps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run K differentiable Newton steps and return the final state + residual.

    Returns:
        body_vel_final: (num_worlds, N_u) body velocities after K steps
        constr_force_final: (num_worlds, num_constraints) constraint forces after K steps
        residual: (num_worlds, N_u + num_constraints) residual vector
    """
    v, lam = body_vel, constr_force

    for k in range(num_newton_steps):
        v, lam = DifferentiableNewtonStep.apply(
            model, contacts, data, config, dims,
            pcr_solver, A_op, preconditioner,
            v, lam,
        )

    residual = AxionResidualAD.apply(
        model, contacts, data, config, dims, v, lam,
    )

    return v, lam, residual


def detached_newton_loss(
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
    num_newton_steps: int = 2,
) -> torch.Tensor:
    """Run K Newton steps WITHOUT gradients, then compute ||residual||^2 WITH gradients.

    The Newton steps refine the NN prediction toward the solution (no gradient
    needed — Newton converges regardless). The loss gradient flows only through
    the final residual evaluation, avoiding vanishing gradients through
    contracting Newton iterations.

    Args:
        model, contacts, data, config, dims: Engine state (must have load_data called)
        pcr_solver, A_op, preconditioner: Solver infrastructure
        body_vel: (num_worlds, N_u) initial body velocities (from NN)
        constr_force: (num_worlds, num_constraints) initial constraint forces (from NN)
        num_newton_steps: Number of Newton iterations to run (detached)

    Returns:
        loss: scalar torch.Tensor (differentiable w.r.t. body_vel and constr_force)
    """
    # Run K Newton steps without gradient tracking
    v = body_vel.detach()
    lam = constr_force.detach()

    for k in range(num_newton_steps):
        # Write current state into engine
        v_reshaped = v.reshape(dims.num_worlds, dims.body_count, 6).contiguous()
        data.body_vel = wp.from_torch(v_reshaped, dtype=wp.spatial_vector, requires_grad=False)

        cf_wp = wp.from_torch(lam.contiguous(), requires_grad=False)
        wp.copy(data._constr_force, cf_wp)
        wp.copy(data._constr_force_prev_iter, cf_wp)

        # Integrate poses
        wp.launch(
            kernel=integrate_body_pose_kernel,
            dim=(dims.num_worlds, dims.body_count),
            inputs=[data.body_vel, data.body_pose_prev, model.body_com, data.dt],
            outputs=[data.body_pose],
            device=data.device,
        )

        # Linearize
        compute_linear_system(model, contacts, data, config, dims)
        preconditioner.update()

        # Solve A dλ = rhs
        data._dconstr_force.zero_()
        pcr_solver.solve(
            A=A_op, b=data.rhs, x=data.dconstr_force.full,
            preconditioner=preconditioner,
            iters=config.linear.max_iters,
            tol=config.linear.tol, atol=config.linear.atol,
        )
        dlam = wp.to_torch(data.dconstr_force.full).clone()

        # Recover dv
        compute_dbody_qd_from_dbody_lambda(model, data, config, dims)
        dv = wp.to_torch(data.dbody_vel).reshape(dims.num_worlds, dims.N_u).clone()

        # Update
        v = v + dv
        lam = lam + dlam

    # Now compute residual WITH gradients, re-attaching to the original inputs.
    # The key: v and lam are detached, but we express them as:
    #   v_final = body_vel + (v - body_vel).detach()
    #   lam_final = constr_force + (lam - constr_force).detach()
    # This gives identity gradient: ∂v_final/∂body_vel = I, ∂lam_final/∂constr_force = I
    v_final = body_vel + (v - body_vel.detach()).detach()
    lam_final = constr_force + (lam - constr_force.detach()).detach()

    residual = AxionResidualAD.apply(
        model, contacts, data, config, dims, v_final, lam_final,
    )

    return torch.sum(residual ** 2)
