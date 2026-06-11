"""Differentiable linear solve via implicit differentiation.

Wraps the PCR solver (A x = b) as a torch.autograd.Function.
Since A = J M⁻¹ Jᵀ + C is symmetric positive semi-definite,
the adjoint solve is: ∂L/∂b = A⁻¹ ∂L/∂x (same operator, same solver).
"""

import torch
import warp as wp

from ostrich.optim import PCRSolver
from ostrich.optim import JacobiPreconditioner
from ostrich.optim import SystemOperator


class DifferentiableLinearSolve(torch.autograd.Function):
    """Differentiable wrapper around PCR linear solve: x = A⁻¹ b.

    Forward: solve A x = b using PCR.
    Backward: solve A g = ∂L/∂x using the same PCR (A is symmetric).

    Args (forward):
        pcr_solver: PCRSolver instance
        A_op: SystemOperator (Schur complement A = J M⁻¹ Jᵀ + C)
        preconditioner: JacobiPreconditioner
        rhs: (num_worlds, num_constraints) torch.Tensor — the right-hand side b
        max_linear_iters: int — PCR iteration budget
        linear_tol: float — PCR convergence tolerance
        linear_atol: float — PCR absolute tolerance

    Returns:
        x: (num_worlds, num_constraints) torch.Tensor — the solution
    """

    @staticmethod
    def forward(
        ctx,
        pcr_solver: PCRSolver,
        A_op: SystemOperator,
        preconditioner: JacobiPreconditioner,
        rhs: torch.Tensor,
        max_linear_iters: int,
        linear_tol: float,
        linear_atol: float,
    ):
        ctx.solver_args = (pcr_solver, A_op, preconditioner, max_linear_iters, linear_tol, linear_atol)

        num_worlds, num_constraints = rhs.shape

        # Convert RHS to warp
        rhs_wp = wp.from_torch(rhs.contiguous(), requires_grad=False)

        # Allocate solution buffer (zero-initialized = initial guess x=0)
        x_wp = wp.zeros((num_worlds, num_constraints), dtype=wp.float32, device=rhs_wp.device)

        # Solve A x = b
        pcr_solver.solve(
            A=A_op,
            b=rhs_wp,
            x=x_wp,
            preconditioner=preconditioner,
            iters=max_linear_iters,
            tol=linear_tol,
            atol=linear_atol,
        )

        # Convert solution to torch
        x_torch = wp.to_torch(x_wp).clone()
        return x_torch

    @staticmethod
    def backward(ctx, grad_output):
        pcr_solver, A_op, preconditioner, max_linear_iters, linear_tol, linear_atol = ctx.solver_args

        num_worlds, num_constraints = grad_output.shape

        # Adjoint solve: g = A⁻¹ (∂L/∂x)
        # Since A is symmetric: A⁻ᵀ = A⁻¹, so we solve A g = grad_output
        grad_wp = wp.from_torch(grad_output.contiguous(), requires_grad=False)

        g_wp = wp.zeros((num_worlds, num_constraints), dtype=wp.float32, device=grad_wp.device)

        pcr_solver.solve(
            A=A_op,
            b=grad_wp,
            x=g_wp,
            preconditioner=preconditioner,
            iters=max_linear_iters,
            tol=linear_tol,
            atol=linear_atol,
        )

        grad_rhs = wp.to_torch(g_wp).clone()

        return (
            None,  # pcr_solver
            None,  # A_op
            None,  # preconditioner
            grad_rhs,  # rhs
            None,  # max_linear_iters
            None,  # linear_tol
            None,  # linear_atol
        )
