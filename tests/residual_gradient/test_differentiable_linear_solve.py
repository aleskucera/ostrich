"""Gradient verification for DifferentiableLinearSolve.

Tests that the implicit differentiation backward (adjoint solve)
matches finite differences: ∂x/∂b ≈ A⁻¹ for the system A x = b.
"""

import newton
import numpy as np
import torch
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig, LinearSolverConfig, NewtonRaphsonConfig
from axion.core.logging_config import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.learning.differentiable_linear_solve import DifferentiableLinearSolve
from axion.optim import JacobiPreconditioner, SystemLinearData, SystemOperator

wp.init()


def build_scene(num_worlds: int = 1) -> newton.Model:
    """Build a simple scene with contacts for testing."""
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.05
    builder.add_ground_plane()

    # Add a box sitting on the ground (creates contact constraints)
    body1 = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
    )
    builder.add_shape_box(
        body=body1, hx=0.5, hy=0.5, hz=0.5,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, mu=0.5),
    )

    return builder.finalize_replicated(num_worlds=num_worlds, gravity=-9.81)


def setup_engine_at_timestep(model, config=None):
    """Create engine, step once to populate contacts, return engine ready for linear solve."""
    if config is None:
        config = AxionEngineConfig(
            nr=NewtonRaphsonConfig(max_iters=10),
            linear=LinearSolverConfig(max_iters=200),
        )

    engine = config.create_engine(model=model, sim_steps=5, logging_config=LoggingConfig())

    state_cur = model.state()
    state_next = model.state()
    control = model.control()
    dt = 0.01

    # Step a few times to get objects into contact
    for _ in range(3):
        contacts = model.collide(state_cur)
        engine.step(state_cur, state_next, control, contacts, dt)
        state_cur, state_next = state_next, state_cur

    # Now do one more step: load_data + compute_linear_system to populate J, C, rhs
    contacts = model.collide(state_cur)
    engine.load_data(state_cur, control, contacts, dt)

    # Initialize body_vel and constr_force
    wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()

    # Integrate poses
    from axion.mechanics import integrate_body_pose_kernel
    wp.launch(
        kernel=integrate_body_pose_kernel,
        dim=(engine.dims.num_worlds, engine.dims.body_count),
        inputs=[engine.data.body_vel, engine.data.body_pose_prev, engine.axion_model.body_com, engine.data.dt],
        outputs=[engine.data.body_pose],
        device=engine.data.device,
    )

    # Compute linear system (populates J, C, rhs)
    from axion.core.linear_system import compute_linear_system
    compute_linear_system(
        engine.axion_model, engine.axion_contacts, engine.data, engine.config, engine.dims
    )

    # Update preconditioner
    engine.preconditioner.update()

    return engine, state_cur, state_next, control


def test_forward_solves_correctly():
    """Verify that DifferentiableLinearSolve produces the same result as direct PCR."""
    print("\n=== Test: Forward solve correctness ===")

    model = build_scene(num_worlds=1)
    engine, *_ = setup_engine_at_timestep(model)

    dims = engine.dims
    rhs_torch = wp.to_torch(engine.data.rhs).clone()

    # Solve via DifferentiableLinearSolve
    x_diff = DifferentiableLinearSolve.apply(
        engine.cr_solver, engine.A_op, engine.preconditioner,
        rhs_torch,
        engine.config.linear.max_iters,
        engine.config.linear.tol,
        engine.config.linear.atol,
    )

    # Solve via direct PCR
    x_direct = wp.zeros((dims.num_worlds, dims.num_constraints), dtype=wp.float32, device=engine.data.device)
    engine.cr_solver.solve(
        A=engine.A_op,
        b=wp.from_torch(rhs_torch),
        x=x_direct,
        preconditioner=engine.preconditioner,
        iters=engine.config.linear.max_iters,
        tol=engine.config.linear.tol,
        atol=engine.config.linear.atol,
    )
    x_direct_torch = wp.to_torch(x_direct).clone()

    err = torch.norm(x_diff - x_direct_torch).item()
    rel_err = err / (torch.norm(x_direct_torch).item() + 1e-10)
    print(f"  Forward solve error: {err:.2e} (relative: {rel_err:.2e})")
    assert rel_err < 1e-4, f"Forward solve mismatch: relative error {rel_err:.2e}"
    print("  PASSED")


def test_backward_finite_difference():
    """Verify backward (adjoint solve) matches finite differences.

    For f(b) = x where Ax = b, we have ∂f/∂b = A⁻¹.
    We check: for a scalar loss L = wᵀx, ∂L/∂b = A⁻¹ w = A⁻ᵀ w.
    """
    print("\n=== Test: Backward (adjoint) via finite differences ===")

    model = build_scene(num_worlds=1)
    engine, *_ = setup_engine_at_timestep(model)

    dims = engine.dims
    rhs_base = wp.to_torch(engine.data.rhs).clone()

    max_iters = 200
    tol = 1e-8
    atol = 1e-8

    print(f"  RHS norm: {torch.norm(rhs_base).item():.6f}")
    print(f"  RHS nonzero count: {(rhs_base.abs() > 1e-8).sum().item()} / {rhs_base.numel()}")

    # Random weight vector for scalar loss
    torch.manual_seed(42)
    w = torch.randn_like(rhs_base)

    # Compute analytical gradient via backward
    rhs_torch = rhs_base.clone().requires_grad_(True)
    x = DifferentiableLinearSolve.apply(
        engine.cr_solver, engine.A_op, engine.preconditioner,
        rhs_torch,
        max_iters, tol, atol,
    )
    print(f"  Solution norm: {torch.norm(x).item():.6f}")
    loss = torch.sum(w * x)
    loss.backward()
    grad_analytical = rhs_torch.grad.clone()

    # Compute finite difference gradient
    # Note: eps must be large enough that the iterative solver (float32)
    # can resolve the difference in solutions
    eps = 1e-3
    grad_fd = torch.zeros_like(rhs_base)

    # Only test a subset of indices (full would be too slow)
    # Pick indices with non-zero RHS (active constraints)
    rhs_np = rhs_base.cpu().numpy().flatten()
    nonzero_indices = np.where(np.abs(rhs_np) > 1e-8)[0]
    if len(nonzero_indices) == 0:
        print("  WARNING: No active constraints, skipping FD test")
        return

    test_indices = nonzero_indices[:min(20, len(nonzero_indices))]
    print(f"  Testing {len(test_indices)} of {len(nonzero_indices)} active constraint indices")

    # Use a single shared buffer for FD solves (avoids CUDA graph caching issues)
    x_buf = wp.zeros((dims.num_worlds, dims.num_constraints), dtype=wp.float32, device=engine.data.device)

    def solve_for_rhs(rhs_tensor):
        """Solve A x = rhs using PCR directly."""
        rhs_wp = wp.from_torch(rhs_tensor.contiguous())
        x_buf.zero_()
        engine.cr_solver.solve(
            A=engine.A_op, b=rhs_wp, x=x_buf,
            preconditioner=engine.preconditioner,
            iters=max_iters, tol=tol, atol=atol,
        )
        return wp.to_torch(x_buf).clone()

    max_rel_err = 0.0
    for idx in test_indices:
        # Perturb +eps
        rhs_plus = rhs_base.clone()
        rhs_plus.view(-1)[idx] += eps
        x_plus = solve_for_rhs(rhs_plus)
        loss_plus = torch.sum(w * x_plus).item()

        # Perturb -eps
        rhs_minus = rhs_base.clone()
        rhs_minus.view(-1)[idx] -= eps
        x_minus = solve_for_rhs(rhs_minus)
        loss_minus = torch.sum(w * x_minus).item()

        fd_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical_grad = grad_analytical.view(-1)[idx].item()

        abs_err = abs(fd_grad - analytical_grad)
        denom = max(abs(fd_grad), abs(analytical_grad), 1e-8)
        rel_err = abs_err / denom
        max_rel_err = max(max_rel_err, rel_err)

        if rel_err > 0.05:
            print(f"  idx={idx}: fd={fd_grad:.6f}, analytical={analytical_grad:.6f}, "
                  f"rel_err={rel_err:.4f}")

    print(f"  Max relative error: {max_rel_err:.6f}")
    assert max_rel_err < 0.1, f"Finite difference check failed: max relative error {max_rel_err:.4f}"
    print("  PASSED")


if __name__ == "__main__":
    test_forward_solves_correctly()
    test_backward_finite_difference()
    print("\n=== All tests passed! ===")
