"""Contact mode freezing for the adjoint backward pass.

The FB complementarity for friction rarely converges to R=0 (residuals O(1e-2)).
For the adjoint, we replace the FB-derived friction compliance with a linearized
version based on the detected contact mode:

  - Sticking (|λ_f| < μ·λ_n): friction locks tangential velocity.
    → small compliance (rigid constraint, like joint compliance)
  - Sliding (|λ_f| ≈ μ·λ_n): friction force is at Coulomb limit.
    → large compliance (force is nearly independent of velocity)

The friction Jacobians J (tangential directions) are kept from the forward solve.
The residuals are zeroed so the IFT assumption (R=0) holds.

See docs/adjoint_warm_start_issue.md for full background.
"""
import warp as wp


@wp.kernel
def freeze_contact_mode_kernel(
    # Friction constraint data
    constr_active_mask_f: wp.array(dtype=wp.float32, ndim=2),
    C_values_f: wp.array(dtype=wp.float32, ndim=2),
    res_c_f: wp.array(dtype=wp.float32, ndim=2),
    # Converged forces
    constr_force_f: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n: wp.array(dtype=wp.float32, ndim=2),
    # Material
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Config
    sticking_compliance: wp.float32,
    sliding_compliance: wp.float32,
):
    """Linearize friction for the adjoint based on the converged contact mode."""
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # Get converged forces
    lambda_t1 = constr_force_f[world_idx, 2 * contact_idx + 0]
    lambda_t2 = constr_force_f[world_idx, 2 * contact_idx + 1]
    lambda_n = constr_force_n[world_idx, contact_idx]

    # Friction coefficient
    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]
    mu = 0.5 * (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1])

    lambda_f_norm = wp.sqrt(lambda_t1 * lambda_t1 + lambda_t2 * lambda_t2 + 1e-10)
    coulomb_limit = mu * lambda_n

    # Deactivate if no normal force
    if lambda_n < 1e-6:
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # Detect mode: sliding if friction force is near the Coulomb limit
    ratio = lambda_f_norm / (coulomb_limit + 1e-10)

    # Hard threshold: sticking if ratio < 0.9, sliding otherwise
    is_sliding = ratio > 0.9
    c_adj = sticking_compliance
    if is_sliding:
        c_adj = sliding_compliance

    C_values_f[world_idx, 2 * contact_idx + 0] = c_adj
    C_values_f[world_idx, 2 * contact_idx + 1] = c_adj

    # Zero residuals so IFT assumption holds
    res_c_f[world_idx, 2 * contact_idx + 0] = 0.0
    res_c_f[world_idx, 2 * contact_idx + 1] = 0.0


@wp.kernel
def freeze_contact_mode_soft_kernel(
    # Friction constraint data
    constr_active_mask_f: wp.array(dtype=wp.float32, ndim=2),
    C_values_f: wp.array(dtype=wp.float32, ndim=2),
    res_c_f: wp.array(dtype=wp.float32, ndim=2),
    # Converged forces
    constr_force_f: wp.array(dtype=wp.float32, ndim=2),
    constr_force_n: wp.array(dtype=wp.float32, ndim=2),
    # Material
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    # Config
    sticking_compliance: wp.float32,
    sliding_compliance: wp.float32,
    temperature: wp.float32,
):
    """Linearize friction for the adjoint using soft sigmoid blending."""
    world_idx, contact_idx = wp.tid()

    if contact_idx >= contact_count[world_idx]:
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # Get converged forces
    lambda_t1 = constr_force_f[world_idx, 2 * contact_idx + 0]
    lambda_t2 = constr_force_f[world_idx, 2 * contact_idx + 1]
    lambda_n = constr_force_n[world_idx, contact_idx]

    # Friction coefficient
    shape0 = contact_shape0[world_idx, contact_idx]
    shape1 = contact_shape1[world_idx, contact_idx]
    mu = 0.5 * (shape_material_mu[world_idx, shape0] + shape_material_mu[world_idx, shape1])

    lambda_f_norm = wp.sqrt(lambda_t1 * lambda_t1 + lambda_t2 * lambda_t2 + 1e-10)
    coulomb_limit = mu * lambda_n

    # Deactivate if no normal force
    if lambda_n < 1e-6:
        constr_active_mask_f[world_idx, 2 * contact_idx + 0] = 0.0
        constr_active_mask_f[world_idx, 2 * contact_idx + 1] = 0.0
        return

    # Soft blending: sigmoid interpolation between sticking and sliding
    ratio = lambda_f_norm / (coulomb_limit + 1e-10)
    exponent = -(ratio - 0.9) / wp.max(temperature, 1e-8)
    # Clamp exponent to avoid overflow in exp
    exponent = wp.clamp(exponent, -20.0, 20.0)
    sigma = 1.0 / (1.0 + wp.exp(exponent))
    c_adj = (1.0 - sigma) * sticking_compliance + sigma * sliding_compliance

    C_values_f[world_idx, 2 * contact_idx + 0] = c_adj
    C_values_f[world_idx, 2 * contact_idx + 1] = c_adj

    # Zero residuals so IFT assumption holds
    res_c_f[world_idx, 2 * contact_idx + 0] = 0.0
    res_c_f[world_idx, 2 * contact_idx + 1] = 0.0


@wp.kernel
def adjoint_regularize_compliance_kernel(
    C_values: wp.array(dtype=wp.float32, ndim=2),
    constr_active_mask: wp.array(dtype=wp.float32, ndim=2),
    gamma: wp.float32,
):
    """Add regularization gamma to all active constraint compliances for the adjoint.

    Modifies the adjoint Schur complement from [J M-1 JT + C] to
    [J M-1 JT + C + gamma*I], preventing ill-conditioning and creating
    a gradient trust region that preserves signal through contacts.
    """
    world_idx, constr_idx = wp.tid()
    if constr_active_mask[world_idx, constr_idx] > 0.0:
        C_values[world_idx, constr_idx] = C_values[world_idx, constr_idx] + gamma
