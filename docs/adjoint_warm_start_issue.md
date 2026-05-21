# Adjoint Gradient Accuracy: Friction Convergence

## Summary

The implicit differentiation backward pass (`step_backward`) produces incorrect
gradients when **friction constraints on the same body as joints** don't converge
to zero residual. The IFT (Implicit Function Theorem) assumes R=0 at the
converged point; when friction residuals are O(1e-2), this assumption breaks and
gradients can be off by orders of magnitude.

**Scenes without friction on joint bodies work correctly** (verified: 2% error for
control gradient, <0.01% for velocity gradient).

## What Was Fixed

### 1. Control target gradient formula (residual_utils.py)

The `control_target_grad_kernel` used the wrong scaling factor:

- **Position mode**: changed `w_lambda * (-1/dt²)` → `w_lambda * (-1/dt)`
- **Velocity mode**: changed `w_lambda * (-1/dt)` → `w_lambda * (-1)`

Derivation from the IFT (see `implicit_gradient.pdf`):
```
dL/d(target) = w^T ∂R/∂target

R_c_position = (q - target)/h + α·λ·h   →   ∂R_c/∂target = -1/h
R_c_velocity = (qd - target) + α·λ·h    →   ∂R_c/∂target = -1
```

Verified: with the pendulum raised above ground (no friction contacts), the
corrected formula gives **2% error** vs finite differences for the control
gradient.

### 2. Friction mode freezing for adjoint (adjoint_friction.py)

Added `freeze_friction_mode_kernel` which runs in `step_backward()` after
`compute_linear_system()`. For each friction contact pair:

- Replaces the FB-derived compliance `C_f = w/dt` with a fixed `friction_compliance`
- Zeros the friction residual so the IFT assumption holds
- Deactivates friction for contacts with negligible normal force

Also zeros normal contact residuals in the adjoint.

**Status**: Implemented but not yet fully verified for the hard case (pendulum
touching ground). The gradient improves but is still inaccurate when both joints
and friction act on the same body. Further investigation needed.

## Root Cause Analysis

### Why friction breaks the adjoint

The Fisher-Burmeister complementarity for friction is highly nonlinear and
typically converges to residuals of O(1e-2), while joint constraints converge to
O(1e-5). When friction residuals are large:

1. The Jacobian `∂R/∂s⁺` at the "converged" point doesn't accurately represent
   the true sensitivity (because R ≠ 0)
2. The Schur complement feedback `M⁻¹ J^T w_λ` nearly perfectly cancels
   `w_u_init`, giving `w_u ≈ 0` (feedback/init ratio = -0.999957)
3. This makes `body_vel_prev.grad ≈ 0` and `target_pos.grad ≈ 0` regardless
   of the true sensitivity

### Per-constraint residuals (pendulum at ground level)

| Constraint type | Active | Max |h_c| | Converged? |
|----------------|--------|---------|------------|
| Joint (5 DOF)  | 5      | 8e-6   | Yes        |
| Control (1)    | 1      | 3e-5   | Yes        |
| Normal (4)     | 4      | 1e-4   | Marginal   |
| **Friction (8)** | **8** | **3e-2** | **No**  |

### Convergence tolerance sweep

| newton_atol | ctrl analytical | ctrl FD | rel error |
|-------------|----------------|---------|-----------|
| 5e-2        | 1.36           | 3.80    | 64%       |
| 1e-3        | 0.016          | 0.004   | ~300%     |
| 1e-4        | 0.022          | 0.018   | 24%       |

At tight tolerance (1e-4), the gradient converges but the FD itself becomes
unstable due to Newton convergence path sensitivity.

## Remaining Work

1. **Validate friction freeze on more scenes** — test with stacked boxes, walking
   robots, etc. where friction on joint bodies is common
2. **Normal contact mode freezing** — similar to friction, the normal contact FB
   may also need linearization for the adjoint
3. **h_d residual** — the dynamics residual is also non-zero at convergence;
   investigate if zeroing it in the adjoint improves results
4. **Sign error in implicit_gradient.pdf** — slide 8 writes `[∂R/∂s⁺]^T w = ∇L`
   but the correct equation from the IFT is `[∂R/∂s⁺]^T w = -∇L`

## Affected Code

- `src/axion/core/residual_utils.py:223-233` — control gradient formula (FIXED)
- `src/axion/core/adjoint_friction.py` — friction mode freeze kernel (NEW)
- `src/axion/core/base_engine.py:386+` — `step_backward()` integration (MODIFIED)

## Test Coverage

- `tests/differentiable_simulator/test_control_gradient.py` — revolute pendulum
  with position control (raised above ground to avoid friction)
- `tests/differentiable_simulator/test_velocity_gradient.py` — box on ground
  (contact-only, no joints)
- Other tests in `tests/differentiable_simulator/` — zero gradient, multi-step,
  pose, optimization, symmetry, contact boundary
