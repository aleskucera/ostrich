# Warm-Start Neural Network: Experiment Findings

## Goal

Train a neural network to predict initial `(body_vel, constr_force)` so the
Newton-Raphson solver converges in fewer iterations.

## Approach Tested

**Option B (direct prediction)**: NN predicts `(v, lambda)` from the previous state
`(pose_prev, vel_prev)`, trained by minimizing `||r(v, lambda)||^2` (the physics
residual). Two backward modes were tested:

- **Approximate**: `grad = A_full^T @ grad_output` where `A_full = [M, -dt*J^T; J, dt*C]`
- **Exact**: `wp.Tape()` autodiff through all nonlinear constraint kernels

No differentiation through solver iterations (that would be Option A).

## Code

- `src/ostrich/learning/torch_residual.py` -- `OstrichResidual`: differentiable residual with approximate `A_full^T` backward
- `src/ostrich/learning/torch_residual_ad.py` -- `OstrichResidualAD`: differentiable residual with exact gradients via `wp.Tape`
- `src/ostrich/learning/warm_start_net.py` -- `WarmStartNet`: configurable MLP, `WarmStartTrainer`: training loop
- `src/ostrich/optim/full_system_operator.py` -- `FullSystemOperator`: full KKT system operator and its transpose
- `examples/train_warm_start.py` -- training and evaluation script

## Findings

### 1. The gradient pipeline works correctly

Verified by initializing raw `(body_vel, constr_force)` parameters from the
solver's converged solution and optimizing with Adam:

- From solver init: loss went **0.198 -> 0.002** (5000 epochs)
- All constraint types (dynamics, contacts, friction) improve smoothly
- Gradients flow correctly through `wp.Tape` for all nonlinear kernels

### 2. Fisher-Burmeister complementarity creates an optimization barrier

The contact normal residual uses:

    FB(a, b) = alpha*a + beta*b - sqrt(alpha^2*a^2 + beta^2*b^2 + eps)

where `a = signed_dist(v)` (gap, depends on body_vel through pose integration)
and `b = lambda_n` (normal contact force).

When `beta*lambda_n >> |a|`: `FB approx a`. The residual asymptotes to
`signed_dist / dt` regardless of how large the contact force gets.

For penetrating contacts (`signed_dist < 0`), increasing `lambda_n` cannot
reduce the residual. Only changing `body_vel` (to move bodies out of
penetration) can reduce it, but `body_vel` is coupled through the dynamics
equation `M*(v - v_prev) = dt*J^T*lambda + f_ext`.

This creates a shallow valley that gradient descent traverses very slowly.

### 3. Two contact constraints dominate the stuck residual

In a test scene with 6 bodies and 768 constraints:

- Only ~26 constraints are active at any timestep (92% inactive)
- **99% of the stuck residual** comes from just 2 contact normal constraints
- Dynamics residual: ~0.05 (converges fine)
- Friction residual: ~0.006 (converges fine)
- Joint/control: 0 (no joints in test scene)
- Contact normal: ~47 (stuck)

### 4. The loss landscape is well-behaved near the solution

Starting from the solver's output, gradient descent easily refines further
(0.198 -> 0.002). The FB barrier only blocks optimization when starting far
from the solution (e.g., from zero or random initialization).

### 5. Optimizer choice does not overcome the barrier

| Optimizer         | Final loss | Notes                                |
|-------------------|-----------|--------------------------------------|
| Adam (lr=1e-4)    | ~10       | Slow but steady descent              |
| L-BFGS            | ~56 (stuck)| Line search fails at FB boundary    |
| Adam + cosine     | ~10       | Same floor, fancier path             |
| Softplus reparam  | ~45       | Force magnitude isn't the bottleneck |
| **Solver target** | **0.15**  |                                      |

### 6. Approximate vs exact gradients

- `A_full^T` backward (linearized Jacobian): plateaus at ~40
- `wp.Tape` backward (exact nonlinear): reaches ~10
- Exact is 4x better but does not overcome the fundamental FB barrier

### 7. NN warm-start trained with residual loss is worse than default

- Default warm-start `(v=v_prev, cf=0)`: avg ~5 solver iterations
- NN trained with residual loss: avg ~7 solver iterations (**worse**)
- The NN's prediction (residual ~4.3) puts the solver in a harder starting
  point than the simple default

## Conclusion

**The `||r(v, lambda)||^2` loss is not suitable for training a warm-start NN
from scratch.** The Fisher-Burmeister contact complementarity creates an
asymptotic barrier that prevents gradient-based optimization from reaching the
solution basin when starting from a generic (zero or NN) prediction.

## Recommended Next Steps

1. **Supervised pretraining** on solver outputs (MSE loss), then fine-tune with
   residual loss (the good basin around the solution is reachable, as shown by
   finding #1)
2. **Hybrid loss**: weighted combination of MSE-to-solver-output and residual
3. **Option A**: differentiate through the unrolled Newton iterations, so the NN
   learns to produce initializations that lead to fast convergence, rather than
   directly minimizing the residual at the predicted point
