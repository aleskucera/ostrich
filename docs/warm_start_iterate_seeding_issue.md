# Why Cross-Step Warm Start Cannot Seed the NR Iterate

## Symptom

Cross-step contact warm start populates `data._constr_force_prev_iter`
with the previous step's converged λ (matched contacts) plus α/β/γ
heuristics (cold-start contacts) so that the FB-friction kernel sees a
non-zero `f_n_prev` at NR iter 0. This works — phase 2.5 of warm-start
ships in `ostrich/collision/warm_start.py`.

The natural next step is "true" warm-start: also seed `data._constr_force`
itself, so the NR iterate at iter 0 starts from the same heuristic guess
instead of from λ = 0. That is, treat the warm start as the actual
initial iterate rather than only as the friction radius.

**Empirically this hurts, not helps.** Even seeding only the *matched*
contacts (literally last step's converged forces, with at most a
tangent-basis re-projection — no heuristic involved) produces:

| obstacle_benchmark, 200 steps | bad steps | picked max ‖r‖² | iter-0 max ‖r‖² |
|---|---|---|---|
| warm completely off                  |   0   | 8.5e-4 | 180 |
| match-only → `_constr_force_prev_iter`, the phase 2.5 design |   1   | 1.2e-3 | 110 |
| match-only → both buffers (true warm start) | **7** | **1.2e-2** | 230 |

Same scene, same matched values, the only difference is whether the
matched λ also lands in `_constr_force`. Bad steps jump 1 → 7; picked
worst-case residual gets 14× worse. Adding the cold-start heuristic on
top of true warm start makes it noisier still — iter-0 max ‖r‖² spikes
to 2.5e5 (i.e. ‖r‖ ≈ 500) on individual steps.

This is a property of the Fisher–Burmeister Newton-Raphson formulation,
not a bug in our heuristic. Documenting it here so future contributors
don't repeat the experiment.

## Reproducer

```bash
# All three runs use cluster reduction + min_iter=2 + warm-start match.
# The variable is which buffers warm_starter writes into.

# Baseline: warm fully off.
python examples/helhest/obstacle_benchmark.py \
  ++engine.enable_contact_warm_start=false \
  logging.hdf5_log_file=data/logs/A_off.h5

# Phase 2.5 design (current): warm only feeds prev_iter.
python examples/helhest/obstacle_benchmark.py \
  logging.hdf5_log_file=data/logs/B_prev_iter_only.h5

# True warm start: also mirror into _constr_force.
# (To reproduce, edit warm_start.py to write matched λ into both
# _constr_force_prev_iter AND _constr_force inside _match_kernel,
# and remove engine.step's _constr_force.zero_().)
```

The "bad steps" count is `(picked_residual_squared > newton_atol²)`,
where `newton_atol = 1e-3` per the engine config.

## Root cause: Jacobian degeneracy at the FB cone boundary

The contact-normal complementarity is enforced through Fisher–Burmeister:

$$
\varphi_{\text{FB}}(\lambda_n,\, g) = \lambda_n + g - \sqrt{\lambda_n^2 + g^2}
$$

where `g = J_n·v + b_n` is the gap-velocity residual that converges to 0
when contact is active. At the converged state, exactly one of the two
KKT corners holds:

  * **separating**: `λ_n = 0`, `g > 0`  → `φ_FB = 0`
  * **touching**:  `λ_n > 0`, `g = 0`  → `φ_FB = 0`

Newton needs the local Jacobian of φ_FB to drive the residual down. Its
partial derivatives are:

$$
\frac{\partial \varphi_{\text{FB}}}{\partial \lambda_n} = 1 - \frac{\lambda_n}{\sqrt{\lambda_n^2 + g^2}},\qquad
\frac{\partial \varphi_{\text{FB}}}{\partial g} = 1 - \frac{g}{\sqrt{\lambda_n^2 + g^2}}
$$

Evaluate these at the two corners:

  * **At `(λ_n=0, g)` with `g ≠ 0`**: `∂φ/∂λ = 1`, `∂φ/∂g = 1 − sign(g)`.
    Non-degenerate in λ — Newton can move λ to satisfy the constraint.
  * **At `(λ_n=warm > 0, g=0)`**: `∂φ/∂λ = 1 − 1 = 0`, `∂φ/∂g = 1`.
    **Zero sensitivity to λ.**

The third row of the Jacobian (the FB row for this contact) becomes
`[0 · Δλ + 1 · Δg = -φ]`. Newton can no longer update λ_n through this
row — the constraint at this point is informative about Δg only.

In a healthy converged state this is harmless: φ_FB = 0, no update is
needed, and Newton stops. But warm-starting puts the iterate *near*
this corner with a slightly inconsistent `(λ_n, g)`:

  * the matched λ_n is from last step's pose,
  * the current `g` is computed from the new pose / new contact normal,
  * even with our prediction step, normal directions and tangent bases
    differ at the 1e-3 level (re-projection through `orthogonal_basis`
    on a slightly rotated normal).

So the iterate sits at `(warm, ε)` with `|ε|` small. Two things go wrong:

1. **The FB row is near-degenerate in λ**, but it's not exactly zero —
   it has order `ε` sensitivity. Solving the Newton system with this
   tiny coefficient amplifies any rhs error into a large Δλ. The next
   iterate overshoots to `(warm + huge, …)`.

2. **The FB function has a kink along its zero set**. From `(warm, 0)`,
   moving in the `g < 0` direction (slight penetration from the new
   pose) gives `φ_FB(warm, -ε) = warm - ε - √(warm² + ε²) ≈ -2ε`, so
   the residual is small. But moving in the `g > 0` direction gives
   `φ_FB(warm, ε) = warm + ε - √(warm² + ε²) ≈ ε`. The function is C¹
   but the curvature flips sign across the zero set; second-order
   information that backtracking and linesearch implicitly assume
   (smooth descent direction) breaks down.

Starting from `λ_n = 0`, the iterate is at `(0, g)` with `g` of order 1.
That's far from any kink, the Jacobian has full rank, and Newton makes
clean progress. The first iter typically lands somewhere in the
*interior* of the cone, away from both corners, and Newton converges
toward the corner from the interior over the next 5–10 iters. This is
the regime the solver was tuned for.

## Why the friction-radius warm start (phase 2.5) is fine

The FB-friction kernel reads `_constr_force_prev_iter` only to compute
the friction cone radius `μ · f_n_prev`, which scales the friction
constraint:

```
||λ_t|| ≤ μ · f_n_prev    (cone constraint on the tangent forces)
```

`f_n_prev` enters as a *parameter* of the current iter's friction FB,
not as the iterate's own coordinate. The friction Jacobian at `λ_t = 0`
(our actual iter-0 state) with `μ·f_n_prev > 0` is well-defined: the
friction cone has positive radius and the FB residual has a smooth
gradient in λ_t. Compare against the cold-start case where
`μ·f_n_prev = 0`: the friction cone collapses to a point, the friction
constraint becomes `λ_t = 0` enforced by a degenerate FB, and friction
Jacobians don't assemble (the `μ * f_n <= 1e-6` guard in
`compute_friction_model` skips them). That's the friction-lag bug
phase-2 warm-start fixes.

So the asymmetry is real and stems from the different roles of the two
buffers:

  * `_constr_force` *is* the iterate. Seeding it lands you at a
    Jacobian corner in the worst place possible.
  * `_constr_force_prev_iter` is a *parameter* of the friction kernel.
    Seeding it just turns friction on at iter 0 instead of iter 1+.

## What "true" warm start would actually require

If we wanted the NR iterate itself to start from a non-zero λ (which
*is* worth chasing — it would unlock 5–10× speedups on the easy steps)
we'd need to change the constraint formulation away from
plain Fisher–Burmeister:

* **Smoothed FB**: replace `√(λ² + g²)` with `√(λ² + g² + δ²)` for a
  small δ > 0. This rounds the corner and gives a positive
  `∂φ/∂λ` everywhere, at the cost of a small bias in the converged
  contact force (force-equilibrium is satisfied with a tiny gap).
  Worth prototyping — δ on the order of `1e-4 N` would be
  invisible to the dynamics but would lift the warm-start
  block. Trade-off: dt-stability could suffer; needs its own
  benchmark.

* **Projected Gauss–Seidel** instead of Newton: PGS is a fixed-point
  iteration on the contact forces directly. It tolerates any starting
  λ — bad initial guesses just slow convergence, they don't make it
  diverge. Mature solvers (Bullet, ODE) use PGS for this reason. But
  PGS converges much slower than NR on stiff systems, so this would
  cost in the regimes where NR currently shines.

* **Active-set pre-solve**: classify contacts as separating /
  touching / sliding before NR runs, then solve a reduced equality-
  constrained system per branch. This is the classical "complementarity
  pivot" approach. Implementation is invasive (a separate solver path
  per active set) but it eliminates the FB corner entirely — λ is only
  a free variable on the touching/sliding branches, where the local
  Jacobian is full-rank.

For now the answer is **don't seed the iterate**, accept the modest
phase-2.5 win (friction radius only), and revisit if the formulation
ever changes.

## See also

* `ostrich/collision/warm_start.py` — phase-2.5 implementation
* [`friction_sticking_issue.md`](friction_sticking_issue.md) — a related
  active-arm degeneracy in the friction NCP, documents the same
  Jacobian-rank-deficiency family of failures
* [`dt_dependence_problem.md`](dt_dependence_problem.md) — another
  FB-NR tuning consideration that interacts with iterate quality
