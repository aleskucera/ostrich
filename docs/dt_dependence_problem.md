# dt-Dependence Problem: Why Contacts Are Harder to Solve at Small Timesteps

## Symptom

When `dt` is reduced from ~10–50 ms (the regime the underlying paper validates) to
~1 ms, contact simulation quality **degrades** rather than improves: bodies jitter,
penetrate visibly, and resting contacts fail to settle. This contradicts the naive
expectation that smaller `dt` means more accurate physics.

The simulation is solving the right physics; the **Newton iteration on the
complementarity system fails to converge within `max_newton_iters` at small `dt`**,
and the unconverged state is what gets rendered.

## Setting

The contact normal constraint follows
[Macklin et al. 2019, *Non-Smooth Newton Methods for Deformable Multi-Body Dynamics*](https://arxiv.org/abs/1907.04587).
The Fisher–Burmeister NCP function is

```
φ_n_FB(C_n, λ_n) = C_n + r·λ_n − √(C_n² + r²·λ_n²)
```

with the paper's complementarity preconditioner (their Eq. 66):

```
r = h² · [J M⁻¹ J^T]_ii
```

The implementation in `src/axion/constraints/contact_constraint.py:94-106` matches
the paper:

```python
precond = wp.pow(dt, 2.0) * effective_mass         # r = h²·W
phi_n, dphi_dc, dphi_dλ = scaled_fisher_burmeister_diff(signed_dist, f_n, 1.0, precond)
res_n_val = phi_n / dt                              # h_n in paper Eq (58)
c_val     = dphi_dλ / dt² + compliance              # S/h² block in paper Eq (53)
```

## Concrete Walkthrough

### Test problem

A 1 kg ball, 1D, resting on a floor at `q = 0`:

| Quantity | Value |
|----------|-------|
| `M`      | 1 kg              |
| `g`      | 10 m/s²           |
| `C_n(q)` | `q` (gap = height) |
| `J`      | 1                 |
| `W = J M⁻¹ J^T` | 1            |
| `λ_n` (correct) | `M·g = 10 N` |

### State at start of one Newton step

Body has 0.1 mm of penetration carried over from previous step, with otherwise
correct multipliers (warm-started case):

```
q⁻       = −1e-4 m
u⁻       = 0
ũ        = u⁻ − h·g = −10·h
λ_n⁰     = 10
u⁰       = 0
q⁰       = q⁻ + h·u⁰ = −1e-4
```

Residuals at this iterate:

```
g (momentum)  = M·(u⁰ − ũ)/h − λ_n⁰  =  0       (warm start: balanced)
φ (FB)        = q⁰ + h²·λ_n⁰ − √(q⁰² + h⁴·λ_n²)
h_n           = φ / h
```

The FB diagonal entry:

```
S = ∂φ/∂λ_n  =  h² · (1 − h²·λ_n / √(q² + h⁴·λ_n²))
```

The denominator splits into two regimes around the FB **smoothing radius** `10·h²`:

- `pen ≪ 10·h²`: inside the smooth corner, `S ≈ 0`.
- `pen ≫ 10·h²`: outside, on the linear arm, `S ≈ h²·β` with `β` close to 1.

| h     | smoothing radius `10·h²` | regime for `pen = 0.1 mm` |
|-------|--------------------------|----------------------------|
| 50 ms | 25 mm                    | inside (smooth)            |
| 8.3 ms| 0.7 mm                   | inside (smooth)            |
| 1 ms  | 10 µm                    | **outside (linear arm)**   |

### One Newton step

Eliminating `δu` from the paper's Eq (53), the Newton update on `λ_n` is

```
δλ_n  =  pen / [ h² · (1 + S/h² + e) ]
```

where `e = contact_compliance` in code. With `e = 0` (current default
`1e-6` is effectively zero):

| h     | `S/h²`  | `pen/h²`  | `δλ_n`        |
|-------|---------|-----------|----------------|
| 50 ms | ≈ 0     | 0.04      | **0.04 N**     |
| 1 ms  | ≈ 0.9   | 100       | **53 N**       |

At 50 ms Newton corrects `λ_n` by 0.4% — converged in one step. At 1 ms Newton
overshoots `λ_n` by **530%**, sending the body upward at 53 mm/s. Subsequent
iterations chase this back, but with `max_newton_iters = 8` and the cold-start /
linesearch dynamics, Newton exits before settling. The leftover error becomes the
visible jitter.

## Why It Happens

The Newton-step formula

```
δλ_n  =  pen / [ h² · (1 + S/h² + e) ]
```

has a `1/h²` factor that comes structurally from the paper's variable substitution
`Δλ' = Δλ·h` in Eq (53), combined with the dynamics block linearization
`M·δu/h − δλ = …`. The denominator term `(1 + S/h² + e)` only damps the blowup
when *something in it is `O(1/h²)`*. The options are:

1. **`S/h² = O(1)`** — happens automatically when penetration sits **outside** the
   FB smoothing radius. But the smoothing radius itself shrinks as `h²`, so by the
   time `h = 1 ms` essentially any non-trivial penetration triggers `S/h² = O(1)`,
   which means the FB *only just barely* damps the overshoot — typically by 2×.

2. **`pen ≪ smoothing radius`** — keeps the numerator in step with the
   denominator. At small `h` the smoothing radius is microscopic, so `pen` would
   have to be sub-µm. Not realistic without warm-starting *and* tight float
   precision.

3. **`e` chosen large enough** — adds an `O(1/h²)`-type contribution to the
   denominator independent of FB state. This is the proposed fix.

The smoothing-radius shrinkage is structural to the FB ratio
`r_b/r_a = h²·W` chosen in the paper. The ratio is locked to the discrete
dynamics' force-to-position sensitivity `∂C_n/∂λ_n = h²·W`, so it can't be tuned
out without leaving the position-level FB framework.

## Why Some Plausible Fixes Don't Work

### Removing the `/dt²` from `c_val`

Hypothesis: "the `1/h²` is what's blowing things up; just take it out."

The `1/h²` is in the **denominator** of `δλ_n`. Removing it shrinks the
denominator and **doubles** the overshoot at h=1 ms (from 53 N to ~100 N). The
`S/h²` term in the matrix is what was bounding the step on the linear arm; switching
it to `S` makes that contribution vanish (since `S ≈ h² → 0`).

### Re-preconditioning the FB arguments

The FB function admits two free scaling parameters
`φ(r_a · a, r_b · b) = 0` with the same complementarity solution. The paper
chooses `(r_a, r_b) = (1, h²·W)`. Same-ratio rescalings (e.g. `(1/h, h·W)`) are
**gauge-equivalent**: identical Newton iterates, only the FB function value
rescales. They don't widen the smoothing radius, which depends only on the ratio.

A different ratio (e.g. `(1, h·W)`) does widen the smoothing radius but introduces
a `1/h` blow-up in the system-matrix diagonal, trading one form of ill-conditioning
for another and breaking the alignment with the dynamics linearization. Net loss.

### Switching to velocity-level constraint

Replacing `C_n` with `Ċ_n` and `r = h·W` makes the smoothing radius scale as `h·g`
instead of `h²·g`, and removes the `1/h` residual scaling. But the constraint then
only enforces non-penetration *velocity*, not position — static penetration drifts
without a Baumgarte position-correction term. Tunable, but adds a parameter and
introduces dissipation.

## The Fix: Compliance Regularization

Adding `e` to the contact-row matrix diagonal kills the `1/h²` overshoot:

```
δλ_n  =  pen / [ h² · (1 + S/h² + e) ]
```

For `e ≳ pen / (h² · M · g)` the right-hand side stays bounded by `M·g` ≈ the
correct multiplier magnitude, so one Newton step lands close to the answer
instead of overshooting.

### Why this is safe

- **Newton-only regularization, not physical compliance.** Paper §8.4: "The
  regularization only applies to the error at each Newton iteration, and the
  solution approaches the original one as the Newton iterations progress." When
  Newton converges, the FB equation is satisfied exactly regardless of `e`.
- **Identical to the `regularization` knob's spirit** (`engine_config.py:68`),
  which is global. `contact_compliance` is the same thing scoped to the contact
  normal block.
- **Already-supported code path.** `c_val = dphi_dλ/dt² + compliance` already
  accepts a non-zero compliance; only the value needs to change.

### Recommended values

For our test problem (`pen = 0.1 mm`, `M = 1 kg`, `g = 10`):

| Strategy | Value |
|----------|-------|
| Constant `e` for h ≤ 1 ms       | `contact_compliance = 10.0`           |
| dt-adaptive (clean version)     | `e(h) = α / (h² · M_typical · g)` with `α ≈ 1` |

The dt-adaptive form sits at machine-zero at `h = 50 ms` (no effect on the
working regime) and grows to `e ≈ 10` at `h = 1 ms` (kills the overshoot
exactly when needed).

### Verification

Per-step indicators that the fix is working:

- `data.iter_count` drops from "always at `max_newton_iters`" to "varies, often
  small" at small `dt`.
- `data.res_norm_sq` at exit is smaller than before.
- A resting body's contact point shows steady `q ≈ 0`, not jitter.
- A penetration creep (constant offset of `q < 0`) means `e` is too large
  relative to `max_newton_iters`. Either reduce `e` or raise `max_newton_iters`.

---

# Implementation Plan

The fix is small. Estimated effort: ~30 min code, plus tuning.

## Phase 1 — Diagnostic baseline (do this first)

Goal: confirm the failure mode before changing anything.

1. Pick a reproducible scene with the symptom — a resting robot at `dt = 1 ms`
   that visibly jitters. Anything in `experiments/2_dt_stability/` likely fits.
2. Enable HDF5 logging:
   - In the relevant `AxionEngineConfig`, set `enable_hdf5_logging = True` and
     `log_constraint_data = True`.
3. Run one short simulation (~1 s of sim time).
4. Read the log and plot per-step:
   - `iter_count` vs step
   - `res_norm_sq` vs step
   - For a chosen contact, `phi_n` and `lambda_n` vs step
5. Confirm at `dt = 1 ms`:
   - `iter_count` saturates at `max_newton_iters = 8`
   - `res_norm_sq` does not converge below `newton_atol²`
   - `lambda_n` oscillates around the steady value
6. Re-run at `dt = 50 ms` for comparison; confirm `iter_count` is small and
   `res_norm_sq` converges.

This pins the diagnosis. **If small-dt iter_count is *not* saturated, the issue
is something else and this fix won't help.**

## Phase 2 — Constant compliance (one-line change)

Goal: confirm the math by trying the simplest possible version.

1. Edit `src/axion/core/engine_config.py`, line 65:
   ```python
   contact_compliance: float = 10.0   # was 1e-6
   ```
2. Re-run the diagnostic scene at `dt = 1 ms`.
3. Expected: `iter_count` drops, jitter visibly reduces. Confirm with the
   same logs from Phase 1.
4. Sanity-check `dt = 50 ms` behavior is unchanged (or slightly slower
   convergence — acceptable).
5. Sweep `contact_compliance ∈ {0.1, 1, 10, 100}` to map out the
   stability/penetration tradeoff.

**Decision point:** if a single constant works for the whole `dt` range you care
about, stop here. Skip Phase 3.

## Phase 3 — dt-adaptive compliance (only if Phase 2 isn't enough)

Goal: make compliance auto-scale so it's silent at large `dt` and active at small
`dt`.

1. Edit `src/axion/constraints/contact_constraint.py`, replacing line 106:
   ```python
   c_val = dphi_dlambda_n / wp.pow(dt, 2.0) + compliance
   ```
   with:
   ```python
   adaptive_compliance = compliance / wp.pow(dt, 2.0)
   c_val = dphi_dlambda_n / wp.pow(dt, 2.0) + adaptive_compliance
   ```
2. Update the same change in the batched kernels lower in the file
   (`compute_contact_core` is shared — check that the change propagates to all
   three kernels: `contact_constraint_kernel`, `batch_contact_residual_kernel`,
   `fused_batch_contact_residual_kernel`).
3. Re-tune `contact_compliance` in `engine_config.py` — the meaning of the
   field has changed. The new value is `α / (M_typical · g)` where `α ≈ 1` and
   `M_typical` is the mass of a representative rigid body. For `M ≈ 1 kg`,
   `g = 10`, set `contact_compliance = 0.1` (which gives `e = 0.1 / h²`,
   i.e. `e = 10` at `h = 1 ms` and `e = 4e-5` at `h = 50 ms`).
4. Re-run Phase 1 diagnostic at multiple `dt` values
   (`{50, 10, 5, 1, 0.5} ms`) and confirm:
   - All `dt` show low `iter_count` and converged `res_norm_sq`.
   - No regression in `dt = 50 ms` quality.

## Phase 4 — Friction (only if friction is also misbehaving)

The same trick applies to `friction_compliance` (`engine_config.py:66`,
`friction_constraint.py:167`). If after Phase 3 friction still has problems at
small `dt`, apply the analogous constant or `1/h` scaling. Note `friction`'s `c_f`
already has `w/dt`, so the scaling exponent for the adaptive version is `1/h`,
not `1/h²`.

Skip this phase unless friction shows independent symptoms.

## Phase 5 — Validation suite

Before merging:

1. Run `experiments/2_dt_stability/sweep_axion.py` (or the closest existing
   benchmark) for `dt ∈ {50, 10, 5, 1, 0.5} ms`.
2. Compare:
   - Trajectory quality (visualize a representative scene).
   - Total energy / drift over time.
   - Per-step solver stats (`iter_count`, `res_norm_sq`).
3. Compare against the MuJoCo baseline at the same `dt`'s if available.
4. Check that adjoint-mode tests (the `adjoint-interventions` branch's main
   focus) still pass — compliance changes the linearized system that the
   adjoint differentiates through.

## Risks and rollback

- **Adjoint sensitivities**: changing the matrix changes the adjoint. If
  adjoint-mode regression appears after Phase 2/3, it's likely due to the
  modified Schur structure, not a bug in compliance per se.
- **Penetration creep**: if a constant `e` is too large for the available
  Newton-iter budget, bodies sit at a steady offset below the floor. Bump
  `max_newton_iters` from 8 → 16 first, then if still creeping, reduce `e`.
- **Rollback**: revert is one-line in `engine_config.py` (Phase 2) or one-block
  in `contact_constraint.py` (Phase 3).

## Out-of-scope alternatives

The following were considered and rejected (see "Why Some Plausible Fixes Don't
Work" above) — leaving them documented to save the next person from re-deriving:

- Removing `/dt²` from `c_val` — makes the small-dt overshoot worse.
- Re-gauging the FB function `(r_a, r_b) = (1/h, h·W)` — pure relabeling, no
  effect on Newton iterates.
- Switching contact normal to velocity-level — works, but adds a Baumgarte
  parameter and dissipates energy. Recommend only if the compliance fix proves
  insufficient.
