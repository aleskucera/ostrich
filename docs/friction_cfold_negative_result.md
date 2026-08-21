# Negative result: scalar C-fold for the friction Newton derivative

**Status: tried, measured, rejected (2026-08-21). Do not re-implement without
reading this.** The experimental code was deleted; this document is the record.

## Context

After the friction cone fix (`clamped_imp_norm` in the `w` denominator,
`compute_friction_model`), the friction row `res_f = v_t + w*lambda_f` has the
correct Coulomb fixed point (`|lambda_f| = mu*f_n` at sliding), but the Picard
iteration on `w = |v_t|/(mu*lambda_n)` contracts slowly near the cone boundary
and forward NR convergence degrades ~25x at nr.max_iters=16.

The obvious remedy — add the missing Newton derivative `dw/dv_t` as a rank-1
row correction `I + (lambda_f (x) v_hat_t)/(mu*lambda_n)` — is **incompatible
with the solve pipeline**, established by code-level review:

- The Schur operator (`optim/system_operator.py`) uses ONE shared `J_values`
  array for both the scatter (`M^-1 J^T x`) and gather (`J ...`) sides:
  `A = J M^-1 J^T + diag(C)` is symmetric *by construction*. A row-side-only
  correction cannot even be stored (no second J array; `C_values` is strictly
  scalar per row, no home for a within-contact 2x2 coupling).
- `optim/pcr_solver.py` is textbook preconditioned Conjugate Residual: valid
  only for symmetric A. Its alpha/beta guards (`yAp > 0`, `zAz_old > 0`)
  cannot detect nonsymmetry; on the indefinite symmetric part that arises near
  incipient sliding, the iteration silently stagnates to max_iters and returns
  the partial solution. Nothing downstream inspects failure.
- The asymmetry is global, not confined to the friction 2x2 block: modified
  friction rows against unmodified normal/joint columns on shared bodies stay
  asymmetric even at the converged sliding point where the block term itself
  becomes symmetric (`-v_hat v_hat^T`).
- The engine's own idiom (contact-normal rows) is the correct pattern: the FB
  scaling `dphi_dc` multiplies `J_hat` on BOTH sides, and the lambda-derivative
  goes into scalar `C >= 0` only.

## The C-fold (the one PCR-compatible descendant) — and why it failed

In a slip-aligned tangent frame the row correction is a scalar
`g = 1 + (lambda_f . v_hat_t)/(mu*lambda_n)` on the parallel row; dividing that
row by `g` restores the geometric J on the row side and absorbs the correction:
`C_par = (w/dt + compliance)/max(g, eps_g)` plus the same `1/g` on that row's
Schur rhs entry.

**Implementation constraint that decided the outcome:** slip-aligned rotation
of the isotropic tangent basis is blocked by a pipeline invariant — friction
lambda slots are componentwise-persistent in a fixed frame across NR iterations
(`newton_step` does `lambda += dlambda`), across backtracking candidates, and
across the warm-start snapshot; `force_projection.py` assumes the components
are in the solver's frame. So the fold was implemented as a diagonal
approximation (both rows scaled by `1/g`), which mis-scales the perpendicular
row (true factor: 1) by up to `1/eps_g = 10`.

**Measurements** (flat mesh, dt 0.03, isotropic wheels mu 0.65/0.65/0.5,
spin (3,-3,0) 5 s / forward (5,5,5) 4 s; mean NR residual; cone occupancy
`|f_t|/(mu*f_n)` med/p90 over loaded contacts):

| arm                          | spin res | spin occ  | fwd res |
|------------------------------|----------|-----------|---------|
| pre-fix reference, nr 16     | 0.0119   | 3.70/19.1 | 0.0020  |
| cone fix, fold off, nr 16    | 0.263    | 1.09/4.0  | 0.0216  |
| cone fix, fold ON,  nr 16    | 0.793    | 1.13/6.3  | 0.0476  |
| fold ON, lagged lambda, 16   | 0.698    | 1.07/5.7  | 0.0807  |
| fold ON, g clamped <=1, 16   | 1.095    | 0.90/5.0  | 0.0925  |
| cone fix, fold off, nr 32    | 0.139    | 1.01/2.2  | 0.0053  |
| cone fix, fold ON,  nr 32    | 0.389    | 1.22/6.7  | 0.0302  |

The fold degraded convergence 2.7-4.2x (spin) / 2.2-5.7x (forward) in every
variant (prescribed lambda, Picard-lagged lambda, one-sided clamp), moved cone
occupancy *away* from 1, and did not improve residual-vs-iteration scaling.
Structural reading: at sliding, dividing the folded row by g -> eps_g scales
the row's `J M^-1 J^T` coupling and its rhs term down by ~10x, degenerating
the update toward a noisy pure-diagonal Picard step — the fold throws away
exactly the coupling information Newton needed.

## What actually works instead

- **More NR iterations**: cone fix alone at nr 32 reaches occupancy med 1.01 /
  p90 2.2 and halves the residual. Plain iteration budget beat every clever
  correction tested.
- **Step-lagged lambda_n + limit floor** (separate experiment): removes
  within-step normal<->friction chatter; best spin tracking and jitter of any
  configuration, at the cost of a residual floor on smooth rolling. Env-gated
  option, requires `warm_start.method=pair_aggregate`.
- A full rank-1 Newton would require a row/column J split plus a nonsymmetric
  Krylov solver (GMRES) — a rewrite of the linear-solve layer.
