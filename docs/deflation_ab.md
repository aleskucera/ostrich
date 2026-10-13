# Deflation A/B: projection deflation vs the additive two-level preconditioner

Follow-up to `learned_preconditioner_feasibility.md`. Every subspace probe
there used the **additive** two-level preconditioner

$$
M^{-1} = D^{-1} + Z (Z^\top A Z)^{-1} Z^\top ,
$$

which with an imperfect $Z$ can push eigenvalues *above* the Jacobi
cluster ("over-corrects"). Several negative verdicts (harmonic-Ritz
recycling, the cross-step pipeline) may therefore be verdicts on the
additive form, not on the subspace. This study re-runs them with
**Galerkin / projection deflation**, which for *any* $Z$ keeps the spectrum
of $M^{-1}PA$ inside that of $M^{-1}A$:

$$
E = Z^\top A Z,\quad Q = Z E^{-1} Z^\top,\quad P = I - AQ,\qquad
\text{solve } PA\,\tilde x = Pb \text{ (Jacobi-PCR)},\quad x = Qb + P^\top \tilde x .
$$

Since $b - Ax = Pb - PA\tilde x$ exactly, the inner residual is the true
one; only the relative target is rescaled by $\|b\|/\|Pb\|$.

## Setup

- Implementation: `test_scripts/deflation.py` (`deflated_pcr`, step-1b
  checks), `test_scripts/deflation_ab.py` (step-1c table). Offline,
  PyTorch float64, dense.
- Harness settings as `precond_lab.py`: TOL = ATOL = 1e-5, MAX_ITERS = 64.
  Iterations = PCR iterations; setup (k products with A for $AZ$) is
  reported separately as k.
- Data: regenerated 2026-10-12 (ostrich fbc3efe, newton 102e812).
  Obstacle `helhest_systems.npz` 335 systems (n 31–67); surface
  `helhest_surface_systems.npz` 322 systems (n 50–170). The engine has
  changed since the feasibility doc, so its numbers are re-measured
  here and the Jacobi bars differ slightly from it.
- **Iteration counts only — no wall time yet.** Fewer iterations is
  not a speed-up until the setup and per-iteration projection cost are
  accounted for (step 2).

## Step 1b — sanity checks (all systems)

| | obstacle | surface |
|---|---|---|
| Jacobi PCR, mean | 19.9 | 39.5 (30 unconverged) |
| Z empty ≡ Jacobi | bit-identical, all systems | bit-identical, all systems |
| exact bottom-8, deflation / additive | 16.3 / 16.5 | 21.4 / 20.6 |
| exact bottom-16, deflation / additive | 10.0 / 10.8 | 12.2 / 12.1 |
| random k=16 (3 seeds), deflation / additive | 27.5 / 32.8 | 39.4 / 41.8 |

- Every converged deflated solve meets the true residual target, and $x$
  matches a dense solve. On obstacle it also matches the stored engine
  solution (surface `x_engine` is often unconverged at the engine's
  16-iteration cap, so it is not a valid reference there).
- **Random Z is clearly worse than Jacobi on obstacle (27.5 vs 19.9).
  This is not a bug.** Interlacing holds (κ_eff never increases), and
  rounding, cluster splitting and a larger initial residual were all
  ruled out. The cause is the right-hand side: projecting out random
  directions moves RHS energy *into* the soft modes (obstacle: RHS
  energy in the soft modes 1.9e-4 → 1.3e-3, ~7×; surface unchanged at
  2.4e-2). The theory bounds κ, not the iteration count, so
  "deflation can never be clearly worse than Jacobi" is too strong. With
  the same bad Z, deflation is still better than additive.

## Step 1c — the A/B table

Held-out split = last 20% of systems (obstacle 67, surface 65). Same Z
given to both methods. Columns: median iterations [p90]; wins/losses =
systems faster/slower than Jacobi. k = columns of Z after rank
revealing; k/n = fraction of the system Z spans.

**Obstacle** — Jacobi 19 [22], mean 18.4

| Z source | k | k/n | additive | wins/loss | deflation | wins/loss |
|---|---|---|---|---|---|---|
| exact k=4 | 4 | .07 | 19 [22] | 5/0 | 19 [22] | 5/0 |
| exact k=8 | 8 | .14 | 19 [21] | 13/0 | 19 [22] | 11/0 |
| exact k=16 | 16 | .27 | 12 [13] | 67/0 | 11 [12] | 67/0 |
| exact k=32 | 32 | .54 | 10 [10] | 67/0 | 6 [8] | 67/0 |
| cross-step (N−1) | 16 | .27 | 13 [25] | 51/16 | 11 [17] | 63/4 |
| recycle-exact (prev) | 16 | .27 | 12 [13] | 64/2 | 11 [12] | 67/0 |
| harmonic m=16 | 16 | .27 | 37 [44] | 0/67 | 20 [27] | 13/50 |
| harmonic m=24 | 16 | .27 | 25 [30] | 2/64 | 16 [20] | 52/8 |
| harmonic m=32 | 16 | .27 | 16 [20] | 52/11 | 12 [17] | 64/3 |
| GCRO-DR accumulated | 16 | .27 | 18 [26] | 31/30 | 14 [21] | 53/11 |
| null(Jᵀ) | 35 | .59 | 15 [16] | 58/0 | 12 [15] | 67/0 |
| joint/control | 18 | .31 | 15 [17] | 67/0 | 7 [7] | 67/0 |
| enriched (null ‖ joint) | 53 | .90 | 17 [19] | 59/4 | 4 [4] | 67/0 |

**Surface** — Jacobi 29 [42], mean 28.7

| Z source | k | k/n | additive | wins/loss | deflation | wins/loss |
|---|---|---|---|---|---|---|
| exact k=4 | 4 | .03 | 19 [34] | 64/0 | 19 [34] | 65/0 |
| exact k=8 | 8 | .06 | 15 [19] | 65/0 | 15 [19] | 65/0 |
| exact k=16 | 16 | .12 | 10 [12] | 65/0 | 10 [13] | 65/0 |
| exact k=32 | 32 | .24 | 10 [11] | 65/0 | 9 [10] | 65/0 |
| cross-step (N−1) | 16 | .12 | 29 [54] | 29/28 | 20 [35] | 65/0 |
| recycle-exact (prev) | 16 | .12 | 11 [16] | 60/2 | 10 [14] | 65/0 |
| harmonic m=16 | 16 | .12 | 27 [42] | 29/33 | 15 [24] | 64/0 |
| harmonic m=24 | 16 | .12 | 19 [37] | 57/6 | 15 [28] | 64/1 |
| harmonic m=32 | 16 | .12 | 13 [25] | 59/3 | 11 [23] | 65/0 |
| GCRO-DR accumulated | 16 | .12 | 13 [23] | 59/2 | 11 [18] | 65/0 |
| null(Jᵀ) | 109 | .82 | 23 [25] | 49/10 | 19 [22] | 60/1 |
| joint/control | 18 | .14 | 29 [42] | 40/16 | 12 [21] | 65/0 |
| enriched (null ‖ joint) | 127 | .95 | 22 [25] | 54/3 | 6 [6] | 65/0 |

The additive column reproduces the earlier probes (recycle-exact 11.9,
harmonic 37.7 / 24.7 / 15.9, GCRO-DR 18.6, null 15.0, joint 14.7,
enriched 16.4 — means, obstacle). Cross-step is 15.0 here vs 14.4 in
`cross_step_stability.py`, which holds out by step (80 systems) rather
than by system. Harmonic sources use a seeded random RHS, so they
differ by a few tenths from the unseeded originals. All converged
deflated solves meet the true residual target (max res/target ≤ 1.00).

### Z sources and how realistic they are

| Source | What Z is | Cost to produce Z | Realistic in the engine |
|---|---|---|---|
| exact k | bottom-k eigvecs of the Jacobi-scaled current A | dense eigensolve, O(n³) | no — ceiling only |
| cross-step | exact bottom-16 of step N−1's last system, reused for all of step N | one eigensolve per timestep, off the critical path | partly (async eigensolve) |
| recycle-exact | exact bottom-16 of the previous solve | eigensolve per solve | no — reference only |
| harmonic m | harmonic Ritz from m Krylov vectors of the previous solve | m×m eigenproblem | m=16 yes; m ≥ 24 exceeds the engine's 16-iteration cap |
| GCRO-DR | carried U, refreshed each solve from [U ‖ 16 Krylov vecs] | QR of 32 vectors + 32×32 eigenproblem | yes |
| null(Jᵀ) | basis of {x : Jᵀx = 0} (A = C there) | SVD of J per Newton iteration; large k | partly |
| joint/control | unit vectors on the 18 always-active joint/control rows | none (AZ = 18 columns of A, E = joint block) | yes |
| enriched | null(Jᵀ) ‖ joint/control | SVD + near-full k | no — k/n ≈ 0.9 is a direct solve |

### Findings

1. **Deflation is never meaningfully worse than additive on the same Z,
   and is much better when Z is imperfect.** With exact Z the two tie
   (within ±0.5 iterations). With approximate Z, additive loses to
   Jacobi on many systems (cross-step surface 28 losses, harmonic m=16
   obstacle 67/67) and deflation removes almost all of those losses.
   The earlier negative verdicts on recycling and the cross-step pipeline
   were largely verdicts on the additive form.
2. **Deflation is not a win against Jacobi for every Z.** Harmonic m=16
   on obstacle still loses (median 20 vs 19, 50/67 slower). Exact k ≤ 8
   on obstacle barely moves (the bad band there is wider than 8).
3. **joint/control is the standout cheap source.** Deflating the 18
   always-active joint/control rows means solving the stiff joint block
   exactly and running PCR on the Schur complement of the rest (static
   condensation). Obstacle 19 → 7, surface 29 → 12, no losses, fixed
   k = 18, nothing to compute for Z. Additive failed with the same Z
   because adding $A_{JJ}^{-1}$ on top of the Jacobi diagonal
   double-counts those rows instead of eliminating them.
4. **Large-k sources are misleading on iteration count.** null(Jᵀ)
   (k/n 0.59 / 0.82) and enriched (0.90 / 0.95) move most of the work
   into the k×k coarse factorization; enriched's 4–6 iterations are
   essentially a direct solve. Step 2 must charge for it.

### Decision gate (step 1 → step 2)

Criterion: with deflation, median ≥ ~20% below Jacobi on BOTH scenes and
no p90 loss vs Jacobi.

| Source | obstacle median | surface median | p90 ok | Verdict |
|---|---|---|---|---|
| joint/control | −63% | −59% | yes | **qualifies, cheapest** |
| GCRO-DR accumulated | −26% | −62% | yes | **qualifies** |
| harmonic m=32 | −37% | −62% | yes | qualifies, but m=32 > engine's 16-iteration budget |
| harmonic m=24 | −16% | −48% | yes | misses on obstacle |
| harmonic m=16 | +5% | −48% | no (obstacle) | fails |
| null(Jᵀ) | −37% | −34% | yes | qualifies, but k large |
| cross-step (N−1) | −42% | −31% | yes | qualifies; carries a per-step eigensolve |

Step 2 (wall-time estimate) still needs to be done. Candidates:
joint/control, GCRO-DR, and cross-step as the expensive reference.

## Step 1d — robustness curve: how accurate must Z be?

`test_scripts/deflation_robustness.py`, plot `docs/deflation_robustness.png`.
Start from the exact bottom-16 eigenvectors $V_k$ of the Jacobi-scaled
$A_s = DAD$ and rotate them by a controlled angle:
$Z_s = V_k\cos\theta + W\sin\theta$ with $W \perp V_k$, so every principal
angle to the exact subspace is exactly θ. Two kinds of error:
**random** (isotropic $W$, 3 seeds) and **spectral** ($W = V_{k+1..2k}$,
i.e. each mode mixes with a neighbouring eigenvector, like Ritz errors).
Angles are measured in the **scaled** space. A first attempt measured
them in the original space, which is meaningless here: diag(A) spans
orders of magnitude, so 5° there is a huge tilt in the space the
preconditioner sees.

Deflation median iterations [additive in brackets], held-out:

| θ (°) | obstacle random | obstacle spectral | surface random | surface spectral |
|---|---|---|---|---|
| 0 | 11 [12] | 11 [12] | 10 [10] | 10 [10] |
| 0.3 | 17 [20] | 11 [13] | 10 [11] | 10 [11] |
| 1 | 22 [31] | 13 [13] | 10 [12] | 10 [11] |
| 3 | 31 [45] | 16 [19] | 11 [13] | 10 [12] |
| 10 | 43 [64] | 22 [29] | 13 [17] | 13 [17] |
| 20 | 57 [64] | 25 [30] | 17 [23] | 17 [21] |
| 30 | 63 [64] | 24 [32] | 20 [28] | 20 [26] |
| 60 | 53 [61] | 29 [31] | 29 [37] | 27 [31] |
| 90 | 26 [37] | 15 [20] | 29 [37] | 27 [32] |
| Jacobi | 19 | 19 | 29 | 29 |

Real k = 16 sources: angle (°) of the i-th scaled bottom eigenvector to
span(Z), p50, and deflation median iterations:

| source | obstacle i=1 / 4 / 8 / 12 / 16 | iters | surface i=1 / 4 / 8 / 12 / 16 | iters |
|---|---|---|---|---|
| cross-step (N−1) | 0 / 0 / 0.1 / 0.7 / 0.7 | 11 | 49 / 50 / 56 / 54 / 44 | 20 |
| recycle-exact | 0 / 0 / 0 / 0.1 / 0.1 | 11 | 0.1 / 0.2 / 0.2 / 0.2 / 0.2 | 10 |
| harmonic m=16 | 74 / 71 / 59 / 54 / 66 | 20 | 30 / 58 / 61 / 59 / 65 | 15 |
| harmonic m=32 | 60 / 62 / 19 / 1.1 / 6.8 | 12 | 1.3 / 1.2 / 4.4 / 3.8 / 4.5 | 11 |
| GCRO-DR | 90 / 62 / 44 / 12 / 7.3 | 14 | 2.4 / 5.9 / 7.0 / 3.2 / 8.7 | 11 |

Spectral gap λ₁/λ₁₇ (scaled), p50: obstacle 7e-6, surface 8e-2.

### Findings

1. **Deflation ≤ additive at every angle**, on both scenes and both
   error types. It also degrades more slowly.
2. **Surface (mild gap) gives a clean, monotone curve.** To pass the gate
   (≤ 23) a uniform error of ≲ 20° is enough; to stay near the ceiling
   (≤ 12) it must be ≲ 5°.
3. **Obstacle (extreme gap) is non-monotone, and one angle is not a valid
   accuracy spec.** The bottom few modes are ~10⁵× softer than the rest
   but carry almost no RHS energy, which is why Jacobi needs only 19
   iterations despite κ ≈ 10⁵. Capturing such a mode *partially* is the
   worst case: its leftover eigenvalue is ≈ λᵢ / sin²θᵢ, still tiny,
   and ‖Pb‖ grows (4–40 × ‖b‖ at 5–30°). A uniform random error of just
   1° already makes it worse than Jacobi, and 90° (ignoring the band)
   beats 10–60°. The real sources that work on obstacle (GCRO-DR,
   harmonic m=32) are exactly those that *ignore* the ultra-soft modes
   (60–90°) but capture modes ~12–16 to within 1–12°. Their RMS angle
   (34–38°) would predict 25+ iterations on the uniform curve; they get
   12–14.
4. **Implication for step 3 (a network).** The NeurKItt projection loss
   (uniform over the subspace) targets the wrong quantity here. On
   obstacle, a network with a uniform error of a few degrees would lose
   to Jacobi. A target must either exclude the ultra-soft, RHS-free
   modes or weight modes by how much they cost iterations (e.g. RHS
   content); the success metric must stay iterations under deflation.

## Files

- `test_scripts/deflation.py` — `deflated_pcr`, step-1b checks.
- `test_scripts/deflation_ab.py` — step-1c table (~20 s on CPU).
- `test_scripts/deflation_robustness.py` — step-1d sweep and plot (~50 s).
