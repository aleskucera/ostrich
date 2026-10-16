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

Candidates for step 2: joint/control, GCRO-DR, and cross-step as the
expensive reference.

## Step 1d — robustness curve: how accurate must Z be?

`test_scripts/deflation_robustness.py`, plot `docs/deflation-study/deflation_robustness.png`.
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

## Step 2 — wall-time estimate

`test_scripts/deflation_cost.py` (~90 s, GTX 1050). Everything is float32,
like the engine.

### Engine facts that shape the estimate

- The dumped systems come from the library-default config: linear tol
  1e-5, **cap 16**. The `examples/conf/engine/ostrich.yaml` config uses tol
  1e-3, cap 26. Both are evaluated.
- With Jacobi at the default config the engine almost never converges:
  57/67 (obstacle) and 60/65 (surface) held-out solves stop at the cap,
  with true relative residual p90 8e-4 / 2e-2 instead of 1e-5. Savings are
  therefore counted on **capped** iterations.
- All worlds iterate until the slowest one converges (`capture_while`
  over `keep_running`), so for W > 1 worlds the cost is set by the
  **maximum** iteration count in the batch, not the median.
- **Cross-step breaks in float32** (surface, tol 1e-3: 48/65 capped,
  residual 0.36). It partially captures ultra-soft modes, which makes E
  too ill-conditioned for single precision (see step 1d). It was dropped.
- **Block-Jacobi on the joint block** (exact 18×18 inverse, Jacobi
  elsewhere, no projection) is no substitute for joint/control deflation:
  obstacle 14 vs 7, surface 16 vs 12 (tol 1e-5, cap 16). The gain comes
  from eliminating the joint block exactly.

### Measurement

The engine's real `PCRSolver` + `SystemOperator` + `JacobiPreconditioner`
run on the dumped systems as one captured CUDA graph, timed for 4 and 32
iterations (convergence disabled). The slope is the cost per iteration;
the intercept is the per-solve part. Deflation is written as Warp kernels
captured in the same graph. The Warp deflated solve matches the offline
`deflated_pcr` (same iteration count on 22/23 checked systems, off by one
on the other; residuals equal up to float32 noise).

Timing is time-based (≥ 0.5 s warm-up, ≥ 1 s measurement per graph). The
GTX 1050 only reaches full clocks under sustained load: with a short
warm-up, Jacobi PCR measured 67 µs per iteration at W = 1 instead of 41.
Earlier numbers in this section were affected and have been replaced.

**Kernel design (optimized for a single world).**
- Z and AZ are stored transposed (k × slots), so neighbouring threads
  read neighbouring memory.
- The 402-long dot products (Zᵀb, E = ZᵀAZ, the projection coefficients)
  are block reductions, one thread block per world (Warp tiles), not
  serial loops.
- A is symmetric, so Zᵀ(Ax) = (AZ)ᵀx. The projection coefficients
  s = E⁻¹(AZ)ᵀx are therefore computed from the matvec's *input*, and
  z = Ax − AZ·s is fused into the matvec's gather kernel. A deflated
  iteration has **one extra launch** compared with Jacobi PCR.
- E⁻¹ comes from a per-k generated, fully unrolled register kernel
  (Cholesky, L⁻¹, L⁻ᵀL⁻¹; one thread per world). A loop version keeping
  the matrix in memory took 1.2 ms at k = 18. Warp's cooperative
  `tile_cholesky` needs MathDx, which fails on this sm_61 GPU; worth
  trying on a newer GPU.

| | W = 1 | W = 64 | W = 1024 |
|---|---|---|---|
| Jacobi PCR: fixed + per iteration | 0.11 ms + 40.7 µs | 0.15 ms + 119 µs | 1.33 ms + 1.24 ms |
| deflation, k = 18: fixed+setup + per iteration | 0.19 ms + 50.3 µs | 1.31 ms + 204 µs | 17.2 ms + 2.19 ms |
| deflation, k = 16: fixed+setup + per iteration | 0.18 ms + 50.2 µs | 1.16 ms + 196 µs | 15.0 ms + 2.08 ms |
| of the setup: A·Z (k = 18) | 21 µs | 302 µs | 4.8 ms |
| of the setup: Cholesky + inverse (k = 18) | 33 µs | 49 µs | 318 µs |
| one A matvec (reference) | 15 µs | 32 µs | 367 µs |
| GCRO-DR Z build, k = 16 (eager torch, not optimized) | 1.38 ms | 36.6 ms | 566 ms |

At W = 1 deflation adds **+80 µs per solve** (A·Z, E, Cholesky, Zᵀb,
the init and final projections) and **+9.6 µs per iteration (+24%)**.
At W ≥ 64 the dense k-column work dominates. A·Z is k matvec-equivalents
(13× a matvec at W = 1024), and each iteration reads AZ twice, which is
more data than the sparse matvec itself. That part is inherent to a
dense Z and was not optimized further (single-world focus).

### Estimate: time per solve

time = fixed+setup + Z build + iterations × per-iteration, all measured.
For W > 1, iterations = expected maximum over the batch. "Ceiling" = zero
overhead (only iterations saved).

**Single world (W = 1), ms:**

| config | Jacobi | joint/control | GCRO-DR k=16 |
|---|---|---|---|
| obstacle, 1e-5/cap16 | 0.11 + 15.5 × 0.041 = **0.74** | 0.19 + 0 + 6.7 × 0.050 = **0.53 (−29%)** | 0.18 + 1.38 + 13.2 × 0.050 = 2.23 (+199%) |
| obstacle, 1e-3/cap26 | 0.11 + 12.5 × 0.041 = **0.62** | 0.19 + 0 + 4.8 × 0.050 = **0.43 (−31%)** | 0.18 + 1.38 + 6.9 × 0.050 = 1.91 (+207%) |
| surface, 1e-5/cap16 | 0.11 + 15.8 × 0.041 = **0.75** | 0.19 + 0 + 12.1 × 0.050 = 0.80 (+6%) | 0.18 + 1.38 + 12.1 × 0.050 = 2.17 (+187%) |
| surface, 1e-3/cap26 | 0.11 + 17.9 × 0.041 = **0.84** | 0.19 + 0 + 9.0 × 0.050 = **0.64 (−23%)** | 0.18 + 1.38 + 8.5 × 0.050 = 1.99 (+137%) |

**Batched (Δ vs Jacobi, measured / ceiling):**

| config | source | W = 64 | W = 1024 |
|---|---|---|---|
| obstacle, 1e-5/cap16 | joint/control | +34% / −52% | +54% / −53% |
| obstacle, 1e-3/cap26 | joint/control | +24% / −58% | +36% / −61% |
| surface, 1e-5/cap16 | joint/control | +124% / 0% | +147% / 0% |
| surface, 1e-3/cap26 | joint/control | +55% / −29% | +75% / −26% |
| any | GCRO-DR k=16 | +1161% to +1904% | +1722% to +2799% |

GCRO-DR k = 8 is no better than k = 16.

### Findings

1. **Single world: joint/control deflation beats Jacobi PCR** in 3 of 4
   configurations: obstacle −29% / −31%, surface at tol 1e-3 −23%. It
   is roughly even on surface at the default config (+6%): there it saves
   only 16 → 12 iterations, which (≈ 34 µs) does not pay for the +80 µs
   setup and the +24% per iteration. These are linear-solve times; PCR is
   ~85% of the step (eisenstat_walker.md).
2. **Batched: joint/control loses** (+24% to +147%). The per-solve
   setup and the per-iteration AZ traffic scale with W·k·N_c, and on
   surface the batch maximum stays at the cap. Faster large-W kernels
   would narrow it, but the ceiling on surface at the default config is 0%.
3. **GCRO-DR is not competitive.** Its Z build (harmonic Ritz: extra A
   products, QR, eigensolve per world, here eager torch) costs 1–2× a whole
   Jacobi solve at W = 1 and far more in batch. Even with a free Z build
   it would gain only at tol 1e-3 (W = 1: obstacle −15%, surface −27%)
   and lose at the default config. Any dense Z, including a network's,
   has the same per-iteration traffic problem at large W.
4. **Accuracy is a second, unmeasured benefit.** At the default config
   Jacobi returns unconverged solutions (residual p90 8e-4 obstacle,
   2e-2 surface), while joint/control deflation reaches 6e-6 / 2e-4. If
   more accurate linear solves cut Newton iterations, the gain grows.
   Measuring it needs an in-engine run, i.e. a change to `src/`, which
   this study does not make.

## Direct-solve baseline: is iterating worth it at all?

Every number above compares PCR variants with each other. The obvious
reference is a direct solve, since the systems are small. Two variants:

- **Dense Cholesky of A** (W × n × n, `cholesky_ex` + two triangular
  solves). "Padded" uses all 402 slots with identity on inactive rows (a
  static shape, as a CUDA graph needs). "Compact" keeps only n_max = the
  scene's maximum active row count (67 / 170). That is optimistic: it needs
  a gather and a static per-scene bound.
- **Body-space solve** (Woodbury, i.e. static condensation onto the body
  DOFs). $A = G M^{-1} G^\top + C$ with $G$ the n × 6N_b constraint
  Jacobian. Helhest has N_b = 4, so $A$ is rank 24 plus a diagonal. Every
  active row has c > 0 (joint 0.011, control 0.22, normal 0.11–0.95,
  friction 1e-6 … 3.3e6, plus reg 1e-6), so

$$
(M + G^\top C^{-1} G)\,v = G^\top C^{-1} b,\qquad x = C^{-1}(b - Gv),
$$

  which needs a 24 × 24 Cholesky per world, independent of the number of
  active contacts. The most-null directions of A (null(Gᵀ), held only by
  c) are the ultra-soft modes that slow PCR. Here they are handled
  exactly, by the per-row division by c.

Setup: float32, GPU (GTX 1050), torch CUDA graph, assembly from the raw
engine arrays included (G scatter, A or K, factor, solve). M is built
outside the graph (the engine has it; `linalg.inv` is not capturable).
cuSOLVER is forced, because MAGMA's batched `potrs` is not capturable.
The PCR side reuses the step-2 engine graph: fixed + E[max iterations over
W systems] × per-iteration. "jc + 8 strength" is joint/control plus 8
contact rows eliminated as unit vectors, picked by the largest AMG
strength $\max_{j\ne r}|A_{rj}|/\sqrt{A_{rr}A_{jj}}$
(`test_scripts/deflation_rowsel.py`). It matches the eigenvector
leverage oracle within ±0.3 iterations. Its deflation timings come from
the dense kernels and are an upper bound.

**Accuracy**, float32 relative residual ‖b − Ax‖/‖b‖ (A, b in float64),
held-out split, p50 / p90 / max. "+1 ref." adds one step of iterative
refinement with the same factor (one A product + one more solve, not
included in the timings):

| solver | obstacle | obstacle +1 ref. | surface | surface +1 ref. |
|---|---|---|---|---|
| dense Cholesky | 9e-8 / 2e-7 / 3e-7 | 7e-8 / 1e-7 / 3e-7 | 1e-7 / 9e-7 / 2e-6 | 1e-7 / 6e-7 / 2e-6 |
| body-space 24 × 24 | 3e-5 / 3e-4 / 9e-4 | 3e-6 / 2e-5 / 8e-5 | 3e-6 / 1e-5 / 4e-5 | 6e-8 / 3e-7 / 2e-6 |

For comparison, the PCR residual p90 is: Jacobi 8e-4 / 2e-2 (default
config), joint/control 6e-6 / 2e-4. Body-space accuracy is limited by the
sticking friction rows (c = 1e-6 → weight 1e6 in K).

**Time per solve, ms** (PCR at yaml 1e-3/cap26; default 1e-5/cap16 in
parentheses):

| obstacle | W = 1 | W = 64 | W = 1024 |
|---|---|---|---|
| body-space 24 × 24 | **0.115** | 0.785 | 11.0 |
| dense compact (67) | 0.164 | **0.721** | **9.1** |
| dense padded (402) | 0.656 | 13.7 | 207 |
| PCR Jacobi | 0.622 (0.747) | 1.87 (1.91) | 19.3 (18.2) |
| PCR joint/control | 0.445 (0.541) | 2.23 (2.41) | 25.5 (27.3) |
| PCR jc + 8 strength | 0.596 (0.645) | 2.79 (3.00) | 34.1 (36.2) |

| surface | W = 1 | W = 64 | W = 1024 |
|---|---|---|---|
| body-space 24 × 24 | **0.108** | **0.788** | **11.1** |
| dense compact (170) | 0.280 | 2.42 | 33.6 |
| dense padded (402) | 0.639 | 13.8 | 207 |
| PCR Jacobi | 0.837 (0.750) | 3.33 (2.11) | 34.4 (21.8) |
| PCR joint/control | 0.656 (0.811) | 4.81 (4.45) | 57.3 (51.2) |
| PCR jc + 8 strength | 0.608 (0.656) | 3.59 (4.01) | 45.2 (50.0) |

### Findings

1. **The body-space direct solve beats every PCR variant at every W on
   both scenes.** It is 3.9–5.6× faster than the best PCR at W = 1, 2.4–2.7×
   at W = 64 and 1.7–2.0× at W = 1024.

   Accuracy:
   - It already beats Jacobi PCR at the caps without refinement (p90 3e-4
     vs 8e-4 on obstacle, 1e-5 vs 2e-2 on surface).
   - With one refinement step (2e-5 / 3e-7) it is about joint/control's
     level on obstacle and better on surface.
   - Dense Cholesky reaches ~1e-7 on both scenes.

   Its cost does not depend on
   the active row count (same time on both scenes) or on the iteration
   count, and it has no batch maximum: every world costs the same.
2. **Dense Cholesky is competitive only when compacted.** On obstacle
   (n_max 67) it is slightly faster than body-space at W ≥ 64. On surface
   (170) it is 3× slower. Padded to 402 slots it loses to PCR.
3. **This supersedes the deflation work for these scenes.** Deflation
   (joint/control, row selection, any network-predicted Z) improves PCR by
   at most ~30% at W = 1 and loses in batch, while the direct solve is
   several times faster everywhere. A network for Z cannot close that gap.
4. **Where PCR should still win.** Its cost grows linearly with system
   size, while the dense solve grows with the cube of 6N_b (body-space)
   or of n (dense). So PCR is expected to win with many bodies (piles,
   large maximal-coordinate robots), with deformables, or when c → 0
   makes C⁻¹ unusable. These cases are not measured. The crossover in
   6N_b on the GPU is unknown. A size-based dispatch (body-space when 6N_b
   is small, PCR otherwise) is the natural engine design.
5. **Not yet measured:**
   - in-engine time per step;
   - the effect of exact linear solves on the Newton iteration count;
   - a fused Warp implementation (the torch version is unoptimized: dense
     G scatter, generic batched potrf);
   - the adjoint, which can reuse the same factor, since A is symmetric
     (offline accuracy check in the next section).

   All of these need a `src/` change.

## Adjoint accuracy: how wrong is the one-step gradient with capped PCR?

`base_engine.step_backward` computes, at the converged state,

$$
d = M^{-1} f,\quad A\,w = J d \;\text{(PCR, forward tol/cap, } x_0 = 0),\quad
w_u = d - M^{-1} J^\top w ,
$$

where $f = -\nabla_u L - h G^\top \nabla_q L$, and $w$ also seeds the pose VJP. With
the defaults (`friction_linearization = "true"`, `adjoint.regularization = 0`),
$A$ is the forward operator.

Offline stand-in: the last NR system of each dumped step (C not zeroed),
25 steps per scene, 8 loss draws each. Loss gradient $f$:

- **random:** Gaussian on all bodies;
- **chassis:** Gaussian on body 0 only, e.g. a loss on the final chassis
  pose.

The reference is a float64 dense solve. Everything else is float32.
"Body-space native" uses the Woodbury identity in the adjoint direction,

$$
w_u = (M + J^\top C^{-1} J)^{-1} f,\qquad w = C^{-1} J\, w_u
$$

(from $A w = J d \Rightarrow J w_u = C w$). It reuses the forward
24 × 24 factor.

Relative error of $w_u$ (the gradient carried to the previous step).
Entries are p50 / p90, then the share of draws with > 10% error:

| solver | obstacle random | obstacle chassis | surface random | surface chassis |
|---|---|---|---|---|
| PCR Jacobi 1e-5/cap16 | 5e-2 / 0.57, 42% | 2e-2 / 0.53, 41% | 4e-2 / 0.16, 22% | 3e-2 / 0.19, 21% |
| PCR Jacobi 1e-3/cap26 (yaml) | 4e-2 / 0.46, 39% | 2e-2 / 0.42, 39% | 8e-3 / 6e-2, 3% | 3e-3 / 5e-2, 6% |
| PCR joint/ctrl 1e-5/cap16 | 6e-3 / 0.15, 14% | 6e-5 / 0.11, 11% | 1e-4 / 8e-3, 0.5% | 6e-5 / 9e-3, 3% |
| PCR joint/ctrl 1e-3/cap26 | 2e-2 / 0.51, 42% | 3e-3 / 8e-2, 9% | 4e-3 / 4e-2, 3% | 6e-4 / 5e-3, 1% |
| body-space w, then $w_u = d - M^{-1}J^\top w$ | 0.14 / 2.1, 51% | 9e-3 / 4e-2, 0% | 1e-3 / 0.28, 18% | 1e-4 / 1e-2, 0% |
| same + 1 refinement step | 6e-3 / 0.13, 15% | 2e-4 / 3e-3, 0% | 3e-6 / 9e-3, 2% | 8e-7 / 4e-4, 0% |
| **body-space native** | **2e-4 / 2e-3, 0%** | **2e-4 / 2e-3, 0%** | **3e-6 / 4e-4, 0%** | **4e-6 / 5e-4, 0%** |

Worst-case cosine between computed and exact $w_u$:

- PCR Jacobi: 0.61–0.72 on obstacle, 0.97–0.99 on surface;
- body-space with the subtraction: −0.51 (obstacle) and 0.31 (surface), random loss;
- body-space native: 1.0000 everywhere (max error 5e-3).

The relative error of $w$ itself, for PCR Jacobi on obstacle, is p50
0.2–0.4 and p90 ~0.9.

### Findings

1. **Capped PCR makes the one-step adjoint substantially wrong on
   obstacle.** At the yaml config, 39% of draws have > 10% error in
   $w_u$, and the worst cosine is 0.68. Joint/control deflation helps only
   partly. Surface is milder (3–6% of draws > 10% at yaml, 21% at
   1e-5/cap16).
2. **The cause is cancellation, not only solver tolerance.** $w_u = d -
   M^{-1}J^\top w$ subtracts two nearly equal vectors when bodies are
   strongly constrained (joints, sticking contact), so any relative error
   in $w$ is amplified. Even an accurate $w$ (body-space, 3e-3) gives 14%
   p50 error in $w_u$ on obstacle through this formula.
3. **The body-space adjoint avoids the subtraction.** It computes $w_u$
   directly as $K^{-1} f$ with the forward factor: two triangular solves,
   ≤ 0.5% error in float32 on every draw. Together with the forward
   result above, the same factor gives a 2–6× faster forward solve and an
   essentially exact, almost free adjoint.
4. **Open.**
   - This is single-step only; errors compound over a horizon through the
     step Jacobians.
   - The offline operator is the last NR system, not exactly the converged
     state.
   - Whether the error matters for optimization (the paper reports ≤ 7%
     FD agreement on its gradient suite; part of that may be linear-solve
     error — untested) needs an in-engine FD check and a trajectory
     optimization run with both adjoints. That is a `src/` change.

## Files

- `test_scripts/deflation.py` — `deflated_pcr`, step-1b checks.
- `test_scripts/deflation_ab.py` — step-1c table (~20 s on CPU).
- `test_scripts/deflation_robustness.py` — step-1d sweep and plot (~50 s).
- `test_scripts/deflation_cost.py` — step-2 iterations under engine settings, GPU timing, estimate (~90 s).
- `test_scripts/deflation_hybrid.py` — hybrid Z ceiling: joint/control ‖ exact bottom-k_c of the condensed contact problem (results not written up here; `deflation_rowsel.py` and `deflation_direct.py` import from it).
- `test_scripts/deflation_rowsel.py` — joint/control + k_c contact rows eliminated, picked by AMG strength vs leverage oracle vs random.
- `test_scripts/deflation_direct.py` — direct-solve baseline: accuracy, GPU timing, PCR comparison (~3 min).
- `test_scripts/adjoint_accuracy.py` — one-step adjoint accuracy: capped PCR vs body-space vs exact (~1 min, CPU).
