# Can We Warm-Start PCR?

The Preconditioned Conjugate Residual solver in `axion/optim/pcr_solver.py`
runs once per Newton-Raphson iteration. NR currently calls it 5–16 times
per simulation step, and PCR inside each call runs up to
`max_linear_iters = 16`. That makes PCR the dominant cost (~85% of step
time per `engine_profiler` runs). Cutting its iters or reusing
information across calls is the most promising remaining lever.

This note enumerates what "warm-starting PCR" could actually mean,
which options are dead ends, and which are worth implementing. Order
roughly matches "obviousness on first read" — the dead ends are first.

## Option 1: Seed `Δλ_{k+1}` with `Δλ_k` (within an NR step)

The natural-sounding option. PCR solves `A_k · Δλ_k = b_k` at NR
iter k, then `A_{k+1} · Δλ_{k+1} = b_{k+1}` at iter k+1. If A and b
change slowly, why not start PCR's iterate at the previous solution?

**Implementation cost**: zero. Today
`base_engine.py:428` does `self.data._dconstr_force.zero_()` right
before each PCR call. Just removing that line is the change.

**Why it doesn't help**: as NR converges, `Δλ → 0` (each Newton step
gets smaller). At a late NR iter where the true `Δλ_{k+1}` is tiny,
seeding with the LARGER `Δλ_k` puts PCR's initial residual
`||b - A·Δλ_k||` *above* `||b||`, since `A·Δλ_k` is order
`||Δλ_k|| · ||A||` while `b = -r(x)` is order `||residual||` which is
small near convergence. PCR then has to spend iterations *undoing* the
warm-start before it can converge. Seed-from-zero is closer to the
truth than seed-from-last-Δλ once Newton is in the basin.

The one regime where this could help is iter 0 — but iter 0 doesn't
have a "previous Δλ". So zero benefit here.

**Verdict**: dead end. Worth a 5-minute test (delete the `.zero_()`,
measure on `obstacle_benchmark`) just to confirm the regression — but
do not invest beyond that.

## Option 2: Cross-step warm-start (seed first PCR call of step N+1 with last step's converged λ)

Same idea, scaled up: at the start of step N+1, NR iter 0 starts at
λ = 0; PCR will solve for Δλ such that `0 + Δλ ≈ λ_converged^{N+1}`.
If the next step's solution is close to the previous step's, why not
warm-start `Δλ` with `λ_converged^N`?

**Implementation cost**: a few lines — copy `_constr_force` into
`_dconstr_force` at the top of NR iter 0, then DON'T zero
`_constr_force` (which would then need to track the cumulative
update — also non-trivial).

**Why it doesn't help — and would actively hurt**: this is
mathematically equivalent to starting NR iter 0 at λ = λ_converged^N.
We tried this directly in the cross-step warm-start branch and
documented the failure in
[`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md):
the FB Jacobian is degenerate at the touching corner `(λ > 0, g = 0)`
where `∂φ_FB/∂λ = 1 - λ/√(λ² + g²) → 0`, and Newton diverges from
any non-zero starting iterate near that corner. Laundering the
warm-start through PCR's initial guess instead of through `_constr_force`
doesn't change the math — Newton still has to evaluate the FB
Jacobian at the warm-started state, and that Jacobian is still
degenerate.

**Verdict**: dead end, same root cause as the iterate-seeding failure.

## Option 3: Reuse the preconditioner across NR iterations

Today `preconditioner.update()` runs every NR iter. The Jacobi
preconditioner is `1/diag(A)`, which costs a full matvec-of-diagonal
plus an element-wise inverse. If A's diagonal barely changes between
NR iters (which it does once Newton is in the convergence basin),
recomputing it each iter is wasted GPU time.

**Implementation cost**: low. Two reasonable strategies:
  * Refresh every K iters (e.g., K = 4): simple, predictable, no
    measurement needed at runtime.
  * Refresh when residual stops dropping fast enough: needs a
    cheap residual-history check, but adapts to scenes.

**Why it might help**: the preconditioner update is a kernel launch
plus one diagonal extraction — small, but adds up over 5–16 NR iters
times 200 steps. Concrete impact needs measurement (the diagonal
extraction is probably memory-bandwidth-bound and may be cheap), but
it's the kind of change that's easy to A/B test cleanly.

**Risk**: a stale Jacobi preconditioner is just a worse preconditioner,
which means PCR iters might GO UP — net loss if those extra PCR iters
cost more than the saved updates. Need to measure on a representative
scene before committing.

**Verdict**: worth a 1–2 day prototype + benchmark. The mechanism is
cheap, the risk is bounded, and the result tells you something about
how much A's diagonal really moves.

## Option 4: Krylov subspace recycling

The textbook answer when the matrix changes slowly across solves.
After each PCR solve, save the basis vectors of the Krylov subspace
`{r₀, A·r₀, A²·r₀, …}` (or rather, an orthogonalized version of
them). Use approximate eigenvectors of A from those basis vectors to
"deflate" the next PCR solve — start the new Krylov iteration in a
subspace orthogonal to the small-eigenvalue directions that slow
ordinary CG/CR.

**Implementation cost**: high. Needs:
  * Modified Gram-Schmidt or reorthogonalization to keep the basis
    well-conditioned;
  * Eigenvector estimation (Rayleigh-Ritz on the saved basis);
  * Storage for ~10–20 vectors per system (small relative to the
    contact-constraint state, but adds GPU memory pressure);
  * Careful integration with CUDA graph capture.

References: Parks, de Sturler et al. ("Recycling Krylov Subspaces for
Sequences of Linear Systems") for the canonical algorithm.

**Why it might help (a lot)**: this is exactly the regime where
recycling pays off — PCR runs many times on slowly-varying
matrices. Published speedups range from 1.5× to 5× depending on how
much the matrix moves between solves. For our case (A changes
within an NR step but slowly; A changes more across steps but is
still structurally similar), the benefit is plausibly real.

**Risk**: implementation complexity. Probably 2–3 weeks of work plus
debugging cycles. Worth it only if Options 3 and 5 don't move the
needle.

**Verdict**: real technique, real wins, big project. Defer until
cheaper options are exhausted.

## Option 5: Adaptive `max_linear_iters` / Eisenstat-Walker forcing

PCR already has tolerance-based early exit
(`pcr_solver.py:_check_residuals_kernel` + the `wp.capture_while`
loop at line 295). At late NR iters where `||b||` is small, PCR
naturally exits in fewer iterations than at iter 0. This is the
existing inexpensive-Newton behavior.

The **forcing-term** technique (Eisenstat-Walker, 1996) takes this
further: tighten the linear tolerance only as the NR residual drops,
so early NR iters tolerate sloppy linear solves and only the final
NR iters demand tight ones. Today we use a fixed `linear_atol`; an
adaptive one keyed off the outer NR residual would let the early
PCR calls exit much sooner.

**Implementation cost**: low. One-line change in the PCR call inside
`nr_loop_step`: pass `tol = max(linear_atol_min, eta * outer_res_norm)`
where `eta` is a small constant (~0.01).

**Why it might help**: profiles show many PCR calls hitting
`max_linear_iters`. Many of those are wasted precision — at NR iter
0–2, the linear system is built from a far-from-converged outer
state, so solving it tightly just locks in noise. Loosening the
early calls saves PCR iters per NR iter.

**Risk**: too-loose early PCR can break NR's Q-quadratic convergence
near the solution. The classical Eisenstat-Walker analysis bounds
this (forcing term must shrink to maintain superlinear convergence),
so the standard schedule is well-tested.

**Verdict**: cheap to prototype, well-understood, likely real wins.
Probably the second-best place to look after Option 3.

## Recommendation

Order to chase if optimizing PCR:

1. **Option 5** (adaptive linear tol): cheapest, most well-trodden,
   likely visible win. Do this first.
2. **Option 3** (preconditioner reuse): cheap prototype, bounded
   risk, gives you data on how stable A's diagonal is.
3. **Option 4** (Krylov recycling): only if 5 and 3 didn't get you
   to your performance target.
4. **Options 1 and 2**: don't pursue. They run into either the
   "Δλ → 0 at convergence" wall (Option 1) or the FB-NR
   iterate-seeding wall (Option 2, see
   [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)).

## Postscript: options 3, 4, 5 measured and ruled out

After implementing options 1 and 2 (and reverting them — they're
dead-ends as documented above), we measured the upper bound for
options 3, 4, and 5 directly. None deliver enough to be worth
pursuing on this codebase at this scale.

### Option 3 (preconditioner reuse): 2% ceiling

Per-component profile of obstacle_benchmark (200 steps, 3200 NR
iters total):

```
phase                count     mean ms     share
linear_system        3200       0.049       9.5%
preconditioner       3200       0.010       2.0%   ← option 3 ceiling
cr_solve             3200       0.386      74.4%
step_or_linesearch   3200       0.047       9.1%
convergence_check    3200       0.026       5.0%
```

`preconditioner.update()` is **2% of NR-iter time**. Skipping every
non-first update saves at most ~1.5% of step time. Run-to-run
variance on wall-clock is 3–5%. The maximum possible win is below
noise, even before counting any PCR-iter increase from a stale
preconditioner.

The PCR-doc pitch claimed "5–15%" payoff; that was wrong. Always
profile before committing.

### Option 4 (Krylov recycling): A is too volatile to recycle

Recycling pays off when A's slow-eigenvector subspace persists
across solves. Measured how much A actually moves between PCR calls
in our setup, using two probes:

  1. `||Δdiag(A)|| / ||diag(A)||` — cheap, captures only diagonal.
  2. `||A·v − A_prev·v|| / ||A·v||` for fixed random `v` — captures
     full A change (diagonal + off-diagonal entries).

Within-step (NR iter k vs k-1, same simulation step):

```
  ||Δdiag(A)|| / ||diag(A)|| :  p50 = 7e-2,  p95 = 8e-1,  max = 8
  ||Δ(A·v)|| / ||A·v||       :  p50 = 0.85,  p95 = 4.5,   max = 3e4
```

A changes by **85% per NR iter on average within a single step**.
The diagonal changes "only" 7%, meaning off-diagonal entries shift
much more than diagonals — plausible because each NR iter's
`apply_newton_step` integrates body pose (a wheel at 1.5 m/s with
a Newton-step-driven velocity update can move several cm per iter),
which alters the geometric alignment between J rows of constraints
sharing a body. The diagonal `||J_i row||² / m_eff + C_ii` depends
on a single row's magnitude (less sensitive); the off-diagonal
`J_i · M⁻¹ · J_j^T` depends on inner products of geometrically
shifting rows (more sensitive).

For comparison: published Krylov-recycling work (Parks & de Sturler;
PETSc) targets relative A-change in the 10–20% range. **Our 85%
makes recycling within a step infeasible.**

Across-step (last iter of step n-1 vs first iter of step n):

```
  ||Δdiag|| / ||diag|| :  p50 = 3e-7  (essentially identical)
  ||Δ(A·v)|| / ||A·v|| :  p50 = 2.7e5  (broken — see caveat)
```

The `A·v` measurement here is contaminated by changing constraint
sets between steps: when contacts come/go, the same slot in the
residual vector now indexes a different physical constraint, so a
fixed probe `v` measures slot-relabeling rather than actual matrix
shift. The diagonal measurement (which masks inactive slots) shows
the *active* part of A is essentially unchanged across steps —
because step n's first iter starts with body_pose copied from
step n-1's picked iter (warm-start trajectory continuity).

To get a clean cross-step A-change number, the probe would need to
track active-constraint identity across steps. Doable but moot:
within-step recycling is by far the dominant regime (3200 calls vs
200 step boundaries), and within-step recycling is dead at 85%
relative change.

**Verdict on option 4: not viable for this solver.** Within-step A
changes too much for any subspace recycling scheme. Reproducer:
`test_scripts/measure_A_stability.py`.

### Option 5 (Eisenstat-Walker): tried and reverted

See [`eisenstat_walker.md`](eisenstat_walker.md) for the full
postscript. Summary: implemented, measured, reverted. The "smoothness
assumptions of E-W's classical proofs" don't hold for FB-NR; loose
early-iter linear solves amplify FB-corner degeneracy, breaking
convergence on impact scenes. Even with the tightest η_max bounds
that didn't break convergence, PCR-iter savings were negligible.

### What's left

Of the five options in this doc, four are now ruled out by
measurement (1 and 2 by analysis, 3 and 4 by direct profiling, 5
by implementation+measurement). The remaining linear-solver-side
ideas are at the *kernel* level rather than the *algorithm* level:

  * **Block-Jacobi preconditioner**: per-body-pair (the cleanest
    starting point). See "Block-Jacobi: which block?" below for
    the design space of block schemes considered and why per-body-
    pair is the recommended first attempt.
  * **CG instead of CR**: ruled out by the source paper's
    Figure 18 (Macklin et al. 2019, §10.4). They tested Jacobi /
    Gauss-Seidel / PCG / PCR on three test cases and found PCR
    "an order of magnitude lower residual for the same iteration
    count compared to other methods." Specifically: PCR's
    monotonicity in residual norm is the right property for our
    inexact-Newton outer loop. PCG can have non-monotone residual
    behavior that breaks early-termination. **Don't swap to CG.**
  * **Sub-kernel optimization** (kernel fusion in the CR inner
    loop; reduce kernel-launch overhead; lower precision in inner
    products). Tedious but tractable, 5–15% plausible.

Or, for a structural rethink that **bypasses the FB corner entirely**
(which has been the dominant source of dead-ends across this entire
investigation), see APGD discussion in solver-side notes — that's a
4–6 week reformulation, not a linear-solver swap.

### Block-Jacobi: which block?

A first attempt used **3×3 per-contact blocks**, grouping each
contact's `(λ_n, λ_t1, λ_t2)` into a 3×3 sub-matrix of A. It failed.
The structural reason becomes clear from the geometry:

For a single contact, `J_n` is along the contact normal; `J_t1` and
`J_t2` are tangent to it — **geometrically orthogonal by
construction**. So `J_n · J_t1ᵀ = 0`, and `J_n · M⁻¹ · J_t1ᵀ ≈ 0`
too (M⁻¹ is mass-diagonal). The 3×3 block of A within a single
contact is **already approximately diagonal**. Inverting it gives
nearly the same M⁻¹ as plain Jacobi. The within-contact coupling
is genuinely small.

The real off-diagonal coupling lives **between different contacts
that share a body**. For a wheel with 4 ground contacts, the
wheel's mass M_wheel⁻¹ couples those 4 contacts through
`J_i · M⁻¹ · J_jᵀ`. For Helhest at 16 bodies × ~3 contacts each:
each body's "constraint group" is ~10–13 dim (3–4 contacts × 3
components + 1 joint).

### Five block schemes considered

For each constraint i, A_ij is non-zero iff i and j share a body
(through the J·M⁻¹·Jᵀ term). The natural block schemes group
constraints by which bodies they touch.

#### 1. Per-body with ownership rule

Each constraint assigned to one canonical body (e.g., the lighter
body, or the lower body-id). No overlap, but body-body constraints
lose half their coupling — only one body's M⁻¹ contribution gets
captured. For our problem the heavier body's contribution is
small anyway (we measured this; chassis-touching off-diagonals are
~0), so the loss is mild but real.

**Drawback**: the ownership rule is arbitrary, and the dropped half
of body-body coupling matters for stiff-joint problems.

#### 2. Per-body-pair (recommended starting point)

For each pair of bodies (B₁, B₂), group all constraints between
them. Ground contacts (b₂ = −1) belong to (b₁, ground) pair. Each
constraint has exactly one pair → **no overlap, no ownership rule**.

Block size: chassis-wheel pair ~5 (joint DOFs); wheel-ground pair
~12 (4 contacts × 3 components). Comparable to per-body sizes.

**Captures** both bodies' contributions to A_ij within the pair —
strictly more than per-body's ownership-rule version.

**Misses** cross-pair coupling through a shared body (e.g.,
(chassis, wheel-1) and (chassis, wheel-2) joint constraints share
chassis, with A_ij ≠ 0 through chassis). For Helhest the chassis
contribution is small (heavy body, M_chassis⁻¹ tiny — measured),
so the missed coupling is small in practice.

**This is the cleanest first attempt** for our problem: simpler
than per-body (no ownership rule), captures more coupling per
block (both bodies), and the thing it misses we already measured
to be negligible on this scene.

#### 3. Per-body with overlap (Restricted Additive Schwarz)

Each constraint joins *every* body group it touches (overlapping
groups). RAS solves each block then writes back only to the
constraint's "owned" rows. Captures both bodies' contributions
AND cross-pair coupling through shared bodies.

**Strictly more capture** than #1 or #2, at the cost of doubled
per-body-block work for body-body constraints and more complex
book-keeping. Convergence theory is mature in domain-decomposition
literature.

**When to choose**: if per-body-pair turns out marginal and the
data shows uncaptured cross-pair coupling is what's limiting us.

#### 4. Schur-on-constraint-types (joint vs contact)

Block-factorize A by constraint type:

```
A = [A_jj  A_jc] = [I              0] [A_jj  0           ] [I  A_jj⁻¹·A_jc]
    [A_cj  A_cc]   [A_cj·A_jj⁻¹    I] [0     S = A_cc − A_cj·A_jj⁻¹·A_jc] [0  I        ]
```

**Invert A_jj exactly** (joints are bilateral equality constraints —
small, well-conditioned). Then approximate the contact-block Schur
complement S = A_cc − A_cj·A_jj⁻¹·A_jc with a Jacobi (or per-pair)
preconditioner.

**Captures all chassis-mediated coupling exactly** because A_jj is
solved without approximation — exactly the cross-pair coupling
that per-body-pair misses. This is the structurally "right" answer
for heavy-articulation problems.

**Cost**: significantly more invasive. Need to identify joint vs
contact rows, factorize A_jj (small, dense, exact), assemble the
Schur complement update matrix-free. The Schur update assembly
involves matvecs through A_jj⁻¹, which we'd compute via a small
direct solver per world.

**Status**: deferred. User has explicitly flagged this as a
"would like to try later" direction. Per-body-pair first; if it
delivers the speedup, Schur-on-types may not be needed. If
per-body-pair is marginal, this is the natural escalation —
particularly attractive for stiff-joint scenes where per-body-
pair's cross-pair miss is the binding limitation.

#### 5. Multi-colored Gauss-Seidel on body groups

Color the body-group graph so adjacent bodies (sharing
constraints) get different colors. Within each color, blocks are
independent — process in parallel. Across colors, sequential.
Each "sweep" gives a forward GS update; SGS adds a backward sweep.

**Captures cross-body coupling** via the sequential structure —
strictly stronger than per-body block-Jacobi.

**Cost**: 4–6 color sweeps per preconditioner-apply (for typical
contact graphs), each is a per-body block-Jacobi step. So ~4–6×
the cost of per-body block-Jacobi for stronger capture. Net wins
only if PCR iter count drops by 4–6×+, which is plausible for
stiff problems but not guaranteed.

**Cost on GPU**: coloring sweeps serialize. Lose 4–6× parallelism
factor on the preconditioner.

### Implementation: per-body-pair (recommended starting point)

**Cost ratio vs plain Jacobi**: ~1000× per preconditioner-apply
(matvec with a 10×10 inverse vs scalar division by a diagonal).
Wins if PCR iter count drops by more than ~1.5× — plausible if
the previously-uncaptured off-diagonals were doing real work.
The measurement above shows ratio ~0.22–0.28 for wheel-ground
pairs, so there's real coupling to capture.

**Implementation pieces** (1–2 weeks):

1. **Pair-assignment** at start of each step. For each constraint,
   form its (b₁, b₂) pair (using b₂=GROUND_SENTINEL for ground
   contacts). Build (pair_idx → list of constraint_idx) mapping.
   Static across NR iters within a step (constraints don't move
   between pairs during a step).
2. **Block extraction**: for each pair group, build A_pair by
   gathering rows of A from the constraint Jacobians J_i for i in
   that pair's group. A_ij = J_i_b1ᵀ·M_b1⁻¹·J_j_b1 +
   J_i_b2ᵀ·M_b2⁻¹·J_j_b2 (sum over BOTH bodies in the pair).
3. **Per-block dense factorization** (Cholesky for SPD, LU as
   fallback). Block sizes ~5–15 dense; Warp tile primitives may
   help with batched factorization.
4. **Apply** as preconditioner: for each pair, solve A_pair·y =
   x_pair with the precomputed factor, scatter back.

**Open questions** to validate during implementation:

* **Conditioning of A_pair near impacts**: small blocks may be
  near-singular when contacts are still "settling". May need a
  per-block diagonal regularization shift.
* **Empty pairs**: pairs with only 1 constraint degenerate to a
  scalar Jacobi entry — handle as a separate fast path.
* **Pair count vs body count**: pairs are roughly num_bodies +
  num_active_contact_pairs. For Helhest, ~20 pairs vs ~16 bodies.
  Marginal scaling difference.

### Postscript: per-body-pair was implemented and measured

Phases 1–4 of the per-body-pair block-Jacobi preconditioner shipped
in commits `86bf36e`, `348edd3`, `c9e8bb9`, `0936b47`. The
implementation in `src/axion/optim/per_body_pair_preconditioner.py`
is correct and validated by 4 phase smoke tests against numpy
references at float32 precision:

  * Phase 1 (pair assignment): consistency checks across 67 sim steps
  * Phase 2 (block extraction): max |A_gpu − A_ref| = 2.67e-7
  * Phase 3 (Cholesky): max |L·Lᵀ − A_pair| = 3.36e-7, all SPD
  * Phase 4 (matvec): max rel error 1e-7 across 4 (α, β) combos

Wired into `base_engine` behind `preconditioner_type` config knob;
default is `"jacobi"` (existing behavior preserved).

**Convergence-rate result on Helhest, 3 replicates (obstacle, 200 steps):**

```
                  sumNR    PCR    PCR/NR    bad   uw_max range
jacobi             1823  42191    23.2     1.3   1.2e-4..5.3e-2
per_body_pair      1854  39832    21.5     2.3   3.6e-2..1.3e-1
```

Per-PCR-iter savings of ~7% — the preconditioner does capture real
coupling, as predicted by the per-body off-diagonal measurement
(0.22–0.28 ratio for wheel-sized groups in
`test_scripts/measure_per_body_block_coupling.py`). NR iter count
unchanged within noise. Worst-case quality slightly degraded but
within the session-wide high-variance regime.

**Wall-clock result is the binding regression.**

End-to-end profiles on RTX A500 laptop GPU:

```
                                jacobi    per_body_pair    ratio
nr_solve mean (16 worlds)      5.34 ms    10.58 ms        1.98×
nr_solve mean (100 worlds)    18.10 ms    39.72 ms        2.20×
total step (16 worlds)         6.26 ms    11.49 ms        1.84×
total step (100 worlds)       19.34 ms    40.98 ms        2.12×
```

**The gap WIDENS with worlds**, contrary to the "more parallelism
will help" hypothesis. Three structural reasons:

1. **Two kernel launches per matvec** (baseline + per-pair) vs
   Jacobi's one. PCR calls matvec on every inner iter, so this
   multiplies through.
2. **Poor memory coalescing** in the per-pair kernel. Each thread
   accesses its own `L_blocks[w, p, :, :]` region; adjacent threads
   in the launch grid (different `w` and/or `p`) read far-apart
   memory. The opposite of what GPUs love. Adding more worlds adds
   more threads but each thread's access pattern is still scattered.
3. **8.8× higher setup cost** per NR iter (4 kernels: pair-assignment
   + block-extract + Cholesky + Jacobi-fallback-update) vs Jacobi's
   1 kernel.

Per-component profile breakdown (16 worlds, 3200 NR iters):

```
phase                   jacobi      per_body_pair    ratio
linear_system          0.050 ms      0.048 ms       similar
preconditioner setup   0.010 ms      0.088 ms        8.8×
cr_solve               0.402 ms      0.785 ms        2.0×
step_or_linesearch     0.049 ms      0.050 ms       similar
convergence_check      0.026 ms      0.027 ms       similar
total per NR iter      0.537 ms      0.999 ms       1.86×
```

**What could plausibly close the gap:**

1. **Different GPU hardware.** A laptop GPU (RTX A500, 16 SMs,
   96 GB/s memory) is the worst case for our access pattern.
   Server GPUs (A100: 108 SMs, 1555 GB/s; H100: similar profile)
   have ~16× more memory bandwidth and 6–10× more SMs. The
   coalescing penalty shrinks with bandwidth, the kernel-launch
   overhead shrinks slightly with newer drivers, and the per-thread
   small-block work amortizes over more SMs. **Plausible** that
   per-body-pair is competitive or faster on A100, but unmeasured.

2. **Warp tile primitives** (`wp.tile_load`, `wp.tile_cholesky`,
   `wp.tile_lower_solve`, etc.). One CUDA block per
   (world, pair_id) instead of one thread; threads cooperate via
   shared memory on the triangular solves. Eliminates the
   coalescing problem and amortizes kernel launch overhead via
   fusion (gather + solve + scatter in one kernel using shared
   memory). Estimated 3–5× speedup on the apply kernel,
   plausibly enough to make per-body-pair competitive on the
   laptop GPU. ~1 week of careful work; the variable per-pair
   block size (5–15) needs padding to a fixed compile-time tile
   shape (16 or 32).

**Why the code stays in the repo (not reverted):**

* The structural argument (off-diagonal coupling exists, capturing
  it reduces PCR iters) was empirically validated. Future work on
  GPU optimization or scaling has all four phases of correct
  implementation to build on.
* Default is `"jacobi"`, so there's no behavior change; the code
  is opt-in via `engine.preconditioner_type=per_body_pair`.
* If/when the simulator runs on server GPUs (A100/H100), the
  hardware change alone may flip the wall-clock verdict; testing
  is a 30-minute experiment.
* Tile-primitives optimization is a real follow-up direction with
  clear engineering scope.

Future contributors who want to revisit: start by measuring on
target GPU first (low cost, high information), then commit to
tile-primitives optimization only if the spectral case is
confirmed there.

### r-factor heuristics for the complementarity preconditioner

The paper's complementarity preconditioner is **already implemented**
in this codebase (see `contact_constraint.py:94`):
`r = h²·effective_mass` for contact-normal, `r = h·effective_mass`
for friction. This is the paper's recommended r-factor.

But: paper's §11 explicitly flags this as a research direction —
"the design space for choosing the r factor in the complementarity
preconditioner is large, and we think that heuristics based on
global information could provide significant performance
improvements."

What "global information" might help:
* Full block-diagonal of A (not just diagonal entry) for r
* System-wide velocity scale or characteristic timescale
* Adaptive r based on outer NR progress (like Eisenstat-Walker
  but for the NCP scaling rather than the linear tol)
* Per-body or per-constraint-island tuning

**Status**: speculative; the paper doesn't propose specifics. Lower
priority than per-body block-Jacobi until we know whether the
linear-solver-side preconditioner improvements pay off.

## See also

* `axion/optim/pcr_solver.py` — current PCR implementation
* `test_scripts/measure_A_stability.py` — the A-volatility probe
  used for option 4's verdict
* [`warm_start_iterate_seeding_issue.md`](warm_start_iterate_seeding_issue.md)
  — why warm-starting the NR iterate (which Option 2 reduces to)
  doesn't work with FB complementarity
* [`eisenstat_walker.md`](eisenstat_walker.md) — option 5
  postscript with implementation data
* `engine_profiler` per-component output — shows PCR's share of step
  time (~74% in current configurations) so any of these options
  has a meaningful step-time ceiling
