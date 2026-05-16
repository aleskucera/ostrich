# Learned Preconditioner / Initial-Guess: Feasibility Study

Companion to [`pcr_warm_start_options.md`](pcr_warm_start_options.md).
Question: can a small GNN learn a better PCR preconditioner `M⁻¹` or a
better initial guess `Δλ₀` than the cheap analytic options, exploiting
the fact that the constraint-graph topology is fixed for one robot?

**Verdict: no, not on this scene with this approach as built.** The
idea is well-posed and there is a large *ideal* ceiling, but the
straightforward GNN + unrolled-PCR training does not reach it — it
collapses back to Jacobi. Cost of finding this out: ~1 day of offline
work (the point of a minimal prototype).

## Setup

Offline harness (`test_scripts/`), entirely in PyTorch, float64:

* `dump_linear_systems.py` — drives Helhest `obstacle_benchmark` with
  the eager NR path, snapshots every per-NR-iter PCR system
  (`A`-inputs + `b` + engine solution) → `data/baselines/helhest_systems.npz`.
  **335 real systems**, active set 31–67 of 402 slots.
* `precond_lab.py` — reconstructs the dense Schur system
  `A = J M⁻¹ Jᵀ + (C+ε)I`, faithful port of `pcr_solver.py`, plus
  Jacobi and per-body-pair preconditioners.
* `precond_gnn.py` / `precond_train.py` — tiny bipartite
  constraint↔body GNN (~23k params), four heads, trained by
  differentiable unrolled-PCR (loss = log₁₀ relres after K=8 steps).
  Train/val = contiguous 80/20 (val = later, post-impact systems).

Reconstruction was verified faithful: `‖A·x_engine − b‖/‖b‖` p50
9.8e-5 (an O(1) value would mean a wrong spatial-vector convention).

## Findings

**1. Per-body-pair is *worse* than Jacobi here.** iters-to-tol mean
Jacobi 20.3 vs per-body-pair 23.5 (worse on 254/335). Contradicts the
PCR-doc premise that cross-pair coupling is small for Helhest — on
obstacle_benchmark it is not. **The honest baseline is Jacobi.**

**2. κ~1e11 is mostly a units artifact; the hard part is κ~1e5.**
Symmetric-Jacobi scaling collapses median κ 6.2e10 → 1.1e5 (5.7
orders). Ruiz equilibration ≈ same and does not beat Jacobi on iters.
So the residual κ~1e5 is **not diagonal-removable → structurally
non-local**. This is *why* Jacobi works (it removes the units κ for
free) and why no diagonal/block scheme beats it.

**3. The residual κ~1e5 is moderately low-rank — strong ideal
ceiling, but rank ≈16.** Σ(1/λ) is 100% in the bottom-8 modes; ideal
rank-k spectral deflation (exact eigenvectors): k=4→22, k=8→15,
**k=16→9**, k=32→6 iters (vs Jacobi ~21). So a "Jacobi + rank-16
correction" *could* roughly halve iters — if the subspace can be
predicted.

**4. The learned heads do not realise the ceiling.**

| VAL (67 post-impact systems) | mean | p50 | p95 | vs Jacobi |
|---|---|---|---|---|
| jacobi (bar) | 18.8 | 19 | 22 | — |
| per_body_pair | 24.8 | 25 | 29 | worse |
| **gnn_lowrank** (rank 16) | **18.8** | 19 | 22 | **ties — degenerates to Jacobi** |
| gnn_x0 | 22.2 | 20 | 35 | worse |
| gnn_diag (theory) | ≥ Jacobi | | | can't beat (finding 2) |

* `lowrank` training loss flat-lined at exactly Jacobi's level by
  epoch 40 and its val stats are *identical* to Jacobi — the optimiser
  drove the correction gains to ≈0, recovering Jacobi. It never found
  the useful non-local subspace by gradient descent.
* `x0` trains (loss drops fast) but the learned guess **increases**
  iters-to-tol — same mechanism as the dead Options 1/2 in the PCR
  doc (a non-zero guess raises the initial residual; Δλ→0 near
  convergence).

## Why the gap, and the one cheap discriminating test

The ceiling (finding 3) uses *exact* per-system eigenvectors; the
learned head must *predict* that 16-dim non-local subspace from
graph+values across an A that moves ~85%/NR-iter. The collapse to
gains≈0 is consistent with two fixable suspects, not necessarily a
fundamental "no":

1. the unrolled-K residual loss has a trivial `gains→0` minimum that
   is easy to fall into and hard to escape;
2. QR-backward through the predicted basis is ill-conditioned when
   columns are near-dependent (likely at init) → unstable gradients →
   optimiser flees to the trivial minimum.

**Discriminating probe (cheap, ~½ day):** a *supervised* low-rank
head — train `U` directly toward the true bottom-16 eigenvectors
(already computed in `eigenspectrum_probe.py`). This separates "can
the GNN represent the subspace at all" from "can the unrolled loss
train it". If supervised *also* fails to beat Jacobi → the concept is
dead for this scene. If it works → the objective is the problem, and
a better training target (spectral loss / supervised pretrain → unroll
finetune) is worth pursuing.

## Recommendation

By the chosen success bar ("fewer iters at any apply cost vs the
baseline"), the result is a clean **negative**: nothing learned beats
Jacobi; `lowrank` ties it, everything else is worse. Recommend running
the supervised-eigenvector ablation before investing further; absent a
positive there, learned preconditioning is not worth pursuing on
Helhest at this scale, and the PCR-doc's Option 5 (Eisenstat–Walker
adaptive linear tol) remains the best cheap lever.

## Postscript — the structural investigation, and the real answer (supersedes the above)

Instead of the supervised ablation we followed the structure. It led
somewhere better: **the answer is a classical, no-ML change.**

* **Per-body-pair (Finding 1) and per-body coarse space (Finding 5)
  both fail** (worse than Jacobi). The bad modes are *not* rigid-body
  coupling; `range(J^T-block)` covers only 1% of the bottom-8 band.
  They live in `null(J)` — self-equilibrated internal constraint
  combinations (contact/friction redundancy: up to 67 constraints on
  4 bodies), plus a large stiff joint/control compliance component.
* **The bad ~16-dim subspace is stable and near-global** (Finding 6/7):
  same-support consecutive overlap p50 1.00; *different*-support
  overlap p50 0.99 (partly because ~46% of its energy sits in the
  always-active joint/control block — block-energy split: joint 39%,
  control 7%, normal 20%, friction 35%).
* **A single *precomputed* universal basis does NOT work out-of-sample.**
  Train-built rank-32 basis "covers" held-out systems by the energy
  metric (p50 1.00) yet as a deflation preconditioner gives **36.9 val
  iters vs Jacobi 18.8** — high coverage ≠ good preconditioner; a fixed
  averaged basis over-corrects well-conditioned modes. Dead.
* **Subspace recycling works, robustly, out-of-sample, with zero ML.**
  Preconditioning each solve with the *previous* solve's bottom-16
  subspace (additive two-level `M⁻¹ = diag(A)⁻¹ + Z(ZᵀAZ)⁻¹Zᵀ`):
  held-out post-impact **18.8 → 12.1 iters (−35%)**, beats Jacobi
  63/67, ≈ ideal (own exact subspace, 11.4). Locally-adapted beats
  globally-fixed because the subspace drifts slowly in the right
  metric even as A's entries move 60–85%.

**This overturns `pcr_warm_start_options.md` Option 4** ("Krylov
recycling — dead, A 85% volatile"): that measured *raw-A* volatility;
the quantity that governs recycling is the *deflation-subspace*
volatility, which is ≈0. Recycling is not dead — it is the recommended
lever.

### De-risking probe REVERSES the recycling recommendation

`recycle_approx.py` replaced the exact `eigh` subspace with the
harmonic-Ritz subspace a real solver gets for free (harvested from the
PCR Krylov residuals on system i−1). Held-out val:

```
            approx∩exact   recycle-HARMONIC   vs Jacobi (18.8)
m=16            0.35          37.9 iters        −101% (2× WORSE)
m=24            0.50          24.9 iters         −32% (worse)
m=32            0.70          16.6 iters         +12% (marginal, m>iters)
```

PCR converges in ~19 iters, so only m≈16–19 Krylov vectors exist "for
free"; at that budget the harvested subspace captures ~35% of the true
bad band and the deflation **over-corrects, ending 2× worse than
Jacobi** (same failure mode as the dead fixed-universal basis). The
idealised −35% required exact eigenvectors the engine cannot cheaply
produce. **Do not build the in-engine recycling solver as specified.**

Caveat: this probe used the weakest recycling form (rebuild from a
single solve's Krylov space). Proper GCRO-DR *accumulates/maintains*
the recycle subspace across many solves and could reach higher quality
— but that is uncertain research, not a cheap win, and must still track
an 85%-volatile A.

### Proper GCRO-DR recycling — also closed

`recycle_gcrodr.py` ran the *correct* recycling algorithm: maintain and
accumulate the recycle subspace across the whole sequence (harmonic
Ritz over [carried U , fresh Krylov] each solve), same free m=16
budget. Held-out val: **19.0 iters vs Jacobi 18.8 (−1%, no gain).**
Diagnostic: the accumulated subspace's overlap with the true bottom-16
**plateaus at ~0.72 and never matures** (first⅓ 0.75 → last⅓ 0.70) —
limited-Krylov harmonic Ritz caps subspace quality and accumulating
systematically-deficient estimates does not average up. (It works early
in the trajectory — easy within-step regime — then degrades through the
post-impact val split.) The recycling direction is now **definitively
closed**, with no remaining "but you didn't try the real algorithm"
objection.

## Architectural resolution — the preconditioner question is MOOT for small-n robots

`dense_vs_pcr_bench.py`. The recon showed the binding constraint is
static-shape CUDA-graph capture (→ pad to N_c=402, mask not compact, run
~26 latency-bound matrix-free PCR iters), *not* numerics. The active
system is tiny (largest active n=45 here). A **batched dense Cholesky at
a fixed graph-safe bucket** is *exact* and faster (100 worlds, incl.
assembling J·M⁻¹·Jᵀ+C):

```
exactness: ||A x_chol − b||/||b|| = 1.5e-14   (PCR only ~1e-3..1e-5)
matrix-free PCR proxy (26 iters, W=100)          10.6 ms
dense factor+solve+assembly  n_max=64   1.76 ms  (6.0× faster)
                             n_max=96   3.92 ms  (2.7×)
                             n_max=128  6.17 ms  (1.7×)
                             n=45 (compact) 1.27 ms (8.4×)
```

Even at n_max=128 the exact dense path beats iterative PCR. It removes
the *entire* preconditioner problem, the convergence loop, the iter cap,
the linear-tol tuning, and the per-NR-iter A-volatility issue (just
refactor — it's cheap). Feasibility is **de-risked**: a hand-written
Warp Cholesky kernel already exists in-engine
(`per_body_pair_preconditioner._factor_pair_blocks_kernel`).

**Honest caveats:** the PCR side is a torch einsum proxy, not the real
Warp scatter/gather kernels — the exact multiplier needs an in-engine
measurement; the assembly proxy is a dense matmul (upper bound; real
sparse scatter is cheaper, so the comparison is conservative against
the dense path). The exactness/simplicity argument is
fidelity-independent and on its own justifies a prototype. Scope: holds
for *small-n* robot classes (Helhest-like, active ≲ a few hundred);
iterative PCR remains correct for genuinely large active sets.

### De-risking REVERSES the speed claim (the "6×" was a bad proxy)

The "6× faster" above was measured against a torch-einsum *proxy* of
PCR. The project's own recorded per-component profile gives the **real**
number: obstacle_benchmark, num_worlds=1, `cr_solve` = **0.386 ms per
NR iter** (captured-graph, the deployed path). Apples-to-apples dense
factor+solve+assembly at W=1: n_max=64 → 0.527 ms, compacted n=45 →
0.452 ms — i.e. the dense path is **~1.4× SLOWER than real PCR at W=1**,
not faster. The proxy was ~25× too slow and produced a misleading
conclusion.

Honest status of the architectural option:

* The *speed* win is **not demonstrated**; at W=1 it is a slowdown.
* The comparison is **overhead-contaminated → inconclusive**, not
  cleanly negative: 0.386 ms PCR is graph-replayed (no dispatch
  overhead); the 0.527 ms dense is torch-eager (full overhead, and a
  64×64 Cholesky at W=1 is pure launch overhead). A graph-captured
  Warp dense path could be faster — unmeasured.
* The only regime where dense might still win — **large batched W**
  (one cuSOLVER call vs 26 latency-bound iters × replays) — is
  **unmeasured for real PCR** (recorded profile is W=1 only).
* The **exactness/simplicity** benefit is real (1e-14 vs ~1e-3; no
  preconditioner, tol tuning, or convergence loop) but is not a speed
  argument.

### Recommendation (final)

No demonstrated cheap performance win exists anywhere — preconditioner
(learned/classical) **or** architectural. The solid, durable outputs of
this study are the de-risked negatives (which save misdirected effort),
the reusable harness, and corrections to `pcr_warm_start_options.md`
(Option 4 recycling; the per-body-pair premise). The direct-dense idea
is *not recommended* on current evidence; it would only be worth a real
in-engine batched A/B if the large-W regime specifically matters (RL
training), and even then with no strong prior it wins. Eisenstat–Walker
(PCR-doc Option 5) remains the one untouched, genuinely-cheap,
independent lever and is the recommended thing to actually pursue.

### (Earlier fallback, now superseded)

No deployable *preconditioner-side* win exists for this problem. The
whole study is a clean, thoroughly de-risked **negative**: learned
preconditioning fails (collapse / worse), structured analytic fails
(worse than Jacobi), exact-subspace recycling works (−35%) but the
subspace is unobtainable from the free Krylov budget, and proper
accumulated GCRO-DR recycling does not rescue it (subspace quality
plateaus ~0.72). Recommend **PCR-doc Option 5 (Eisenstat–Walker
adaptive linear tolerance)** — a fully independent lever (stop
over-solving early NR linear systems; nothing to do with the
preconditioner) that none of these failure modes touch. One
scene/robot/trajectory throughout; that caveat stands but does not
change the recommendation.

## Files

`test_scripts/{dump_linear_systems,precond_lab,precond_gnn,precond_train,equilibration_diag,eigenspectrum_probe,coarse_probe,ideal_target_stability,recycle_probe,recycle_validate}.py`,
data `data/baselines/helhest_systems.npz`, log `runs/precond_focus.log`.
