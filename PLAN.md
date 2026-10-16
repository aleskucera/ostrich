# Study: predicting the free FB coefficients to cut Newton iterations

Branch: `study/fb-coeffiecients`

## Idea

The Fisher–Burmeister NCP function has free parameters that leave the
complementarity solution unchanged but change the path Newton takes to it
(Macklin et al. 2019, §7, "complementarity preconditioning"). If a better
per-contact choice than the current analytic heuristic exists, predicting it
could reduce Newton iterations per step, and so simulation time.

### What is free

| Parameter | Code | Real DoF |
|---|---|---|
| Contact `(α, β)` = `(CONTACT_FB_ALPHA, dt²·w)` | `constraints/contact_constraint.py:109,127` | 1 per contact: the ratio `r_n = β/α` |
| Friction `r` = `w` | `constraints/friction_constraint.py:453` | 1 per contact (isotropic) |
| Schedule of `r` across NR iterations | — | `r` may change between iterations; the root set does not |

`w = [J M⁻¹ Jᵀ]_ii`. Not free (they change the solution): `fb_eps_sq`,
physical compliance, choice of NCP function.

**Caveat: `r` is free only for the exact FB function, and neither NCP function
in the repo is exact.**

- Contact uses smoothed FB: `φ_ε = 0 ⇔ a·b = ε/(2αβ)`, so the converged gap
  is `ε / (2·k_n·dt²·w·λ_n)` and moves with `k_n` (default
  `ε = contact_fb_smooth_eps_sq = 1e-8`).
- Friction uses plain FB, but on a smoothed slip norm `√(|v_t·dt|² + 1e-8)`
  (≥ 1e-4 m). In sticking this gives `W ≈ a²/(2·gap·(r·|imp| + a))`, which
  depends on `r`, strongly near the cone edge (`gap → 0`). The residual also
  never checks `φ_f = 0` directly.

So `r` is approximately free only where those smoothings are negligible.

Note: friction uses `r = w`, while the paper prescribes `r = h·w`; the force
argument is weighted ~`1/h` more than in the paper.

## Plan

Each step can end the study early.

1. **Profile NR iteration counts** → verify: per-step NR-iteration
   distribution, fraction of steps hitting `nr.max_iters`, and the stopping
   rule, on representative scenes.
   - The engine exits early: `capture_while` stops once *every* world has
     `‖res‖ < nr.atol`, with `backtrack_min_iter ≤ iters ≤ max_iters`
     (`core/base_engine.py:42-81,490-507`). So fewer iterations do translate
     into less work — but in multi-world batches the slowest world sets the
     count.
   - Still open: wall-clock share of the NR loop (needs the CUDA-graph
     profiler, `profiling.mode`).
2. **Oracle ceiling** → verify: iterations-to-tolerance as a function of
   global multipliers `k_n·r_n`, `k_f·r_f` swept over ~10⁻²…10².
   - One fixed `k` best everywhere → retune the constant, stop.
   - Best `k` varies a lot per step → prediction is justified; then a
     per-contact oracle to see what per-contact choice adds.
3. **Analytic candidates** → verify: iterations vs. the step-2 oracle for
   - friction `r = h·w` (paper's choice),
   - `r_i ≈ h² / [A⁻¹]_ii` estimated from the previous step,
   - adaptive `r` across NR iterations.
4. **Learn** (only if step 2 shows a large gap that step 3 does not close) →
   verify: small per-contact MLP predicting a multiplier on the heuristic `r`
   (multiplier 1 = current behaviour), trained on a proxy loss such as the
   residual after K unrolled NR iterations (`learning/unrolled_newton.py`).

Relevant prior negative results: `docs/learned_preconditioner_feasibility.md`
(learned PCR preconditioner collapsed to Jacobi),
`docs/developer/warm_start_nn_findings.md` (warm-start NN stuck at the FB
barrier).

## Results

### Step 1 — NR iteration profile

Script: `dev/fb_study/profile_nr_iters.py` (Helhest `cruise` and `skid` scenes
from `dev/relax_bench.py`, production engine config, dt = 0.03, 1 world,
40 settle steps + 120 drive steps, eager mode).

```
uv run python dev/fb_study/profile_nr_iters.py --json step1.json
```

Production protocol (`nr.max_iters = 16`, `nr.atol = 1e-5`, min 4), drive phase:

| scene | NR iters mean | at cap | PCR / NR iter | final ‖res‖ p50 | final ‖res‖ max |
|---|---|---|---|---|---|
| cruise | 15.97 | 98 % | 5.5 | 6.5e-5 | 4.3e-3 |
| skid | 15.97 | 99 % | 14.3 | 8.1e-4 | 12.8 |

Residual vs. NR iteration (converge protocol, cap 64, drive phase, p50 / p90):

| iter | 1 | 3 | 4 | 5 | 6 | 8 | 10 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| cruise p50 | 3.4 | 1.3 | 5e-2 | 6e-4 | 1.4e-4 | 2.4e-4 | 8.7e-5 | 7e-5 | 4e-5 | 5e-5 |
| skid p50 | 7.6 | 2.9 | 1.4 | 0.41 | 0.11 | 7e-3 | 1.7e-3 | 5e-4 | 3e-4 | 3e-4 |
| skid p90 | 10 | 8.9 | 5.4 | 5.1 | 7.3 | 1.8 | 1.5 | 0.36 | 1.3e-3 | 1.5e-3 |

Findings:

1. **The stopping rule almost never fires.** `nr.atol = 1e-5` is below the
   residual plateau (~1e-5…1e-4 for cruise, ~3e-4 for skid), so even with a
   64-iteration cap, 67 % (cruise) and 92 % (skid) of drive steps never
   reach it. In practice, the engine runs a fixed 16 iterations per step.
   **Changing `r` cannot reduce time under the current config**; only a
   reachable tolerance (or a smaller `max_iters`) turns faster convergence
   into fewer iterations.
2. **Cruise:** iterations 1–4 are almost identical every step (p10 ≈ p90).
   Convergence is then fast: iterations 4→6 reach ~1e-4, after which the
   residual plateaus and oscillates. About 10 of the 16 iterations add
   nothing. The deterministic first iterations look like a
   warm-start / friction-lag effect (friction is off at iter 0), not an `r`
   effect.
3. **Skid:** the pre-asymptotic phase is long (p50 needs ~10 iterations to
   reach 1e-3) with a heavy tail (p90 still 0.36 at iteration 16, max 12.8).
   That is the regime where the paper shows `r` matters (Fig. 17), so this is
   where `r` could help.
4. The plateau's cause is unknown (float32, PCR inexactness at
   `linear.tol = 1e-3`, or the lagged friction fixed point). Since `r` does
   not change the solution, it is unlikely to move the plateau.

Implications for step 2:

- Measure **iterations to reach 1e-3 / 1e-4** (read from the per-iteration
  residual history) and **residual at a fixed budget**, not
  `iter_count` under the shipped atol.
- Focus on skid-like (saturated friction) and impact steps; cruise has
  little to gain from `r`.
- Two cheaper levers appeared that are independent of `r`: a reachable
  `nr.atol` (1e-4 would cut cruise from 16 to ~8 iterations on average; skid
would still hit the cap on most steps), and the
  4 near-deterministic startup iterations.

Not yet measured: wall-clock share of the NR loop (CUDA-graph profiler),
multi-world batches (where the slowest world sets the count), other robots.

### Step 2 — oracle sweep over global multipliers

Knobs added: `compliance.contact_fb_r_scale`, `compliance.friction_fb_r_scale`
(default 1.0; multiply `precond` in `contact_constraint.py` /
`friction_constraint.py`). Script: `dev/fb_study/oracle_sweep.py`.

```
uv run python dev/fb_study/oracle_sweep.py --cap 32 --json step2.json   # ~14 min
```

Method: paired steps (every `k` solves the same step from a snapshot of the
`k = 1` reference trajectory), residual re-measured with `k = 1` after every
NR iteration, and a noise baseline (the same step solved twice at `k = 1`;
the engine is not deterministic run to run). "Iters to tol" uses the engine
rule (≥ 4 iterations); unreached = 33. 1-D sweeps: `k_n ∈ {0.01 … 100}`
with `k_f = 1`, `k_f ∈ {0.01, 0.03, … 100}` with `k_n = 1`.

**Contact `k_n`** (drive phase, 120 steps):

| | cruise: iters to 1e-3 | cruise: to 1e-4 | skid: to 1e-3 | skid: best ‖res‖ by iter 8 (geomean) |
|---|---|---|---|---|
| k = 1 | 5.25 | 8.55 | 16.0 (12 unreached) | 2.9e-2 |
| noise oracle (best of 2× k = 1) | 5.12 | 7.81 | 15.3 | 2.7e-2 |
| best fixed `k_n` | 5.10 (k_n = 3) | 8.34 (3) | 16.0 (1) | 1.3e-2 (3) |
| per-step oracle over k_n | 5.02 | 6.69 | 11.0 | 2.1e-3 |
| k_n = 0.3 / 0.1 / 0.01 | 6.2 / 33 / 33 | 32.8 / 33 / 33 | 23.9 / 31.8 / 33 | 3.6e-2 / 9.6e-2 / 1.1 |

- `k_n < 1` looks much worse, but much of that is the smoothed-FB solution
  moving (gap ∝ 1/k_n), not slower convergence. One cruise step, min
  residual own / on the k = 1 ruler: k_n = 1: 1.4e-5 / 1.4e-5;
  0.3: 4.0e-5 / 5.4e-4; 0.1: 7.0e-4 / 6.8e-3; 0.01: 1.5e-3 / 0.37;
  10: 1.6e-5 / 2.2e-5. So `k_n ≤ 0.1` also converges slower in its own
  metric, but 0.3 does not. For `k_n ≥ 1` the shift is below ~1e-5 and the
  comparison is fair.
- `k_n ∈ [1, 10]` is a broad, flat optimum. No fixed `k_n` beats `k = 1` on
  iteration count.
- Cruise and settle: nothing to gain.
- Skid: the per-step best `k_n` varies (best counts: k=1 42, 3 → 31,
  10 → 24, 0.3 → 13, 100 → 10). The per-step oracle reaches 1e-3 in 11.0
  instead of 16.0 iterations. That is an **upper bound**: the oracle picks
  the min of 7 noisy runs, while the noise baseline only picks the min of 2
  (which alone gives −4.5 %). Real headroom from per-step `k_n` on skid is
  likely ~20–30 %, and only in this hard-slip regime.

**Friction `k_f`: not interpretable — `k_f` is not a free parameter here.**
Every `k_f ≠ 1` (including the paper's `k_f = dt`) looks catastrophically
worse on the `k = 1` ruler. Checked on one cruise step: with `k_f = 3` or
`0.3` the engine converges *in its own residual* as well as `k_f = 1` does,
and body velocities agree to ~6e-5 m/s, but friction forces differ by up to
5 N (of 14 N), so the `k = 1` residual at that point stays ~0.05–0.2.
Likely causes: (1) the smoothed slip norm makes the sticking/transition `W`
depend on `r` (see the caveat under "What is free"); (2) the friction row is
`v_t + W·λ_f` with `W` built from the previous iterate's forces (lagged fixed
point, paper eq. 33–35), and the residual never checks `φ_f = 0` directly;
(3) redundant contacts (13 contacts on 3 wheels) make the force
distribution non-unique. Judging `k_f` needs an `r`-free friction metric
(cone violation, slip–gap complementarity, max-dissipation alignment
computed from `v_t`, `λ_f`, `λ_n`).

**Decision for the plan:**

- Contact: no constant retune is warranted (`k_n = 1` is already at the flat
  optimum). Per-step prediction has modest, uncertain headroom, only on
  hard-slip steps, and (step 1) pays off only if the stopping tolerance is
  reachable.
- Friction: blocked until an `r`-free friction metric exists. The finding
  that the converged friction forces depend on `r` is worth understanding
  independently of this study: it suggests the shipped residual does not
  certify the friction complementarity conditions.
