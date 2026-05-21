# Friction Sticking Issue: Why Static Friction Under-Constrains the Body

## Symptom

Apply a tangential force well below the Coulomb static-friction limit to
a body resting on a surface — the body should stay at rest. Instead the
body slides at a steady velocity, with the friction force settling at
some intermediate value below the applied force (so net force ≠ 0 and
the body accelerates until it reaches a velocity-dependent equilibrium).

This is **dt-independent** — same outcome at `dt = 50 ms` and `dt = 1 ms`,
unlike the contact-normal dt-dependence problem documented in
[`dt_dependence_problem.md`](dt_dependence_problem.md). It's a separate
bug.

## Reproducer

`experiments/2_dt_stability/diagnose.py` with the friction-mode flags:

```bash
python experiments/2_dt_stability/diagnose.py --dt 0.05  --mu 0.5 --fx 2.0
python experiments/2_dt_stability/diagnose.py --dt 0.001 --mu 0.5 --fx 2.0
```

Setup: 1 kg sphere, mu = 0.5, gravity 10 m/s² (so static-friction limit
= mu·M·g = 5 N). Apply a 2 N horizontal force each step (well under 5 N).
Expected: body stays at rest. Observed:

| run                 | v_x_final | f_t (median) | iter_mean |
|---------------------|-----------|---------------|-----------|
| dt=0.05  ef=1e-6    | 0.43 m/s  | 0.57 N        | 6         |
| dt=0.05  ef=2e-2    | 0.43 m/s  | 0.57 N        | 6         |
| dt=0.001 ef=1e-6    | 0.43 m/s  | 0.57 N        | 16        |
| dt=0.001 ef=2e-2    | 0.43 m/s  | 0.57 N        | 16        |

`f_t` ≈ 0.57 N in all cases — way below the 2 N needed to balance the
applied force. Body accelerates with the residual 1.43 N until reaching
a steady-state velocity.

Notably: `friction_compliance = 2e-2` (the value `sweep_axion.py` uses,
24,000× the engine default of `1e-6`) makes **no difference** in this
test. Whatever sticking the user sees in their experiments isn't coming
from this knob.

## Root cause: active-arm degeneracy of the friction NCP

The friction NCP is enforced via a Fisher–Burmeister formulation in
`compute_friction_model` (`src/axion/constraints/friction_constraint.py:32-85`):

```
v_t   = J_t · u                          (tangential velocity)
d_t   = v_t · dt                          (impulse-level disp)
limit = mu · |λ_n_prev| · dt              (Coulomb impulse limit)
gap   = limit − min(|λ_f_prev|·dt, limit) (slack to the cone)

phi_f = scaled_FB(|d_t|, gap, 1, precond)
w     = precond · (|d_t| − phi_f) / (precond·|λ_f_prev|·dt + phi_f + ε·dt)
```

The friction Schur diagonal:
```
c_f = w / dt + friction_compliance
```

The two regimes are:
- **Sliding**: `|d_t| > 0`, `gap = 0`. Then `phi_f ≈ 0`, `w ≈ |v_t|/|λ_f|`,
  c_f is `O(1/dt)` — naturally regularized.
- **Sticking**: `|d_t| → 0`, `gap > 0`. Then `phi_f ≈ 0`, `w → 0`, and
  c_f reduces to `friction_compliance` only.

**The sticking regime is the default state** (we want it most of the
time), and `w → 0` means the FB linearization has **no curvature in
`λ_f`**. This is the same "active-arm degeneracy" that breaks the contact
normal at the resting solution — the constraint can't push `λ_f` toward
the value that satisfies dynamic equilibrium because `∂φ_f/∂λ_f = 0`.

Newton converges (per its tolerance) to whatever `λ_f` value makes
`res_f = v_t + w·λ_f = 0`. With small `w` and small but nonzero `v_t`,
that gives a finite small `λ_f` — exactly the 0.57 N we observe — even
though physical equilibrium needs `λ_f = 2 N`.

The body then slides at the `v_t` that balances the residual equation,
not at zero.

## Why the friction case is different from the contact case

For contact normals we fixed this by adding `compliance / dt²` to the
diagonal — a regularization that auto-scales with dt and dominates
when the FB curvature `S/h²` collapses. The same recipe applied to
friction would give `compliance / dt`, but that doesn't help the
sticking regime structurally:

- For contact: regularizing the matrix diagonal with `e/dt²` damps
  Newton's per-iteration step from `pen/h²` to `pen·dt²/(h²·e) = pen/e`,
  i.e. independent of dt. That's the win.
- For friction in sticking: regularizing with `e_friction/dt` would
  similarly damp the step, but the *natural curvature* `w/dt` is
  already much larger than any reasonable `e/dt` would be, so the
  `e/dt` term is dominated and contributes nothing.

The friction case fails not because Newton overshoots (it doesn't), but
because Newton **converges to the wrong fixed point** — the FB function
has flat regions in `λ_f`-space at the sticking corner, so the
"converged" answer is whatever the dynamics happen to land on.

## What the paper does about it

Macklin et al. 2019 acknowledge this but their evaluation never tests
static sticking under continuously-applied tangential force. Their
demos (Fetch tomato, Allegro grasp) all involve dynamic contact with
constant motion, where friction is in the sliding regime and `w/dt` is
large enough.

Section 8.1 mentions that in the friction-constrained system "we found
line search would often cause the iteration to stall and make no
progress", which is the same observation under a different name —
Newton can't make progress because the FB derivatives in the sticking
direction are degenerate.

## Fix candidates

In rough order of structural-correctness vs implementation-cost:

### (1) Call the already-defined warm-start two-pass — cheapest
`base_engine.py:247` defines `compute_warm_start_forces` which solves
twice per step: once for normal+joint forces only (friction inactive
because `λ_n_prev = 0`), then a second time with friction enabled now
that `λ_n_prev` is non-zero. The function is implemented but **never
called** from any code path.

If friction's failure mode in this test is partly because Newton starts
from `λ_f = 0` and only activates friction in iteration 2 — by which
point the body has already accumulated horizontal velocity in iteration
1's friction-less solve — then the two-pass warm start could help.
Cost: ~5 lines in `engine.py:step()` to invoke it. Risk: changes
adjoint sensitivities.

Verify by enabling and re-running the diagnostic. If `f_t` jumps from
0.57 to 2.0 N: this was the issue. If not: move on.

### (2) Better friction compliance — partial structural fix
The FB function's flatness in `λ_f` at the sticking corner is the
underlying problem. Adding a stronger compliance specifically in the
sticking regime — e.g. `c_f = w/dt + e_stick * (1 − sliding_fraction)`
where `sliding_fraction` is some indicator of how saturated the cone
is — would force a non-zero curvature in `λ_f` even when `w → 0`. This
is what proximal-map methods do (the paper's Section 2.1 "Relaxation
methods").

Cost: small kernel change in `friction_constraint.py:compute_friction_model`.
Risk: changes the friction-cone behavior in transitions; needs tuning.

### (3) Velocity-level constraint with Baumgarte position correction
Replace the FB formulation with a `v_t = 0` velocity constraint plus a
position-correction Baumgarte term to recover stuck-position. This is
what most production solvers (Drake, MuJoCo) do for sticking. Cost:
significant rewrite of `friction_constraint.py`. Risk: large; would
change behavior in dynamic scenes too.

### (4) Switch friction NCP to minimum-map
Section 4.1 of the paper. Minimum-map has a sharp kink instead of a
smooth corner, which avoids the flat-derivative problem at the cost of
non-smooth Newton iterations. Cost: change in
`compute_friction_model`. Risk: convergence becomes harder to predict
because of the kink.

## Recommended next step

Try fix (1) first since it's a 5-line change and the implementation
already exists. Re-run the diagnostic at both dt's; if `f_t` reaches
2 N and `v_x_final = 0`, that's the fix. If not, the bug is in
`compute_friction_model` itself and we move to (2).

## References

- Reproducer: `experiments/2_dt_stability/diagnose.py --mu 0.5 --fx 2.0`
- Code path: `src/axion/constraints/friction_constraint.py:compute_friction_model`
  (line 32) and `compute_friction_core` (line 93)
- Existing-but-unused warm-start: `src/axion/core/base_engine.py:compute_warm_start_forces`
  (line 247)
- Macklin et al. 2019, "Non-Smooth Newton Methods for Deformable
  Multi-Body Dynamics", §3.4 "Friction" and §8.1 "Line Search and
  Starting Iterate"
