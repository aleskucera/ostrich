# Relaxation Branch — Implementation Plan

Execution plan for `RELAXATION_BRANCH.md`. Written after reading the engine; several things
are much cheaper than that document assumes, and two things are riskier.

---

## 0. What the code review changed

### 0.1 The FB nonlinearity is localized to two call sites

Every constraint block already factors into `(J_hat, residual, C_diag)`. Complementarity
enters only through scalar weights:

| | current | relaxed | file:line |
|---|---|---|---|
| friction | `v_t, w_x, w_y = compute_friction_model(...)`<br>`res_f = v_t + w·λ`, `C_f = w/dt + compliance` | **`w = 0`** → `res_f = v_t`, `C_f = compliance` | `constraints/friction_constraint.py:306` |
| contact | `phi, dphi_dc, dphi_dλ = FB(signed_dist, f_n, ...)`<br>`res_n = phi/dt`, `J_hat = dphi_dc·J`, `C = (dphi_dλ+compliance)/dt²` | **`(signed_dist, 1, 0)`** → `res_n = signed_dist/dt`, `J_hat = J`, `C = compliance/dt²` | `constraints/contact_constraint.py:98` |

R1 and R2 are each a branch inside one `@wp.func`. The tangent frame, effective mass,
force-application Jacobian and `d_spatial` accumulation are literally untouched — which is
what §4 predicted, but the mechanism is even more contained than that.

### 0.2 The normal→friction coupling is ALREADY lagged

`friction_residual_kernel` reads `data.constr_force_prev_iter.n`, not the current iterate.
The cone limit uses the *previous* NR iterate's normal force, so the friction block has no
Jacobian entry w.r.t. `λ_n` today. §4's "decouples from the normal block" is therefore not a
new benefit — what R1 actually removes is (a) the FB nonlinearity and (b) the *staggered
fixed-point* that lagged coupling forces NR to resolve. §8's attribution needs to say which.
Phase 2 below is the control that separates them, and it costs no code.

### 0.3 There is a hidden iteration floor that will inflate R1's apparent win

`core/base_engine.py:55` `_update_newton_iter_kernel` forces `keep_running=1` until
`iter_count >= nr.backtrack_min_iter` (default 2), with this comment:

> the friction kernel early-exits at iter 0 because `constr_force_prev_iter` is zeroed at
> `engine.step` start, which makes the iter-0 residual "friction-blind"

Under R1 that early exit (`friction_constraint.py:381`, `mu_max·force_n_prev <= 1e-6`) must
be **removed** — bilateral friction does not depend on `λ_n`, so the row is valid at iter 0.
The warmup floor then becomes unnecessary too. So part of any measured iteration drop is the
floor disappearing, not physics. **Report R1 at both `backtrack_min_iter=2` and `=0`, and
report the full rung at `=0` as well.** Without this the headline number is not attributable.

### 0.4 Risk the source document does not flag: R1 may converge *worse*

`C_f = w/dt + compliance`. Today `w` is an effective-mass-scaled weight that acts as an
adaptive diagonal regularizer on the friction block. Setting `w = 0` leaves
`C_f = compliance.friction = 1e-6` — a near-zero diagonal on 2/3 of the constraint rows,
feeding a saddle-point system solved by PCR. Removing the complementarity is not
unambiguously a conditioning win; it may cost PCR iterations even as it saves NR iterations.

This is a *reason to run the experiment*, not a reason to doubt it — but the branch should
not be built on the assumption that the answer is known. Log NR and PCR separately (§8
already asks for this) and sweep `compliance.friction` for the bilateral rung specifically.

---

## 1. Mechanism: how a rung is selected

New sub-config in `core/engine_config.py`, matching the existing sub-config style:

```python
@dataclass(frozen=True)
class RelaxationConfig:
    friction: str = "cone"             # "cone" | "bilateral"
    contact:  str = "complementarity"  # "complementarity" | "equality"
    control:  str = "compliant"        # "compliant" | "prescribed"
    gyro:     bool = True              # False → drop f_gyro
```

Mapped to `int` and threaded to the kernels as a scalar argument, branched on inside the two
core `@wp.func`s.

**Why a runtime scalar and not a build-time kernel factory.** A factory closing over
`wp.constant` would eliminate the dead branch at compile time and avoid carrying the FB
code's register pressure into the relaxed rung. But it restructures module-level kernel names
that six other modules import by name, and it is a larger change to make before knowing
whether the effect exists. The uniform runtime branch is free in divergence terms (every
thread in a launch takes the same side); its only cost is occupancy from max-over-branches
register allocation, which affects **wall-clock but not NR/PCR iteration counts** — and §8
already names iteration counts as the primary metric. If wall-clock turns out to be the
deciding number for the runtime ladder, convert to a factory then, with the physics already
validated.

Touch list: 2 core `@wp.func`s, the 4 kernel variants per constraint
(`*_residual_kernel`, `*_constraint_kernel`, `batch_*`, `fused_batch_*`), and their launch
sites in `core/residual.py` and `core/linear_system.py`. Mechanical.

The `batch_*` variants only run when `linesearch.enabled=True` (default off). Thread the flag
through them in the same pass; if that slips, add an explicit `raise` on
`linesearch.enabled and relaxation != default` rather than letting the linesearch silently
evaluate a different physics than the Newton step.

---

## 2. Build order

Each phase states its verification. No phase advances until its check passes.

### Phase 0 — measurement harness, before any physics change

`data.iter_count` and `cr_solver.iter_count` already exist; PCR counts are currently only
copied out under `logging_config.hdf5.enabled` (`base_engine.py:230`). Add a forward-only
bench in `dev/` that reads both without requiring HDF5.

- forward-only, `differentiable=False` (the existing `results/scalability_axion_*.json`
  numbers include the backward pass — do not reuse as baseline, per §8)
- fixed `nr.max_iters`, identical initial states and control sequences per rung
- log: NR iters/step, PCR iters/NR step, per-block residual norms, wall-clock/step,
  trajectory, per-contact `λ_n`, `λ_f`

→ **verify:** two consecutive baseline runs agree bit-for-bit on iteration counts.

### Phase 0.5 — R0, drop `f_gyro`

One line in `constraints/dynamics_constraint.py`. Exercises the config plumbing end to end on
a change whose physics answer is known in advance (a 3-wheeled robot at 0.3 m/s has
negligible `ω × Iω`).

- certificate `‖ω × Iω‖·dt / ‖M·Δu‖`, computed inline

→ **verify:** trajectory delta below solver tol on cruise, certificate reports ~0. A
non-trivial certificate here means the certificate machinery is miscalibrated, and it is far
cheaper to find that out now than in Phase 1.

### Phase 1 — R1 bilateral friction

`compute_friction_model` returns `(v_t, 0, 0)`; delete the `mu_max·force_n_prev` early exit
under this mode.

→ **verify (equivalence):** flat hard ground, high `mu`, low tractive demand — bilateral
matches the cone rung to solver tolerance.

→ **verify (continuity, stronger):** sweep tractive demand upward and check the two rungs'
divergence goes to zero *continuously* as `‖λ_f‖/(μλ_n) → 1⁻`. The flat-ground test alone
mostly proves the tangent frame survived; a discontinuity before saturation would pass it and
still be a bug.

→ **verify (ordering):** hard skid turn — the rungs must now differ, with bilateral the
optimistic one (over-predicts achievable yaw).

→ **measure:** NR/PCR at `backtrack_min_iter ∈ {0, 2}` for both rungs (per §0.3), and a
`compliance.friction` sweep for the bilateral rung (per §0.4).

**Certificate**, with the guard the source document omits:

```
saturation_k = ‖λ_f,k‖ / max(μ_k · λ_n,k, λ_floor)
```

Unguarded, this diverges exactly where it means least — a near-airborne wheel with `λ_n → 0`
reports enormous saturation while carrying no load. Under R1+R2 combined it is worse, since
`λ_n` may be negative and the ratio flips sign. Report load-weighted, and define the
interaction with R2's certificate explicitly: `λ_n < 0` **supersedes** `saturation > 1` on
the same contact (a wheel that should be airborne has no meaningful friction budget).

### Phase 2 — the attribution control (config only, no code)

Promoted from §7 "parametric side-experiment, low expectations", because §0.2 makes it the
only clean way to separate two hypotheses. Run the 2×2:

| | cone kept | cone removed |
|---|---|---|
| **cone binds** | full (`μ = 0.7/0.4`) | — |
| **cone never binds** | `μ = 50` | R1 bilateral |

If `μ = 50` is as fast as bilateral, the cost is *binding / mode switching* — §2's
hypothesis. If `μ = 50` stays slow, the cost is the FB nonlinearity and lagged fixed-point
themselves, and the "stick–slip is the driver" claim in §2 is wrong. Either result is
publishable; without this cell neither is supported.

Note `μ = 50` also degrades conditioning on its own — which is precisely §2's argument
against parametric relaxation, so a slow `μ = 50` is consistent with two stories. Report the
`C_f` diagonal magnitudes alongside so they can be told apart.

### Phase 3 — R2 normal equality

`compute_contact_core` uses `(signed_dist, 1, 0)` in place of the FB triple. Collision
detection untouched, contact set not fixed — per §5.

- certificate: `λ_n < 0` = "this contact should have separated"

→ **verify:** flat cruise where nothing should lift off — matches full rung to tol.
→ **verify:** curb — `λ_n < 0` fires on exactly the contacts the full rung reports as
separating. This is a discrete predicate, so it should match exactly, not approximately;
any mismatch is a real defect in the detector.

**Interaction to control:** `warm_start` seeds `λ_n` and its cold-start terms
(`cold_impact = m_eff·(−v_n)/dt`) assume `λ_n ≥ 0`. Under R2 that assumption is gone. Hold
`warm_start` config identical across rungs and check whether seeding helps or hurts the
equality rung separately — do not let it vary silently.

### Phase 4 — R1 + R2 = the "kinematic rolling" rung

The interesting combination. Compare against `helhest_stack`'s settle on the same scenes
(two-engine calibration). Both certificates active; interaction rule from Phase 1 applies.

### Phase 5 — R3a prescribed motion

The control row is `R_c = (qd − target_vel) + α·λ·h` with `α = 1/(h·ke)`
(`core/residual.py:250`). R3a is `α = 0` **exactly**.

Worth being precise about why this is structural and not parametric: `α` enters *linearly*
and `0` is exactly attainable, so this is the ideal-actuator assumption represented exactly —
not `ke = 1e6` approaching it with the conditioning blow-up §2 warns about. It is the one
place where the two categories touch.

Risk: exact zero diagonal on the control block removes its regularization entirely. Same
family of concern as §0.4, and possibly worse. Have `compliance` available as a fallback and
report whether it was needed.

### Phase 6 — R3b full DOF elimination — **conditional, not scheduled**

§2's own unknown-count table says NCP removal dominates; R3b is the most invasive change
(`model_builder.py`, `engine_dims.py`, `data_views.py`, **and the adjoint**) on the axis
expected to matter least. Gate it on Phases 1–5: if R1/R2 deliver the speedup and R3a covers
the actuator axis for the attribution study, R3b buys only unknown-count, and may not be
worth the adjoint risk.

If it does run: finite-difference gradient check **before** anything downstream consumes
`λ = H⁻ᵀ∇J`. §6's warning is correct and the failure is silent.

---

## 3. Scenes

S1–S5 as specified in §8. One gap: no scene makes two certificates fire *simultaneously and
disagree* about which rung to promote — which is the case the runtime ladder must actually
handle. S3-oblique (curb + lateral friction demand) is already that scene in substance;
label it as the interaction case rather than contact-dominant, or add S6.

S1 is not optional — it sets the certificates' false-positive rate, and Phase 0.5 gives an
early read on it.

---

## 3.5 Results — Phases 0 through 3

Harness: `dev/relax_bench.py`. Helhest on flat ground, 40-step settle under the full
formulation (every rung starts bit-identical), then 40 driven steps. Engine config mirrors
`examples/conf/engine/ostrich.yaml`, `dt=3e-2`, forward-only, eager mode, 1 world, RTX A500,
`converge` protocol (`nr.atol=1e-3`, `max_iters=64`).

> **Correction.** An earlier version of this section concluded that removing the friction cone
> never helps and that R2 should be built before R1. Both claims were artefacts of running the
> harness at `builder.rigid_gap = 1.0`, copied from `examples/helhest/obstacle_benchmark.py`.
> That is the top of the project's range (newton's default is 0.1; the helhest *drive* scenes
> use 0.5) and it is wrong for a flat-ground cruise scene. The corrected results are below.
> What survives from that analysis is marked as such.

### The governing variable: unloaded contacts, not `rigid_gap`

Collision proposes candidates out to `rigid_gap`, and most carry essentially no load. The
cone formulation absorbs them for free: the friction budget is `mu*lambda_n`, which scales
*continuously* with load, so a barely-loaded contact is automatically barely-relevant.

A bilateral rung has no such scaling. Every contact past the `mu*lambda_n > 1e-6` gate gets an
**unbounded** multiplier — full-strength no-slip regardless of whether it carries any load.
Measured directly (cruise, `rigid_gap` 0.1 vs 0.2):

| gap | pairs in contact | lambda_n | clear the 1e-6 gate |
|---|---|---|---|
| 0.1 | 3 wheels | 99 – 271 | all |
| 0.2 | 3 wheels **+ chassis** | chassis **1e-4 – 7e-4** | **all** |

At `rigid_gap >= 0.2` the chassis starts generating candidates carrying six orders of
magnitude less load than a real wheel contact. They are physically irrelevant, they clear the
gate, and they pin the chassis. The `1e-6` threshold was tuned for a formulation where it only
had to skip *exactly*-zero contacts; for a bilateral rung it **is** the entire active-set
decision, and it is ~4 orders of magnitude too permissive.

`relaxation.friction_load_gate` makes it configurable. Setting it to `1e-2` — between the two
populations, with ~4 orders of margin each way — removes the `rigid_gap` dependence entirely:

| gap | rung | NR/step | `\|omega\|` | ‖Δ chassis‖ |
|---|---|---|---|---|
| 0.2 | full | 5.25 | 1.936 | — |
| 0.2 | bilateral, gate 1e-6 | 61.23 | 0.743 | 0.835 (welded) |
| 0.2 | **bilateral, gate 1e-2** | **5.15** | 1.935 | **7.4e-4** |
| 0.5 | bilateral, gate 1e-2 | 5.17 | 1.935 | 7.4e-4 |

So `rigid_gap` was a proxy: it controls how many unloaded candidates exist. The real variable
is that **a relaxed rung needs a load-proportional active-set criterion, because it gave up
the continuous one the cone provided for free.**

### Falsified: "per-contact no-slip welds a multi-point wheel"

An earlier version of this document claimed that pinning several material points of a rigid
wheel over-constrains it and forbids spin. **That is wrong**, and the test that killed it is
worth recording. Instrumenting the separation of each pair's load-carrying contacts, split
into normal and tangential components:

| `rigid_gap` | normal spread | tangential spread | `\|omega\|` bilateral |
|---|---|---|---|
| 0.02 | 0.0000 | 0.1100 | 1.935 |
| 0.1 | 0.0000 | 0.1100 | 1.935 |
| 0.2 | 0.0000 | 0.1100 | 0.497 |
| 0.5 | 0.0000 | 0.1100 | 1.180 |

A wheel resting on a plane carries **two** load-bearing contacts separated by 11 cm, and
bilateral friction handles them perfectly. Normal spread is zero at every gap, including the
gaps where the rung breaks — so it is not the discriminator either. The geometry is unchanged
across the whole sweep; only the contact *count* changes, via the unloaded chassis pair.

The algebra agrees: pinning two points requires `(omega x d) || n`. Separation along the axle
or along the rolling direction satisfies this identically, so it obstructs nothing. Only a
separation with a normal component would obstruct spin, and real ground patches have none.

`bilateral_patch` is retained because one row per patch is cheaper than N, not because
multi-point patches are unsound.

### Independent failure: skid-steer is kinematically incompatible with no-slip

The Helhest has three non-steerable wheels. Commanding (+8, -8, 0) rad/s has **no** no-slip
solution at all — the three wheels' instantaneous centres cannot agree. Skid-steering *is*
controlled slipping.

This is not fixable by any gate, and measurement confirms it: on skid, `friction_load_gate=1e-2`
leaves `|omega| = 0.000` and ‖Δ‖ = 0.18 (marginally *worse* than ungated, 19.77 vs 17.38
NR/step). Locking is the correct least-squares response to an infeasible constraint set.

**R1 is therefore restricted to near-straight-line motion on this robot, permanently.**

### R2 has the same disease and does not yet have the cure

`friction_load_gate` protects only the friction block. The normal equality still forces
`signed_dist = 0` on distant candidates, and there is no `mu*lambda_n` analogue to scale them
down — the constraint is equally strong at 1 mm and at 0.5 m. Measured: `kin_roll` with the
friction gate applied still diverges at `rigid_gap = 0.5` (`lambda_n ~ -9.7e8`, then NaN).

R2 needs its own gate — a distance predicate, skipping the equality where
`signed_dist > threshold`. Note this is exactly what §5 warned about ("that's a prediction
problem, not a constraint swap"), with one mitigating difference: the predicate is evaluated
fresh from current geometry each iteration rather than carried over from the previous step.
**Open design decision, not yet implemented.**

### Where this leaves the branch

The rungs are usable exactly where the runtime ladder needs them — benign, few-contact,
unsaturated regimes — and unusable in the demanding scenes the attribution study cares about.
That is a real result for the ladder (screen cheaply, promote on certificate) and a real
problem for the study.

Build order stands as originally written (R1, then R2, then combined). The Phase-1-first
ordering was never the problem; the harness was, and then the active-set criterion was.

The one structural lesson that generalises: **every structural relaxation needs to re-supply
the active-set criterion that the complementarity it deleted was providing implicitly.** The
cone supplied it via a load-proportional budget; the normal NCP supplies it via
`lambda_n >= 0`. Delete either and the criterion has to come back explicitly, or unloaded /
distant candidates are enforced at full strength. That is one line for friction
(`friction_load_gate`) and an open question for contact.

### Caveats

Two scenes, 40 steps, one world, one GPU, eager mode, no wall-clock (the harness syncs per NR
iteration by design). No seed sweep, no curb scene. Iteration counts are censored by the
64-iteration cap wherever a rung reports 64. `no_gyro` showed a convergence failure on skid
(residual 3.6 at `rigid_gap=1.0`) that is still unexplained.

---

## 4. Open questions, re-prioritized

Q3 from §10 (can the Newton factorization be retained so the adjoint is one back-substitution)
should move **before Phase 1**, not "early". If the answer is no, the DWR certificate costs a
full extra solve per candidate and the runtime ladder's economics change qualitatively — that
is a premise of the branch, not a detail.

Q1, Q2, Q5 are genuinely deferrable. Q4 (forward-only throughput at 128/256/512/1024 worlds)
falls out of the Phase 0 harness for free.
