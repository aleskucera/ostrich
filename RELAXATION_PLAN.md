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

## 3.5 Results so far — Phases 0, 0.5 and 1

Harness: `dev/relax_bench.py`. Scene: Helhest on flat ground, `mu` 0.9, 40-step settle under
the full formulation (so every rung starts from a bit-identical state), then 40 driven steps
at 2 rad/s on all three wheels. Engine config mirrors `examples/conf/engine/ostrich.yaml`,
`dt=3e-2`, forward-only, eager mode, 1 world, RTX A500. `converge` protocol
(`nr.atol=1e-3`, `max_iters=64`) — iterations-to-convergence.

### Phase 0.5 — dropping `f_gyro`: confirmed negligible, as predicted

| | NR/step | PCR/NR | certificate | ‖Δ chassis‖ |
|---|---|---|---|---|
| full | 5.22 | 12.06 | — | — |
| no_gyro | 6.55 | 15.32 | 8.5e-4 | 1.6e-5 |

The certificate `‖ω × Iω‖·dt / ‖M·Δu‖` reports ~1e-3 and the trajectory delta is ~1e-5, i.e.
the certificate correctly predicts its own irrelevance. That is the false-positive check S1
exists for, and it passes. The NR/step difference is run-to-run noise in when the tail
iterations trip `atol`, not a real cost difference.

### Phase 1 — R1 does not work as specified, for two separate reasons

**(a) A per-contact velocity equality is not "rolling without slipping".** It says *this
material point is pinned*. Pinning two distinct points of a rigid wheel confines its motion to
the axis through them, which forbids spin. Measured: the rung converges cleanly — 6.00
NR/step, residual 7.8e-5, better than baseline — to `|ω_wheel| = 0.000` with the robot
displacing 1e-6 m. It is not a conditioning failure. It converges accurately to welded.

No-slip therefore belongs to the contact *patch*, not to each sampled point. `friction=
"bilateral_patch"` imposes one no-slip row per `(shape0, shape1)` pair. This is also why
`helhest_stack`'s settle is a 3×3 system — the same constraint, correctly counted.

**(b) Bilateral friction does NOT decouple from the normal block.** Collision proposes
candidates out to `rigid_gap` (1.0 m in these scenes), most of them not touching. The cone
rung ignores them for free — zero normal force means zero friction budget. A bilateral rung
has no budget to be zero, so without the `mu·λ_n ≤ 1e-6` gate it pins the chassis to contacts
that are metres away, and the robot cannot move at all. The gate has to stay.

That gate is a lagged dependence on `λ_n`. So R1 removes `λ_n` from the constraint *row* but
keeps it for *active-set selection* — and the active set is what the iteration count actually
responds to. §4's "decouples from the normal block" does not survive contact with the
collision pipeline.

### Phase 2 — the attribution control refutes the branch's central hypothesis

The 2×2 from §2, run as config only:

| | cone kept | cone removed |
|---|---|---|
| **cone binds** (μ=0.9, saturation 2.6) | full — **5.22** NR/step, 12.06 PCR/NR | — |
| **cone never binds** | μ=50 — **5.22** NR/step, 11.27 PCR/NR | bilateral_patch — **24.45** NR/step, 23.09 PCR/NR |

μ=50 costs *exactly* what the full model costs and lands within 6.6e-4 m of its trajectory.
Making the cone never bind is free. Removing it structurally costs **4.7× more NR iterations**
and 9× more total PCR iterations.

**Stick–slip mode switching is not the cost driver in this engine.** RELAXATION_BRANCH.md §2
states it is ("per the author") and builds the whole build order on it. The FB cone is
approximately free here; what costs is selecting which contacts carry load, and that is the
*normal* complementarity, not the friction cone.

A `compliance.friction` sweep over 1e-8 → 1e-6 → 1e-4 does not rescue the bilateral rungs
(24–60 NR/step throughout), so §0.4's conditioning hypothesis is also not the explanation.
The mechanism is active-set churn: an unregularized equality row whose membership is decided
by a lagged normal load.

### What this implies for the build order

**R2 should go first, not R1.** The load-carrying set is what the solver is spending its
iterations on, and R2 is the rung that eliminates the question — under a normal equality every
detected contact carries load by fiat, which also removes the ambiguity that forces R1 to keep
its `λ_n` gate. R1 is only coherent *after* R2, as part of the combined kinematic-rolling rung
(§4 of the source document, Phase 4 here) — not as the standalone first rung.

The DOF-elimination axis (R3) is untouched by this result and remains conditional.

### Caveats

Single scene, 40 steps, one world, one GPU, eager mode, no wall-clock (the harness syncs per
NR iteration by design). The skid scene and the seed sweep are not run. These numbers are
strong enough to reorder the build order, not to close the question.

---

## 4. Open questions, re-prioritized

Q3 from §10 (can the Newton factorization be retained so the adjoint is one back-substitution)
should move **before Phase 1**, not "early". If the answer is no, the DWR certificate costs a
full extra solve per candidate and the runtime ladder's economics change qualitatively — that
is a premise of the branch, not a detail.

Q1, Q2, Q5 are genuinely deferrable. Q4 (forward-only throughput at 128/256/512/1024 worlds)
falls out of the Phase 0 harness for free.
