# Relaxation Branch — Structural Physics Relaxations in Ostrich

Working plan for a branch that adds **alternative constraint formulations** to Ostrich, so the
same engine can run the same robot under different named physical assumptions.

Written 2026-08-04 to be picked up cold. Companion documents live in the `helhest_stack`
scratchpad (`relaxation-axes-design.md`, `adaptive-fidelity-literature-map.md`) and cover the
research framing and prior art; **this file is implementation only**.

---

## 1. Why this branch exists

`helhest_stack` (the navigation stack for the same robot) plans with a very cheap rollout model:
`(x, y, yaw)` integrated kinematically from wheel speeds, and `(z, pitch, roll)` from a
quasi-static 3×3 Newton settle that forces all three wheels onto the terrain. It runs ~22,000
rollout-steps/ms. It is wrong in specific, nameable ways: no slip limit, no wheel liftoff, no
motor stall, no momentum.

The research question is *which* of those omissions actually changes the planner's decision, and
when. Answering it needs a family of models that differ by **exactly one named assumption at a
time**, sharing everything else — same integrator, same collision pipeline, same solver. That is
only possible inside one engine.

Ostrich is that engine. This branch adds the relaxed formulations.

**Two distinct uses, don't confuse them:**

- **Attribution study** — run every rung on a scene set and measure which assumption changes the
  candidate *ranking*. Cost is irrelevant here; correctness of attribution is everything.
- **Runtime ladder** — a cheap rung screens candidates, a subset is promoted to an expensive
  rung. Cost is everything here.

---

## 2. The key distinction: structural, not parametric

There are two ways to "relax" a constraint, and only one of them is useful.

| | Parametric | Structural |
|---|---|---|
| Ideal motors | `target_ke = 1e6` | **eliminate the wheel DOF and its multiplier** |
| No slip | `mu = 50` | **replace the friction NCP with a bilateral equality** |
| Persistent contact | `contact_fb_smooth_eps_sq ↑` | **replace normal complementarity with an equality** |

Parametric relaxation keeps the system size and the constraint class, changing only conditioning.
It is **often slower**: `ke → ∞` makes the control constraint hard (`α = 1/(h²·ke + h·kd) → 0`),
and scaling down the inertia term removes diagonal dominance. Softening (`compliance ↑`,
`eps_sq ↑`) does help conditioning, but modestly.

Structural relaxation removes unknowns and/or removes complementarity. That is where
order-of-magnitude gains live — and each rung is a *textbook assumption* ("rolling without
slipping") rather than a fudge factor, which is also what makes the violation certificates
well-defined. **You cannot write "the residual of `eps_sq = 1e-4`."**

Rough unknown counts for the Helhest model (chassis + 3 wheels), illustrative:

```
full Ostrich
  bodies              4 × 6   = 24
  revolute joints     3 × 5   = 15
  control             3 × 1   =  3
  contacts   3 normal + 6 fric =  9
                              ────
                               51    NCP, non-smooth, active set flips

wheels slaved + spin prescribed
  bodies              1 × 6   =  6
  joints / control            =  0
  contacts                    =  9
                              ────
                               15

+ bilateral friction / normal equality
                               15    but LINEAR — no FB, no active set
```

The NCP removal is expected to dominate. **Stick–slip mode switching is the known cost driver in
this engine** (per the author), so deleting the complementarity should matter more than deleting
unknowns.

---

## 3. What makes this tractable: the residual is already block-separable

`src/ostrich/core/residual.py` assembles the residual from five independent kernel launches, each
adding into a shared spatial residual plus its own constraint block:

```
unconstrained_dynamics_kernel  →  res.d_spatial
joint_residual_kernel          → +res.d_spatial, res.c.j
control_residual_kernel        → +res.d_spatial, res.c.ctrl
contact_residual_kernel        → +res.d_spatial, res.c.n
friction_residual_kernel       → +res.d_spatial, res.c.f
```

**An alternative implementation of a block is additive, not invasive.** Write
`friction_bilateral_residual_kernel` alongside `friction_residual_kernel` and select it at build
time. Nothing else needs to know.

Other relevant landmarks:

- `src/ostrich/constraints/friction_constraint.py` — FB on the (elliptical) Coulomb cone;
  `resolve_friction_frame` builds the tangent frame; `mu_perp < 0` is an isotropy sentinel
- `src/ostrich/constraints/dynamics_constraint.py:48` —
  `h_d = M·(u − u_prev) − (f_ext + f_g + f_gyro)·dt`
- `src/ostrich/mechanics/complementarity.py` — `scaled_fisher_burmeister{,_diff}`
- `src/ostrich/core/engine_config.py:162` — `ComplianceConfig`
- `src/ostrich/core/{linear_system,newton_step,linesearch,residual}.py` — the solve
- `examples/helhest/common.py` — free-joint base, revolute wheels in `TARGET_VELOCITY`,
  `TARGET_KE = 150`, `friction_axis_local = (0,1,0)`, μ 0.7 lateral / 0.4 rear

---

## 4. R1 — Bilateral friction (no-slip). **Do this first.**

### The assumption

Rolling without slipping: relative tangential velocity at each contact is zero, and the friction
force required to achieve it is unbounded.

### What changes

Replace the FB cone constraint with a **linear velocity constraint**:

```
res.c.f[k]  =  (v_rel · t1)  +  α·λ_f1       and same for t2
```

where `v_rel` is the relative velocity of the two contacting bodies at the contact point, `t1/t2`
the tangent frame, `α` the compliance term.

**Unchanged:** the tangent frame construction (`resolve_friction_frame`), the effective-mass
helper, and how `λ_f` is applied back into `d_spatial`. The force-application Jacobian is
identical.

**Removed:** the FB call, the dependence on `constr_force_prev_iter.n` (the cone's coupling to
the normal force), `mu`, `mu_perp`. The multiplier becomes unbounded and sign-free.

### Why it should be much faster

- No complementarity → no active set → **no stick–slip mode switching**
- The constraint row is **linear in body velocities** → the friction block's Jacobian is constant
  within a step, no re-linearization
- Decouples from the normal block (no `λ_n` in the friction row)

Expect Newton iteration counts to drop sharply. This is the hypothesis the branch exists to test.

### Equivalence test (must pass)

On a scene where friction never saturates — flat ground, low tractive demand, high `mu` in the
reference — the bilateral rung must match the full NCP rung **to solver tolerance**. If it
doesn't, the tangent frame or the force application diverged, not the physics.

Second test: a scene that *does* saturate (hard skid turn). The two must now differ, and the
bilateral rung must be the *optimistic* one — it will over-predict achievable yaw, because
unbounded friction lets the wheels do whatever is asked.

### Violation certificate

The reason this rung is useful for the method: the recovered `λ_f` is unbounded, so you can
check it against the cone that *would* have applied:

```
saturation_k  =  ‖λ_f,k‖ / (mu_k · λ_n,k)
```

`> 1` means the no-slip assumption demanded more friction than physically available. This is
computable from the cheap rung's own output — no expensive solve. It is the friction axis's
violation indicator, and it is the primary deliverable of R1 beyond the speedup.

---

## 5. R2 — Normal contact. **Harder than it looks; read before starting.**

The naive plan was "fix the contact set and impose `signed_distance = 0` as an equality." That is
not right, for four reasons:

1. **Which contacts?** In the full model, collision detection proposes candidates and
   complementarity decides which are active. Fixing the set means *deciding* it — from the
   previous step, from a geometric predicate, or by fiat. That's a prediction problem, not a
   constraint swap.
2. **Contact points move.** Wheel–terrain contact migrates as the robot rolls. There is no fixed
   pair of points to constrain, and re-detection is most of the geometry cost anyway — so fixing
   the set saves less than it appears.
3. **Equality produces adhesion.** `g = 0` as a bilateral constraint lets `λ_n` go *negative* —
   the terrain pulls the wheel down. Physically impossible. This is exactly what
   `helhest_stack`'s settle does, and why its residual blows up on curbs.
4. **Contact count is dynamic** → buffer sizing, warm start, graph capture all assume a maximum.

### The formulation that actually works

**Do not fix the set.** Keep collision detection exactly as-is. Replace only the *complementarity*
on the detected set with an equality:

```
res.c.n[k] = signed_dist_k + α·λ_n,k         (instead of FB(signed_dist, λ_n))
```

Then **the adhesion is the feature**: `λ_n,k < 0` means "this contact should have separated" — a
wheel that ought to be airborne. That is a discrete, exact predicate, valid precisely where a
linearised estimate would not be, and it is the contact axis's violation certificate.

This mirrors `helhest_stack` exactly: its settle forces all wheels down and its `normal_loads`
returns a negative load in the same situation. Same information, same detector, two engines.

### Caveats

- The rung is *qualitatively* wrong when contacts should break, not merely less accurate. That's
  a deliberate, documented approximation, and the certificate flags it.
- Speedup is likely smaller than R1's, since collision detection cost is unchanged and the normal
  block is smaller than the friction block.
- Combined with R1 this gives a coherent **"kinematic rolling" rung** — structurally the Ostrich
  analogue of the `helhest_stack` settle, inside the same engine. That combination is the
  interesting one.

---

## 6. R3 — DOF elimination (ideal actuators). Real surgery.

### R3a — cheap partial version, do this first

Keep the wheel bodies. Replace the revolute-joint + control-constraint pair with a single
**prescribed-motion constraint** on the wheel spin. Removes the control block (`res.c.ctrl`), but
not the body DOF. Much less invasive; a useful intermediate that tests the plumbing.

### R3b — full elimination

Slave the wheels to the chassis with prescribed relative spin: the wheel ceases to be an
independent body and becomes attached geometry with a known angular velocity. Removes 18 body
DOF, 15 joint multipliers, 3 control multipliers.

**Touches:** `core/model_builder.py`, `core/engine_dims.py` (body/joint counts, constraint
offsets), index maps in `core/data_views.py`, and **the adjoint path**.

> ⚠️ Getting the adjoint wrong here breaks silently. The DWR violation certificate that the whole
> research method depends on is `λ = H⁻ᵀ∇J` — if the adjoint is inconsistent with the reduced
> system, the certificate produces plausible numbers that are wrong. **Add a finite-difference
> gradient check before trusting anything downstream.** (`helhest_stack` learned this the hard
> way: a Warp `sample_field` position-gradient bug threw friction gradients off ~47% and was
> invisible under uniform fields.)

---

## 7. Not in scope for this branch

- Parametric dials (`compliance.*`, `contact_fb_smooth_eps_sq`) — worth a cheap side experiment
  as a possible "release valve" if stick–slip cost blows up, but **expect little**, and they are
  not rungs.
- Inertia relaxation (`s·M(u − u_prev)`, dropping `f_gyro`) — only coherent once R3 removes DOF;
  on its own it degrades conditioning and goes singular when contacts break. Revisit later.
  Exception: dropping `f_gyro` alone is one line, well-posed, and has an exact free certificate
  `‖ω × Iω‖·dt / ‖M·Δu‖` — cheap to try any time.
- Terrain remeshing from a live heightmap (needed for the runtime ladder, not for the study).

---

## 8. Measurement harness

For every (rung × scene × seed), log:

- **NR iterations per step** and **PCR iterations per NR step** ← the primary cost metric.
  Engine-internal, noise-free, and it tests every prediction in §2 directly.
- wall-clock per step, forward-only, no backward pass
- final residual norms per block
- convergence failures / conditioning blow-ups
- full trajectory `(x, y, yaw, z, roll, pitch)`
- per-contact `λ_n`, `λ_f`, saturation ratio, `min λ_n`

**Controls that will invalidate results if skipped:**

- **fix the NR iteration budget across rungs** — removing or stiffening constraints changes
  conditioning and the number of unknowns; without this you will attribute solver effects to
  physics, and the whole attribution claim collapses
- identical initial states and identical control sequences across rungs
- forward-only timing (existing `results/scalability_axion_*.json` numbers almost certainly
  include the backward pass — do not reuse them as the cost baseline)

### Scenes

Four, plus a compound. Each is designed so one assumption is *dominant*, not isolated — axes
genuinely couple, so run the full factorial and report interactions.

| | Scene | Setup | Dominant |
|---|---|---|---|
| S1 | Cruise (null) | flat hard ground, 0.3 m/s, straight / gentle arc | none — expect all rungs to agree |
| S2 | Turn into a constraint | corner or obstacle at fixed lateral offset; candidates differ in wheel differential | **friction (R1)** |
| S3 | Curb, over vs. around | 0.2 m step with a bypass; perpendicular **and oblique** | **contact (R2)** |
| S4 | Momentum | stop/hazard at a distance; tractive demand kept under the cone | inertia |
| S5 | Stairs | compound | demo only, never for attribution |

S1 is not optional — it establishes the false-positive rate of the certificates.

20–50 randomized seeds per scene (curb height, approach angle, μ, speed, initial pose). The
pattern already exists in `results/terrain_batch/seed_*.json`.

---

## 9. Build order

1. **R1 bilateral friction** + both equivalence tests
2. Measure NR/PCR iteration counts, R1 vs full, on S1–S4
3. **R2 normal equality** with the `λ_n < 0` certificate
4. R1+R2 combined = the "kinematic rolling" rung; compare against `helhest_stack`'s settle on the
   same scenes (calibration of the two-engine gap)
5. R3a prescribed-motion constraint
6. R3b full DOF elimination + finite-difference adjoint check
7. Parametric side-experiment, low expectations

Each rung is its own build and its own CUDA graph. **Rungs do not need to coexist within one
launch** — the runtime ladder is `launch build_0 × N candidates`, then
`launch build_1 × K′ promoted`. This removes any need for per-world parameter arrays.

---

## 10. Open questions to resolve early

1. Are `shape_material_mu`, `shape_mu_perp`, `shape_friction_axis_local` per-world (`ndim=2`)?
   Check `core/model.py`. Affects nothing for R1 (which drops them) but matters for parametric
   experiments.
2. Does `add_joint_free` support per-DOF modes, or does R3 need a new joint type in the builder?
3. Can the Newton factorization be retained after the solve so the adjoint is a single
   back-substitution? The DWR certificate's affordability depends on this.
4. Forward-only throughput at realistic shapes: 128 / 256 / 512 / 1024 worlds × 5 / 10 / 25 steps.
   Needed to size the runtime ladder.
5. How does terrain enter for these scenes — static mesh built once, or rebuilt per run? Check
   `examples/helhest/obstacle.py` and the `terrain_traversal` experiment.
