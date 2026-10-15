# Why Helhest Junior jitters when it turns

Answer to `jitter_investigation_brief.md`. Everything below was re-derived on
this machine, headless, against this working tree. Where the brief's numbers
are quoted they are labelled as such.

## The short answer

The turn is not what breaks. **Sliding** is what breaks, and a three-wheel
skid-steer cannot rotate without sliding while it can drive in a straight line
without sliding at all.

Ostrich's friction row does not bound the tangential force by `mu * f_n`. It
bounds it from *below* at slip and not at all from above. The moment a contact's
tangential impulse reaches the Coulomb limit, `mu` drops out of the row entirely
and the row degenerates into "`v_t` antiparallel to `lambda_f`" — a condition any
magnitude satisfies. The force is then set by whatever tangential load the rest
of the system happens to apply. In effect every loaded wheel is *welded* to the
ground, in both directions, with unbounded strength.

A straight line never notices: rolling is compatible with a weld, so `v_t ~ 0`,
the force stays inside the cone, and the model is correct there. A spin-in-place
requires all three wheels to slide, so it drives every contact into exactly the
saturated branch where the model has lost `mu`. The robot then fights its own
welds: it reaches 7-25 % of the yaw rate its wheel command implies, its wheels
reach only ~45 % of their *own* commanded speed, and the yaw only advances in
bursts, on the steps where a weld is broken by a wheel losing normal load.

That is the mechanism. Two further defects amplify it, and both are specific to
harness B — which is why the two harnesses disagree.

## Reconciling the two harnesses: A is right, B is broken three ways

The engine sources are byte-identical between the two checkouts
(`ostrich-odinsim` is a git worktree of this repo; `md5sum` of
`friction_constraint.py`, `contact_constraint.py` and `base_engine.py` all
match). So the divergence is entirely scene and config, and harness A can be
reproduced here. It reproduces on every row:

| harness A row | A reported (yaw / std / jerk) | reproduced here |
|---|---|---|
| contact 1e-10 | −1.171 / 0.392 / 11.4 | −1.214 / 0.349 / 9.28 |
| contact 1e-6 | −1.210 / 0.033 / 1.2 | −1.211 / 0.038 / 1.22 |
| 1e-10, newton_iters 32 | — / — / 20.5 | −1.094 / 0.588 / 17.68 |
| 1e-10, dt 0.01 | — / — / 28.5 | −0.773 / 0.675 / 27.52 |

(Harness A's own configuration and metric: analytic ground plane, isotropic
wheels 0.8/0.4, k_p 250, dt 0.02, 1.5 s settle, 4 s window, sampled every
2 steps.)

**Harness A's numbers are correct and its compliance result is real.** Nothing
in it needs discarding.

Harness B differs from A in three ways, and *each one independently* destroys
the turn. All rows below are harness A's metric at contact compliance 1e-6, so
only the named variable moves:

| configuration | rate_std |
|---|---|
| plane, isotropic 0.8/0.4 — **harness A** | **0.035** |
| plane, isotropic 0.5/0.2 | 0.030 |
| plane, **anisotropic** 0.8/0.4 lat + 0.9/0.6 long | 0.444 |
| plane, "anisotropic" 0.8/0.4 + 0.8000001/0.4000001 | 0.180 |
| plane, anisotropic 0.5/0.2 + 0.9/0.6 (B's wheels) | 0.484 |
| **mesh**, isotropic, reduction `none` | 0.462 |
| mesh, isotropic, `fps` K=8 | 0.365 |
| mesh, isotropic, `cluster` K=8 (B's default) | 0.755 |
| mesh, anisotropic, cluster K=8 — **harness B** | 0.75-0.79 |

Lateral friction magnitude does nothing (0.035 vs 0.030) — the brief already
observed this, and the mechanism above explains *why*: once saturated, `mu` is
not in the row.

Harness B stacks anisotropy **and** the mesh **and** cluster reduction. Its
jitter is saturated by whichever defect you did not turn off, which is exactly
why nothing in harness B ever responded to any knob. Every "ruled out" line in
the brief was measured inside that saturated regime.

## Defect 1 — the friction cone is one-sided

`src/ostrich/constraints/friction_constraint.py`, `compute_friction_model`,
both the isotropic (line ~311) and elliptical (line ~352) branches:

```python
clamped_imp_norm = wp.min(raw_imp_norm, limit)
gap = limit - clamped_imp_norm          # == max(0, limit - |f|), never negative
phi_f = scaled_fisher_burmeister(d_t_norm, gap, 1.0, r)
denominator = r * raw_imp_norm + phi_f + denom_eps
w = r * (d_t_norm - phi_f) / denominator
```

The row the solver actually enforces is `res_f = v_t + w * lambda_f = 0`.
Setting `res_f = 0` and substituting `w` gives `phi_f * (d + r*raw) = -d*eps`,
i.e. `phi_f = 0`, i.e. `d_t_norm * gap = 0`. Because `gap` is clamped at zero
from below, that is satisfied by

* `d_t_norm = 0` (stick) with **any** `|f|`, including `|f| > mu*f_n`; or
* `gap = 0`, which means `|f| >= mu*f_n` — again with no upper bound.

Unclamping `gap` does not fix it (it drives `w` to 0 and welds harder). The
defect is that in the saturated regime the *denominator* normalises by the
previous iterate's own force magnitude, so `w = |v_t| / |lambda_prev|` and the
fixed point is scale-free in `lambda_f`.

**Standalone confirmation.** One mass, one contact, `mu*N = 800 N`, one step,
iterating exactly this update:

| tangential load | ostrich `f_t` | ostrich `v_t` | exact Coulomb `f_t` | exact `v_t` |
|---|---|---|---|---|
| 200 N | −200.0 | 0.0000 | −200.0 | 0.0000 |
| 700 N | −700.0 | 0.0000 | −700.0 | 0.0000 |
| 900 N | −817.3 | 0.0248 | −800.0 | 0.0300 |
| 2000 N | −1912.5 | 0.0263 | −800.0 | 0.3600 |
| 5000 N | −4826.7 | 0.0520 | −800.0 | 1.2600 |
| 20000 N | −17768.9 | 0.6693 | −800.0 | 5.7600 |

Below the limit it is exact. Above it, `f_t` tracks the applied load instead of
saturating, and the contact barely slips.

**Confirmation in the sim.** Cone occupancy is
`||(f_x/mu_x, f_y/mu_y)|| / f_n` per loaded contact — 1.0 is the Coulomb
boundary. Mesh harness, correct engine-side contact indexing:

| command | cone median | frac > 1 | total \|f_t\| / robot weight | rel. jitter (std/mean) |
|---|---|---|---|---|
| settle | 0.00 | 0 % | 0.00 | — |
| forward 1 rad/s | 0.03 | 2 % | 0.05 | 0.04 |
| forward 5 rad/s | 0.67 | 21 % | 0.50 | 0.09 |
| spin 0.5 rad/s | 1.29 | 80 % | 0.99 | 0.37 |
| spin 3 rad/s | 3.48 | 93 % | 2.50 | 1.00 |

This is the whole "why a turn and not a straight line" in one table. Even the
*gentlest* spin — 0.5 rad/s on the wheels, achieving a pathetic 0.034 rad/s of
yaw, 7 % of kinematic — is already 80 % outside the cone, whereas forward
driving at ten times the ground speed is 21 % outside.

Downstream consequences, measured during a commanded 3 rad/s spin:

* Wheels reach +1.5 / −1.5 rad/s of a commanded +3 / −3 (a straight line tracks
  4.95 of 5.00). The joint servo (`lambda = ke * (target - qd)`, k_p = 250)
  is applying ~440 N·m per wheel and still losing — 3.3x the Coulomb torque
  that wheel's normal load can support.
* Chassis yaw reaches 23-41 % of the rate implied by the wheels it does turn.
* Total tangential contact force is 2.5-3.5x the robot's weight, applied
  0.35 m below the CoM.

**It is not a convergence artifact.** Solving ten times harder converges *onto*
the wrong fixed point:

| solver settings | ‖residual‖ | yaw std / \|mean\| |
|---|---|---|
| nr 16, linear 26, tol 1e-3 (default) | 1.41e-2 | 1.03 |
| nr 64 | 5.71e-3 | 0.95 |
| nr 64, linear 200, tol 1e-8 | 1.40e-3 | 1.14 |

Residual down 10x, jitter unchanged, cone violation slightly *worse*. Also
falsified as causes by direct measurement: contact compliance, friction
compliance (1e-8 → 1e-2), warm start, linesearch, linear-solver iterations,
triangle size.

### The one-line fix

Normalise by the cone limit, not by the previous iterate's force — i.e. use
`clamped_imp_norm` in the denominator. Sub-limit behaviour is untouched
(`clamped == raw` there), and in the saturated regime
`w = d/limit = |v_t| / (mu * f_n)`, so `|lambda_f| = mu * f_n` exactly. In the
toy above it reproduces exact Coulomb to within 2 % at every load. In the sim:

| | \|f_t\| / weight | wheels (cmd ±3) | yaw, % of kinematic | cone median |
|---|---|---|---|---|
| current | 3.47 | +1.46 / −1.50 | 41 % | 3.5-5.8 |
| with the fix | 1.18 | +2.52 / −2.65 | 64 % | 1.2-1.4 |

Relative yaw jitter also flattens from 0.85-1.06 (turn-speed dependent) to a
constant 0.59. It does **not** remove the rocking — that is defect 3.

The patch is saved at
`/tmp/claude-1000/-home-kuceral4-projects-ostrich/387b76fc-a730-442f-80c9-07ef3f7701f9/scratchpad/cone_fix.patch` (env-gated on `OSTRICH_FRICTION_SLIP_NORM=clamped`,
default bit-exact with today). **It is not applied.** It changes every sliding
contact in the engine, so the paper's terrain and scalability numbers would all
move; that is a decision, not a bug fix to land quietly.

## Defect 2 — the anisotropic branch is discontinuous at `mu_x == mu_y`

The row in the table above worth staring at:

* plane, isotropic `mu = 0.8 / 0.4` → rate_std **0.035**
* plane, anisotropic `mu_x = 0.8/0.4`, `mu_y = 0.8000001/0.4000001` → **0.180**

The coefficients are numerically the same to seven digits. The only thing that
changed is which branch of `compute_friction_model` runs — and the answer gets
5x worse. The two branches are not the same function in the limit: the
isotropic branch pairs `FB(|v_t|dt, mu*f_n*dt - |f|dt)`, the elliptical branch
pairs `FB(mu*|v_t|dt, f_n*dt - |f|dt/mu)`. They differ by a factor of `mu^2` in
how force slack is weighted against slip, and the weights come back as
`w_x = w_tilde/mu_x^2`, `w_y = w_tilde/mu_y^2`.

This is consistent with the existing note that the `mu_x == mu_y` hard branch
must not be unified: the branches genuinely disagree, and the hard branch is
hiding the disagreement rather than resolving it. With real anisotropy
(0.9/0.6 longitudinal) the penalty is 13x (0.035 → 0.444).

Harness A never applied anisotropy (the brief verified this) — so harness A has
been running the good branch all along. Harness B applies it. That is the single
largest difference between them.

## Defect 3 — cluster reduction is unstable on a flat mesh

Every contact a wheel makes with the flat mesh has the *same* normal, so
`cluster_normal_dot_thresh = 0.996` matches all of them, and any two within
`cluster_pos_thresh = 5 mm` are declared duplicates. The reducer keeps "the
deepest" of each cluster — but on a perfectly flat mesh the depth differences
are geometric noise, so *which* points survive changes every step. The wheel's
support set (5-6 kept out of ~12 candidates) hops around its contact patch,
the friction lever arms jump with it, and normal load sloshes between wheels.

Mesh, 3 rad/s spin, measured per physics step:

| reduction | chassis roll peak-to-peak | steps with a wheel unloaded | yaw std/\|mean\| |
|---|---|---|---|
| `cluster` K=8 (default) | **6.33°** | **27.5 %** | 1.02 |
| `fps` K=8 | 0.36° | 5.8 % | 0.80 |
| `cluster` K=32 | 0.28° | 4.4 % | 0.82 |
| `none` | 0.34° | 5.8 % | 0.91 |
| analytic plane (6 contacts, reducer is a no-op) | 0.34° | — | 0.50 |

The plane never had this problem because a cylinder-plane contact yields 2
points per wheel, below the reduction budget, so the reducer never fires.

**This is the release valve for the stick-slip.** A per-step trace makes the
cycle explicit — the rear wheel is the only wheel whose lateral direction
opposes the spin, and the yaw rate is high exactly on the steps its contact set
is empty:

```
   t      wz  |   N_L    N_R   N_rear |   z_L    z_R  z_rear | contacts per wheel
3.00  -1.410  |    71   1043       -0 | 0.349  0.350   0.351 | [5, 7, 8]
3.03  -1.955  |   241    721        0 | 0.349  0.349   0.347 | [8, 4, 0]   <- rear airborne
3.06  -0.233  |   647      0      872 | 0.350  0.354   0.350 | [7, 8, 7]   <- right lifted 4 mm
3.09  -0.118  |   356      0      335 | 0.350  0.355   0.350 | [8, 0, 5]
3.12  -0.105  |   380      0      412 | 0.350  0.350   0.350 | [8, 0, 4]
3.15  -1.302  |    30   1375        0 | 0.349  0.350   0.353 | [3, 8, 4]   <- right lands, rear lifts
3.18  -1.999  |   298    595        0 | 0.350  0.350   0.352 | [8, 5, 0]
3.21  -0.133  |   701      0      613 | 0.350  0.356   0.350 | [7, 7, 8]
```

Yaw is welded near −0.1 rad/s while the rear wheel is down, and releases to
−2.0 (near the kinematic 2.88) the moment it is not. Wheel lift is 4-46 mm and
chassis roll reaches ±3° — real motion, not solver noise.

## The severity scales with tangential demand, not with turning

Mesh harness, chassis roll peak-to-peak and wheel vertical excursion:

| command | achieved | roll p-p | wheel z p-p |
|---|---|---|---|
| spin 0.5 rad/s | −0.033 (7 % kin) | 0.00° | 0.0 mm |
| spin 1.0 | −0.061 (6 %) | 0.00° | 0.1 mm |
| spin 2.0 | −0.157 (9 %) | 0.02° | 0.5 mm |
| spin 3.0 | −0.645 (22 %) | 4.82° | 44 mm |
| spin 6.0 | −1.904 (33 %) | 15.65° | 172 mm |
| **forward 10 rad/s** | 3.43 m/s | **178.5°** | **4.6 m** |

Driving *straight* at 3.5 m/s on flat ground throws the robot 4.6 m into the air
and flips it. Turning is not special — it is just the cheapest way for this
robot to demand a large tangential force, because rotating requires slip at
every wheel and driving requires none.

## A note on the metric

`jerk_rms` carries no information beyond `rate_std` and the sampling interval.
For rate noise that decorrelates between samples, `jerk_rms ≈ rate_std·√2/Δt`:

* harness A: 0.392·√2/0.04 = 13.9 vs 11.4 reported
* harness B: 0.75·√2/0.03 = 35 vs 24-38 reported

Two consequences. First, the brief's item 3 is correct and the two harnesses'
jerk numbers were never commensurable. Second — and this is what hid defect 3 —
`jerk_rms` **inverts the ordering** on ground representation, because it also
rewards a slower turn:

| | jerk_rms | rate_std/\|mean\| | roll p-p |
|---|---|---|---|
| plane | 51-59 | 0.50 | 0.34° |
| mesh | 36-38 | 1.02 | 6.33° |

The mesh has *lower* `jerk_rms` and is unambiguously worse. That is why "not the
ground representation, <15 % in jerk_rms" was recorded as ruled out when it is
in fact one of the three causes.

Also worth knowing: harness A samples every 2 steps, which aliases away half the
variance. Its clean configuration reads `rate_std` 0.035 at `chunk=2` and 0.173
sampled every step.

Use `rate_std / |yaw_rate|`, or band-limited yaw-rate power on a fixed grid, or
— best of all, because it is the actual physics — **cone occupancy**.

## What holds in both harnesses

Predictions, all verified here on both grounds:

1. Lateral `mu` has no effect on a saturated turn, on either ground, at any
   compliance. (Because `mu` is not in the row.) Confirmed: 0.035 vs 0.030.
2. Contact compliance is worth ~10x **only** in the clean configuration
   (analytic plane, isotropic wheels). Add either anisotropy or the mesh and the
   effect vanishes. Confirmed on both.
3. Anisotropic wheels cost ~13x on the plane, and nothing on the mesh (already
   saturated by the reducer). Confirmed.
4. `fps` or a large `cluster` K removes 20x of the chassis rocking on the mesh
   and none on the plane. Confirmed.
5. The fix to defect 1 raises achieved yaw rate 1.5-6x and cuts total tangential
   force from ~3x to ~1.2x robot weight, on both grounds. Confirmed.

## Recommended order of work

1. **Defect 1** (cone) — this is the physics bug and it is present in every
   scene the engine simulates, including the paper's. Patch is written and
   validated against analytic Coulomb; the question is what it does to the
   published results, which needs a rerun, not a decision here.
2. **Defect 2** (anisotropic branch) — reformulate the elliptical branch so it
   reduces continuously to the isotropic one. The `mu_x == mu_y` hard branch
   currently masks a real discrepancy. A regression test on the
   `mu_y = mu_x*(1+1e-7)` case would have caught this.
3. **Defect 3** (cluster reducer) — either switch the default to `fps`, or make
   the cluster representative selection deterministic under ties instead of
   "deepest", which is noise on a flat surface. Cheap and low-risk.
4. Then retire `jerk_rms` from `turn_jitter.py` and report `rate_std/|rate|`
   plus cone occupancy.

## Reproducing

Scratch harnesses (headless, `RenderingConfig(vis_type="null")`), in
`/tmp/claude-1000/-home-kuceral4-projects-ostrich/387b76fc-a730-442f-80c9-07ef3f7701f9/scratchpad/`:

| file | what it does |
|---|---|
| `harness.py` | instrumented `TurnJitterSimulator`, scriptable command sequence |
| `toy.py` / `toy2.py` | one-contact friction row vs analytic Coulomb; no engine needed |
| `repro_A.py` | reproduces harness A exactly (plane, isotropic, A's metric) |
| `final.py` | the three-way disentangling table |
| `straight.py` | cone occupancy, straight vs spin, correct engine-side indexing |
| `sweep3.py` | contact-reduction policy sweep |
| `sweep4.py` | solver-hardness sweep |
| `trace.py` | per-step stick-slip trace |
| `cone_fix.patch` | the defect-1 patch, env-gated, not applied |

Read cone occupancy and per-wheel loads from `solver.data.constr_force.n/.f`
indexed against `solver.ostrich_contacts.contact_shape0/1` — **not** against
`sim.contacts`, which is the pre-reduction Newton contact list and does not
share indices with the force vector.

---

## Resolution (added 2026-08-21, same day)

All three defects were addressed; the engine is Macklin et al. 2019
(arXiv:1907.04587) and the cone defect is a one-line deviation from its Eq. 32
(the clamped FB slack makes phi_FB(d, 0) = 0 identically, amputating the
paper's Fig. 6 beyond-cone penalization).

Shipped (branch `relaxation`):
1. **Cone fix** — `clamped_imp_norm` in the w denominator: evaluates the
   paper's Eq. 35 W at its own fixed point, so |f_t| = mu*f_n at sliding.
2. **`nr.w_relaxation` = 0.5** — damped Picard on w across NR iterations;
   restores convergence the corrected cone costs (residual 0.28 -> 0.11 at
   nr 16, cone p90 -> 1.6, jitter halved again). Default 1.0 is bit-exact.
3. **fps K=8 reducer** as the wheeled-robot default (roll 4-7 deg -> ~1 deg).
4. **Regression test** `tests/test_friction_cone_bound.py` (fails on the old
   formulation at 2.9x the cone).
5. **Viewer clock fix** — CUDA-graph replays froze `clock.time`, so scripted
   demos never left their first phase in the GL viewer.

End state on this harness (mesh spin, dt 0.03, aniso wheels): yaw −2.15 rad/s
(75 % of kinematic, was 23 %), rate_std/|mean| 0.10 (was ~1.0), roll 0.14°
(was 4–7°), cone occupancy p90 ≈ 1.2 (was 13–19). The mu_x==mu_y branch
discontinuity (defect 2) no longer measurably bites once the solve converges
(eps-anisotropy within noise); anisotropic wheels are now the smoothest
configuration. Wheels stay cylinders: a capsule doubles mesh turn rate again
(−2.15 vs −1.1, jitter 0.11 vs 0.35) but adds ~0.8 m phantom lateral width —
per-scene choice, documented in common.py. Rejected along the way:
post-hoc cone projection, lambda_n step-lag (superseded by w-relaxation),
and the scalar C-fold (docs/friction_cfold_negative_result.md).
