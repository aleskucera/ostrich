# Gradient difficulty-ramp: baseline gradient quality vs horizon

Goal: show, fairly, *why* each baseline is excluded from the box gradient
experiment. We run the same inverse problem (K-knot wheel-velocity spline fit
to a recorded real trajectory, MSE on chassis XY) at increasing horizons:

- **short = 1 s, K=3** — robot rolls ~0.7 m on flat ground; the box front face
  is at x~1.0 m so this is pre / onset of contact ("easy" regime).
- **(mid = 3 s, K=6)** — first climb.
- **full = 6 s, K=10** — full box climb ("hard" regime).

Each engine runs at its own native dt (Ostrich 100 ms, MJX 5 ms, Semi-Implicit
/ XPBD 0.5 ms, Brax 1 ms). 3 seeds, Adam, grad-clip 1.0.

## Hardware caveat
All numbers below are from a 4 GB RTX A500 laptop. MJX is not installed here;
Brax generalized and the full clean matrix should run on the 24 GB machine
(dasenka). Brax generalized capsule OOMs at 4 GB; generalized is also very slow
(~150 s/iter at 1 s).

## Results (local, this laptop)

| engine | short (1 s) | full (6 s) | failure mode |
|--------|-------------|------------|--------------|
| Ostrich | converges, loss 0.12 -> 0.016 | converges, 0.074 | works throughout |
| MJX | (dasenka) | converges (slow), 0.27-0.35 | works throughout |
| Semi-Implicit | descends 0.019 -> 0.002 | no basin, 0.7-1.3 | contact collapse |
| Brax-positional | converges 0.038 -> 0.0006 | gradients explode (|g|~1e14), diverges | contact collapse |
| Brax-generalized | converges (slow) 0.039 -> 0.023 | NaN gradients | contact collapse |
| Brax-spring | NaN gradient (iter 0) | NaN gradient | broken always |
| XPBD | dead gradient (~0), loss drifts 0.0128 -> 0.0134 | (dasenka) | uninformative always |

## Three failure modes (the story)

1. **Contact-specific collapse** (Semi-Implicit, Brax-positional, Brax-generalized):
   provably descend on the easy pre-contact problem, then break the moment the
   stiff box climb enters the horizon. The cleanest "we did not rig the baselines"
   evidence: same engine, same code, same robot; only the contact event differs.
   - Brax-positional is the sharpest case: loss -> 0.0006 at 1 s, |g| -> 1e14 and
     divergence at 6 s. Confirmed on flat-ground control too (descends cleanly).
2. **Always-broken gradients** (Brax-spring NaN, XPBD dead/zero): fail even on the
   trivial pre-contact horizon.
3. **Survivors** (Ostrich, MJX): converge at every horizon.

## Brax wheel-geometry note (fairness)
Brax has no cylinder geometry. Capsule wheels are stable forward in spring and
generalized but UNSTABLE in positional (chassis launches to z~1.6 m at rest), so
positional uses spheres. The gradient failures are robust to wheel model: spring
NaNs with both sphere and capsule. So spheres for positional are not a handicap;
they are the only stable option there.

## Dojo note
Dojo has a differentiable SphereBoxCollision, but only as an experimental,
manually-wired primitive (examples/simulation/object_collision_development/);
no high-level API exposes it. Dojo is CPU-only and dormant since April 2023, so
it is out of scope for the batched GPU comparison regardless. The earlier
"planar surfaces only" claim was too strong and has been corrected in the paper.

## Files
- optimize_axion.py / optimize_mjx.py / optimize_semi_implicit.py / optimize_xpbd.py
  / optimize_brax.py (--pipeline positional|spring|generalized, --wheel sphere|capsule)
- run_ramp.sh — full matrix for dasenka
- plot_ramp.py — converged-loss-vs-horizon figure
- ramp_*_{short,1s}.json — local validation results
- BRAX_FINDINGS.md — detailed Brax investigation
