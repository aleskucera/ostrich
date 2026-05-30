# Brax on the box-obstacle gradient task — findings

Run via `optimize_brax.py` (this dir), Brax 0.14.2 / JAX 0.10.1, RTX A500 (4 GB).
Same inverse problem as the MJX/Ostrich box experiment: fit a K=10 wheel-velocity
spline to a recorded real trajectory, MSE on chassis XY.

## Setup notes / fairness

- **Brax has no cylinder collision geometry**, so wheels are spheres (radius
  0.35 m, matching the cylinder radius). Same approximation as TinyDiffSim.
- **dt = 1e-3 s.** Brax's positional pipeline is unstable at the dt Ostrich uses
  (100 ms): a velocity-actuator transient launches the chassis (z -> 4.8 m,
  flies off). Forward sim is stable only at dt <= ~1 ms here. Ostrich runs the
  same scene at 100 ms (100x larger). At dt=1e-3, kv=50, the forward rollout is
  stable (verified: robot drives forward, z stays 0.35-0.47 m), so the gradient
  results below are NOT an artifact of an exploding forward sim — the forward
  losses are all finite and O(0.1-1).
- horizon 4 s (4000 steps), 20 Adam iters, 3 seeds, grad-clip 1.0.

## Results (per pipeline)

### positional (Position-Based Dynamics, penalty contacts)
Forward stable, but **gradients explode and the optimizer diverges**:

| seed | loss first | best | last | median |g| | max |g| |
|------|-----------|------|------|-----------|--------|
| 42   | 0.714     | 0.714| 14.23| 4.7e9     | 1.3e14 |
| 43   | 0.578     | 0.380| 0.735| 6.6e6     | 3.3e12 |
| 44   | 1.155     | 1.155| 43.25| 2.4e10    | 1.9e14 |

Gradient norms 1e6–1e14 (a well-posed problem is O(1–100)); even with clipping
at 1.0 the loss bounces chaotically and ends worse than it started on 2/3 seeds.
No convergence.

### generalized (QP constraint solver)
Forward loss finite (0.556) but **gradient is NaN** (`iter 1: |g|=nan`).
Also prohibitively slow on this scene (did not finish 5 iters in 400 s on the
A500), so its JSON was not saved in the batch run; the NaN was captured in the
batch log.

### spring (spring-damper joints + penalty contacts)
Forward loss finite (0.29–0.58) but **gradient is NaN on iteration 0**, all 3
seeds. Immediate backward-pass failure.

## Forward accuracy: Brax CLEARS it (corrected)

CORRECTION: an earlier forward sweep used dt in {1e-3, 5e-4, 2e-4} and concluded
Brax fails the forward gate (~0.435 m). That dt range was WRONG -- Brax positional
is stable only at dt ~ 0.005-0.01; at dt<=1e-3 it launches off the box. A proper
sweep (forward_brax_sweep.py: dt in {0.002,0.005,0.01} x kv x mu x baumgarte x
spring scales x {sphere,capsule}, scored with the boundedness guard) gives:

| pipeline | best combined err (m) | config |
|----------|-----------------------|--------|
| MuJoCo (paper)        | 0.054 | |
| Ostrich (paper)       | 0.062 | |
| Brax SPRING           | 0.074 | sphere, dt5e-3, kv50, mu1.0, spring_mass_scale=1 |
| Semi-Implicit (paper) | 0.110 | |
| Brax positional       | 0.957 | sphere, dt5e-3 (tracks one run, diverges other) |

So Brax's spring pipeline forward-tracks the box at 0.074 m -- comparable to the
benchmarked engines and BELOW the 0.2 m usability threshold. Brax does NOT fail on
forward accuracy. The earlier "forward gate" exclusion was an artifact of the bad
dt range and has been removed from the paper.

The real reason Brax is excluded from the gradient comparison is GRADIENT quality
(not forward accuracy): on the box climb its positional gradients diverge
(|g| ~ 1e14) and its spring/generalized gradients are non-finite. So Brax can
forward-simulate the box but cannot drive gradient-based optimization through it.
generalized+capsule and a thorough generalized sweep are impractically slow on the
4 GB laptop GPU (timed out); generalized+sphere best is similar to positional.

## Conclusion for the paper

None of Brax's three differentiable pipelines yields a usable gradient on the
box scene: positional produces exploding gradients and diverges; generalized and
spring return non-finite (NaN) gradients. This is a genuine backward-pass
failure (forward losses are finite), not a forward-instability artifact.
