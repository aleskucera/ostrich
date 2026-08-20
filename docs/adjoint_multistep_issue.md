# Adjoint Gradient Accuracy: Multi-Step Trajectories

## Summary

The control-parameter gradient through a **multi-step** trajectory on the
Helhest model comes back essentially **zero**, where finite differences say it
is order 1. Single-step gradients on the same model are accurate, so this is
specific to accumulating a parameter gradient across steps, not to the
per-step adjoint.

Reproduce with `tests/differentiable_simulator/test_helhest_gradient.py`,
which currently fails on `test_multi_step` for this reason.

## Measured

Helhest, 5 steps at `dt = 0.01`, loss seeded on final body velocity,
d(loss)/d(`joint_target_vel`) per wheel:

| wheel | analytic | finite difference | rel err |
|---|---|---|---|
| left  | -0.0001 | +0.3755 | 1.0004 |
| right | -0.0004 | -1.2167 | 0.9997 |
| rear  | -0.0004 | -0.9363 | 0.9996 |

A relative error of ~1.0 across every wheel is the signature of the analytic
value being zero rather than merely inaccurate.

The same file's single-step cases are healthy on the same model and engine
config, which is what makes this a multi-step-specific problem:

| case | max rel err |
|---|---|
| straight drive (1 step)     | 0.032 |
| differential turn (1 step)  | 0.003 |
| **multi-step (5 steps)**    | **1.000** |

## Ruled out

**Finite-difference step size.** `test_helhest_gradient` previously used
`eps = 1e-4`, which sits in the roundoff-dominated regime for this solve
(forward tolerance 1e-8, so FD noise ~ tol/eps) and made the *FD reference*
the inaccurate side on the single-step cases. Sweeping 1e-6 .. 1e-2 showed
agreement improving monotonically with larger eps, and the analytic value never
moved. `eps` is now `1e-3`, which fixed the two single-step cases — and left
multi-step unchanged at ~1.0. So the multi-step failure is not an FD artifact.

**`zero_gradients()` placement in the backward loop.** The test calls
`engine.data.zero_gradients()` inside the reverse loop, and that method does
zero `joint_target_vel.grad`, so it looked like each step was wiping the
previous step's accumulated parameter gradient and leaving only step 0's
contribution. Hoisting the call out of the loop changes nothing — the result
is still ~1e-4. So the accumulation is not being clobbered by the caller.

Note that `TrajectoryBuffer.load_step` *does* restore the incoming adjoint
(`data.body_vel_grad <- self.body_vel.grad[step_idx + 1]`), so the state
adjoint chain is at least wired.

## Not yet investigated

Whether `step_backward` accumulates into `joint_target_vel.grad` or overwrites
it, and whether `save_gradients` propagates the pose/velocity adjoint correctly
across the step boundary for this model. Both are cheap to check by dumping
`joint_target_vel.grad` after each reverse step and seeing whether it grows,
holds, or collapses.

## Why it matters

Multi-step BPTT is what every trajectory-optimisation experiment in this repo
relies on, so a vanishing parameter gradient here is more consequential than
the single-step accuracy questions in
[adjoint_warm_start_issue.md](adjoint_warm_start_issue.md). Prior bugs in this
same area (tape reset scaling gradients by call count; a shared contacts buffer
corrupting the semi-implicit baseline's adjoint) suggest looking at buffer
aliasing and accumulation semantics first.
