# test_stribeck_friction: why the 50-step settle is load-bearing

## Summary

`tests/differentiable_simulator/test_stribeck_friction.py` is the slowest test
in the gradient suite (113 s of 207 s), and almost all of that is settling the
box: `measure_deceleration` is called 5 times, each settling for 50 steps.

The settle looks wasteful and mostly is -- but it cannot be cut much, and the
obvious optimisation (spawn the box at its resting height instead of dropping
it) actively breaks the test. Both were measured; this note records the numbers
so the next person does not repeat the attempt.

## The settle is genuinely needed

`ratio slow/fast`, asserted `> 1.5`, current settings are spawn z=0.6 with 50
settle steps:

| settle_steps | spawn z=0.6 (current) | spawn z=0.5 (at rest) |
|---|---|---|
| 10 | -- | 1.9556 |
| 15 | -- | 1.9504 |
| 20 | -- | **1.3772** (fails) |
| 25 | **0.7176** (fails) | 2.0619 |
| 30 | 1.9565 | 2.0619 |
| 40 | 2.0619 | **1.3772** (fails) |
| 50 | 2.0620 | -- |

At the current spawn height the sequence converges: 0.72 -> 1.96 -> 2.06 ->
2.06, settled by 40. So 50 is a reasonable choice with a little margin, not an
arbitrary large number. 40 reproduces the current figures exactly (2.0619 vs
2.0620, identical lateral/longitudinal), which is a ~20% saving on this test if
you want it, but it sits right at the convergence knee -- 30 is already wrong.

## Spawning at rest makes it unstable

The box has half-height 0.5 and spawns at 0.6, so it free-falls before
settling: of the 50 steps roughly 13 are fall, ~6 are the bounce, and the last
30 are idle (z pinned at 0.499677, vz already down at 1e-5). Removing the fall
by spawning at z=0.5 looks like free money.

It is not. At z=0.5 the result stops converging and starts alternating --
passing at 15, 25 and 30, failing at 20 and 40. Starting in marginal contact
(0.0003 m above rest) evidently gives a different and less repeatable initial
contact resolution than arriving from a clean fall.

## Underlying sensitivity

`measure_deceleration` integrates **one** step at `measure_dt = 0.005` and
reports `(v_start - v_end) / (measure_steps * measure_dt)`. A single step means
one stick/slip decision by the solver sets the whole measurement, which is why
the reading is so sensitive to the contact state entering it.

## What would actually make it fast

Average the deceleration over enough steps that the result is a slope rather
than one step's stick/slip outcome. Once the measurement is robust, check
whether it is still sensitive to `settle_steps`; if not, the settle can be cut
and the box spawned at rest, which would take this test from 113 s to roughly
20 s. Until then the 50-step settle is load-bearing and should stay.
