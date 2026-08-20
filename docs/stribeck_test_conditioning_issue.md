# test_stribeck_friction is ill-conditioned

## Summary

`tests/differentiable_simulator/test_stribeck_friction.py` measures the
slow-slip / fast-slip deceleration ratio over a **single** implicit-Euler step.
The result is chaotic with respect to how long the box settles beforehand: the
same quantity lands on 0.72, 1.38, 1.95 or 2.06 depending on `settle_steps`,
with no trend. It passes at the current settings, but that is not evidence the
measurement is sound.

This blocks the obvious runtime fix. The test is the slowest in the suite
(113 s of 207 s), and the settle is where the time goes, but the settle count
cannot be reduced without changing the number being asserted on.

## Measured

`ratio slow/fast`, asserted `> 1.5`. Spawn height 0.5 is the box's resting
height; 0.6 is the current default, from which it free-falls first.

| settle_steps | spawn z=0.5 | spawn z=0.6 |
|---|---|---|
| 10 | 1.9556 | — |
| 15 | 1.9504 | — |
| 20 | **1.3772** (fails) | — |
| 25 | 2.0619 | **0.7176** (fails) |
| 30 | 2.0619 | 1.9565 |
| 40 | **1.3772** (fails) | — |
| 50 | — | 2.0620 (current) |

Increasing the settle makes it pass, then fail, then pass again. The
`stribeck_lateral_only` sub-test moves the same way -- its longitudinal ratio
reads 2.1165 at the current settings and 0.3940 at `settle_steps=15`.

Two settings reproduce the current numbers *exactly* (z=0.5 with 25 or 30
steps: 2.0619 vs 2.0620, and identical lateral/longitudinal figures), which is
tempting as a ~2x speedup. It should not be taken: 20 and 40 fail at the same
spawn height, so those two values are a knife's edge rather than a converged
regime.

## Cause

`measure_deceleration` kicks the box to `vx0` and integrates **one** step at
`measure_dt = 0.005`, then reports `(v_start - v_end) / (measure_steps *
measure_dt)`. A single step means one stick/slip decision by the solver decides
the entire measurement, so any difference in the contact state entering that
step -- which is what the settle determines -- can move the answer between
regimes rather than perturb it.

Note the box also free-falls before settling: it has half-height 0.5 but spawns
at 0.6, so of the 50 settle steps roughly 13 are fall, ~6 are the bounce, and
the last 30 are idle (z pinned at 0.499677, vz already down at 1e-5). The waste
is real; it just cannot be removed while the measurement stays this sensitive.

## What would fix it

Average the deceleration over enough steps that the result is a slope rather
than one step's stick/slip outcome, and check that the reading is stable
against `settle_steps` before trusting it. Once the measurement is conditioned,
the settle can be cut to ~10 steps and the box spawned at rest, which should
take this test from 113 s to roughly 20 s.
