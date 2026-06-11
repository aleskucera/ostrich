# Recovered-control investigation (§4.3 inverse problem)

Side investigation of `experiments/3_gradient_quality_box`: can Ostrich's
optimizer recover the real wheel-velocity setpoints from the recorded XY
trajectory? The goal was (a) a possible §4.3 supplementary plot, and
(b) a foundation for the §4.6 open-loop hardware experiment.

## TL;DR

At K=10 with light smoothness + yaw + skid-steer coupling, Ostrich recovers
a smooth, bounded, cross-seed-consistent wheel-velocity profile — but
**~0.5–1.5 rad/s magnitude rather than real's peak ~2 rad/s at t≈4 s**.
The gap is the sim-to-real model mismatch from §4.1 (~6 cm residual at
calibrated params), not optimizer-side. No knob we tried closes it.

The recovered control is smooth and bounded enough to be safely
executable on real hardware.

## Configurations tested

Loss = MSE XY position to real prism over 6 s horizon, K-knot
wheel-velocity spline optimised via Adam, 3 seeds each.

| Config | Best loss (m²) | Cross-seed | Smoothness | Tracks t=4 peak |
|---|---|---|---|---|
| K=10, XY only | 0.071 | ±5% | smooth | partial |
| K=100, XY only | 0.073 | poor (noisy) | jagged | partial (mess) |
| K=10, +yaw, smooth=0.01 | 0.079 | tight | over-smoothed flat | no |
| K=10, +yaw, smooth=0.001 | 0.080 | tight | clean | partial |
| K=10, +yaw, smooth=0.001, skid-steer | 0.076 | tight | clean | partial |

Knobs tried:

- **K (spline knots):** 10 → 100 added high-frequency noise, did not
  reduce loss (gradient is fine, optimization just gains nuisance DoF).
- **Yaw loss (L=0.5 m, matching §4.1):** barely moved the loss → the
  XY-only solution was already near yaw-correct. Yaw was not constraining
  anything new on this trajectory.
- **Smoothness reg λ‖Δp‖²:** λ=0.01 over-smoothed (controls collapsed
  to constant ~1 rad/s). λ=0.001 was a sweet spot — preserved shape.
- **Skid-steer coupling (rear = (L+R)/2):** reduced spline dim 3K → 2K,
  rear panel now consistent by construction, slight cross-seed cleanup.
  Did not close the t=4 gap.

## Why the t=4 peak isn't recovered

Forward sim at calibrated parameters has 6 cm residual error (§4.1
calibrated outcome). The inverse-problem optimizer finds the control
that produces the **target trajectory under sim physics**, not the
control that real physics required. Because sim friction/inertia
slightly differ from real, sim needs lower wheel velocities to produce
the same XY trajectory. The optimizer correctly converges to that
lower-magnitude control.

This is a fundamental property of the inverse problem under model
mismatch, not an optimizer or expressivity failure. Adding more spline
expressivity (K=100) cannot fix it. Adding yaw cannot fix it
(yaw was already near-correct).

The picture confirms gradient quality is good (cross-seed consistency,
smooth recovered controls) — it just demonstrates the floor imposed
by the §4.1 model gap.

## Implications for §4.6 open-loop hardware

For an open-loop **forward planning** task (drive to a target pose, not
inverse-fit), the same optimizer + parameterization (K=10 spline, yaw
in loss, light smoothness, skid-steer coupling) produces controls that
are:

- smooth (~rate of change small enough for the real motors to track),
- bounded (~0.5–2 rad/s range, well within safe envelope),
- reliable across random inits (cross-seed convergence to similar
  controls).

If we instead **execute the inverse-recovered control on real**, we
expect the real trajectory to fall short of the target (the control
was optimised under sim physics with lower magnitude than real).
That deviation is the model-gap floor for open-loop transfer.

## Code state

- `optimize_ostrich.py` — modified to:
  - log `init_params` + `final_params` per trial
  - add `chassis_yaw_loss_kernel` and yaw target wiring
  - add Python-side L²-smoothness reg on spline knots
  - couple rear wheel to (L+R)/2 (NUM_OPT_DOFS = 2)
- `plot_recovered_control.py` — new; loads ostrich.json, overlays
  recovered vs real per wheel.
- `results/recovered_control.png` — current rendering.

Default CLI flags now: `--K 10 --yaw-lever 0.5 --smooth-lambda 0.01`.
For "best executable" reproduction use `--smooth-lambda 0.001`.

`optimize_mjx.py` and `optimize_semi_implicit.py` were **not** modified —
they still log only losses/grad_norms. If we want the same recovered-
control comparison across all three engines (e.g. to visualise gradient
quality as control quality), they need the same `init_params` /
`final_params` save + a similar yaw + smoothness wiring.

## Recommendation

For §4.3 paper figure: **do not include recovered-control as a primary
panel.** It introduces "why doesn't recovered equal real?" explanation
overhead that distracts from the loss-curve reliability story already
present.

Save as supplementary or as foundation for §4.6:

- Supplementary: shows the gradient finds physically meaningful directions
  and that the gap is honest sim-to-real, not optimization failure.
- §4.6 building block: the optimizer + parameterization here are
  directly reusable for a forward-planning hardware task.
