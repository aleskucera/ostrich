# Experiment 3 (box, v2) — random-IC final-pose

Variant of `experiments/3_gradient_quality_box`. Same scene
(helhest_junior + static box obstacle, calibrated physics from
`1_sim_to_real_box`), same K=10 wheel-velocity spline, but a different
optimisation task:

- **No GT trajectory**. The optimiser sees only a *final* target pose.
- **Random initial conditions per trial** (chassis xy + yaw).
- **Random target per trial** (target xy + yaw, past the box).
- **N independent trials per engine** for robust statistics.
- **Real-robot deployable**: the resulting wheel-velocity spline can be
  replayed open-loop on the real helhest_junior — the loss encourages
  smooth, low-magnitude controls that reach the target at rest.

## Loss

```
L = w_pos    · ‖xy(T) − xy_target‖²              # final position
  + w_yaw    · (1 − cos(ψ(T) − ψ_target))        # final heading (wrap-safe)
  + w_vel    · (1/τ) Σ_{t∈tail} ‖v_xy(t)‖²       # zero terminal velocity
  + w_smooth · (1/T) Σ_t ‖u[t+1] − u[t]‖²        # control smoothness
  + w_reg    · (1/T) Σ_t ‖u[t]‖²                 # control magnitude
```

Default weights:

| term       | weight | rationale |
|---|---|---|
| `w_pos`    | 1.0    | anchor (≈ same scale as expected pos² ≈ 0.04 m²) |
| `w_yaw`    | 0.5    | yaw matters but pos dominates |
| `w_vel`    | 0.3    | enforce stopping; soft penalty so it doesn't fight pos |
| `w_smooth` | 1e-3   | real-robot motor friendliness |
| `w_reg`    | 1e-5   | discourage huge wheel speeds without dominating |

Velocity is computed via forward finite differences over the last
``terminal_tail_frac = 0.10`` of the horizon. Single-step terminal-vel
penalty has very sparse gradients; spreading over the tail encourages
deceleration over the last ~0.6 s rather than a sudden braking step.

## Randomisation

```
IC:     xy ∈ [−0.3, +0.3] m,    yaw ∈ [−15°, +15°]
target: xy = (3.0, 0.0) ± 0.3 m, yaw ∈ [−15°, +15°]
```

The box sits at x ≈ 1.37 m; the target at x = 3.0 m is past the box,
forcing the robot to cross it. Each trial:

1. Draw (IC, target) from a deterministic RNG stream seeded with
   `--seed-base` (so all engines see the *same task set* — fair
   comparison).
2. Draw random spline init from a separate RNG stream offset by 1000
   (so spline-init noise is independent of task noise).
3. Optimise for `--iterations` steps (default 100).

## Trial-level metrics

```json
{
  "xy_final": [3.05, -0.12],     "yaw_final": 0.08,
  "pos_error_m": 0.13,           "yaw_error_rad": 0.05,
  "terminal_speed_mps": 0.18,
  "control_jerk": 8.4,           // Σ_t ‖Δu‖
  "success": true                // pos<0.2m AND terminal_speed<0.3m/s
}
```

## Aggregate metrics (the "robust" headline)

- **Success rate** across N trials
- **Median final-pose error** (and IQR)
- **Median terminal speed**
- **Control jerk distribution**

These show up in the figure as box plots per engine.

## Engines

| script | engine | gradient mechanism | key knobs |
|---|---|---|---|
| `optimize_ostrich.py` | Ostrich | implicit adjoint | dt=0.10, lr=0.1 |
| `optimize_mjx.py`   | MJX   | JAX BPTT + capsule wheels | dt=5e-3, lr=0.05, clip=1.0, β1=0.9 |
| `optimize_semi_implicit.py` | Newton SemiImplicit | Warp tape BPTT | dt=5e-4, lr=0.02, clip=1.0 |

Per-engine defaults inherited from the calibrated `1_sim_to_real_box`
physics + `3_gradient_quality_box` optimizer tuning. No re-tuning needed.

## Wall-time per engine (3090, default settings)

| engine | per trial | 25 trials | bottleneck |
|---|---|---|---|
| Ostrich | ~70 s    | ~30 min | warm iter @ ~0.5 s, 100 iters |
| MJX   | ~30 min  | ~12 hr  | 200 s/iter (BPTT through box) × 100 |
| SI    | ~60 min  | ~25 hr  | ~20 min cold capture + 100 warm @ ~4 s |

For a quick smoke check use `--iterations 30 --num-trials 5` first.

## Reproduce

```bash
# full run (~38 hr if all three engines; mostly MJX + SI)
bash experiments/3_gradient_quality_box2/run_experiment.sh

# one engine only
bash experiments/3_gradient_quality_box2/run_experiment.sh --ostrich
bash experiments/3_gradient_quality_box2/run_experiment.sh --mjx
bash experiments/3_gradient_quality_box2/run_experiment.sh --semi-implicit

# quick screening
bash experiments/3_gradient_quality_box2/run_experiment.sh --iterations 30 --num-trials 5

# regenerate figure
python experiments/3_gradient_quality_box2/plot_results.py
```

`run_experiment.sh` skips already-existing JSON files in `results/` so
it's safely resumable.

## Real-robot test

The intended downstream use: deploy one of the trial's optimised splines
**open-loop** on the real helhest_junior (no tracking controller). The
loss formulation guarantees:

1. The robot reaches a known target — no need for a recorded reference
   to imitate.
2. It ends at rest — no overshooting past the target.
3. Controls are smooth — real motors track the command profile cleanly.

The per-trial JSON saves enough info (IC, target, spline, full sim
trajectory) to deploy any trial directly. The comparison across engines
shows which one's *simulated* gradient produces the most reliable
*real-world* outcome.

## See also

- `experiments/3_gradient_quality_box/RESULTS.md` — same scene, GT-tracking
  variant. The two are complementary: trajectory-match (exp 3 box)
  measures gradient fidelity vs a known answer; final-pose (exp 3 box 2)
  measures gradient utility for a deployable task.
- `experiments/1_sim_to_real_box/` — source of the calibrated physics
  parameters each engine uses here.
