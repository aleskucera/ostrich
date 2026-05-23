# Experiment 2 (box) — accuracy vs timestep

How does each engine's trajectory error change with the simulation timestep
`dt`, and where does it stop being usable? Drives the same `helhest_junior +
box` scene as `experiments/1_sim_to_real_box` with the real recorded wheel
setpoints, sweeps `dt` for each engine at its yaw-tuned best params, and
overlays the resulting accuracy-vs-dt curves.

The headline result: **Axion stays usable at ~10× larger `dt` than MuJoCo for
the same trajectory accuracy on this scene.**

## Pipeline (fully reproducible from data)

```
run_accuracy_vs_dt.py   --->   results/accuracy_vs_dt.json   --->   plot_dt_vs_error.py   --->   results/dt_vs_error.png
```

No hand-copied numbers anywhere — `plot_dt_vs_error.py` loads the JSON; the
"~10× larger usable Δt" annotation is computed from the data, not typed in.

```bash
# regenerate everything from scratch (~5–6 min)
python experiments/2_dt_stability_box/run_accuracy_vs_dt.py
python experiments/2_dt_stability_box/plot_dt_vs_error.py
xdg-open  experiments/2_dt_stability_box/results/dt_vs_error.png
```

## What gets measured

For each engine, for each `dt` in its grid, the script runs the box scene on
2 GT runs (`18_04_51` + `18_10_33`, the clean ones used in
`1_sim_to_real_box`) and records per-`(engine, dt, gt)`:

- `combined_with_yaw` — position L2 + `L · yaw RMSE` (the same yaw-aware
  metric the 1_sim_to_real_box sweeps optimise, `L = 0.5 m`),
- a **stability flag**: no NaN AND chassis `z ∈ [0.05, 2.0] m` AND robot
  drove ≥ 0.5 m past the far edge of the box.

The plot then categorises each point as:

| marker | meaning |
|---|---|
| filled circle | **usable** — stable AND error ≤ accuracy threshold (0.5 m) |
| open circle | stable but error > threshold (runs, not usable) |
| `X` | broken — NaN, diverged, or didn't pass the box |

## Engine configs swept (from 1_sim_to_real_box)

```python
Axion:  mu_front=0.8, mu_rear=1.5, mu_rolling=0.7, compliance.contact=1e-7
MuJoCo: μ=1.5, tor=2.0, kv=1000, solref0=0.005, condim=6, integrator=implicitfast
```

These are exactly the per-engine tuned best params from the
`1_sim_to_real_box` yaw-aware re-tune; this experiment only varies `dt`.

## Reading the figure

- **MuJoCo (pink)** — flat plateau of ~0.055 m from `dt = 10⁻⁴` to `10⁻²`
  (5 orders of magnitude!), then a sharp wall: 0.118 m at `dt = 0.02`,
  0.24 m at 0.03, 1.17 m at 0.05 (above threshold), NaN at 0.1.
- **Axion (blue)** — flat plateau of ~0.065–0.10 m across `dt = 5×10⁻³` to
  `3×10⁻¹`, no wall anywhere in this range.
- The shaded arrow shows `(Axion's max usable dt) / (MuJoCo's max usable dt)`
  — read directly from the data, currently ~10×.

## Caveats

- "Usable" is defined by the 0.5 m threshold and the stability criterion
  above; both are encoded in the JSON for repeatability.
- The two GT runs (`18_04_51`, `18_10_33`) are the clean, low-drift ones from
  the real dataset — runs with more lateral slip would not change the
  per-engine `dt` ranking, only the absolute error level.
- This is the box-scene counterpart of `experiments/2_dt_stability/`; that
  one uses a synthetic scripted drive with randomized obstacle parameters.
  Here the drive is real recorded setpoints, so the result maps directly to
  what you'd get running `examples/helhest_junior/replay_real.py` at each
  `dt`.
