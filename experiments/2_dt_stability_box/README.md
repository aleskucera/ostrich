# Experiment 2 (box) — accuracy vs timestep

How does each engine's trajectory error change with the simulation timestep
`dt`, and where does it stop being usable? Drives the same `helhest_junior +
box` scene as `experiments/1_sim_to_real_box` with the real recorded wheel
setpoints, sweeps `dt` for each engine at its yaw-tuned best params, and
overlays the resulting accuracy-vs-dt curves.

The headline result: **Axion stays usable at ~20× larger `dt` than MuJoCo for
the same trajectory accuracy on this scene.**

## Pipeline (fully reproducible from data)

```
run_accuracy_vs_dt.py   --->   results/accuracy_vs_dt.json   --->   plot_dt_vs_error.py   --->   results/dt_vs_error.png
```

No hand-copied numbers anywhere — `plot_dt_vs_error.py` loads the JSON; the
"~20× larger usable Δt" annotation is computed from the data, not typed in.

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
MuJoCo: μ=1.2, tor=0.3, kv=1000, solref0=0.005, condim=6, integrator=implicitfast, wheel_geom=capsule
```

These are exactly the per-engine tuned best params from the
`1_sim_to_real_box` yaw-aware re-tune; this experiment only varies `dt`.

## Reading the figure

The figure (`plot_dt_vs_error.py`) uses a **0.2 m** accuracy threshold (the red
shaded band marks error above it). "Usable" = stable AND under that threshold.
All MuJoCo numbers below are the current **turning config** (`tor=0.3`, capsule).

- **Semi-Implicit (orange)** — narrowest range. Floor ~0.096 m at
  `dt = 5×10⁻⁴`; diverges/NaNs at `dt ≥ 7×10⁻⁴` (penalty contact blows up).
  Usable wall: ~`0.0005 s`.
- **MuJoCo (pink)** — floor ~0.065 m, flat from `dt = 10⁻⁴` up to a few
  ×10⁻³, then climbs through the 0.2 m threshold around `dt ≈ 0.02 s` and
  degrades steeply (multi-metre error by `dt = 0.05` and up; the solver stays
  finite but the trajectory is meaningless). Usable wall (0.2 m): ~`0.02 s`.
- **Axion (blue)** — floor ~0.063 m, flat across almost two decades; stays
  under threshold to `dt ≈ 0.4 s`, first instability/NaN at `dt = 0.7 s`.
  Usable wall (0.2 m): ~`0.4 s`.

Resulting usable-`dt` ratios (Axion vs MuJoCo) depend on the threshold:

| threshold | Axion max | MuJoCo max | Axion / MuJoCo |
|---|---|---|---|
| 0.2 m (figure) | 0.4 s | 0.02 s | **~20×** |
| 0.5 m | 0.5 s | 0.03 s | ~17× |

Axion / Semi-Implicit is ~800× (≈ three orders of magnitude).
The figure's headline annotation (computed from the data) is the **0.2 m**
value, ~20×.

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
