# Experiment 3 (box) — gradient quality

Box-obstacle counterpart of `experiments/3_gradient_quality`. Each
differentiable engine uses its native gradient mechanism to optimize a
K-knot wheel-velocity spline so the helhest_junior matches a recorded
real-robot trajectory while crossing the box. The figure overlays the
loss-vs-iteration curves so the relative gradient quality is visible
at a glance.

Calibrated physics params come from the yaw-aware best of
`experiments/1_sim_to_real_box` so the optimizer isn't fighting a
mis-tuned simulation — it's purely a gradient-quality test.

## Pipeline (reproducible from data)

```
optimize_<engine>.py   --->   results/<engine>.json   --->   plot_results.py   --->   results/gradient_quality_box.png
```

- `optimize_<engine>.py` does the optimization for one engine, multiple
  trials with different random initializations (`--num-trials`,
  `--seed-base`). Saves per-trial loss curves.
- `plot_results.py` loads every `results/*.json` and overlays them
  (median + min/max band per engine).

```bash
# regenerate everything (Axion only for now; ~30 min at iters=50 trials=3)
./experiments/3_gradient_quality_box/run_experiment.sh

# or with a shorter horizon / fewer iters for a quick look
ITERATIONS=15 NUM_TRIALS=2 HORIZON=4.0 \
    ./experiments/3_gradient_quality_box/run_experiment.sh

# plot whatever's in results/
python experiments/3_gradient_quality_box/plot_results.py
```

## What each script does

| script | engine | gradient mechanism | status |
|---|---|---|---|
| `optimize_axion.py` | Axion | adjoint (`AxionDifferentiableSimulator`) | ✅ working |
| `optimize_semi_implicit.py` | Newton SemiImplicit | warp tape / BPTT | ⏳ TODO |
| `optimize_mjx.py` | MuJoCo MJX | `jax.grad` / BPTT | ⏳ TODO (needs JAX + JAX junior model) |
| `optimize_tinydiffsim.py` | TinyDiffSim | CppAD | ⏳ TODO (separate codebase) |

For now only Axion is implemented; the others will be added as their
junior + box setups become available. `plot_results.py` picks up any
present `*.json` automatically, so adding a new engine doesn't require
plot changes.

## Optimization setup

- **Scene**: `helhest_junior` over the box (matches `1_sim_to_real_box`).
- **Physics params**: `mu_front=0.8, mu_rear=1.2, mu_rolling=0.7,
  compliance.contact=1e-7, dt=0.02` (Axion's tuned best; engine-specific
  knobs come from each engine's calibration when added).
- **Parameterization**: `[K, 3]` array of control points (default K=10)
  interpolated linearly to per-step `[T, 3]` wheel-velocity commands for
  the `[left, right, rear]` DOFs.
- **Loss**: mean squared XY distance between sim chassis (body 0) and
  the recorded real prism trajectory shifted to the sim's chassis-start
  frame. Only XY — Z is dominated by the box climb, which adds noise
  without informing the wheel-velocity spline.
- **Optimizer**: Adam on the spline params with cosine LR decay, `lr=0.1`.

## Per-engine `axion.json` schema

```json
{
  "simulator": "Axion",
  "gt": "2026_05_20-18_10_33",
  "K": 10, "lr": 0.1, "iterations": 50,
  "horizon_s": 6.0, "dt": 0.02,
  "num_trials": 3, "seed_base": 42,
  "trials": [
    {"seed": 42, "losses": [...], "wall_s": 1750.0, "best_loss": 0.041},
    {"seed": 43, "losses": [...], ...},
    ...
  ]
}
```

## Caveats

- The loss is on the chassis pose, but the real GT is the *prism* point
  (offset 0.11 m forward of the chassis). We shift the target by the sim's
  chassis-start position so initial XY align; the residual prism-vs-chassis
  offset is small (~11 cm constant) and gets folded into the loss baseline.
  If we add yaw-dependent terms later, this should switch to comparing
  prism-tracked positions on both sides.
- Optimization is slow at fine `dt` (Axion adjoint ~35 s/iter at
  `dt=0.02` over a 4 s horizon). Coarser `dt` (0.05) is much faster and
  per `2_dt_stability_box` only marginally less accurate — but the
  adjoint kernels were validated at `dt≤0.05` so we keep `dt=0.02` for
  safety in v1.
