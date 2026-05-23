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
| `optimize_mjx.py` | MuJoCo MJX | `jax.grad` / BPTT through `mjx.step` | ✅ written (needs `jax` + `mujoco-mjx`) |
| `optimize_semi_implicit.py` | Newton SemiImplicit | warp tape / BPTT | ⏳ TODO |
| `optimize_tinydiffsim.py` | TinyDiffSim | CppAD | ⏳ TODO (separate codebase) |

### MJX setup

```bash
# in a separate venv (axion's main env doesn't carry JAX)
pip install jax mujoco-mjx

python experiments/3_gradient_quality_box/optimize_mjx.py
```

Defaults: `dt=5e-3`, `horizon=6s` → 1200 BPTT steps. Extrapolating from a
measured `1.1 GB / 2 s` BPTT footprint (typical mjx config), this should
land around **~0.7 GB peak GPU memory** at the default — well within any
modern GPU. dt=1e-2 (still in MJX's accuracy plateau on this scene) halves
that again if memory matters.

The MJX script uses the same junior + box MJCF template + tuned best
params (`μ=1.5, tor=10, condim=6, implicitfast, kv=1000`) as
`experiments/1_sim_to_real_box/sweep_mujoco.py`, so the only difference
vs the forward sweep is `mjx.step` + `jax.grad` for the adjoint.

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

## Speed

`AxionDifferentiableSimulator.diff_step()` captures the full
forward+backward physics+loss into a CUDA graph when
`use_cuda_graph=True` (the default in our `make_configs`). With it:

| | per iter | per-step adjoint | vs realtime (6 s sim) |
|---|---|---|---|
| **dt=0.10 (current default)** | **~0.50 s warm (~1.3 s cold)** | ~8 ms | **~12× faster than realtime** |
| dt=0.05 | ~1.0 s warm (~3.4 s cold) | ~8 ms | ~6× faster than realtime |
| dt=0.05, *no* CUDA graph | ~35 s warm (~45 s cold) | ~290 ms | ~6× **slower** than realtime |

The 47× speedup over the no-CUDA-graph version is entirely the missing
kernel-launch overhead — the box adds zero adjoint cost. Per-step we
match the original `experiments/3_gradient_quality` Axion throughput
(~9 ms/step adjoint).

A full **50-iter × 3-trial** optimisation finishes in ~80 s at the
default dt=0.10.

### Why dt=0.10 (not 0.05)?

From `experiments/2_dt_stability_box`, Axion's forward error plateau
extends from `dt=0.005` to `dt≈0.30` (combined error 0.063→0.088 m
on this scene). dt=0.10 sits comfortably in the plateau yet halves the
sim-step count vs dt=0.05 — for free. Larger dt is fine too:

| `--dt` | per-iter warm | converged loss (3 trials) |
|---|---|---|
| 0.05 | ~1.0 s | ~0.045 m² |
| **0.10 (default)** | **~0.50 s** | **~0.072 m²** |
| 0.20 (untested here) | ~0.25 s | likely ~0.10 m² |

The slightly higher *optimisation* loss at coarser dt isn't a gradient
regression — it's that the K=10 spline finds a different optimum per
dt (the spline has to compensate for the engine's per-dt response). The
key signal (3 random seeds converging within ±5%) holds at every dt
we've tried.

## Caveats

- The loss is on the chassis pose, but the real GT is the *prism* point
  (offset 0.11 m forward of the chassis). We shift the target by the sim's
  chassis-start position so initial XY align; the residual prism-vs-chassis
  offset is small (~11 cm constant) and gets folded into the loss baseline.
  If we add yaw-dependent terms later, this should switch to comparing
  prism-tracked positions on both sides.
