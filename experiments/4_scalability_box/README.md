# Experiment 4 (box) — scalability

Box-obstacle counterpart of `experiments/4_scalability`. Sweeps the number of
parallel worlds (each running the same helhest_junior + box scene from `1_box`
/ `2_box` / `3_box`) for each differentiable engine, measuring median time
per iteration and peak GPU memory. The figure shows how each engine scales —
both in compute (does it stay fast at high world counts?) and in memory
(when does it OOM?).

## Pipeline

```
<engine>_sim.py --num-worlds N --save results/<engine>_<N>.json
       |        |
       +--------+----------> run_sweep.sh (sweeps N for every engine)
                                |
                                v
                          results/<engine>_<N>.json
                                |
                                v
                          plot_results.py
                                |
                                v
                          results/scalability_box.png
```

## What each script does

| script | engine | gradient mechanism | replicated-worlds via |
|---|---|---|---|
| `axion_sim.py` | Axion | implicit adjoint | `builder.finalize_replicated(num_worlds=N)` |
| `mjx_sim.py` | MuJoCo MJX | `jax.grad` BPTT + `jax.vmap` | `jax.tree_util` batched `mjx.Data` |
| `semi_implicit_sim.py` | Newton SemiImplicit | Warp tape BPTT | `builder.finalize_replicated(num_worlds=N)` |

All three:
- Same scene: helhest_junior + static box obstacle (from `1_box`)
- Same target: real prism trajectory from `run_2026_05_20-18_10_33` (from `3_box`)
- Same K=10 wheel-velocity spline, broadcast identically across all worlds
- Same horizon 6 s, calibrated physics per engine (from `1_box`)
- 5 iterations of gradient descent (measuring per-iter throughput, not convergence)

**Why "same input across all worlds"?** This is a *throughput* test, not a
diversity test. With identical inputs, all worlds simulate the same trajectory
and produce the same gradient — so memory measures pure batch-replication
overhead and time measures pure per-world compute. Matches the convention in
`experiments/4_scalability/`.

## Reproduce

```bash
# full sweep (~2.5 hr total on a 24 GB GPU)
bash experiments/4_scalability_box/run_sweep.sh

# one engine only
bash experiments/4_scalability_box/run_sweep.sh --axion
bash experiments/4_scalability_box/run_sweep.sh --mjx
bash experiments/4_scalability_box/run_sweep.sh --semi-implicit

# one (num_worlds, engine) data point
python experiments/4_scalability_box/axion_sim.py --num-worlds 64 \
    --save experiments/4_scalability_box/results/axion_64.json

# regenerate the figure
python experiments/4_scalability_box/plot_results.py
```

The sweep:
- Iterates powers of two from 1 up to the configured ceiling per engine
  (Axion: 131k, MJX: 64, SI: 32 — past that we expect OOM).
- Skips already-existing result files (resumable).
- Stops a given engine's loop on the first OOM/crash.

## Wall-time per `num_worlds` run

| engine | warm 5-iter cost | cold compile per run | typical sweep cost |
|---|---|---|---|
| Axion         | ~0.5 s × 5 = 2.5 s    | ~60 s (module compile cached after run 1) | ~30 min for 18 points |
| MJX           | ~17 s × 5 = 85 s      | ~78 s JIT compile (per process) | ~25 min for 7 points |
| Semi-Implicit | ~4.3 s × 5 = ~22 s    | **~20 min** (CUDA-graph capture, per process) | ~90 min for 5 points |

SI is the long pole — each `num_worlds` value triggers a fresh CUDA-graph
capture. Could be avoided by sharing the optimizer across `num_worlds`
settings inside one process, but that's a non-trivial refactor.

## Per-engine configuration

| param | Axion | MJX | SI |
|---|---|---|---|
| dt | 0.10 | 5e-3 | 5e-4 |
| sim steps per iter (6 s horizon) | 60 | 1200 | 12 000 |
| friction / contact model | implicit, μ=0.8/1.2 | implicitfast, μ=1.5, condim=6, tor=10 | penalty, μ=0.05, ke=8e4 |
| collision-shape workaround | none | wheels: cylinder→capsule, wheel↔wheel disabled | none |
| gradient stabilisation | none needed | grad clip 1.0 | grad clip 1.0 |

Each engine's dt comes from `experiments/2_dt_stability_box` (the largest dt
inside its accuracy plateau on this scene).

## Memory measurement

Peak GPU memory is reported as **NVML absolute** (total used by the process,
sampled at 100 Hz on a daemon thread). This captures activation buffers,
allocator fragmentation, and (for MJX) JAX's preallocation pool. We require
`XLA_PYTHON_CLIENT_PREALLOCATE=false` for MJX so NVML peaks reflect actual
allocations rather than the JAX pre-allocator. The `peak_gpu_mb` field in
each JSON is the engine-native peak (Warp's `get_mempool_used_bytes` for
Axion/SI, JAX's `peak_bytes_in_use` for MJX) and is included for context but
not used in the headline plot (it understates real GPU footprint).

## Output JSON schema

```json
{
  "simulator": "Axion" | "MJX-grad" | "Semi-Implicit",
  "num_worlds": 64,
  "median_time_ms": 95.2,
  "peak_gpu_mb": 215.0,
  "peak_gpu_mb_nvml_absolute": 1837.0,
  "peak_gpu_mb_nvml": 215.0,
  "time_ms": [...],
  "K": 10, "dt": 0.1, "duration_s": 6.0, "iterations": 5
}
```

## See also

- `experiments/3_gradient_quality_box/RESULTS.md` — same calibration / target /
  spline setup, gradient quality at num_worlds=1.
- `experiments/4_scalability/` — flat-ground counterpart (Axion + MJX only,
  K=30 spline to synthetic chassis-pose target).
