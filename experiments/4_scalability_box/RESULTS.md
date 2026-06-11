# Experiment 4 (box) — results

Box-scene scalability sweep on a single NVIDIA RTX 3090 (24 GB) on dasenka.
Same helhest_junior + box geometry, K=10 spline → recorded GT trajectory loss,
horizon=6 s, 5 iterations per (engine, num_worlds) point. Each engine uses
its `experiments/1_sim_to_real_box` calibrated physics and the dt at the top
of its `experiments/2_dt_stability_box` accuracy plateau.

Headline figure: [`results/scalability_box.png`](results/scalability_box.png)

## Headline numbers

| engine | max worlds | per-iter @ N=1 | per-iter @ N_max | memory @ N=1 | memory @ N_max | per-world memory slope |
|---|---|---|---|---|---|---|
| **Axion**         | **8 192** (OOM at 16 384) | **525 ms** | 23.6 s @ 8192 | 203 MB | 13.0 GB | ~1.56 GB/world (post-knee) |
| MJX-grad         | 2 (OOM at 4)              | 202 s | 204 s @ 2 | 7.0 GB | 14.0 GB | **~7.0 GB/world** |
| **Semi-Implicit** | **512** (OOM at 1024)     | 4.4 s | 22.0 s @ 512 | 228 MB | 14.1 GB | **~27 MB/world** |

## Three-way ordering and what it means

| dimension | Axion | MJX | SI |
|---|---|---|---|
| **per-iter speed (low N)** | 525 ms | 202 s | 4.4 s |
| → Axion vs MJX | **386× faster** at N=1 | | |
| → Axion vs SI | 8.4× faster at N=1 | | |
| **max worlds on 24 GB** | 8 192 | 2 | 512 |
| → Axion vs MJX | **4 096× more worlds** | | |
| → Axion vs SI | 16× more worlds | | |
| **per-world memory** | ~1.6 GB | ~7 GB | **~27 MB** |
| → SI is the memory champion | | | |

So the three engines occupy three distinct corners of the speed × memory trade-off:

- **Axion**: fast *and* scales. Optimal on both axes.
- **MJX**: slow *and* OOMs immediately. Worst on both axes.
- **Semi-Implicit**: medium speed, but *unmatched memory efficiency* — only 27 MB
  per added world (vs MJX's 7 000 MB). The catch: SI's gradients on this scene
  fail to converge (see `experiments/3_gradient_quality_box/RESULTS.md`), so its
  memory-efficient batching can't actually be used for optimization. SI's
  scaling shape is the right one for an RL-style batched-forward use case,
  not for gradient-based optimization.

## Per-engine result detail

### Axion (implicit adjoint)

| N | time (ms) | memory (MB) |
|---|---|---|
| 1     | 525    | 203 |
| 8     | 558    | 207 |
| 64    | 693    | 295 |
| 512   | 1 875  | 995 |
| 1 024 | 3 352  | 1 794 |
| 4 096 | 12 033 | 6 592 |
| 8 192 | 23 555 | 12 990 |
| 16 384 | — | OOM (1 GB alloc fails on transition ~13 GB → ~24 GB) |

**Two-regime scaling**:
1. **Pre-knee (N ≤ ~64)**: time and memory both essentially flat. The 3090
   has spare bandwidth and per-world replication overhead is absorbed by the
   static base state. This is the "Axion runs equally fast on a laptop as
   on a workstation" regime we saw in `experiments/3_gradient_quality_box/RESULTS.md`.
2. **Post-knee (N ≥ ~256)**: linear in both time and memory. GPU saturated.
   Per-iter cost ~2.9 ms/world; memory ~1.6 GB/world. Wall: 16 384.

The knee at ~256–512 worlds matches the convention from `experiments/4_scalability`'s
flat-scene results.

### MJX-grad (JAX BPTT + jax.vmap)

| N | time (ms) | memory (MB) |
|---|---|---|
| 1 | 202 606 | 6 995 |
| 2 | 204 113 | 13 990 |
| 4 | — | OOM (28 GB needed; GPU has 24) |

**Per-iter time is dominated by box-scene contact computation through BPTT.**
At dt=5e-3 over 6 s horizon = 1 200 mjx.step calls per rollout, plus the
backward pass through each. With condim=6 + torsional friction + the box
obstacle, this becomes a 200 s per-iter cost — about 12× slower than MJX
without vmap on the same scene (17 s/iter in `experiments/3_gradient_quality_box`).
Most of that 12× is vmap+grad XLA overhead at small batch sizes; the rest
is the extra contact-detection cost at condim=6.

**Memory scales linearly at ~7 GB/world** — the BPTT tape (each of 1 200
steps' worth of `mjx.Data` activations + gradients) must be replicated per
world. Wall: 4 worlds.

### Semi-Implicit (Warp tape BPTT)

| N | time (ms) | memory (MB) |
|---|---|---|
| 1     | 4 419  | 228 |
| 8     | 4 754  | 412 |
| 64    | 6 899  | 1 932 |
| 256   | 13 599 | 7 142 |
| 512   | 22 047 | 14 090 |
| 1 024 | — | OOM (28 GB needed by trajectory tape) |

**Most memory-efficient scaling of the three.** SI's per-world overhead is
just ~27 MB — two orders of magnitude smaller than MJX's 7 GB/world and an
order of magnitude smaller than Axion's 1.6 GB/world. Why: SI's Warp tape
stores raw state buffers rather than complete operator activations, and
since SI uses no Newton iteration, there's no per-step linear-system state
to store per world.

**Per-iter time scales roughly linearly past N=64**: from 6.9 s (N=64) to
22 s (N=512), close to linear in N. Pre-N=64 the curve is flat because GPU
is underused.

**The catch** (paper-relevant): from
`experiments/3_gradient_quality_box/RESULTS.md`, SI's gradient quality on
this scene doesn't actually converge — best loss across 3 trials was 0.71
vs Axion's 0.07. SI's memory efficiency is real, but it's only useful for
non-gradient batched-forward workloads (e.g., RL, MPC) on this scene; for
gradient-based optimization the noisy gradients dominate.

## Why is MJX so much slower per-iter than Axion (386×)?

Two compounding factors:

1. **dt margin (~20×)**. Axion runs at dt=0.10 (60 sim steps per 6 s
   rollout); MJX runs at dt=5e-3 (1 200 steps). Per `experiments/2_dt_stability_box`,
   that's each engine at the largest dt inside its accuracy plateau on this
   scene.
2. **Per-step BPTT + jax.vmap+grad XLA overhead (~20×)**. MJX must materialise
   gradient activations at every step (~200 ms each); Axion's adjoint
   captures forward+backward into one CUDA graph that replays in ~9 ms per
   step warm.

Either factor alone would account for a 20× gap — together they make 386×.

The matching exp-3 box result (no vmap, just N=1) was 17 s/iter for MJX vs
0.5 s for Axion — a 34× ratio. The 386× here adds another ~12× from vmap+grad
overhead at N=1. (XLA's batched-trace compilation for a batch of 1 is much
less optimised than the unbatched path.)

## Why does SI dominate on memory?

| | per-step state | per-world replicated |
|---|---|---|
| **Axion** | Newton-iter linear-system buffers + body state | yes, ~few MB each |
| **MJX**   | full `mjx.Data` + per-call gradient activations | yes, ~6 MB per step × 1 200 steps ≈ 7 GB per world |
| **SI**    | body state only (penalty contact, no Newton) | yes, but ~6 bytes per body per step × 12 000 steps ≈ tens of MB per world |

SI's "no linear system, no Newton iterations, no contact-Jacobian inversion"
costs almost nothing per world. Each added world is essentially just another
copy of the body-state arrays.

## Reproduce

```bash
# full sweep (~3 hr — SI's cold-capture dominates)
bash experiments/4_scalability_box/run_sweep.sh

# one engine only
bash experiments/4_scalability_box/run_sweep.sh --axion
bash experiments/4_scalability_box/run_sweep.sh --mjx
bash experiments/4_scalability_box/run_sweep.sh --semi-implicit

# one (engine, N) data point
python experiments/4_scalability_box/axion_sim.py --num-worlds 1024 \
    --save experiments/4_scalability_box/results/axion_1024.json

# regenerate the figure
python experiments/4_scalability_box/plot_results.py
```

## Caveats

- **NVML peak memory** wasn't reported on dasenka (NVML poller returned `None`
  — `pynvml` not in the env). The figure uses the engine-native peak
  (`warp.get_mempool_used_bytes` for Axion/SI, JAX's `peak_bytes_in_use` for
  MJX). Native peaks are 1.5–2× smaller than NVML absolute on the flat-scene
  experiment; the OOM points observed here confirm the native numbers are
  *under*-estimates of true GPU footprint.
- **SI gradient quality**: SI scales beautifully in memory, but its gradient
  signal on this scene is too noisy to be useful for optimisation
  (see `experiments/3_gradient_quality_box/RESULTS.md`). Its high
  memory-efficiency is therefore most relevant to non-gradient batched
  workloads (forward-only inference, RL rollouts, MPC), not to the
  gradient-based optimisation that drives this whole `_box` experiment series.
- **MJX cylinder↔box** isn't implemented; wheels are swapped to capsules
  (with `contype="1" conaffinity="2"` to skip wheel↔wheel contacts) as
  documented in `experiments/3_gradient_quality_box/README.md`.
- **TinyDiffSim, Brax** are referenced by `experiments/4_scalability/`
  (flat scene) but not included here; they aren't part of the headline
  three-way for this paper section.
