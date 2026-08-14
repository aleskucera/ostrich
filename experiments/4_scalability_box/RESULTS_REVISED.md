# Revised scalability results (2026-08-14, reviewer-response reruns)

All runs on dasenka RTX 3090 24 GB, GT-tracking task, 6 s horizon, 5 iterations,
median of warm iterations, compile excluded. Memory reported BOTH ways: engine-native
allocator peak (`peak_gpu_mb`) and NVML absolute/delta (pynvml installed this time —
fixes the paper's footnote which claimed NVML but shipped native peaks).

MJX env: fresh `.venv-mjx` (jax 0.10.1+cuda12, mujoco/mjx 3.9.0), solver patched to a
fixed 10-iteration `_while_loop_scan` (stock MJX's dynamic `while_loop` is not
reverse-differentiable at all — must be disclosed in the paper). Scripts:
`mjx_sim_ckpt.py --checkpoint {none,step,sqrt}`.

## Headline table

| engine | N | s/iter | alloc MB | NVML abs | NVML delta | world-iter/s |
|---|---|---|---|---|---|---|
| MJX plain BPTT | 1 | 47.7 | 1437 | 3361 | 2616 | 0.02 |
| MJX plain BPTT | 8 (max) | 48.5 | 11495 | 18895 | 18150 | 0.16 |
| MJX plain BPTT | 16 | OOM (22.45 GiB alloc req) | | | | |
| MJX + jax.checkpoint(step) | 1 | 51.2 | 320 | 1311 | 566 | 0.02 |
| MJX + jax.checkpoint(step) | 256 | 61.2 | 515 | 1829 | 1080 | 4.19 |
| MJX + jax.checkpoint(step) | 8192 | 248.1 | 12921 | 19079 | 18058 | 33.0 |
| Ostrich | 1 | 0.51 | 203 | 1247 | 659 | 1.94 |
| Ostrich | 512 | 1.85 | 995 | 2049 | 1461 | 276 |
| Ostrich | 8192 | 23.6 | 12990 | 14023 | 13435 | 347 |
| Ostrich | 16384 | OOM | | | | |
| Semi-Implicit | 1 | 4.19 | 228 | 4423 | 3835 | 0.24 |
| Semi-Implicit | 512 (max) | 21.1 | 14090 | 19367 | 18779 | 24.3 |
| Semi-Implicit | 1024 | OOM | | | | |

## What changed vs the paper

1. **Paper's MJX numbers do not reproduce in a modern env**: 1.4 GB/world (not 7.0),
   OOM at 16 worlds (not 4), ~48 s/iter. The whole MJX column must be re-measured.
2. **Reviewer 2 is right about checkpointing, more than expected**: one line of
   `jax.checkpoint` around `mjx.step` costs only ~7-25% time (recompute is nearly free —
   the rollout is kernel-launch-bound) and removes the memory ceiling: checkpointed MJX
   reaches 8192 worlds, same as Ostrich. `sqrt` two-level checkpointing adds nothing
   over per-step at these sizes.
3. **The surviving claim is throughput, not memory**: at matched 8192-world batch,
   Ostrich 23.6 s/iter vs checkpointed MJX 248 s/iter (10.5x); peak throughput
   347 vs 33 world-iter/s (10.5x); single-world latency 0.51 s vs 51 s (100x).
   Driver: 60 steps/rollout at h=0.1 s vs 1200 at h=5 ms, and an adjoint with no
   store-vs-recompute trade-off. Gradient *quality* (92% vs 20% task success, exp 3)
   is unaffected by checkpointing.
4. **Ostrich & SI paper numbers reproduce exactly** (0.51 s/iter N=1, 23.5 s N=8192,
   OOM 16384; SI 4.19 s/iter N=1, max 512, OOM 1024) — now with NVML methodology.
   Note: the paper's SI "4.3 s/iter" figure originates from THIS task, not the box2
   control-synthesis task (28.7 s/iter there); Sec. IV-B must be corrected accordingly.
5. SI footnote: even here its loss increases across iterations — its gradients are
   unusable for descent on this scene, consistent with exp 3.

Raw JSONs: `results/mjx_ckpt_{none,step,sqrt}_N.json`, `results/ostrich_nvml_N.json`,
`results/si_nvml_N.json` (this directory, now version-controlled).
