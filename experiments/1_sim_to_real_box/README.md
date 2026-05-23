# Experiment 1 (box) — sim-to-real over a box obstacle

Box-obstacle counterpart of `experiments/1_sim_to_real`. Drives every physics
engine with the same recorded wheel commands from a real **helhest_junior**
run **over the box**, and scores how well each engine reproduces the measured
trajectory **including the climb (Z)**, using the same prism-tracked
comparison the single-engine replay (`examples/helhest_junior/replay_real.py`)
uses.

## Pipeline

```
synced rosbag .h5  ──prepare_gt.py──►  data/run_*.json  ──sweep_<engine>.py──►  results/sweep_<engine>.json  ──plot_results.py──►  results/box_sim_to_real.png
```

1. **`prepare_gt.py`** — converts `~/rosbags_experiment/synced/run_*.h5` into
   per-run GT JSONs in `data/`. The wheel commands are already remapped to
   sim DOF order `[L,R,rear]` and sign-flipped so forward = positive on every
   wheel; the real trajectory is the prism point aligned (start at origin,
   heading +X, z relative). All from `examples/helhest_junior/replay_real.py`
   so the convention matches the validated single-engine replay.
2. **`sweep_axion.py`** — reuses `HelhestJuniorReplaySimulator` (junior model
   + box, CUDA-graph replay). Sweeps `dt`, `mu_front`, `mu_rear`.
3. **`sweep_mujoco.py`** — junior-geometry MJCF + box authored inline (matches
   `HelhestJuniorConfig` dimensions/masses; same wheel-axis convention so the
   same GT commands drive it). Sweeps `dt`, `kv`, `mu`.
4. **`plot_results.py`** — 3-panel figure: top-down XY, prism Z vs time, and
   the combined 3D L2 accuracy bar chart.

## Metric

`common_box.score()` tracks the prism point in sim, cross-correlates forward-x
to absorb the wheel-vs-pose stream-zeroing offset (~0.3–0.5 s — a known
data-side artifact, see the replay docs), then computes the combined 3D L2 of
(Δx, Δy, Δz) over the overlap of valid samples. Returns combined / xy / z.

## Quickstart

```bash
# 1. Build GT JSONs (one-time; ~seconds)
python experiments/1_sim_to_real_box/prepare_gt.py --all

# 2. Sweep both engines
python experiments/1_sim_to_real_box/sweep_axion.py \
    --dt 0.05 --mu-front 0.6 0.8 --mu-rear 0.35 0.6 0.8 1.0

python experiments/1_sim_to_real_box/sweep_mujoco.py \
    --dt 0.001 0.002 0.005 --kv 1000 4000 --mu 0.4 0.8 1.2

# 3. Plot
python experiments/1_sim_to_real_box/plot_results.py --run 18_10_33
```

## Current result (Axion vs *fully-tuned* MuJoCo, 2 runs: 18_04_51 + 18_10_33)

| Engine | best combined 3D L2 | best params | dt |
|---|---|---|---|
| **MuJoCo** | **0.048 m** | `μ=1.5, tor=10, condim=6, integrator=implicitfast` | 0.001 |
| Axion | 0.056 m | `mu_front=0.8, mu_rear=1.0` | 0.05 |

Essentially tied (within tuning noise) — but at **50× different timesteps**.
MuJoCo needs dt≈10⁻³ s to land here; Axion does it at dt=5·10⁻² s.

### MuJoCo tuning journey (why the strawman matters)

| stage | best | what was added |
|---|---|---|
| initial sweep (dt × kv × μ) | 0.104 m | `condim=3`, default torsional |
| stage 1 (+ integrator, condim, solref, ground torsional) | 0.066 m | **`condim=6`** + torsional 0.5 |
| stage 2 (refine: higher μ + higher torsional) | 0.049 m | μ=1.2, torsional 5.0 |
| stage 3 (probe edges) | **0.048 m** | converged floor |

The single biggest lever was **`condim=6` with torsional friction ≥ 2.0** —
pyramidal cone with `condim=3` has no torsional component, so the rear wheel
skids freely on box impact and the chassis acquires a spurious yaw (exactly
the same physics as the Axion `mu_rear` tuning). With it, MuJoCo joins the
floor at ~0.048 m. The remaining ~5 cm of error is the chassis Z baseline
drifting down a couple of cm (contact compliance during the climb/descent)
and a slightly lower climb peak — small residual physics.

## Adding more engines

Mirror `sweep_mujoco.py`'s structure: build a junior-geometry model in the new
engine's format with the box obstacle and three velocity-controlled wheel
joints (axis `+Y`, positive = forward), step it with the GT command timeseries,
collect `[N,7]` chassis poses (`px,py,pz, qx,qy,qz,qw`), and call
`common_box.score(pose, dt, gt)`. Save `results/sweep_<engine>.json` in the
same schema and `plot_results.py` will pick it up automatically.

## Notes & caveats

- All engines must use **junior geometry** (r=0.35, track=0.73, wheelbase=0.75)
  because the GT was recorded with that robot — `HelhestJuniorConfig` is the
  source of truth. The MJCF here mirrors it.
- Drive signal is the **commanded** wheel velocity (`/joint_setpoint`); using
  measured wheel velocity is worse (per the replay analysis).
- For the runs where the real prism position curves while the IMU heading
  stays straight (e.g. `17_58_43`, flagged "yaw fold" + TS gap at the box),
  the divergence is a data-side artifact, not a physics gap — don't tune to
  them. The default sweeps use `18_04_51` (cleanest) + `18_10_33` (clean +
  curved end).
