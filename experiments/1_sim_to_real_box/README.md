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

## Metric — position + yaw, NOT position alone

`common_box.score()` tracks the prism point in sim, cross-correlates forward-x
to absorb the wheel-vs-pose stream-zeroing offset (~0.3–0.5 s, a known
data-side artifact), then computes:

  combined_with_yaw = sqrt( <|Δp|²> + (L · RMSE(Δyaw))² )

where Δp is the prism position error (3D) and Δyaw is the chassis heading
error (rad, relative to start). The lever arm `L = 0.5 m` (≈ half wheelbase)
converts the yaw RMSE into a position-equivalent that a point at the chassis
tip would see; this is the single number the sweeps minimize.

**Why not position L2 alone?** A simulator that over-cranks torsional friction
and refuses to rotate at all can hide its missing yaw dynamics behind a
low position L2 — the chassis stays near the average of the real curve and
"wins." The yaw term explicitly penalizes that degeneracy (see the MuJoCo
tuning journey below).

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

## Current result (both engines fully tuned, yaw-aware metric, 2 runs: 18_04_51 + 18_10_33)

| Engine | combined error (pos+yaw) | yaw RMSE | best params | dt |
|---|---|---|---|---|
| **MuJoCo** | **0.055 m** | **2.7°** | `μ=1.5, tor=2, condim=6, implicitfast` | 0.001 |
| Axion | 0.061 m | 3.2° | `mu_rear=1.5, compliance.contact=1e-7` | 0.05 |

Essentially tied (~6 mm apart) at **50× different timesteps**. Both engines
produce yaw responses of the same magnitude (real is ~0–3° depending on run);
neither hides behind a locked chassis. MuJoCo needs dt≈10⁻³ s; Axion does it
at dt=5·10⁻² s.

> ### Why I no longer report the L2-only "MuJoCo 0.048 m" number
>
> An earlier sweep that minimized **position L2 alone** put MuJoCo at
> `tor=10`, which produced **literally 0° chassis yaw on every run** — a
> degenerate "robot refuses to rotate" solution that wins position L2 by
> hugging the average real path. With the yaw-aware metric the optimum
> moves to `tor=2` and the chassis actually rotates. The 0.048 was a
> metric artifact, not a real physics win.

### MuJoCo tuning journey

| stage | best | what was added |
|---|---|---|
| initial sweep (dt × kv × μ) | 0.104 m | `condim=3`, default torsional |
| stage 1 (+ integrator, condim, solref, torsional) | 0.066 m | **`condim=6`** + torsional 0.5 |
| stage 2 (higher μ + higher torsional) | 0.049 m | μ=1.2, torsional 5.0 |
| stage 3 (probe edges) | **0.048 m** | converged floor |

The single biggest lever was **`condim=6` with torsional friction ≥ 2.0** —
pyramidal cone with `condim=3` has no torsional component, so the rear wheel
skids freely on box impact and the chassis acquires a spurious yaw.

### Axion tuning journey (mirror)

| stage | best | what was added |
|---|---|---|
| initial sweep (mu_front × mu_rear) | 0.056 m | exposed via `--mu-front/--mu-rear` |
| stage 1 (mu_rolling, the torsional analog) | 0.057 m | **flat — no improvement** |
| stage 2 (contact `ke`, `compliance.contact`, mu_rear) | 0.054 m¹ | ~2 mm of headroom |
| **switch to yaw-aware metric** | **0.061 m** | honest, no degeneracy |

¹ NB: `ShapeConfig.ke/kd/kf` are not consumed by the Axion solver (Axion uses
its own compliance model); the `--ke` CLI knob is kept for signature parity
but has no effect.

**Notable asymmetry:** Axion's `mu_rolling` — the natural analog of MuJoCo's
torsional friction — is *flat* across 0–5.0 (Δ ≈ 4 mm). Axion's plain friction
model already handles the box-kick yaw without needing a separate torsional
channel. By contrast, MuJoCo needs `condim=6` + torsional friction explicitly
enabled, and the L2-only metric pushed it into the degenerate locked-yaw
solution — both of which the yaw-aware metric exposes.

The remaining ~5–6 cm of error on both engines is real residual physics — the
chassis Z baseline drifts down ~2 cm during the climb/descent (contact
compliance) and neither engine perfectly matches both real runs' yaw
simultaneously (real is ~0° on 18_04_51 and ~+1° on 18_10_33; both sims
overshoot to a few degrees).

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
