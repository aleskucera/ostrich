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

## Current result (all 3 engines fully tuned, yaw-aware metric, scored on the
## discriminator run `18_10_33`; 18_04_51 used as a sanity floor during
## tuning but not reported — see "Per-run breakdown" below for why)

| Engine | combined (pos+yaw) | yaw RMSE | best params | dt |
|---|---|---|---|---|
| **MuJoCo** | **0.074 m** | **4.4°** | `μ=1.5, tor=10, condim=6, implicitfast` | 0.001 |
| Axion | 0.092 m | 5.9° | `mu_rear=1.2, ke=150, compliance.contact=1e-7` | 0.05 |
| Semi-Implicit | 0.189 m | 6.5° | `μ=0.05, ke=8e4, k_d_act=200, joint_attach_ke=1e6` | 0.0005 |

MuJoCo wins by ~2 cm over Axion (at 50× different timesteps); SemiImplicit
is ~2.5× behind MuJoCo on this discriminator.

### Per-run breakdown — why we don't report on `18_04_51`

| Engine | `18_04_51` (easy, all engines tie) | `18_10_33` (discriminator) |
|---|---|---|
| Axion | 0.032 m | 0.092 m |
| MuJoCo | 0.034 m | 0.074 m |
| Semi-Implicit | 0.031 m | 0.189 m |

All three engines land within **3 mm of each other** on the easy run — that
run carries no signal between engines and dilutes the comparison when
averaged in. We keep it in the sweeps as a cherry-pick guard (single-run
tuning is fragile, see the SemiImplicit journey below), but the headline
table and the plot report only the discriminating run's score.

> ### Fixing a metric bias: dt-aware settle (commit TODO)
>
> The chassis is spawned 15 cm above its rest height and falls onto the
> wheels. The earlier `replay_real.py` settled for only 60 steps regardless
> of dt — adequate for Axion/MuJoCo's implicit contact (converges in a few
> steps) but **way too short for SI's penalty contact on a 90 kg chassis at
> dt=5e-4 (30 ms of settle)**. The chassis was still falling when recording
> started, baking a constant ~15 cm Z offset into the SI score — a 5–8 cm
> chunk of `combined`. Switching to a dt-aware settle (≥ 500 ms) drops SI
> from 0.169 → 0.110 m. Implicit-contact engines barely move (0.061→0.062).

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

### SemiImplicit tuning journey

| stage | best | what was learned |
|---|---|---|
| initial (mu=0.5, ke=4e4) | 1.5 m | wheels barely spun — `k_d=0` ok for Axion, broken for SI |
| add `k_d_act` (velocity gain) | 2–3 m or NaN | torque applied but unstable at dt=0.001 |
| drop to dt=5e-4 | 0.226 m | stability margin recovered |
| refine ke/μ/k_d_act at dt=5e-4 | 0.204 m | first floor; tuning is fragile |
| add `joint_attach_ke` (engine knob) | **0.169 m** | library default 1e4 too soft for 90 kg chassis; 1e6 + lower μ=0.05 is robust on both runs (1e6 + μ=0.1 destabilises 18_10_33). Use `tune_semi_implicit.py` to keep iterating. |

**SI is genuinely harder to use here, on three axes:**
1. **`k_d=0` makes the wheels not spin** (in `TARGET_VELOCITY` mode, SI consumes
   `target_kd` as the velocity-feedback gain; Axion doesn't). One forgotten
   knob → silent ~1.5 m error.
2. **`ke` had to be ~10× exp-1's helhest value** because the junior chassis is
   heavier and contact must be stiffer to keep the wheels from sinking into
   the box during the climb. Even so, the climb plot shows the chassis Z
   sinking to −18 cm — a spring-damper compression artifact, not real motion.
3. **The stable parameter region is narrow** — many neighbouring configs
   diverge (combined of 14, 18, 28, 100 m, or NaN). MuJoCo and Axion don't
   show this; their tuning is robust.

The 0.204 m floor is ~4× the implicit-contact engines. Most of that gap is the
chassis-Z sink, not the in-plane error.

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
