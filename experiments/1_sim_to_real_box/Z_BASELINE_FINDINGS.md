# Box sim-to-real: z-axis baseline correction

This note documents the z-axis (climb) handling in the box sim-to-real benchmark
(`experiments/1_sim_to_real_box`): the artifact we found, why it happened, how we
fixed it, and how the fix applies across the three engines. It also records the
turning-config decision that preceded it, since the two are entangled in the
current figure/scores.

## TL;DR

- **Symptom:** MuJoCo's z (prism-elevation) curve sat ~2 cm below every other
  engine and below the real data — a constant DC offset, not a dynamics
  difference.
- **Root cause:** z was being zeroed at `sim[0]` (the spawn instant). The chassis
  spawns slightly above its resting height and drops onto its wheels in the first
  ~0.2 s. MuJoCo's softer contact settles ~2 cm, so reading the baseline mid-drop
  shifted MuJoCo's *entire* z curve down. The real data is already settled at
  t=0, so it has no such drop.
- **Fix:** zero z against the **median z over a settled pre-box window
  [0.3, 1.0] s** instead of `sim[0]`. x and y are still zeroed at `sim[0]`.
- **Where:** a single shared function `common_box.score()` — so the rule is
  identical for all three engines. It only removes whatever spawn-settle each
  engine actually has.
- **Result:** MuJoCo z RMSE 25.3→19.2 mm (run 18_10_33) and 20.5→15.1 mm
  (18_04_51); combined error 67→64.7 mm. Ostrich (z 14.1 mm) and Semi-Implicit
  (z 37.9 mm) essentially unchanged — they don't settle, so the rebase is a
  no-op for them.

## How z is resolved, step by step

All scoring goes through `common_box.score(sim_pose, sim_dt, gt)`:

1. **Prism-track the chassis pose.** Convert each engine's chassis pose `[T,7]`
   (position + quaternion) to the *prism point* — the spot the real total-station
   measured (`PRISM_OFFSET = [0.11, 0, 0.10]` m in the chassis frame):
   ```python
   sim = prism_track(sim_pose, PRISM_OFFSET)
   ```
   z here is the world-frame prism height, which rises both when the chassis
   climbs and when it pitches over the box — apples-to-apples with the real prism.

2. **Zero x, y at spawn; zero z at the settled baseline** (`common_box.py:106-112`):
   ```python
   sim = sim - sim[0]                          # x,y relative to spawn (no transient)
   lo = int(round(Z_SETTLE_LO / sim_dt))       # 0.3 s
   hi = min(int(round(Z_SETTLE_HI / sim_dt)), sim.shape[0])  # 1.0 s
   if hi > lo:
       z_baseline_rel = float(np.median(sim[lo:hi, 2]))
       sim[:, 2] -= z_baseline_rel             # rebase z on settled flat-ground height
   ```
   The key asymmetry: **x, y use `sim[0]`; z uses the median over [0.3, 1.0] s.**

3. **Score normally.** Cross-correlate forward-x to find the stream-zeroing time
   shift (`best_time_shift`, z-independent), interpolate sim onto the real
   timestamps, take `dz = sz - rz`, and `z = sqrt(mean(dz²))`. Same for x, y; the
   combined metric folds yaw in via the lever arm.

## Why the [0.3, 1.0] s window

- **After 0.3 s** — the spawn-drop transient is over; the chassis rests on its
  wheels.
- **Before 1.0 s** — the robot has travelled only ~0.02–0.11 m, far short of the
  box near-face (~1 m away), so it is still on flat ground. The median z over
  this window *is* the flat-ground resting height, the physically correct z=0.
- **Median, not mean** — robust to residual jitter/bounce in the window.

Constants live in `common_box.py`:
```python
Z_SETTLE_LO = 0.3  # s
Z_SETTLE_HI = 1.0  # s
```

## Same code, different effect across engines

`score()` is defined **once** and called identically by every engine:
`sweep_ostrich.py:65`, `sweep_mujoco.py:175`, `sweep_semi_implicit.py:68`. No engine
has its own z handling. The engines differ only in how they produce the raw
`pose`; from the pose onward the pipeline is byte-for-byte identical — this is the
apples-to-apples guarantee.

The rebase subtracts each engine's *own* settled height, so the amount removed
depends on how much that engine settles:

| Engine          | spawn-settle        | rebase removes | net z effect              |
|-----------------|---------------------|----------------|---------------------------|
| MuJoCo          | ~2 cm (soft contact)| ~2 cm offset   | z RMSE 25.3→19.2 / 20.5→15.1 mm |
| Ostrich           | ~0 (stiff contact)  | ≈0             | unchanged (14.1 mm)       |
| Semi-Implicit   | ~0                  | ≈0             | unchanged (37.9 mm)       |

A single uniform rule ("z=0 is your settled flat-ground height") that happens to
be a no-op for Ostrich/SI and only bites MuJoCo's spurious spawn-drop. No engine
gets special treatment.

`plot_results.py` reads the already-rebased `sim_rel` straight from the jsons
(`plot_results.py:92`), so the figure and the stored scores share the same
corrected z — the plot does no separate z handling.

## Caveat / alternative

This assumes the flat-ground resting height is the right z=0 for all engines,
which matches how the real prism z was zeroed (relative to its settled start). The
alternative is spawn-relative for everyone (`sim = sim - sim[0]`, artifact
included) — but then MuJoCo is penalised for a spawn-height *convention*, not its
climb dynamics. We chose the settled baseline ("option 1").

## Context: the turning-config decision (entangled with these numbers)

The current MuJoCo config is the **turning** config, adopted before the z fix:

- The earlier auto-tuner cranked torsional friction to `tor=10` because the
  position-only metric rewarded suppressing yaw — MuJoCo drove nearly straight
  even though the real robot clearly turns over the box.
- A finer `tor` grid (see `diag_mujoco_yaw.py`) showed `tor≈0.3` (capsule wheels,
  `condim=6`) recovers the real drift magnitude. Both engines still drift the
  *wrong sign* (+y) vs the real (−y) box-crossing kick, but the magnitude is
  right and the chassis actually rotates.
- We adopted `tor=0.3` capsule as canonical MuJoCo. Ranking became
  **Ostrich < MuJoCo < Semi-Implicit** (vs the old MuJoCo "win" that came from
  suppressing the turn).

Current MuJoCo best params:
`dt=0.002, kv=1000, mu=1.2, tor=0.3, solref0=0.005, condim=6,
integrator='implicitfast', wheel_geom='capsule'`.

## Final numbers (after z fix, turning config)

Combined error (position + yaw, L=0.5 m), averaged over the 2 runs:

| Engine          | combined | yaw RMSE | z RMSE (18_10_33) | config                          |
|-----------------|----------|----------|-------------------|---------------------------------|
| Ostrich           | 0.062 m  | 3.4°     | 14.1 mm           | μ_r=1.2, ke=150, Δt=0.05        |
| MuJoCo          | 0.065 m  | 3.0°     | 19.2 mm           | μ=1.2, tor=0.3, Δt=0.002        |
| Semi-Implicit   | 0.110 m  | 3.6°     | 37.9 mm           | μ=0.05, ke=80000, Δt=0.0005     |

## Files touched

- `common_box.py` — added `Z_SETTLE_LO/HI` constants and the settled-baseline
  z rebase in `score()`.
- `results/sweep_ostrich.json`, `results/sweep_mujoco.json`,
  `results/sweep_semi_implicit.json` — rescored with the new z baseline.
- `results/box_sim_to_real.png` — regenerated; copied to
  `ostrich_paper/figures/box_sim_to_real.png`.

## Still open

- Commit the `common_box.py` z-fix + updated jsons + figure (not yet committed).
- `ostrich_paper/sections/04_experiments.tex` prose may still cite the old
  MuJoCo-best ~54 mm ranking, which now contradicts the figure (Ostrich is best).
