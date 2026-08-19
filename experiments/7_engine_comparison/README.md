# Engine comparison: Ostrich vs AGX Dynamics vs Project Chrono on helhest_junior

Quantifies strengths and weaknesses of the Ostrich solver against two industrial
engines — Algoryx AGX Dynamics 2.42.1.0 (hybrid direct/iterative, inside the
`agx-env` distrobox container) and Project Chrono 9.0.1 (NSC complementarity,
micromamba env at `~/.local/opt/chrono-env`) — on ONE robot: helhest_junior
(3-wheel skid-steer, 106.2 kg, `examples/helhest_junior/common.py`).

## Ground truth

Recent field bags from `~/projects/helhest_stack/bags/` (Aug 2026, flat-ish
outdoor, one surface), pose from the odin lidar-inertial unit (14.5 Hz, gap-free),
wheel commands from `/joint_setpoints` (100 Hz, already in sim order
[left, right, rear], forward-positive, tracking gain ~1.0):

| bag | role | content |
|---|---|---|
| fast_experiment0 | tune/eval | 339 s, 264 m, incl. turn-in-place |
| fast_experiment1 | tune/eval | 125 s, 92 m, aggressive turning |
| calibrate | tune/eval | 125 s, arcs/spins, +1438° net yaw |
| motors0 | **holdout** | 284 s, 211 m — never touched during tuning |
| steps_air | actuator ID | in-air setpoint steps → τ ≈ 0.10 s, gain ≈ 1.0 |

`prepare_gt.py` (run with the helhest_stack venv — needs `rosbags`) converts bags
to `data/gt_*.json`; `actuator_id.py` fits the motor response. The measured
actuator lag is identical hardware for every engine, so the bridge pre-filters
commands with the first-order τ = 0.10 s once, host-side (`common.prepare_commands`).

## Metric

Open-loop replays over multiple minutes diverge in every engine, so a whole-run
alignment mostly measures compounding luck. `common.window_score` cuts each run
into 15 s windows, re-anchors the sim to the real pose at each window start
(SE(2) + z offset), and scores position + yaw RMSE per window
(`combined = sqrt(pos_rmse² + (0.5 m · yaw_rmse)²)`); a bag's score is the mean
over windows, a config's score the mean over the three GT bags. Sim chassis frame
= front-axle center at hub height = the real robot's `base_link` (verified
against helhest_stack `RobotParams`), so both tracks follow the same body point.

## Layout

```
prepare_gt.py, actuator_id.py     bag → JSON conversion (helhest_stack venv)
common.py                         GT loading, command prep, window scoring
engines/bridge.py                 JSON job/result subprocess bridge
engines/chrono_replay_junior.py   pychrono runner (chrono-env python)
engines/agx_replay_junior.py      AGX runner (inside agx-env distrobox)
engines/ostrich_runner.py         in-process ostrich runner (this venv, GPU)
engines/verify_runner.py          smoke + tilt-probe + replay checks per engine
sweep.py <engine>                 axes A+B: staged sweeps, defaults + tuned + holdout
run_speed_dt.py <engines>         axis C: accuracy + real-time factor vs dt
run_scenarios.py <engines>        axis D: step16 / turn_radius / rock_field
plot_results.py                   regenerates results/summary.md + figures
```

All numbers in reports are regenerated from `results/*.json` by
`plot_results.py` — nothing is hand-copied.

## Robot model parity

All three engines share the spec of `examples/helhest_junior/common.py` (chassis
= two boxes 78.8375 + 10.8625 kg, wheels r 0.35 m 5.5 kg, track 0.73 m,
wheelbase 0.75 m, velocity-servo hinges about +Y) and self-check masses at
startup. AGX uses the existing `agx_helhest/demos/helhest_junior.agxPy` class
(cylinder wheels, AGX-derived inertia); Chrono ports the tier1 model (48-gon
hull wheels — Bullet cylinder-on-plane is a degenerate single contact point);
ostrich uses `create_helhest_junior_model` (cylinder collision).

## Verified engine findings along the way

- **AGX default friction creeps.** The stock `IterativeProjectedConeFriction`
  SPLIT solve lets the robot slide steadily down a 20° incline (~0.65 m/s at
  dt 0.01, dt-dependent) where static friction should hold; `DIRECT` friction
  holds exactly. This matches AGX manual §9.8's warning that SPLIT contacts give
  "viscous" friction on wheels — measured here with `verify_runner.py agx`.
- **Chrono NSC holds exactly** on the same probe, but `SetRollingFriction` must
  never be used (complementarity rolling friction locks chassis yaw — tier1
  finding, inherited).
- **All engines at defaults over-rotate ~2×** vs the real robot's turn gain
  (α ≈ 2 understeer from skid-steer scrub on this surface) — the dominant error
  term the friction sweeps must absorb.
- **No engine reproduces α ≈ 2 after tuning**: Chrono converges to α ≈ 1.1
  (over-turns), AGX ≈ 4.4 and ostrich ≈ 4.3 (under-turn) at their
  accuracy-tuned params — the turn response brackets the real robot from both
  sides depending on the friction formulation.
- **Ostrich stalls in the loose-rock field** (stops at the first rock row where
  AGX/Chrono push through) — consistent with its known multi-contact impact
  stress case (see `examples/helhest/obstacle_benchmark.py`). Its two scenario
  reps also differ visibly (GPU nondeterminism), while AGX and Chrono are
  bit-identical across reps.
- **Ostrich is the only engine whose accuracy *improves* with larger dt**
  (best window error at dt = 0.1 s), consistent with the implicit
  position-level formulation; the CPU engines run ~60× more steps per second
  but need small dt for their best accuracy.

## Wheel-terrain extensions (the second round)

Isotropic Coulomb on rigid ground is a representational limit for skid-steer:
it ties lateral to longitudinal force, pinning the turn gain far from the real
α ≈ 2. Four extensions tested (see `results/summary.md` for the table):

- **Anisotropic friction wins.** AGX oriented friction (`sweep_aniso.py agx`,
  chassis-frame primary direction, DIRECT solve) hits α ≈ 2.0 and holdout
  **1.54 m**; ostrich `mu_perp` (`sweep_aniso.py ostrich`) reaches α ≈ 2.5–2.8
  and holdout **1.72 m** — both roughly halving their isotropic-tuned error.
  Ostrich caveat: the contact **averages** wheel and ground μ per axis, so the
  ground μ must be lowered (0.2) for the effective lateral to reach the winning
  range; wheel-side longitudinal is boosted to compensate.
- **Chrono SCM** (Bekker/Janosi via the source build, `eval_scm.py`) is the best
  *unfitted* model for normal driving (fast0 2.53 m, holdout 2.37 m with the
  measured Janosi K) but stays at α ≈ 3.5 — the Mohr angle barely moves α, so
  the lateral resistance is dominated by shear-displacement dynamics.
- **agxTerrain** (`terrain: soil`) needs `agxTerrain.TerrainWheel` +
  `configureContactMaterial` — plain rigid wheels get ZERO traction on the soil
  surface. With it, straight-line slip is spot-on (10% ≈ the measured forward
  gain), but turn-in-place over-resists (α 6–17) and multi-minute replays
  diverge mid-bag. Not competitive without custom soil calibration.

## Qualitative axes (not in the tables)

- **Differentiability**: Ostrich only (analytic adjoint / warp tape). AGX and
  Chrono expose no gradients.
- **GPU batch**: Ostrich only (multi-world CUDA graphs; 8192 worlds on this
  robot in the older box experiment). AGX and Chrono are CPU per-process here.
- **Licensing / deployment**: Chrono is BSD open source; AGX needs a license
  file with per-launch online validation; Ostrich is in-house.

## Results

See [results/summary.md](results/summary.md) (generated).
