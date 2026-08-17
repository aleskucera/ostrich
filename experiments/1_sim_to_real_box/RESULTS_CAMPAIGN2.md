# Campaign-2 generalization results (2026-08-17)

Four new real runs (`ostrich0..3`, Odin lidar-inertial localization, mcap bags)
replayed in all three engines at the **frozen campaign-1 calibrated
parameters** — a pure generalization test of the contact calibration on new
trajectories, new start poses, a re-measured obstacle, and 3x speed range
(0.12–0.45 m/s). Produced by `eval_campaign2.py`; overlays in
`results/eval_campaign2.png`.

## Setup

- **Obstacle**: EUR pallet (1.2 x 0.8 x 0.144 m), per-run pose fitted from the
  lidar clouds (`fit_pallet.py`: sharpest-edge RANSAC yaw + density-coverage
  translation; edge RMS 3–7 mm). Independent validation: the real climb onset
  lands on the fitted near face to ±1.5 cm on all four runs.
- **Frame registration**: the real `base_link` rides **+0.23 m ahead** of the
  sim chassis origin (front axle), measured by registering sim-vs-real climb
  onset against the fitted near face (spread ±0.015 m over 4 runs). Carried
  by `prism_offset` (sim tracks that point; scene box shifts by it).
- **Motor tracking**: the only re-fit quantity (the LLC firmware changed
  2026-07-27). One scalar command scale per engine, calibrated on the pre-box
  flat cruise only: sim pre-box speed matched to real (identical smoothing
  pipeline). Scales: Ostrich 0.936, MuJoCo 0.945, SI 0.939. Note: Ostrich and
  SI already reproduce most of the real ~6 % tracking deficit through their
  own actuator models (finite k_p / k_d_act), so the scale is a wash for them;
  MuJoCo's stiff kv=1000 servo needs it (0.151 -> 0.144 m).

## Combined error (position RMSE + yaw lever, m) — motor-calibrated pass

| run      | Ostrich | MuJoCo | SemiImplicit (frozen) | SI (dt/2 rescue) |
|----------|---------|--------|-----------------------|------------------|
| ostrich0 | 0.147   | 0.142  | 0.501                 | 0.223            |
| ostrich1 | 0.254   | 0.194  | 31.96 (diverged)      | 0.189            |
| ostrich2 | 0.110   | 0.064  | NaN (diverged)        | 0.134            |
| ostrich3 | 0.248   | 0.178  | 0.292                 | 0.197            |
| **mean** | **0.190** | **0.144** | **diverges 2/4**  | **0.186**        |

Pass-1 (fully frozen, cmd_scale=1.0) means: Ostrich 0.193, MuJoCo 0.151,
SI 11.5 (diverges), SI-rescue 0.155.

## Findings

1. **Ostrich and MuJoCo generalize**: ~0.14–0.19 m mean over 17–21 s runs of
   4–5 m including the pallet climb, ~2.5–3x their campaign-1 fitted error
   (0.062 / 0.065 m) — no divergence, x(t) near-perfect on every run.
2. **SemiImplicit's calibration does not transfer**: at its frozen (already
   fragile) config it blows up at box impact on 2 of 4 runs, with chaotic
   run-to-run sensitivity (a run stable in one pass diverges in the other at
   a 6 % command change). Halving dt (2.5e-4, params otherwise frozen)
   restores stability and lands at 0.155–0.186 m — i.e. the accuracy
   transfers but the *stability* does not; the explicit contact stiffness
   tuning is scene-fragile.
3. **Residual error structure** (both implicit engines): (a) post-pallet
   lateral drift — the real robot picks up ~0.3 m of rightward drift crossing
   the pallet on the two fast runs, which both sims underpredict; (b) a
   pallet-top pitch phase — sims hold a nose-up plateau (rear wheel still on
   ground; wheelbase 0.75 m vs 0.8 m pallet depth is knife-edge), while the
   real robot rides level at z≈0.13 m (soft tires climb earlier, pallet
   presses into the grass). z RMSE stays ≤ 0.04 m everywhere.

## Files

- `eval_campaign2.py` — the harness (frozen params, motor calibration, both
  passes saved). `results/eval_campaign2.json` (untracked, 13 MB: includes
  trajectories), `results/eval_campaign2_si_rescue.json` (dt/2 rescue).
- `plot_campaign2.py` -> `results/eval_campaign2.png`.
- GT JSONs in `data/` (untracked): box pose from `fit_pallet.py`,
  `prism_offset=[0.23,0,0]` frame registration.
