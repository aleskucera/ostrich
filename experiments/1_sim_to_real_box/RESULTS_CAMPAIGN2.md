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

## Yaw analysis and motor/turn-resistance identification (ident_motor_turn.py)

The residual error is heading: the commands carry large differential content
(ideal skid-steer predicts 16–32° net yaw), the real robot executes ~30 % of
it, and at frozen params the sims execute almost none
(`results/eval_campaign2_yaw.png`). The causal chain, none of it identifiable
from campaign-1's near-straight runs:

1. At k_p=250 the sim wheels do not track the commanded differential at all
   (−3 %): the soft velocity servo lets the yaw-resisting load drag both
   wheels to a common speed. `mu_rear` is a dead knob in this regime.
2. A stiff servo (k_p=4000, ~80 % differential tracking) makes the sim
   OVER-turn — turn resistance must then come from friction.
3. Anisotropic wheel friction decouples it: longitudinal grip stays at the
   campaign-1 values (0.8/1.2), lateral (skid) mu is identified from
   turn-content data. Isotropic refits can't do this without corrupting the
   drive behavior.

Train on ostrich0+1, test on ostrich2+3 (contact params frozen):
identified **k_p=4000, mu_lateral=2.0, compliance.friction=1e-3** → train
0.16 m, **test 0.13 m, all-4 0.15 m, yaw RMSE 2.2–2.4°** (frozen baseline
0.19 m / 3.7°). Before/after heading traces:
`results/eval_campaign2_yaw_ident.png`. MuJoCo would need the symmetric
treatment (its kv=1000 is already stiff; its yaw response is also low) — not
done yet.

**MuJoCo, symmetric treatment:** same train/test protocol over its turn
knobs (rear sliding mu x torsional; ground/box mu dropped to 0.4 so the
wheel values bind under MuJoCo's elementwise-max friction combine). The
identification CONVERGES BACK to the frozen campaign-1 config (rear mu=1.2,
tor=0.3) — every lower-resistance setting over-turns and scores worse.
MuJoCo's stiff kv=1000 servo meant it already sat at the constant-mu plateau
that Ostrich needed the k_p + anisotropic-lateral fix to reach.

**Final identified head-to-head (held-out ostrich2+3 / all-4):**

| engine  | test mean | all-4 mean | yaw RMSE |
|---------|-----------|------------|----------|
| Ostrich (k_p=4000, aniso lat=2.0, fc=1e-3) | 0.133 | ~0.15 | 2.2–2.4° |
| MuJoCo (= frozen c1)                        | 0.122 | 0.144 | 2.8°     |

Statistically indistinguishable at n=2 test runs with ±0.01–0.02 run
structure: both engines land on the same constant-efficiency floor.

**Why the remaining under-turn is structural, not a parameter:** the real
per-run turn efficiency (real yaw regressed on the ideal skid-steer
prediction) is 0.11 / 0.21 / 0.13 / 0.30 across the four runs — a 3x spread
correlated with driving speed (grass-tire lateral resistance falls with slip
speed). Constant-mu Coulomb friction realizes exactly one efficiency, and
every knob (front/rear lateral split, torsional, k_p up to 12000, friction
compliance) saturates at the same ~0.13–0.16 m plateau. The identified sim's
yaw RMSE (2.2–2.4°) already equals the residual of the best per-run linear
gain fit (1.0–3.0°) — the constant-efficiency noise floor. Going further
needs slip-speed-dependent (Stribeck-like) lateral friction in the engine.

## Files

- `eval_campaign2.py` — the harness (frozen params, motor calibration, both
  passes saved). `results/eval_campaign2.json` (untracked, 13 MB: includes
  trajectories), `results/eval_campaign2_si_rescue.json` (dt/2 rescue).
- `plot_campaign2.py` -> `results/eval_campaign2.png`.
- GT JSONs in `data/` (untracked): box pose from `fit_pallet.py`,
  `prism_offset=[0.23,0,0]` frame registration.

## Stribeck velocity-dependent friction: breaking the constant-mu plateau (2026-08-17)

Engine feature (branch `stribeck-friction`, merged): per-shape
mu_stiction_scale / v_stribeck; mu multiplied by 1+(scale-1)*exp(-|v_t|/v_s)
inline in the friction residual (Picard in Newton; adjoint picks up d(mu)/dv
via tape-VJP automatically; sentinel-gated, bit-exact off).

Identification on the 14-run dataset (train ostrich0/1/10/13, test = other
10, mu_long pinned 0.8/1.2, cmd_scale 0.937), `ident_stribeck.py`:

| config | test comb | all-14 comb | all-14 yaw |
|---|---|---|---|
| constant-mu best (k_p=10000, mu_lat=0.8) | 0.202 | 0.236 | 7.37 deg |
| **Stribeck best (k_p=4000, mu_lat=0.4, scale=4.0, v_s=0.3)** | **0.197** | **0.189 (-20%)** | **5.65 deg (-23%)** |

The winning corner is exactly the predicted shape: LOW base lateral mu (0.4,
governs turns at slip speed) with STRONG low-speed stiction (x4 = 1.6
effective at rest, matches the straight-run constant-mu identification) and
v_s=0.3 m/s — i.e. the model now spans the measured 0.11-0.30 per-run turn
efficiency that no constant mu could. Gains come almost entirely from yaw.

Note: the replay path leaks ~35-40 MB GPU per simulator construction;
`ident_stribeck.py` isolates each config in a subprocess
(`_stribeck_worker.py`) as a workaround. Leak is a separate cleanup TODO.
