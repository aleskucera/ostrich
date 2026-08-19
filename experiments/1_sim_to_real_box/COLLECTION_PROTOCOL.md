# Box-traversal data collection protocol (campaign 2, 2026-08)

Pre-registered 2026-08-15, before collecting the new runs. Addresses Reviewer 2's
sample-size objection (>=5 trials) and the run-selection transparency gap
(previously 2 of 8 runs were used with the exclusion rationale in prose only).

## Protocol

1. **>= 6 traversals** of the 16 cm box (target: 5 clean after the quality
   gate; collect extras rather than re-judging marginal runs).
2. Identical logging to campaign 1: joint setpoints + total-station prism
   track, synced into `~/rosbags_experiment/synced/run_*.h5` so the existing
   `prepare_gt.py` pipeline applies unchanged.
3. Better localization per the new setup; keep the prism mount and
   `PRISM_OFFSET` unchanged (or record the new offset alongside the runs).
4. Optional but valuable: 2-3 traversals of a second configuration (different
   box height or approach angle) as a generalization check.
5. Drive style: same class of maneuver as campaign 1 (approach, climb,
   descend, continue); no mid-run stops.

## Clean-run criterion (pre-registered, coded)

`prepare_gt.py` stamps every converted run with a machine-checked verdict
(`gt["quality"]`), printed at conversion time. A run is CLEAN iff all of:

| check | threshold | catches |
|---|---|---|
| prism sample coverage | >= 90% | tracking loss |
| max tracking gap | <= 0.85 s | total-station dropouts |
| max yaw step | <= 0.5 rad | heading fold artifacts |
| x progress | >= box far edge + 0.3 m | incomplete traversal |
| z climb peak | >= 0.06 m | did not actually climb |
| mean abs wheel cmd | >= 0.5 rad/s | idle/aborted run |

Thresholds calibrated on campaign 1 before campaign 2 (good crossers show
0.75-0.79 s climb occlusion at ~0.94 coverage; the historically excluded runs
sit at 0.81-0.88 coverage). Campaign-1 verdicts under this gate: 18_10_33
CLEAN; 17_55_19 (never climbed), 17_56_52/17_59_53/18_04_51/18_07_11 (never
crossed), 17_58_43 (coverage), 18_09_11 (coverage+gap) REJECTED — i.e.,
campaign 1 provides exactly one gate-clean run, which motivates this campaign.

Selection rule: use ALL clean runs (no further discretion). If fewer than 5
runs are clean, collect more rather than relaxing thresholds. Threshold
changes after seeing data must be reported as such.

## After collection

1. `python experiments/1_sim_to_real_box/prepare_gt.py --all`
2. Re-run the three engine sweeps (`sweep_ostrich.py`, `sweep_mujoco.py`,
   `sweep_semi_implicit.py`) over the clean set; report per-run and aggregate
   combined error (yaw-aware metric).
3. Update Fig. sim-to-real + Sec IV-A with n = number of clean runs; report
   the rejected runs and which check failed.
