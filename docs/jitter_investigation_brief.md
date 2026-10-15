# Brief: why does Helhest Junior jitter when it turns?

You are picking up an investigation that has already produced contradictory
measurements. Your job is to find the actual cause, and to reconcile or discard
the numbers below -- not to trust them. Where this brief states a number, treat
it as "someone measured this once"; re-derive anything load-bearing.

## The phenomenon

A three-wheel skid-steer (Helhest Junior: chassis + 3 driven wheels, wheel radius
0.35 m) cannot rotate without sliding its wheels sideways. When commanded to spin
in place it rotates in a visibly juddering, stick-slip fashion. Driving straight
on the same ground is smooth. Metric used throughout:

    yaw_rate   mean achieved yaw rate (rad/s)
    rate_std   std of yaw rate; a smooth rotation holds it near zero
    jerk_rms   rms yaw ACCELERATION (rad/s^2) -- the stick-slip signature

## Reproduce it in one command

    cd /home/kuceral4/projects/ostrich
    .venv/bin/python examples/helhest_junior/turn_jitter.py

Flat triangulated ground, no keyboard, scripted loop (settle / spin left / spin
right / forward / back), live yaw-rate and yaw-accel plots in the viewer, and a
per-phase summary printed to the terminal. One loop at dt=0.03:

    SPIN LEFT   yaw_rate -0.660  rate_std 0.665  jerk_rms 23.21
    SPIN RIGHT  yaw_rate  0.702  rate_std 0.673  jerk_rms 23.91
    forward     yaw_rate  0.002  rate_std 0.026  jerk_rms  1.05
    back        yaw_rate -0.006  rate_std 0.036  jerk_rms  1.65

Run it headless (much faster, no GPU/viewer issues) by constructing
`TurnJitterSimulator` with `RenderingConfig(vis_type="null")` and calling
`_run_simulation_segment()` + `clock.advance()` in a loop. See
`/tmp/claude-1000/-home-kuceral4-projects-ostrich/409dcadb-a353-4ab4-a18b-ede350bf1f57/scratchpad/`
for throwaway harnesses that already do this (`smoke_jitter.py`,
`mesh_vs_plane.py`, `sweep_jitter.py`) -- copy, do not trust.

## The contradiction you must resolve

Two harnesses, nominally the same robot / engine / command, disagree by ~80x on
how much the known knobs matter.

**Harness A** -- `ostrich-odinsim/examples/helhest_junior/odin_sim/turn_probe.py`
(runs on dasenka). Reported, on newton's analytic ground plane:

| configuration | yaw_rate | rate_std | jerk_rms |
|---|---|---|---|
| contact 1e-10 | -1.171 | 0.392 | 11.4 |
| contact 1e-8  | -1.127 | 0.387 |  9.4 |
| contact 1e-6  | -1.210 | 0.033 |  1.2 |
| 1e-10, newton_iters 32 | | | 20.5 (MORE iterations is WORSE) |
| 1e-10, dt 0.01 | | | 28.5 (SMALLER dt is WORSE) |

**Harness B** -- `ostrich/examples/helhest_junior/turn_jitter.py` (this repo),
averaged over both spin phases:

| configuration | jerk_rms |
|---|---|
| plane, compliance 1e-6  | 27.1 |
| plane, compliance 1e-10 | 29.2 |
| mesh,  compliance 1e-6  | 26.0 |
| mesh,  compliance 1e-10 | 24.7 |
| mesh,  1e-6, dt 0.02    | 33.3 |
| mesh,  1e-6, dt 0.01    | 39.7 |

In B, compliance does nothing and the ground representation does nothing. In A,
compliance is worth 10x. Both agree only that smaller dt is worse.

## Known differences between the harnesses (start here)

1. **Harness A's anisotropic friction was never applied.** VERIFIED: `set_wheel_friction`
   is defined in `odin_sim/sim.py` but never called from `__init__`, and
   `contact_probe.FlatProbe.build_model` calls `create_helhest_junior_model`
   passing only `friction_left_right` / `friction_rear` -- not `friction_long_*`.
   So every row in turn_probe labelled "aniso" was isotropic; what actually
   varied was lateral friction magnitude. Harness B *does* apply anisotropy
   (passes `friction_long_*` through, and the main repo's `common.py` sets
   `friction_axis_local` + `mu_perp` when `mu_long is not None`). This alone may
   explain the divergence -- check it first.
2. **k_p.** A uses 250 (odin_sim default). B uses 250 too (set in
   `examples/conf/helhest_jitter.yaml`), but the repo-wide
   `conf/control/helhest/velocity.yaml` is 150. k_p dominates turn rate: on a
   heightfield, 150 gave yaw_rate -0.13 / jerk 5.4 and 250 gave -0.84 / jerk 27.9.
   Note 150 "improves" jerk mostly by barely turning at all -- do not mistake
   that for a fix.
3. **dt and sampling.** A runs dt=0.02 sampled every 2 steps (0.04 s); B runs
   dt=0.03 sampled every step. jerk is a second difference of a sampled angle,
   so the sampling interval is part of the metric. Consider whether the two
   jerk_rms numbers are even commensurable, and prefer a sampling-invariant
   measure (e.g. band-limited yaw-rate power, or contact-force variance).
4. Engine configs were diffed and are otherwise identical (nr 16, linear 26,
   compliance joint 6e-10 / contact 1e-6 / friction 1e-8, contacts max_per_world
   256, cluster reduction max_per_pair 8, warm start on, linesearch off).

## Things already ruled out (re-check cheaply if convenient)

- Not the ground representation, in harness B: analytic plane vs triangulated
  mesh differ by <15% in jerk_rms.
- Not the triangle size: cell 0.06 / 0.12 / 0.25 give jerk 27.8 / 21.8 / 24.1.
- Not lateral friction magnitude, in harness B: 0.5/0.2 vs 0.8/0.4 gave 27.8 vs 26.3.
- Not the renderer or real-time pacing: the effect is identical headless.

## Where the code is

    ostrich/examples/helhest_junior/turn_jitter.py     harness B (flat mesh, scripted)
    ostrich/examples/conf/helhest_jitter.yaml          its config
    ostrich/examples/helhest_junior/control_world.py   same robot on helhest_stack worlds
    ostrich/examples/helhest_junior/common.py          create_helhest_junior_model, friction attrs
    ostrich/src/ostrich/constraints/contact_constraint.py
                                                       contact diagonal = (dphi_dlambda_n + compliance)/dt^2
    ostrich-odinsim/examples/helhest_junior/odin_sim/turn_probe.py     harness A
    ostrich-odinsim/examples/helhest_junior/odin_sim/contact_probe.py  harness A's FlatProbe

## Constraints

- Never add "Co-Authored-By: Claude" to commit messages.
- GL viewer: on this laptop OpenGL is deliberately pinned OFF the NVIDIA card
  (Xid 13 / "CUDA error 719"); see `docs/gl_viewer_gpu_contention.md`. The iGPU
  caps the viewer near 20 fps regardless of scene. Work headless.
- dasenka (`ssh dasenka`) has the odinsim tree at
  `/local/kuceral4/projects/ostrich-odinsim` and 2x RTX 3090. GPU 0 often has the
  user's own jobs -- check `nvidia-smi` before taking a card.

## What a good answer looks like

A mechanism, not a knob: which term in the solve is oscillating, why a turn
excites it and a straight line does not, and a prediction that holds in BOTH
harnesses once they are made to agree. If the two harnesses cannot be reconciled,
say precisely which one is wrong and why.
