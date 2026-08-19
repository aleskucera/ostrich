# Helhest two-wheel balance: what we learned (2026-08-14/15)

Goal: make the robot stay on two wheels (rear-wheel wheelie at the ~47°
geometric balance pitch), learned via differentiable simulation with the
corrected adjoint (pose-VJP + true friction linearization, engine defaults).

## Result summary

| approach | outcome |
|---|---|
| Open-loop spline + gradients (any adjoint) | max ~0.86 s balance; curriculum with time-preserving warm starts + best-iterate handoffs NEVER extends past its warm start — the exponential-sensitivity wall of open-loop stabilization, not a gradient defect. |
| Hand PD, rear wheel only | zero authority: 18% alive at any gain, either sign. |
| Hand PD, all wheels (kp=40 on pitch err, kd=0.3–0.8 on pitch rate, clip 25 rad/s) | 100% alive at 3 s **but drives away** (8.4 m drift — an accelerating wheelie, not balance) and collapses by 6 s (80% alive, 30 m). Stability island: kp≥90 or kd≥3 destabilize. |
| Direct-term policy gradient (contract per-step control grads with features) | diverges once feedback gains matter — the omitted du/ds pathway IS the loop. |
| Exact closed-loop BPTT (policy state-Jacobian injected into the adjoint sweep, FD-verified) | correct signs, ~2x scale bias (adjoint-conditioning family); on 3 s horizons the closed-loop adjoint EXPLODES (|g|~1e7–1e9, Lyapunov growth) and training walks downhill from any good start. |
| + mitigations (1.5 s train horizon, inject-scale 0.3 damping, rot-dominant loss wrot=400/wpos=1, alive-gated best tracking) | **first genuine learning win**: PD-warm linear improves on the hand PD across all disturbance kicks — loss −14%, pitch RMS −16% (0.236→0.199 rad), 100% alive at 3 s. |
| Drift curriculum (wpos 1→5) and small residual MLP on top | saturate at the round-2 policy; the driving-wheelie local minimum holds; nothing survives 6 s yet. |

Loss-design trap discovered on the way: with threshold orientation loss (or
quadratic with high wpos), "fall early and stop" scores ≈ "balance while
driving away" — best-iterate tracking gets fooled by lucky fallen rollouts.
Fixed by making orientation dominate (wrot=400 quadratic, wpos small) and
gating best-acceptance on alive fraction.

## Files

- `helhest_balance_feedback.py` — closed-loop rollouts; linear / MLP /
  residual (PD-base + zero-init MLP) policies; exact closed-loop BPTT
  (`--exact-bptt`, `--inject-scale`); disturbance kicks (`--kick-std`,
  self-calibrating pitch-rate axis); `--eval-only` metrics (alive, drift,
  pitch RMS); `--replay` for the GL viewer.
- `helhest_balance_bundled.py` — spline formulation, now with `--duration`,
  time-preserving `--init-spline/--init-duration` warm starts, best-iterate
  `--save-spline`.
- Policies from the investigation live in the session scratchpad
  (`overnight/*.npz`); the best current controller is `r2_linwarm.npz`
  (learned refinement of PD kp=40/kd=0.3).

## Where the remaining problem lives

1. **Local minimum**: the accelerating wheelie is an attractor; reaching
   stand-still balance from it conflicts with pitch-hold along the damped
   gradient direction. Escape likely needs either undamped-but-stable
   gradients (see 2) or a formulation that removes the attractor: spawn AT
   equilibrium at rest, train pure disturbance rejection (kicks only), where
   u≈0 is near-optimal and drift pressure does not fight pitch.
2. **Closed-loop BPTT conditioning**: the honest fix for the explosion is
   truncated/segmented BPTT windows (stop-gradient every k steps) instead of
   scalar injection damping — unbiased within windows, bounded growth. ~50
   lines in the existing sweep.
3. Optionally richer actuation authority (torque-level control or higher
   clip) and wheel-speed features for active deceleration.

The machinery (exact closed-loop BPTT through hard contact, FD-verified) is
the durable outcome — first system-level use of the corrected adjoint for
policy learning, and the round-2 improvement proves end-to-end learning
works when horizon/conditioning are respected.
