# Genesis Differentiable Simulation — Backward Findings (post PR #2742)

Companion document to `GENESIS_BACKWARD_BUG.md`. That file documents the **original ABD hang** in Genesis 0.3.8 – 0.4.6 that was *fixed* by PR [#2742](https://github.com/Genesis-Embodied-AI/Genesis/pull/2742). This file documents what we found about Genesis differentiable simulation **after** PR #2742 — which removes the hang and produces correct gradients for simple cases, but reveals a separate structural limitation in how velocity-control gradients propagate through articulated kinematic chains.

Tested at PR #2742 head `d6036bd` with `quadrants==0.7.7` on Linux + CUDA 12.8 + RTX A500.

---

## TL;DR

| Gradient path | Result |
|---|---|
| Backward through pure ABD (no contacts) | ✅ Fixed — CPU + CUDA both work, correct gradients |
| Backward through contact + **initial-condition** path | ✅ Works — `O(1)` magnitude, dynamics-aware |
| Per-step velocity → **same-DOF** position (e.g. chassis linear vel → chassis pos) | ✅ Works — `O(1e-2)`, all steps contribute, contacts present or absent |
| **Per-step velocity → coupled-DOF position** (e.g. **wheel** rotation vel → chassis pos via articulation/friction) | ❌ `O(1e-5)` noise, **last-step exactly zero**, contacts present or absent |

The cell that's broken is exactly the gradient path used by every gradient experiment in the ostrich paper.

---

## Test 1 — CUDA backward (the original ABD-hang bug, post-fix)

Setup: single free body, 5 steps, per-step `set_dofs_velocity(ctrl)` with grad-tracking. (`pr2742_cuda_repro.py`)

| Backend | Result |
|---|---|
| Released Genesis 0.4.6 + CUDA | ❌ JIT compilation hang (the original `#2537` bug) |
| PR #2742 head `4696cc8` + quadrants 0.7.5 + CUDA | ❌ `CUDA_ERROR_ILLEGAL_ADDRESS` at `kernel_forward_velocity.grad` `cuLaunchKernel` |
| PR #2742 head `d6036bd` + quadrants 0.7.7 + CUDA | ✅ Backward completes, `ctrl.grad = [-0.0015, 0, 0, 0, 0, 0]` matches CPU |

Both originally-reported failure modes resolved by the dynamic-loop refactor and the zero-copy-to-Quadrants migration (PR #2748). Quadrants `>=0.7.7` is required (0.7.5 lacks `MatrixTensor.has_grad`, breaks at `scene.reset()`).

---

## Test 2 — Initial-condition gradient through contacts

Setup: `gs.morphs.Box` falling from `pos=[0, 0, 0.5]` onto `gs.morphs.Plane` under gravity, dt=0.005, 50 steps. CPU. `enable_collision=True`, `disable_constraint=False`. Gradient of `(final_pos - target)²` w.r.t. `init_pos`.

```
Initial pos: [[0.0, 0.0, 0.5]]
After 50 steps: [[0.0, 0.0, 0.187]]   ← fell + impacted + came to rest
Backward OK (23.9 s)
init_pos.grad = [-0.40, 0.00, 0.27]
  norm = 4.85e-01
  trivial loss-only grad = [-0.40, 0.00, 0.90]
  Equal? False  ← dynamics + contact contributes real signal
```

**Verdict: works.** Gradient through impact dynamics is `O(1)` magnitude and dynamics-aware.

---

## Test 3 — Per-step velocity to a same-loss-DOF (sanity baseline)

Setup: single free body (no wheels), MJCF freejoint, per-step `set_dofs_velocity` setting the **linear DOFs** (0,1,2) that directly determine chassis position. dt=0.005, T=20, gravity on. Tested with and without a ground plane.

```
=== A: No ground (no possible contact) ===
final_pos:  [0.20, 0.00, 0.495]
loss:       0.2461
grad_norm:  2.22e-02
first step: [-3.00e-03, 0.00, 3.95e-03]
last step:  [-3.00e-03, 0.00, 3.95e-03]  ← non-zero, all steps contribute equally

=== B: With ground (but box doesn't actually contact it) ===
[identical to A, bit-perfect]
```

**Verdict: works perfectly when the controlled DOF appears directly in the loss.** This is the "good baseline" — what a working per-step gradient looks like.

---

## Test 4 — Per-step velocity to coupled DOFs (the broken case)

Setup: Helhest chassis + 3 hinge-jointed wheels, dt=0.005, T=20. CPU. Per-step `set_dofs_velocity` to **wheel** DOFs (6, 7, 8) — i.e. wheel rotation rates. Loss is `(chassis_pos - target)²`. Chassis position only changes via the mass-matrix coupling from wheel rotation (no contacts) or wheel-ground friction (with contacts).

This is exactly the pattern of `experiments/3_gradient_quality/optimize_ostrich.py`: spline → per-step wheel velocity trajectory → chassis pose.

```
=== A: No ground (wheel rotation → chassis pos via mass coupling only) ===
chassis pos: [3.6e-10, 0.00, 0.318]
loss:        0.2527
grad_norm:   4.03e-05    ← ~3 orders of magnitude smaller than Test 3
first step:  [-2.26e-05, -2.26e-05, -2.26e-05]
last step:   [0.00, 0.00, 0.00]   ← structurally zero

=== B: With ground (wheel rotation → chassis pos via friction) ===
chassis pos: [0.018, -7.5e-08, 0.360]   ← chassis actually moved 1.8 cm via friction
loss:        0.2321
grad_norm:   8.60e-06    ← even smaller; contacts attenuate further
first step:  [1.23e-06, 1.23e-06, -8.17e-06]
last step:   [0.00, 0.00, 0.00]   ← structurally zero
```

**Verdict: broken.** Two structural anomalies:
1. **Gradient magnitude is 3–4 orders of magnitude smaller** than Test 3 even though both are 20-step per-step velocity setups on the same simulator. The forward simulation IS exercising the wheel→chassis pathway (chassis moves 1.8cm horizontally in case B), but backward isn't propagating the sensitivity.
2. **Last-step gradient is exactly `[0, 0, 0]`**, not small-but-finite. This is a kernel-level no-op, not floating-point noise.

These numbers are **bit-identical** across two PR head versions (`4696cc8` and `d6036bd`) separated by an active refactor cycle. Not a transient bug — a deterministic structural property of the current backward implementation.

---

## Diagnosis

The broken path is **per-step `set_dofs_velocity` to child DOFs whose effect on the loss is mediated by the kinematic chain** (mass-matrix coupling and/or contact friction). Contacts are *not* the root cause — case 4A has no contacts and still fails. What's broken is the backward pass through the articulation Jacobian when child-DOF velocities are set per-step inside the time loop.

Working paths:
- Initial-condition gradient through contacts (Test 2)
- Per-step velocity to a DOF that *directly* appears in the loss (Test 3)
- Backward through pure articulated dynamics with no per-step control (this thread, earlier tests)

Broken path:
- Per-step velocity to a *child* DOF, loss is a function of a *parent* DOF (Test 4)

This last case is the one needed for control-trajectory optimization of articulated robots, which is what every paper-experiment optimizes.

---

## Implications

### For Genesis users doing control-trajectory optimization

Genesis (post-PR #2742) is **not suitable** for gradient-based optimization of per-step control trajectories applied to actuated child joints (wheels, fingers, etc.) when the loss is measured on a parent body. Tasks that won't work:

- Trajectory optimization of a wheeled robot to reach a target pose
- Inverse-dynamics / control-fit problems against real-robot rollouts
- Receding-horizon MPC where the cost is on a body-frame state

Tasks that *do* work:
- Initial-state gradient through impact dynamics
- Forward simulation at any complexity (already used as `sweep_genesis_blender.py` in Experiment 2)
- Trajectory optimization where the parameter and the loss DOF coincide (e.g. push a free body to a target)

### For our paper (`ostrich`)

Every gradient experiment optimizes a per-step wheel-velocity trajectory and reads chassis pose — exactly the broken cell. Genesis cannot be a fair baseline for:

- `1_sim_to_real` — parameter fit, requires backward through trajectory
- `3_gradient_quality` — K-knot spline → per-step velocity → chassis pose
- `5_terrain_traversal` — waypoint optimization

Genesis remains useful for `2_dt_stability` as a forward-only visualization tool, where it's already integrated as `sweep_genesis_blender.py`.

**Recommended paper framing:** Document Genesis as "tested but excluded from gradient experiments" with a citation back to this evidence file.

---

## Environment

| | |
|---|---|
| OS | Arch Linux, kernel 6.19.11 |
| GPU | NVIDIA RTX A500 Laptop |
| CUDA | 12.8, driver 590.48.01 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.12 |
| Genesis | PR #2742 head `d6036bd` |
| `quadrants` | 0.7.7 (PyPI) |

## Files

- `pr2742_cuda_repro.py` — original CUDA-hang repro; **now passes**, preserved for regression detection.
- `phase0_cpu_contacts.py` — subprocess harness for early Phase 0 tests (drop / wheels). The wheels-PARTIAL there had MJCF state-read confusion; the cleaner Test 4 above uses `state.pos[0]` and confirms the structural limitation.
- `GENESIS_BACKWARD_BUG.md` — historical record of the original ABD-hang bug (now fixed).
