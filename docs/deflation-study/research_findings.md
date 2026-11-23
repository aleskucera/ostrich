# Research findings

Prior work found while writing up the deflation study. Short notes, with
how each item relates to our setup and where it could go in
`latex-theory/`.

## AGX Dynamics (Algoryx): split direct/iterative solver

Researched 2026-10-25 from the AGX user manual and the Momentum docs. The
full papers have not been read yet.

**What AGX does**
- The default solver is a "split" solver that runs three passes per step:
  1. Direct: ordinary constraints (hinges, prismatics, …) and all contact
     *normal* forces.
  2. Iterative: *friction*, with the normal forces from pass 1 fixed. The
     docs describe Projected Gauss–Seidel (implied, not stated outright).
  3. Direct again: constraints and normals, with the friction forces from
     pass 2 fixed.
  The Momentum docs describe the order slightly differently ("all joints
  are first solved directly, and then in a last pass, the normal and
  frictional contacts are solved"), so the details vary with page and
  version.
- The docs claim residuals at least 100× smaller than the purely
  iterative solver, and stable mass ratios up to about 1:100,000.
- Direct solver: a block-pivot QP/LCP method ("similar to Newton–Raphson
  without damping or line search"). It runs on their own sparse
  symmetric-indefinite factorization (SABRE, BLAS-3). It is robust for
  high mass ratios and many joints, and slow for piles and
  volume-filling scenes.
- The solve type (direct, iterative, or both) can be set per joint and per
  contact material. The docs call "direct for everything, including
  contacts" the most stable and accurate setting, and recommend iterative
  for large stacking scenes.
- Foundations: Lacoursière's SPOOK stepper (Lacoursière & Linde), with
  regularized, compliant constraints plus a "stable direct-iterative
  splitting scheme". The regularization plays the role of our C in
  A = J M⁻¹ Jᵀ + C.

**How it relates to our setup**

| | AGX split solver | Our joint/control deflation (DPCR) |
|---|---|---|
| Solved exactly | joints + contact normals | the 18 joint/control rows |
| Solved iteratively | friction (PGS) | all contact rows, normals included (PCR) |
| Coupling | block Gauss–Seidel-style passes between separate solvers, each with the other's forces fixed | Galerkin projection inside one Krylov solve; PCR runs on the exact Schur complement of the joint block |
| Exact part exact for the coupled system? | only once the passes agree | yes, at every iterate (Zᵀr = 0) |
| Complementarity | inside the solvers (pivoting, projection) | in the outer Newton loop; each linear system is SPD |
| Direct alternative | sparse factorization of the full system | body-space Woodbury, a 24×24 Cholesky (step 2c) |

- **Same structural insight:** solve stiff, always-active articulation rows
  directly and contact rows iteratively. AGX (and Tomcin et al. 2014:
  "iterative for contacts, direct for articulation") reached it on
  engineering grounds. Our step 1c reached it from linear algebra:
  joint/control was the best cheap Z (19 → 7 iterations on obstacle,
  29 → 12 on surface).
- **Different coupling:** AGX alternates between solvers, i.e. a
  multiplicative block splitting. We fold the direct part into the Krylov
  method as a projection. This is a third option next to the projection
  (deflation) and additive two-level forms in chapter 3.
- **Different split of the contact rows:** AGX puts the normals in the
  direct part. In our data the friction rows have the extreme compliances
  (c from 1e-6 to 3e6), and these limit the accuracy of the body-space
  solve. A Z that also covers the normal rows is close to our
  "joint/control + k_c contact rows" variant (`deflation_rowsel.py`).
- **Same pull towards direct solves:** AGX defaults to direct for
  robustness. This is consistent with our steps 2c and 2d, where a direct
  body-space solve beats every PCR variant at Helhest scale and gives an
  accurate adjoint.

**Possible use in the write-up:** a related-work paragraph in §3 (hybrid
methods) or §4 (next to the joint/control Z source).

**Related papers to read before citing**
- C. Lacoursière, *Ghosts and Machines: Regularized Variational Methods for
  Interactive Simulations of Multibodies with Dry Frictional Contacts*,
  PhD thesis, Umeå University, 2007. The source of SPOOK.
- Peiret et al., "Schur Complement-based Substructuring of Stiff Multibody
  Systems with Contact", ACM TOG 2019. It combines a direct subsystem solve
  with an iterative interface solve, which is closest to our
  static-condensation reading of joint/control deflation.
- Tomcin et al., Eurographics 2014, on hard articulation constraints with
  contacts (direct for articulation, iterative for contacts).

**Sources**
- AGX user manual, overall structure:
  https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/overall_structure.html
- AGX, creating stable simulations:
  https://www.algoryx.se/documentation/complete/agx/tags/latest/doc/UserManual/source/creating_stable_simulations.html
- Momentum, stable simulations:
  https://www.algoryx.se/documentation/complete/Momentum/source/Stable%20Simulations.html
- Lacoursière, SPOOK / split solver report:
  https://webapps.cs.umu.se/uminf/reports/2011/009/part1.pdf
- Peiret et al. 2019: https://doi.org/10.1145/3355621

## Idea: body-space / constraint-space hybrid for better adjoint gradients

Discussed 2026-11-02. Tested offline 2026-11-12: see **Results** at the end
of this entry, which correct parts of the reasoning below (the role of the
subtraction, the ρ split rule, and which rows cause the amplification).

**Motivation.** The ostrich adjoint (`base_engine.step_backward`) computes
d = M̃⁻¹f, solves A w = J d with capped PCR and forms w_u = d − M̃⁻¹Jᵀw.
The subtraction amplifies the relative error of w by about the stiffness
ratio of the stiff rows (1/(mc) in the scalar example of §5.3), regardless
of which solver produced w. A full body-space solve removes the subtraction
(§5.3: 0% of draws > 10% error, vs ~40% for PCR on obstacle). However, it
needs c_i > 0 for every row, and a dense K costs (6N_b)³ once body–body
contacts fill it in. The hybrid moves only the rows that cause the
cancellation into body space.

**Construction.** Split the active rows into stiff rows S and soft rows F.
- Eliminate λ_S in body space. This gives a stiffened mass matrix
  M̂ = M̃ + J_Sᵀ C_S⁻¹ J_S (6N_b × 6N_b, SPD).
- Eliminate Δu. This gives a soft-row system A_F = J_F M̂⁻¹ J_Fᵀ + C_F
  (n_F × n_F), solved with PCR (matrix-free, one M̂ solve per product) or
  Cholesky.
- By Woodbury, A_F = A_FF − A_FS A_SS⁻¹ A_SF. This is **exactly the Schur
  complement that joint/control deflation (unit-vector Z) iterates on.** In
  the forward solve the hybrid and DPCR are the same Krylov method on the
  same matrix. They differ only in how A_SS⁻¹ is applied: DPCR uses a
  Cholesky factor of E = A_SS (n_S × n_S), the hybrid uses one of M̂
  (6N_b × 6N_b). The difference that matters is the adjoint formula.
- Hybrid adjoint: d̂ = M̂⁻¹f, A_F w_F = J_F d̂, w_u = d̂ − M̂⁻¹J_Fᵀ w_F,
  w_S = C_S⁻¹ J_S w_u. Multiplied out, w_u = K⁻¹f, so it is exact in exact
  arithmetic.

**Split rule.** Use the per-row stiffness ratio
ρ_i = (J_i M̃⁻¹ J_iᵀ) / c_i: the inertia part over the compliance part of
diag(A), both already computed during assembly. Rows with ρ_i ≫ 1 go to
body space; the rest stay in constraint space. Structure matters too.
Joints form a tree, and ground contacts touch only one body, so both keep
M̂ tree-sparse, which allows an O(N_b) factorization without fill-in
(Baraff-style). Body–body contacts close loops, so they are the natural
constraint-space rows. In Helhest every row is of the first kind, and the
hybrid reduces to full body space. Its regime is manipulation, piles and
self-contact.

**Hypothesized effect on gradient quality.**
- An error in w_F reaches the gradient as δw_u = −M̂⁻¹J_Fᵀ δw_F, so
  (relative error of w_u) ≈ ρ̂ × (relative error of w_F), with
  ρ̂_i = J_i M̂⁻¹ J_iᵀ / c_i ≤ ρ_i.
- Two effects: (1) the stiff rows leave the subtraction, because their
  forces come from a multiplication w_S = C_S⁻¹J_S w_u; (2) the soft rows
  become safer, because M̂ is "heavier" in the directions the stiff rows
  hold. In the scalar example with one stiff and one soft row on the same
  body, ρ̂_F ≈ c_S/c_F, which is below 1 if the split is right.
- With ρ̂ ≲ 1, the gradient error ≈ the PCR error, without amplification.
  Removing the stiff rows also conditions A_F well: joint/control DPCR
  needed 7–9 iterations, under the cap of 16. So PCR should converge to
  its tolerance, and **PCR vs Cholesky on A_F should make little
  difference.** Today both factors are bad at once: large ρ and capped
  PCR.
- With both parts direct, the hybrid is just K⁻¹f in another order: its
  gradient ≈ "body-space native". Then it is about cost and feasibility,
  not about better gradients than full body space.

**Solver choice per part.**
- M̂: always Cholesky. It is applied inside every PCR iteration, so an
  inexact inner solve would make the operator vary between iterations and
  require a flexible Krylov method. The adjoint reuses the factor.
- A_F: Cholesky if n_F is small, especially if it holds rows with large ρ̂
  that must stay there. PCR if n_F is large (piles, dense grasps).

**Open problems / limits.**
- Rows with c = 0 (truly rigid) cannot enter body space and have ρ = ∞.
  The hybrid does not fix them. One possible direction: project w_u onto
  the null space of the rigid rows instead of subtracting (not worked out).
- Relevant to Kučera's `exp/smoothed-gradients` branch (ca21b64). It
  lowers the adjoint's contact compliance to 1e-6, and says 0 is "exact".
  This raises ρ of the normal rows, so it probably worsens today's
  cancellation and pushes those rows towards c = 0.
- Prior art for the split itself: minimal coordinates with a Delassus
  matrix for contacts (Featherstone-style engines, Carpentier et al.) is
  the rigid limit of this split; AGX (direct joints, iterative contacts);
  Peiret et al. 2019 (substructuring); Baraff 1996 (linear-time Lagrange
  multipliers for trees). Not found in the literature: choosing the split
  by ρ_i to remove the adjoint cancellation, in maximal coordinates with
  compliant constraints.

**Cheapest test (offline).** Extend `test_scripts/deflation/adjoint_accuracy.py`
with the hybrid adjoint: put the joint/control rows (or the rows with ρ_i
above a threshold) into M̂, solve A_F in float32 with capped PCR and with
Cholesky, and add both as rows of the §5.3 table. Prediction: errors close
to "body-space native", and nearly equal for PCR and Cholesky on A_F. If
PCR and Cholesky differ a lot, ρ̂ is not small and the split threshold
needs tuning. A stronger test is to rescale the masses and compliances of
the recorded systems (mass ratio × stiffness ratio grid) and look for
regimes where full body space, the hybrid and constraint space each win.

**Possible use:** the core of a paper on "choosing the linear solver in
differentiable contact simulation", with the hybrid as the middle regime
that targets gradient quality (deflation only targets solve speed).

**Results (offline, 2026-11-12).** Script:
`test_scripts/deflation/adjoint_hybrid.py` (~20 s, CPU). The setup is the
same as in §5.3 / `adjoint_accuracy.py`: 25 systems per scene (the last NR
iteration of each step), 8 random loss vectors f each, all solvers in
float32 (the engine's precision). The reference is the float64 solution of
the same linear system, w_u = K⁻¹f. So this measures only the error the
linear solve adds to one adjoint step, not the true gradient of the
simulation (no linearization errors, no Newton truncation, no horizon, no
FD comparison). "Amplification" is the exact factor ‖M̂⁻¹J_Fᵀw_F‖/‖w_u‖.

Share of draws with > 10% error in w_u (median error in brackets),
obstacle scene:

| A_F solver | engine (S = none), amp ≈ 30 | hybrid S = joint/ctrl (18 rows), amp ≈ 2.5 |
|---|---|---|
| PCR cap 16 (engine today) | 42% (5.4e-2) | 2% (2.6e-4) |
| PCR cap 16 + 1 IR step | 38.5% (1.5e-2) | 0% (1.0e-5) |
| PCR cap 16 + 2 IR steps | 37.5% (1.1e-2) | 0% (8.4e-6) |
| PCR cap 32, no IR | 38.5% (1.1e-2) | 0% (3.0e-6) |
| PCR cap 48, no IR | 2.5% (3.3e-3; 42% > 1%) | 0% (3.0e-6) |
| Cholesky | 0% (8.2e-6) | 0% (2.3e-6) |
| Cholesky + 1 IR step | 0% (1.0e-7) | 0% (1.0e-7) |

Terrain scene, same rows: engine 25.5% → 4.5% → 4.0% | cap 32 4.0% |
cap 48 2.0% | Cholesky 0%; hybrid 2% → 0% → 0% | 0% | 0% | 0%. Other
splits (both scenes): S = all contacts (amp ≈ 10–12) and S = ρ > 1e2
(amp ≈ 7–8) give 0% with capped PCR. S = contact normals only changes
nothing (amp stays ≈ 30). Full body space (Cholesky of K) has median
2.8e-4 on obstacle, worse than the hybrid, because the sticking friction
rows put 1/c ≈ 5e5 into K.

Iterative refinement (IR) here works on the full adjoint KKT system
M w_u + Jᵀw = f, J w_u − C w = 0. The residual r1 = f − M w_u − Jᵀw,
r2 = −(J w_u − C w) needs only products (no C⁻¹, so it also works for
c = 0 rows). The correction is solved with the same split and solver and
added. A step contracts the error by η ≈ amplification × (relative
error of the inner solve), and converges only if η < 1.

Findings:
- **The subtraction alone is not the problem.** With an exact w (float32
  Cholesky of A), the engine's formula gives ~1e-5, and one IR step brings
  every variant to the float32 floor (~1e-7). This matches the known
  instability of the Sherman–Morrison–Woodbury formula and its known cure
  (see prior art below). The §5.3 sentence "The formula, not the solver,
  limits the accuracy" overstates it: the subtraction amplifies the solver
  error by ~20–60, and the capped solve combined with that amplification
  limits the accuracy.
- **IR does not rescue the engine under the iteration cap.** With capped
  PCR, η = 30 × (PCR error 0.1–0.4) > 1, so the corrections are as wrong
  as the original. IR is no better than giving PCR the same number of
  iterations, and worse than one long run (each IR step restarts PCR and
  discards its Krylov subspace, like restarted GMRES).
- **The hybrid makes the iterative regime work.** It lowers the
  amplification (30 → 2.5) and leaves a well-conditioned A_F (PCR
  converges in ~16–20 iterations, while the engine's A is not converged
  at 48). Result: 2% at the engine's budget, 0% with one IR step or cap 32.
- **Corrected split rule.** What matters is the share of the loss push a
  row group absorbs, not the per-row ρ_i. In Helhest the joints, which hold
  each wheel to the chassis in 5 of 6 directions, absorb most of a random
  push: moving the 18 joint/control rows cuts the amplification ~12×, more
  than any ρ threshold. Moving the contact normals does nothing. (A 1D toy
  with a single bar suggested the opposite, because the bar constrains only
  one direction.) The structural joint/control split is also the best in
  float32 precision: the joints' moderate c (≈ 0.011) keeps M̂ well
  conditioned, while the sticking friction rows (c ≈ 2e-6) are better left
  in constraint space, where a tiny c only adds to the diagonal.
- Caveat: the PCR helper stops at an absolute tolerance of 1e-5, so small
  IR corrections in the hybrid sometimes run 0 iterations. This understates
  IR there, and does not affect the engine rows (corrections ran all 16).

**Prior art for the amplification (found 2026-11-12).** The engine's
adjoint evaluates K⁻¹f = M̃⁻¹f − M̃⁻¹Jᵀ(C + JM̃⁻¹Jᵀ)⁻¹JM̃⁻¹f, i.e. the
Woodbury formula with base M̃, so the mechanism is known:
- E. L. Yip, "A note on the stability of solving a rank-p modification of
  a linear system by the Sherman–Morrison–Woodbury formula", SIAM J. Sci.
  Stat. Comput. 7 (1986) 507–513.
- B. Hashemi, Y. Nakatsukasa, "Instability of the Sherman–Morrison formula
  and stabilization by iterative refinement", arXiv:2510.01696 (2025). It
  gives the same mechanism: large intermediates and a small solution
  produce cancellation. Its fix is fixed-precision IR, where one step
  restores backward stability. Our data agree for direct solves.
- L. Shu et al., "Same Physics, Different Gradients: A State-of-the-Art
  Report on Differentiable Physics Simulation", arXiv:2611.12162
  (Nov 2026). It names convergence-aware adjoints (tolerance, conditioning,
  precision) as an open problem. In the part read, it does not discuss the
  dual (constraint-space) adjoint formula or this amplification.
- Not checked yet: whether Delassus-based differentiable engines (e.g. Le
  Lidec et al. 2024, Kamino) recover velocity gradients with the same
  subtraction. That decides how general the issue is.

**Updated publication view.** The amplification mechanism itself is not
novel, and for direct solvers the answer is "add one IR step". The open
problem is the budget-limited iterative adjoint: the textbook fix fails
(η > 1), and a structural split fixes it by making η small. Still missing:
the FD and optimization evidence inside the engine (a `src/` change), and
the check of other engines.

**Which simulators have the amplification, and when it hurts (2026-11-12).**
Checked from the papers: Simple (Le Lidec, Carpentier et al. 2024,
arXiv:2409.07107) and Nimble (Werling et al. 2021, arXiv:2103.16021) use
the same Woodbury-form structure as ostrich. Engines that solve the full
KKT system (Dojo, Belbute-Peres et al. 2018), work in primal space (MJX,
MuJoCo Warp, Brax), or differentiate solver iterations (Warp XPBD,
DiffTaichi, Qiao et al. 2021 reverse PGS) do not. Kamino is not
differentiable.

The structure is present when all three hold:
1. **The gradient is the derivative of the converged step solution**, not
   of the solver's iterations.
2. **The velocities are eliminated.** The engine solves a Schur-complement
   (Delassus) system for the constraint forces, then reconstructs the
   velocities or their derivatives as free motion + M⁻¹Jᵀ × (constraint
   quantity). Writing the system with constraint forces is not enough:
   solving the full KKT system jointly has no such subtraction.
3. **The derivative passes through that elimination.** The mode does not
   matter: ostrich does it in reverse mode (adjoint w_u = d − M⁻¹Jᵀw),
   Simple in forward mode (dv⁺/dθ = dv⁺/dθ|_λ − Δt M⁻¹J_cᵀ C A⁻¹ B(…)).
   The method does not matter either: reverse-mode AD of a forward pass
   λ = A⁻¹b, v = v_free + M⁻¹Jᵀλ produces the same adjoint formula.
   Rigid or compliant constraints do not matter (Nimble is rigid, ostrich
   compliant).

It damages the gradient only if, in addition:
4. **The constraint solve is inexact**: an iterative solver stopped by a
   cap or tolerance, low precision (float32), or both. Its relative error ε
   is what gets amplified.
5. **The amplification ‖M⁻¹Jᵀw‖/‖w_u‖ is large in the scene**: stiff
   constraints that absorb most of the loss push (joints in Helhest,
   clamping/sticking contacts). It depends on the scene and on the loss
   direction; ~20–60 on our systems.

The gradient error is then ≈ amplification × ε, and one IR step repairs it
only if that product is < 1.

| | 1 | 2 | 3 | 4 inexact solve | 5 large amp. | harmful? |
|---|---|---|---|---|---|---|
| Ostrich | ✓ | ✓ | ✓ adjoint | ✓ capped PCR, float32, GPU | ✓ joints | yes: 42% / 25% bad single-step gradients |
| Simple | ✓ | ✓ Delassus | ✓ forward mode | ✗ direct QR, likely float64 | possible | probably not |
| Nimble | ✓ | ✓ A = JM⁻¹Jᵀ | ✓ | ✗ direct pseudoinverse, likely float64 | ✓ clamping contacts (true value exactly 0) | probably not |

Not verified: the precision Simple and Nimble actually use (inferred from
their CPU C++ code bases, Pinocchio and DART), and other dual engines
(e.g. Jade). Publication angle: the structure is common in dual-formulation
differentiable engines and harmless with direct float64 solves. It becomes
harmful when such an engine moves to the GPU setting (float32, budgeted
matrix-free iterative solves), which is where ostrich is and where
Kamino-style engines are heading. A convincing demonstration outside
ostrich would run Simple's derivative formula with an iterative float32
solve.

**Claim for a paper (2026-11-12).** The hybrid combination of body space
and constraint space restores, in maximal coordinates with compliant
constraints, what minimal-coordinate engines get for free: the elimination
of the joints from the constraint system. And it does so without giving up
the maximal coordinates that GPU parallelism favours.
(Context: Simple and Nimble work in minimal coordinates, so their
constraint systems hold only contact rows. The joint-space mass matrix is
the rigid limit, c_joint → 0, of the hybrid's M̂ = M̃ + J_Sᵀ C_S⁻¹ J_S.
Engines that could benefit are maximal-coordinate dual engines: ostrich,
Kamino if made differentiable, SPOOK/AGX-style engines.)

**In-engine trajectory test of the adjoint solve (2026-11-13).** Script:
`test_scripts/deflation/adjoint_trajectory.py` (~25 s per scene, GPU +
CPU). It runs the real engine on the obstacle and terrain scenes (same
setup as `dump_linear_systems.py`: dt 0.03, 25 steps, wheel target 6.0,
default config), saves the rollout in a `TrajectoryBuffer`, and repeats
the backward pass on that one trajectory with different adjoint solvers.
Only the linear solve inside `step_backward` is swapped (the script
replaces `engine.cr_solver.solve`, rebuilds the dense active-set system
from `A.data`, solves it on the CPU and writes the result back). No change
in `src/`. The gradient is dL/d(wheel target velocity) at every step
before an end step T ∈ {5, 10, 15, 20, 25}. Losses: 8 random velocity
weight vectors at T (40 gradients per scene) and chassis x / y position
at T (10 gradients per scene).

Background: rerunning `tests/differentiable_simulator` at the production
PCR budget (16 iterations, tol 1e-5) passed 13 of 13 scripts with
unchanged gradients, but the adjoint PCR never reached the cap there (at
most 7 iterations), so those tests do not exercise the capped regime.

Harness checks (all pass): dense A vs the engine's matrix-free operator
3e-8; CPU port of PCR vs the engine's GPU PCR, gradient difference 2e-6;
elimination run to convergence in float64 vs the dense reference 7e-7.

*Result 1: error against the converged adjoint* (dense float64 solve of
the same adjoint systems). Whole horizon (all steps before T), default
forward pass, relative error p50 / p90, share of gradients with > 10%
error, worst cosine:

| adjoint solver | obstacle, velocity loss | obstacle, position loss | terrain, velocity loss | terrain, position loss |
|---|---|---|---|---|
| engine (GPU PCR, cap 16) | 2.2e-1 / 6.5e-1, 60%, 0.75 | 3.0e-2 / 3.7e-1, 30%, 0.91 | 1.4e-1 / 3.1e-1, 65%, 0.93 | 7.4e-2 / 4.9e-1, 40%, 0.93 |
| PCR cap 32 | 6.0e-2 / 3.2e-1, 45%, 0.84 | 9.3e-3 / 8.6e-2, 10%, 0.98 | 2.3e-2 / 7.5e-2, 5%, 0.99 | 2.1e-2 / 1.1e-1, 10%, 0.98 |
| joint/ctrl eliminated, P = diag(A), cap 16 | 9.3e-3 / 1.3e-1, 12%, 0.93 | 5.5e-4 / 5.3e-2, 10%, 0.995 | 1.6e-2 / 3.9e-2, 0%, 0.998 | 1.5e-2 / 9.8e-2, 10%, 0.98 |
| joint/ctrl eliminated, P = diag(A_F), cap 16 | 6.0e-4 / 1.6e-2, 0%, 0.997 | 2.7e-4 / 2.6e-3, 0%, 0.9998 | 4.4e-3 / 1.6e-2, 0%, 0.9995 | 8.8e-3 / 6.0e-2, 0%, 0.998 |

The engine's adjoint PCR reached the cap in 59% (obstacle) and 61%
(terrain) of the solves; with the elimination and diag(A_F) in 25% and
40%. A tight forward pass (`--forward tight`) gives the same picture.
Against the thresholds fixed before the run: claim 1 (the capped solve
makes the gradient materially wrong) holds, 30–65% of the gradients are
off by more than 10%. Claim 2 (the elimination repairs it at the same
budget) holds: no gradient above 10%, better than PCR with twice the
budget in every cell, median error lower by a factor 370 / 110 / 32 / 8
(the last, terrain position loss, misses the 10× mark; 10 gradients).
Single run; the forward pass is not bit-reproducible on the GPU, so
repeated runs change the numbers slightly.

*Result 2: error against finite differences* (`--fd`, tight forward pass,
one wheel target perturbed for all steps, eps 1e-2 and 1e-3, gradient
summed over steps). This was meant as a sanity check of the reference and
turned out to be the more important result:

- While the forward Newton loop converges (2–3 iterations per step), the
  converged adjoint matches FD to 0.1% for velocity losses. Control on
  the gradient tests' flat-ground scene
  (`--scene flat --fd --dt 0.01 --drive-velocity 3`): 0.0–0.1% up to
  T = 15. Correction (same day): these steps are free flight. The robot
  starts at z = 0.6 and the contacts detected before landing carry less
  than 0.005 N, so this does not validate the adjoint through contact.
- From the first step where Newton hits its iteration cap, which is the
  landing (step 5 at
  dt 0.03 and wheel target 6.0, in the obstacle scene and on flat ground
  alike; step 19 in the flat control), the converged adjoint departs from
  FD: 15–47% at T = 6, 57–65% at T = 8, 84–145% at T = 10–12 in the
  obstacle scene, where FD is still stable (eps 1e-2 and 1e-3 agree to
  12% or better). Later, and from T = 8 in the terrain scene, FD itself is unstable
  and cannot serve as a reference.
- The adjoint solver makes no visible difference there: reference, engine
  PCR and the elimination have the same error against FD (e.g. T = 8:
  0.569 / 0.540 / 0.569).
- Raising the Newton cap from 20 to 100 lowered the forward residual by
  a factor 10–100 but did not change the mismatch (ad-hoc run, not in the
  repo). So unconverged Newton iterations alone do not explain it. The
  cause is in the next entry.
- Position-loss gradients could not be validated: over these short
  horizons the FD values are at single-precision resolution (eps 1e-2 and
  1e-3 disagree), and where they agree (flat control, T = 15) the adjoint
  is 32% off.

Consequences for idea 7.2:
- The numerical claim is confirmed in the engine at trajectory level: the
  capped adjoint solve loses 20–60% against the converged adjoint, and
  the elimination with diag(A_F) removes that at the same budget. The
  diag(A_F) preconditioner is needed; deflation with diag(A) leaves
  10–12% bad gradients in three of four cells.
- The practical claim is not supported on these scenes. In the regime
  where PCR is capped, the converged adjoint is itself 50–100% away from
  FD, so repairing the linear solve does not bring the gradient closer to
  the true derivative. The capped-PCR regime and the regime where the
  adjoint is unreliable appear to coincide (both start when the wheels
  load up and Newton stops converging).
- Before 7.2 can claim better gradients, either the larger error has to
  be understood and removed, or a scene has to be found where the
  converged adjoint matches FD while PCR is still cut off by the cap.

Housekeeping: the two scratch scripts behind `tab:adjoint-elimination`
are now in the repo as `test_scripts/deflation/adjoint_equivalence.py`
and `test_scripts/deflation/deflation_forward_precond.py`; rerun
2026-11-13, numbers unchanged.

**Why the adjoint disagrees with FD once contacts carry load (2026-11-13).**
Investigation with ad-hoc scripts (session scratch, not in the repo), on
the flat-ground scene and the obstacle scene at dt 0.03, wheel target 6.0,
tight solver settings (200 PCR iterations, tol 1e-8, 20 Newton
iterations), single time steps, loss L = w·u⁺ with random w, gradient
w.r.t. the three wheel targets.

1. *The onset is the landing.* The first step with a loaded contact
   (480–580 N normal force) is step 5 at dt 0.03 and step 19 at dt 0.01;
   before it the largest normal force is below 0.005 N. All agreement
   with FD reported above is free flight. `test_helhest_gradient.py`
   (1 and 5 steps of 0.01 s from z = 0.6) runs entirely in free flight.
2. *The matrix of the Newton step is not the Jacobian of the residual
   once constraint forces are nonzero*, and the adjoint reuses that
   matrix. Test: reproduce one Newton step with an exact linear solve and
   compare the residual change along the step with the predicted −R. At
   λ = 0 the prediction is exact. At the exit iterates of loaded steps
   the dynamics rows are off by 41–77% of |R| (the mismatch vanishes when
   the pose is held fixed, so it is the dependence of the force
   directions and lever arms on the new pose, a term ∝ dt²·λ), and the
   normal rows by 15–28% (cause not isolated).
3. *Friction rows treat the weight w as a constant* (the code comments
   call this Picard / frozen), but for a sliding contact w is
   proportional to the slip speed. For the loaded sliding contacts at
   flat step 6, the velocity block of a friction row pair has singular
   values (1.064, 1.063) in the engine's matrix and (1.063, 0.000) in the
   FD Jacobian: the true rows do not respond to a change of slip speed
   along the slip direction, the engine's rows model sliding friction as
   viscous.
4. *Forward pass.* With this matrix Newton converges linearly and stalls:
   exit residual 1e-2 to 1e-3 after 20 iterations, 1e-3 after 100, 5e-5
   after 3000, never the 1e-6 requested. Exact dense linear solves give
   the identical residual history, so the linear solver is not the
   cause. With the FD Jacobian (friction lag frozen at the engine's exit
   lag) Newton reaches 1e-5 from the engine's exit iterate in 2–4
   iterations.
5. *Adjoint.* At that root, relative error against the gradient obtained
   by re-solving the step with perturbed targets:

   | step | IFT, FD Jacobian | IFT, engine matrix | engine `step_backward` | FD through `engine.step` |
   |---|---|---|---|---|
   | flat 4 (free flight) | 0.0% | 0.0% | 0.0% | 0.0% |
   | flat 5 | 0.1% | 58% | 58% | 26% |
   | flat 6 | 0.1% | 48% | 48% | 6% |
   | flat 7 | 0.2% | 70% | 70% | 77% |
   | obstacle 5 | 0.1% | 4.3% | 4.3% | 8.9% |
   | obstacle 6 | 0.1% | 6.5% | 6.5% | 10% |

   The implicit-function-theorem gradient is right when the matrix is
   the Jacobian. The engine's adjoint equals what its matrix implies, so
   its error comes from the matrix, not from the unconverged state. FD
   through the engine is also off, because the forward step is not
   converged; it is not a ground truth under load. Replacing one row
   group of the engine matrix by the true rows (flat 5 / 6 / 7): friction
   rows → 18 / 14 / 12%, normal rows → 29 / 32 / 51%, dynamics rows →
   54 / 46 / 68%. For this gradient the friction rows matter most.
6. *Friction above the Coulomb limit.* At the exit of loaded steps the
   contacts carry |λ_f| = 1.1 to 2.0 × μλ_n in all configurations, and
   still after 3000 iterations. Scaling |λ_f| of these contacts by 0.8 to
   1.2 leaves the friction-row residual unchanged to three digits: above
   the limit the rows constrain only the direction of the friction force
   (`gap` clamps at 0 and the weight divides by the unclamped lagged
   magnitude). The system with the lag tied is therefore degenerate
   there, which is why item 5 uses the lag-frozen system.
7. *Ruled out:* the adjoint linear solver, the forward linear solver, the
   `pose_vjp` switch. The `frozen` friction mode of the adjoint is worse
   (40–70% single-step error at flat step 5).
8. *Limits.* Five loaded steps in two scenes, one loss type, one
   parameter. Steps with all contacts sticking (flat 8, 9) could not be
   analysed this way: the FD-Jacobian root search diverged, although the
   single-step adjoint is also 28–45% off FD there. The cause of the
   normal-row mismatch is open. An A/B with `relaxation.friction =
   bilateral_patch` was inconclusive (adjoint off before landing too).

Consequence for idea 7.2: under load the dominant error of the gradient
is the linearization that the Newton step and the adjoint share, not the
adjoint's linear solve. The same linearization makes Newton hit its
iteration cap. The solver error that 7.2 removes stays second-order until
that is fixed.

**Scene check: contact-free articulated chains (2026-11-20).** Script:
`test_scripts/deflation/adjoint_articulated.py` (same harness as
`adjoint_trajectory.py`; ~2 min per comparison, ~25 min for the 12-link FD
check). Question: is there a scene where (a) the converged adjoint agrees
with FD and (b) the engine's adjoint PCR still hits the cap? Only there can
a better adjoint solve give a better gradient. Scene: N links in a row,
revolute joints (axis world x, so the chain swings in the y–z plane), pivot
fixed, no collision shapes, gravity, every joint velocity-controlled
(ke = 100), link masses alternating by a factor 1 or 100, dt 0.01, 40
steps, default config. Gradient: dL/d(joint target velocity) of every joint
at every step before T ∈ {5, 10, 20, 30, 40}. Losses: 8 random velocity
weights, tip y and tip z position (tip x has zero gradient; an earlier run
with tip x produced meaningless rows).

New solver variant: `joints_*` eliminates only the joint rows and iterates
on the control rows. Without contacts S = joint/control rows is every row,
so `elim_*` degenerates to a direct solve (0 iterations); `joints_*` is the
non-degenerate analogue of keeping contact rows in F.

*Result 1: against the converged adjoint*, velocity losses, whole horizon
(40 gradients), rel. error p50, share > 10%, adjoint PCR cap hits:

| adjoint solver | 6 links, ratio 1 | 6 links, ratio 100 | 12 links, ratio 1 | 12 links, ratio 100 |
|---|---|---|---|---|
| engine (PCR cap 16) | 5.7e-2, 20%, cap 99% | 3.1e-2, 2%, cap 86% | 7.1e-2, 32%, cap 100% | 5.4e-2, 18%, cap 86% |
| PCR cap 32 | 6.4e-3, 0% | 7.5e-3, 0% | 1.7e-2, 0% | 7.3e-3, 0% |
| joints eliminated, diag(A_F) | 9e-6, 0% (6 its) | 1.3e-3, 0% (5 its) | 1.4e-3, 0% (14 its) | 1.6e-3, 0% (6 its) |
| joint + control eliminated (direct) | 1e-6 | 1e-6 | 2e-6 | 1e-6 |

Forward Newton converges in 2–4 iterations at every step (no contacts).

*Result 2: against FD* (12 links, ratio 100, tight forward, eps 1e-2;
FD spread between eps 1e-2 and 1e-3 is 7–38%, so FD resolves a few %).
Velocity losses, relative error of the step-summed gradient:

| T | reference | engine | joints eliminated |
|---|---|---|---|
| 5–20 | 1.0–5.3% | 1.5–6.2% | 1.0–5.3% |
| 30 | 1.9–2.7% | 5.0–5.2% | 1.8–2.7% |
| 40 | 2.7–5.3% | 11.6–13.8% | 2.7–5.4% |

Tip z: reference 0.8–3.7% vs engine 1.8–7.4% (T ≥ 10). Tip y at T ≤ 10 and
tip z at T = 5: FD unusable (spread ≥ 0.45, or FD = 0). The 6-link FD
check gave the same picture at a smaller gap (reference ~1%, engine 1–4%).

Findings:
- **The gate passes.** Without contacts the linearization is right
  (converged adjoint within FD resolution), and the engine's adjoint PCR
  hits the cap in 86–100% of solves. At T = 40 the engine is 12–14% off
  FD, against 3–5% for the exact adjoint, so the solver error is visible
  against the true derivative and grows with the horizon.
- **Eliminating only the joint rows is as good as the exact adjoint
  against FD**, at 5–14 iterations. The engine's error grows with chain
  length (6 → 12 links: p50 5.7e-2 → 7.1e-2); the joints-only variant
  needs more iterations for the longer uniform chain (6 → 14), so the
  control block is not free either.
- **Position losses are limited by the absolute tolerance, not the cap.**
  Their adjoint right-hand sides are small (‖b‖ 5e-4 to 2e-2, vs ~0.4 for
  velocity losses), so atol = 1e-5 stops PCR at a relative accuracy of
  only 1e-2 to 1e-3: PCR cap 32 gives exactly the cap-16 numbers (12
  links, ratio 100: p50 3.5e-2, max 0.37, 20% > 10%). With deflation some
  right-hand sides fall below atol entirely and the joints-only variant
  returns a zero gradient (cosine 0). The engine uses the same atol. An
  absolute stopping test does not fit an adjoint solve, whose right-hand
  side scales with the loss; this is the "adjoint stopping test" item of
  §7.2 and needs a relative (or C⁻¹-weighted) criterion.
- Limits: one chain topology, one control gain, single runs, FD at
  float32 resolution (a few %). No contacts, so this shows the joint part
  of the claim only.
