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
