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
