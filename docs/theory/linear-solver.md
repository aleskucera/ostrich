# Linear Solver

At each Newton iteration, the core computational task is solving the Schur complement system for the constraint force update \(\Delta\boldsymbol{\lambda}\). This page describes the structure of that system and how Ostrich solves it efficiently on the GPU.

---

## The Schur Complement System

After eliminating \(\Delta\mathbf{q}\) and back-substituting \(\Delta\mathbf{u}\) (see [Numerical Solution](./numerical-solution.md)), the problem reduces to a single linear system for \(\Delta\boldsymbol{\lambda}\):

\[
\underbrace{\left[ \hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top + \mathbf{C} \right]}_{\mathbf{A}} \Delta\boldsymbol{\lambda} = \underbrace{\hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \mathbf{h}_\text{dyn} - \mathbf{h}_c}_{\mathbf{b}}
\]

The matrix \(\mathbf{A} \in \mathbb{R}^{n_c \times n_c}\) has a clear physical interpretation:

- \(\hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top\) is the **effective mass matrix** in constraint space. Each entry \((i, j)\) measures how a unit force in constraint direction \(j\) accelerates the bodies involved in constraint \(i\). The diagonal entry for constraint \(i\) is the **effective mass**:

\[
a_{ii} = \hat{\mathbf{J}}_i \, \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}_i^\top
\]

- \(\mathbf{C}\) is the block-diagonal **compliance matrix** that softens the constraints (see [Notation](./notation.md#jacobians-and-system-derivatives)).

### Structure and Properties

The matrix \(\mathbf{A}\) is **symmetric positive semi-definite** (SPSD):

- **Symmetric**: \((\hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top)^\top = \hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top\), and \(\mathbf{C}\) is diagonal.
- **Positive semi-definite**: For any vector \(\mathbf{v}\), \(\mathbf{v}^\top \mathbf{A} \mathbf{v} = \|\mathbf{\tilde{M}}^{-1/2} \hat{\mathbf{J}}^\top \mathbf{v}\|^2 + \mathbf{v}^\top \mathbf{C} \mathbf{v} \geq 0\). The compliance \(\mathbf{C} \succ 0\) makes it strictly positive definite in practice.

This SPSD structure is what enables the use of Krylov solvers designed for symmetric systems.

---

## Preconditioned Conjugate Residual (PCR)

Ostrich solves the Schur complement system using the **Preconditioned Conjugate Residual (PCR)** method — the symmetric counterpart of GMRES, designed specifically for SPSD systems.

The Conjugate Residual method exploits the symmetry of \(\mathbf{A}\) to build an orthogonal basis for the Krylov subspace \(\mathcal{K}_k(\mathbf{A}, \mathbf{b})\) using short recurrences, requiring only two matrix-vector products per iteration rather than a full Gram-Schmidt process. This makes it highly efficient for large, sparse systems.

### Matrix-Vector Product

The key operation in each PCR iteration is computing \(\mathbf{A} \mathbf{v}\) for a trial vector \(\mathbf{v}\). Because \(\mathbf{A}\) is never assembled explicitly (it would be \(n_c \times n_c\) and dense), this product is computed in two GPU kernel passes:

1. **Expand**: \(\mathbf{p} = \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top \mathbf{v}\) — map constraint forces to velocity space via the mass-weighted Jacobian transpose.
2. **Contract**: \(\mathbf{A}\mathbf{v} = \hat{\mathbf{J}} \mathbf{p} + \mathbf{C} \mathbf{v}\) — accumulate per-body contributions back into constraint space and add compliance.

### Convergence Criteria

The solver stops when either:

- The **relative residual** drops below a tolerance: \(\|\mathbf{b} - \mathbf{A}\Delta\boldsymbol{\lambda}\| / \|\mathbf{b}\| < \varepsilon_\text{rel}\)
- The **absolute residual** drops below a tolerance: \(\|\mathbf{b} - \mathbf{A}\Delta\boldsymbol{\lambda}\| < \varepsilon_\text{abs}\)
- The **maximum iteration count** \(K_\text{max}\) is reached (the solver returns the best iterate found so far)

---

## Jacobi Preconditioner

Without preconditioning, PCR convergence is governed by the condition number of \(\mathbf{A}\), which can be large when constraints have vastly different effective masses (e.g., a contact between a 1 kg box and a 100 kg robot). The **Jacobi (diagonal) preconditioner** rescales the system to improve conditioning.

The preconditioner is the inverse of the diagonal of \(\mathbf{A}\):

\[
\mathbf{P} = \text{diag}(\mathbf{A})^{-1} = \text{diag}\!\left(\hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top + \mathbf{C}\right)^{-1}
\]

Each diagonal entry for constraint \(i\) is computed as:

\[
p_i = \frac{1}{a_{ii}} = \frac{1}{\hat{\mathbf{J}}_i \, \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}_i^\top + c_{ii}}
\]

where \(c_{ii}\) is the compliance of constraint \(i\). In physical terms, \(1/p_i\) is the effective mass seen by constraint \(i\), plus its compliance. The preconditioner maps from the space of constraint forces to a normalised, dimensionless space where all constraints are approximately unit-scaled.

The preconditioned system solved by PCR is:

\[
\mathbf{P}^{1/2} \mathbf{A} \mathbf{P}^{1/2} \, \hat{\Delta\boldsymbol{\lambda}} = \mathbf{P}^{1/2} \mathbf{b}
\]

applied symmetrically to preserve the SPSD property. In practice, Jacobi preconditioning is applied via two cheap vector scaling operations per iteration, with no factorization required — making it ideal for GPU parallelism.

A small **regularization** term \(\varepsilon_\text{reg}\) is added to the diagonal before inversion to prevent division by zero for inactive constraints:

\[
p_i = \frac{1}{a_{ii} + \varepsilon_\text{reg}}
\]

---

## Summary

| Property | Value |
|:---|:---|
| System type | Symmetric positive semi-definite |
| Solver | Preconditioned Conjugate Residual (PCR) |
| Preconditioner | Jacobi (diagonal inverse of \(\mathbf{A}\)) |
| System size | \(n_c \times n_c\) (number of active constraints) |
| \(\mathbf{A}\) storage | Never assembled; applied as matrix-vector product |
| Parallelism | All constraint rows computed independently on GPU |

→ **Next**: [Adjoint Method](./adjoint-method.md)
