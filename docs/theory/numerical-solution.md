# Numerical Solution

The previous section established a large, non-smooth system of nonlinear equations, \(\mathbf{h}(\mathbf{x}) = \mathbf{0}\), that captures the complete physics of the simulation at each time step. This system cannot be solved analytically and requires a powerful iterative approach. This section describes the numerical strategy Ostrich uses to find the solution.

Ostrich employs an **inexact non-smooth Newton-type method**. The core idea is to start with a guess for the solution \(\mathbf{x}\) and iteratively refine it. In each iteration \(k\), we linearize the nonlinear function \(\mathbf{h}(\mathbf{x})\) at the current guess \(\mathbf{x_k}\) and solve the resulting linear system for a step \(\Delta \mathbf{x}\).

The linear system to be solved at each iteration is:

\[
\mathbf{A}(\mathbf{x_k}) \Delta \mathbf{x} = -\mathbf{h}(\mathbf{x_k})
\]

where \(\mathbf{A} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}\) is the **System Jacobian**.

## Linearizing the System and Eliminating Δq

The state vector we are solving for is \(\mathbf{x} = [\mathbf{q}, \mathbf{u}, \boldsymbol{\lambda}]\), and the corresponding update step is \(\Delta\mathbf{x} = [\Delta\mathbf{q}, \Delta\mathbf{u}, \Delta\boldsymbol{\lambda}]\). The full Jacobian \(\mathbf{A}\) is a 3x3 block matrix of partial derivatives. However, we can simplify this system dramatically before solving.

The key lies in the linearization of the **kinematic equation**, \(\mathbf{h_\text{kin}} = \mathbf{q}^+ - \mathbf{q}^- - h \cdot \mathbf{G}(\mathbf{q}^+) \cdot \mathbf{u}^+ = \mathbf{0}\). Taking its derivative gives us a direct relationship between the update steps \(\Delta\mathbf{q}\) and \(\Delta\mathbf{u}\). With some simplification (treating \(\mathbf{G}\) as constant for the linearization), we get:

\[
\Delta\mathbf{q} - h\mathbf{G}\Delta\mathbf{u} = \mathbf{0} \quad \implies \quad \Delta\mathbf{q} = h\mathbf{G}\Delta\mathbf{u}
\]

This powerful relationship tells us that the change in configuration is determined by the change in velocity. We can now use this to eliminate \(\Delta\mathbf{q}\) from the entire system *for the purpose of solving the problem*.

Consider the linearization of a general constraint residual \(\mathbf{h_c}\), which depends on \(\mathbf{q}\), \(\mathbf{u}\), and \(\boldsymbol{\lambda}\):

\[
\frac{\partial \mathbf{h_c}}{\partial \mathbf{q}}\Delta\mathbf{q} + \frac{\partial \mathbf{h_c}}{\partial \mathbf{u}}\Delta\mathbf{u} + \frac{\partial \mathbf{h_c}}{\partial \boldsymbol{\lambda}}\Delta\boldsymbol{\lambda} = -\mathbf{h_c}
\]

Now, we substitute \(\Delta\mathbf{q} = h\mathbf{G}\Delta\mathbf{u}\):

\[
\frac{\partial \mathbf{h_c}}{\partial \mathbf{q}}(h\mathbf{G}\Delta\mathbf{u}) + \frac{\partial \mathbf{h_c}}{\partial \mathbf{u}}\Delta\mathbf{u} + \frac{\partial \mathbf{h_c}}{\partial \boldsymbol{\lambda}}\Delta\boldsymbol{\lambda} = -\mathbf{h_c}
\]

Grouping the \(\Delta\mathbf{u}\) terms gives:

\[
\left( h \frac{\partial \mathbf{h_c}}{\partial \mathbf{q}}\mathbf{G} + \frac{\partial \mathbf{h_c}}{\partial \mathbf{u}} \right) \Delta\mathbf{u} + \frac{\partial \mathbf{h_c}}{\partial \boldsymbol{\lambda}} \Delta\boldsymbol{\lambda} = -\mathbf{h_c}
\]

This defines the matrices for our reduced system. The term in parenthesis is precisely the **System Jacobian** (\(\hat{\mathbf{J}}\)), and the multiplier of \(\Delta\boldsymbol{\lambda}\) is the **Compliance Matrix** (\(\mathbf{C}\)).

By applying this substitution to the whole system, we eliminate \(\Delta\mathbf{q}\) entirely, arriving at the final 2x2 block KKT system that is actually solved in practice:

\[
    \begin{bmatrix}
        \mathbf{\tilde{M}} & -\hat{\mathbf{J}}^\top \\
        \hat{\mathbf{J}} & \mathbf{C}
    \end{bmatrix}
    \begin{bmatrix}
        \Delta\mathbf{u} \\
        \Delta\boldsymbol{\lambda} \\
    \end{bmatrix}
    =
    -\begin{bmatrix}
        \mathbf{h_\text{dyn}} \\
        \mathbf{h_c}
    \end{bmatrix}
\]

## Schur Complement: A Strategy for Efficiency

Solving the full KKT system directly is still inefficient. The matrix is large, sparse, and indefinite. Ostrich employs the **Schur complement method** to create an even smaller, better-behaved system to solve. The strategy is to algebraically eliminate the velocity update \(\Delta \mathbf{u}\) and form a system that solves *only* for the constraint force update \(\Delta \boldsymbol{\lambda}\).

From the first block row of the KKT system, we express \(\Delta\mathbf{u}\) in terms of \(\Delta\boldsymbol{\lambda}\):

\[
\Delta\mathbf{u} = \mathbf{\tilde{M}}^{-1} (\hat{\mathbf{J}}^\top \Delta\boldsymbol{\lambda} - \mathbf{h_\text{dyn}})
\]

Substituting this into the second block row gives the **Schur complement system**:

\[
    \left[\hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \hat{\mathbf{J}}^\top + \mathbf{C} \right] \Delta\boldsymbol{\lambda} =
    \hat{\mathbf{J}} \mathbf{\tilde{M}}^{-1} \mathbf{h_\text{dyn}} - \mathbf{h_c}
\]

This system is smaller, symmetric positive-semidefinite, and amenable to massive parallelism, making it ideal for a **Conjugate Residual (CR)** iterative solver.

## The Complete Newton Iteration: A Step-by-Step Guide

With this theoretical foundation, we can now outline the exact sequence of computations performed within a single Newton step. This process computes the full update vector \(\Delta\mathbf{x} = [\Delta\mathbf{q}, \Delta\mathbf{u}, \Delta\boldsymbol{\lambda}]\) and applies it to the current state estimate \(\mathbf{x_k} = [\mathbf{q_k}, \mathbf{u_k}, \boldsymbol{\lambda_k}]\).

---
**Step 1: Assemble System Components**
At the current state \(\mathbf{x_k}\), the solver evaluates all required terms: the residual vectors (\(\mathbf{h_\text{dyn}}, \mathbf{h_c}\)) and the system matrices (\(\hat{\mathbf{J}}, \mathbf{C}, \mathbf{\tilde{M}}\)).

---
**Step 2: Solve for Impulse Update (\(\Delta\boldsymbol{\lambda}\))**
This is the core of the computation. The Schur complement system is formed and solved for the first part of our update vector, \(\Delta\boldsymbol{\lambda}\) (the constraint force update). This is done using a Preconditioned Conjugate Residual solver with a Jacobi Preconditioner.

---
**Step 3: Back-substitute for Velocity Update (\(\Delta\mathbf{u}\))**
With \(\Delta\boldsymbol{\lambda}\) known, the velocity update component, \(\Delta\mathbf{u}\), is calculated directly using the back-substitution formula derived from the first row of the KKT matrix:

\[
\Delta\mathbf{u} = \mathbf{\tilde{M}}^{-1} \left( \hat{\mathbf{J}}^\top \Delta\boldsymbol{\lambda} - \mathbf{h_\text{dyn}} \right)
\]

---
**Step 4: Recover the Full Update Vector (\(\Delta\mathbf{x}\))**
Now we recover the final missing piece of the update vector, \(\Delta\mathbf{q}\), using the same linearized kinematic relationship that we used for the elimination:

\[
\Delta\mathbf{q} = h \cdot \mathbf{G}(\mathbf{q_k}) \cdot \Delta\mathbf{u}
\]

With all three components computed, we assemble the **full Newton step vector**:

\[
\Delta\mathbf{x} = \begin{bmatrix} \Delta\mathbf{q} \\ \Delta\mathbf{u} \\ \Delta\boldsymbol{\lambda} \end{bmatrix}
\]

This vector represents the complete, linearized search direction for the entire system state.

---
**Step 5 (Optional): Parallel Step-Size Selection**

At each Newton iteration, an optional parallel step-size search can be used to select a better update magnitude than the standard full step. Rather than a sequential backtracking rule, Ostrich evaluates a discrete set of \(N\) candidate step sizes \(\{\alpha_0, \ldots, \alpha_{N-1}\}\) **simultaneously** in a single GPU pass. For each candidate \(\alpha_i\), a trial state is formed:

\[
\mathbf{x}_{\text{trial},i} = \mathbf{x_k} + \alpha_i \, \Delta\mathbf{x}
\]

The residual \(\|\mathbf{h}(\mathbf{x}_{\text{trial},i})\|^2\) is computed for all candidates in parallel, and the step size that minimises it is selected. The candidate set is split into two clusters designed for different situations:

1. **Conservative cluster** — \(N_c\) step sizes logarithmically spaced in \([\alpha_\text{min},\, \alpha_\text{safe}]\) (default: 32 steps in \([10^{-6},\, 0.05]\)). These small steps are safe when Newton overshoots badly.
2. **Optimistic cluster** — \(N_o\) step sizes linearly spaced in \([1 - \delta,\, 1 + \delta]\) (default: 32 steps with \(\delta = 0.1\), centred around the full Newton step \(\alpha = 1\)). These explore near the standard Newton step when the solver is close to convergence.

The two sets are merged, sorted, and the exact value \(\alpha = 1.0\) is forced to be present so the standard Newton step is always among the candidates.

---
**Step 6: Update the Full State Vector**
Finally, the entire state vector is updated in a single, unified operation using the computed full step \(\Delta\mathbf{x}\) and the optimal step size \(\alpha\) from the line search:

\[
\mathbf{x_{k+1}} = \mathbf{x_k} + \alpha\Delta\mathbf{x}
\]

This single vector update is equivalent to updating each component individually:

\[
\begin{align}
\mathbf{q_{k+1}} &= \mathbf{q_k} + \alpha \Delta\mathbf{q} \\
\mathbf{u_{k+1}} &= \mathbf{u_k} + \alpha \Delta\mathbf{u} \\
\boldsymbol{\lambda_{k+1}} &= \boldsymbol{\lambda_k} + \alpha \Delta\boldsymbol{\lambda}
\end{align}
\]

This brings us to the end of one Newton iteration. The process repeats from Step 1 with the new state \(\mathbf{x_{k+1}}\) until the norm of the residual vector \(\|\mathbf{h}(\mathbf{x_{k+1}})\|\) falls below a specified tolerance, or the maximum iteration count is reached.

---

## Best-Iterate Backtracking

Newton's method is not monotone — the residual can temporarily increase during intermediate iterations before converging. To guard against accepting a poor final iterate, Ostrich uses a **best-iterate backtracking** strategy.

After each Newton iteration \(k\), the full state \(\mathbf{x_k}\) and its residual norm \(\|\mathbf{h}(\mathbf{x_k})\|^2\) are saved as candidates. Once the Newton loop finishes, the solver does **not** necessarily keep the final iterate. Instead, it selects the candidate with the **minimum residual norm** from a minimum iteration index \(k_\text{min}\) onwards:

\[
k^* = \underset{k \geq k_\text{min}}{\arg\min} \; \|\mathbf{h}(\mathbf{x_k})\|^2
\]

The state is then restored to \(\mathbf{x}_{k^*}\). The threshold \(k_\text{min}\) (default: 5) prevents selecting a very early iterate that may have a low residual only because it is close to the initial guess, not because it is physically correct.

This strategy is especially robust for non-smooth problems where Newton can overshoot in the final steps.

→ **Next**: [Linear Solver](./linear-solver.md)
