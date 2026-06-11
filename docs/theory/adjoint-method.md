# Implicit Gradient and Adjoint Method

Ostrich is a **differentiable simulator**: it can compute exact gradients of any scalar loss \(\mathcal{L}\) with respect to initial conditions, control inputs, or physical parameters. This enables gradient-based optimisation of robot behaviour directly through the physics engine.

This section explains the mathematical foundation of how these gradients are computed efficiently using the **implicit function theorem** and the **adjoint method**.

---

## Simulation as a Computational Graph

A simulation run of \(T\) time steps can be viewed as a chain of state transitions:

\[
\mathbf{s}_0 \;\longrightarrow\; \mathbf{s}_1 \;\longrightarrow\; \cdots \;\longrightarrow\; \mathbf{s}_T \;\longrightarrow\; \mathcal{L}(\mathbf{s}_T)
\]

where the **state** at each step is \(\mathbf{s}_k = [\mathbf{q}_k,\, \mathbf{u}_k,\, \boldsymbol{\lambda}_k]\) (configuration, velocity, and constraint forces), and \(\mathcal{L}\) is a scalar loss evaluated on the final state.

Each transition \(\mathbf{s}_{k-1} \to \mathbf{s}_k\) is not a simple explicit function — it is defined **implicitly** by solving the nonlinear system from [The Nonlinear System](./non-linear-system.md):

\[
\mathbf{R}(\mathbf{s}^+,\, \mathbf{s}^-,\, \mathbf{a}^-,\, \boldsymbol{\theta}) = \mathbf{0}
\]

where:

- \(\mathbf{s}^+ = \mathbf{s}_k\) — the **next** state (solved for)
- \(\mathbf{s}^- = \mathbf{s}_{k-1}\) — the **current** state (known)
- \(\mathbf{a}^-\) — the actuation applied at this step (control targets)
- \(\boldsymbol{\theta}\) — world parameters (masses, inertias, friction coefficients, gains, etc.)

The solver finds \(\mathbf{s}^+(\mathbf{s}^-, \mathbf{a}^-, \boldsymbol{\theta})\) such that \(\mathbf{R} = \mathbf{0}\), but there is no explicit closed-form formula for this mapping.

---

## The Gradient Problem

To optimise \(\mathcal{L}\), we need to backpropagate through the computational graph. The key quantity needed at each step is the **state transition Jacobian** \(\frac{\mathrm{d}\mathbf{s}^+}{\mathrm{d}\mathbf{s}^-}\), which tells us how a perturbation to the current state propagates to the next state.

Computing this Jacobian directly is problematic: there is no explicit formula for \(\mathbf{s}^+(\mathbf{s}^-)\), and automatic differentiation through the Newton solver iterations would be extremely expensive and memory-intensive.

Instead, Ostrich uses the **implicit function theorem** to compute the Jacobian exactly without unrolling the solver.

---

## Implicit Function Theorem

Since the solution \(\mathbf{s}^+(\mathbf{s}^-, \mathbf{a}^-, \boldsymbol{\theta})\) always satisfies \(\mathbf{R}(\mathbf{s}^+, \mathbf{s}^-, \mathbf{a}^-, \boldsymbol{\theta}) = \mathbf{0}\), differentiating both sides with respect to \(\mathbf{s}^-\) gives:

\[
\frac{\mathrm{d}\mathbf{R}}{\mathrm{d}\mathbf{s}^-} = \frac{\partial \mathbf{R}}{\partial \mathbf{s}^+} \frac{\mathrm{d}\mathbf{s}^+}{\mathrm{d}\mathbf{s}^-} + \frac{\partial \mathbf{R}}{\partial \mathbf{s}^-} = \mathbf{0}
\]

Rearranging yields the **implicit gradient**:

\[
\boxed{\frac{\mathrm{d}\mathbf{s}^+}{\mathrm{d}\mathbf{s}^-} = -\left[\frac{\partial \mathbf{R}}{\partial \mathbf{s}^+}\right]^{-1} \frac{\partial \mathbf{R}}{\partial \mathbf{s}^-}}
\]

This is exact and requires only the partial derivatives of \(\mathbf{R}\) — quantities that are already computed during the forward solve. The same formula applies for gradients with respect to \(\mathbf{a}^-\) and \(\boldsymbol{\theta}\).

---

## The Adjoint Method

Substituting the implicit gradient into the chain rule gives:

\[
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{s}^-} = \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{s}^+} \frac{\mathrm{d}\mathbf{s}^+}{\mathrm{d}\mathbf{s}^-} = -\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{s}^+} \left[\frac{\partial \mathbf{R}}{\partial \mathbf{s}^+}\right]^{-1} \frac{\partial \mathbf{R}}{\partial \mathbf{s}^-}
\]

Naïvely, evaluating this requires forming and inverting the \(n \times n\) matrix \(\partial \mathbf{R}/\partial \mathbf{s}^+\), where \(n = n_q + n_u + n_c\). This is expensive when \(n\) is large.

The **adjoint trick** exploits the fact that \(\mathrm{d}\mathcal{L}/\mathrm{d}\mathbf{s}^+\) is a **row vector** (\(1 \times n\)), so the full product is a row vector times a matrix times a matrix. By reordering the multiplication and defining the **adjoint vector** \(\mathbf{w}^\top \in \mathbb{R}^{1 \times n}\) as the solution to:

\[
\mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \mathbf{s}^+} = -\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{s}^+}
\]

or equivalently in transposed form:

\[
\left[\frac{\partial \mathbf{R}}{\partial \mathbf{s}^+}\right]^\top \mathbf{w} = -\nabla_{\mathbf{s}^+} \mathcal{L}
\]

the gradient with respect to all inputs follows from a single matrix-vector product:

\[
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{s}^-} = \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \mathbf{s}^-}, \qquad
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\mathbf{a}^-} = \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \mathbf{a}^-}, \qquad
\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\boldsymbol{\theta}} = \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \boldsymbol{\theta}}
\]

**The key advantage**: regardless of the number of inputs (parameters \(\boldsymbol{\theta}\) may have thousands of entries), only **one** linear system needs to be solved per time step — the adjoint system for \(\mathbf{w}\).

---

## The Adjoint Linear System

The adjoint system \([\partial \mathbf{R}/\partial \mathbf{s}^+]^\top \mathbf{w} = -\nabla_{\mathbf{s}^+}\mathcal{L}\) expands using the block structure of \(\partial \mathbf{R}/\partial \mathbf{s}^+\) derived from the KKT matrix. Writing \(\mathbf{w} = [\mathbf{w}_q,\, \mathbf{w}_u,\, \mathbf{w}_\lambda]\):

\[
\begin{bmatrix}
\mathbf{I} & \mathbf{0} & \mathbf{0} \\
\mathbf{G}^\top & \mathbf{\tilde{M}} & \mathbf{J}^\top \\
\mathbf{0} & -h\mathbf{J} & h\mathbf{C}
\end{bmatrix}
\begin{bmatrix} \mathbf{w}_q \\ \mathbf{w}_u \\ \mathbf{w}_\lambda \end{bmatrix}
=
\begin{bmatrix} \nabla_{q^+}\mathcal{L} \\ \nabla_{u^+}\mathcal{L} \\ \mathbf{0} \end{bmatrix}
\]

The right-hand side has \(\nabla_{\lambda^+}\mathcal{L} = \mathbf{0}\) because the loss typically does not depend directly on the constraint forces.

Eliminating \(\mathbf{w}_q\) from the first block row (which gives \(\mathbf{w}_q = \nabla_{q^+}\mathcal{L}\)) and substituting into the remaining two rows:

\[
\begin{bmatrix}
\mathbf{\tilde{M}} & \mathbf{J}^\top \\
-h\mathbf{J} & h\mathbf{C}
\end{bmatrix}
\begin{bmatrix} \mathbf{w}_u \\ \mathbf{w}_\lambda \end{bmatrix}
=
\begin{bmatrix}
\nabla_{u^+}\mathcal{L} + h\mathbf{G}\,\nabla_{q^+}\mathcal{L} \\
\mathbf{0}
\end{bmatrix}
\]

Applying the same Schur complement strategy as the forward solve (eliminating \(\mathbf{w}_u\)):

\[
\boxed{
\left[\mathbf{J} \mathbf{\tilde{M}}^{-1} \mathbf{J}^\top + \mathbf{C}\right] \mathbf{w}_\lambda = \mathbf{J} \mathbf{\tilde{M}}^{-1} \left(\nabla_{u^+}\mathcal{L} + h\mathbf{G}\,\nabla_{q^+}\mathcal{L}\right)
}
\]

\[
\mathbf{w}_u = \mathbf{\tilde{M}}^{-1}\!\left(\nabla_{u^+}\mathcal{L} + h\mathbf{G}\,\nabla_{q^+}\mathcal{L} - \mathbf{J}^\top \mathbf{w}_\lambda\right)
\]

This is **the same Schur complement system** as the forward solve, with a different right-hand side. Ostrich therefore reuses the same PCR solver and Jacobi preconditioner (already computed during the forward pass) to solve the adjoint system at negligible additional cost.

---

## Full Backward Pass

The complete backward pass through a simulation of \(T\) steps proceeds as follows:

1. **Initialise**: set \(\nabla_{\mathbf{s}_T}\mathcal{L}\) from the loss gradient at the final state.

2. **For each step** \(k = T, T-1, \ldots, 1\):
   a. Recompute \(\partial \mathbf{R}/\partial \mathbf{s}^+\) at the stored forward state \(\mathbf{s}_k\).
   b. Solve the adjoint system for \(\mathbf{w}_\lambda\) via PCR, then recover \(\mathbf{w}_u\).
   c. Compute the gradient passed to the previous step:
   \[
   \nabla_{\mathbf{s}^-}\mathcal{L} = \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \mathbf{s}^-}
   \]
   d. Accumulate parameter gradients if needed:
   \[
   \nabla_{\mathbf{a}^-}\mathcal{L} \mathrel{+}= \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \mathbf{a}^-}, \qquad
   \nabla_{\boldsymbol{\theta}}\mathcal{L} \mathrel{+}= \mathbf{w}^\top \frac{\partial \mathbf{R}}{\partial \boldsymbol{\theta}}
   \]

3. The final result \(\nabla_{\mathbf{s}_0}\mathcal{L}\) gives the sensitivity of the loss to the initial conditions.

!!! note "Memory requirement"
    The backward pass requires storing the full state \(\mathbf{s}_k\) for all \(T\) steps. For long rollouts, this can be significant. Gradient checkpointing (recomputing forward states on demand) can trade compute for memory if needed.
