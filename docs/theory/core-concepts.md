# Core Concepts

This section provides a high-level mathematical overview of how Ostrich formulates and solves the physics simulation problem. Understanding these concepts is essential for grasping the theoretical foundation underlying the simulator's robust and unified approach.

---

## Mathematical Foundation

Ostrich's physics engine is built on a unified mathematical framework that treats all physical phenomena—articulated body dynamics, contact interactions, and joint constraints—as a single, coupled system of equations. This approach provides superior stability and accuracy compared to traditional methods that handle these phenomena separately.

### Articulated Bodies

Ostrich represents articulated body systems using generalized coordinates and velocities that describe the system's configuration and motion. The dynamics are governed by:

\[
\mathbf{\tilde{M}}(\mathbf{q}) \Delta\mathbf{u} = h \mathbf{f}_{\text{ext}} + h \mathbf{J}^T(\mathbf{q}) \boldsymbol{\lambda}
\]

This captures how velocity changes result from external forces and constraint forces. The meaning and derivation of these constraint impulses will be explained in [Gauss's Principle of Least Constraint](./gauss-least-constraint.md).

!!! note "Mathematical Notation"
    For detailed definitions of all symbols (\(\mathbf{q}\), \(\mathbf{u}\), \(\mathbf{\tilde{M}}\), \(\mathbf{J}\), \(\boldsymbol{\lambda}\), etc.), see the [Notation](./notation.md) page.

### Contact and Constraint Formulation

Physical interactions are mathematically encoded as constraints:

* **Joint Constraints** (bilateral): Enforce exact geometric relationships between bodies
* **Contact Constraints** (unilateral): Prevent body interpenetration  
* **Friction Constraints**: Model stick-slip behavior through complementarity conditions

These constraints create a system mixing equalities and inequalities, requiring specialized mathematical treatment to solve simultaneously.

---

## Solution Approach

Ostrich's approach follows a four-step mathematical progression:

### 1. Constraint Formulation

First, we mathematically formulate how articulated bodies and their interactions are represented as constraint equations. This establishes the mathematical foundation for describing joints, contacts, and friction.

→ **Next**: [Constraints Formulation](./constraints.md)

### 2. Optimization Principle  

We apply **Gauss's Principle of Least Constraint**, which provides a principled way to determine how the system should evolve when subject to constraints. This principle frames constraint enforcement as an optimization problem.

→ **Next**: [Gauss's Principle of Least Constraint](./gauss-least-constraint.md)

### 3. Nonlinear System

The optimization principle, combined with time discretization, leads to a large nonlinear system of equations that must be solved at each time step. This system encodes all physical laws simultaneously.

→ **Next**: [Nonlinear System](./non-linear-system.md)

### 4. Numerical Solution

We numerically solve this nonlinear system using a specialized Newton-type method designed to handle the non-smooth nature of contact and friction, with a GPU-parallel linear solver at its core.

→ **Next**: [Numerical Solution](./numerical-solution.md)

### 5. Differentiable Simulation

Because each time step is defined by an implicit equation \(\mathbf{R}(\mathbf{s}^+, \mathbf{s}^-, \mathbf{a}^-, \boldsymbol{\theta}) = \mathbf{0}\), the implicit function theorem provides exact gradients of any loss with respect to initial conditions, controls, or physical parameters — without unrolling the solver. The adjoint method makes this computation as cheap as one additional linear solve per time step.

→ **Next**: [Adjoint Method](./adjoint-method.md)

---

## Why This Unified Approach?

Traditional physics engines handle dynamics, contacts, and joints in separate phases, leading to:

* **Instability** in tightly coupled systems
* **Drift** and constraint violation accumulation  
* **Artificial softness** in joints and contacts

Ostrich's unified mathematical formulation addresses these issues by:

* **Solving everything simultaneously** — no artificial sequencing
* **Position-level constraint enforcement** — eliminates drift by design
* **Principled optimization framework** — mathematically grounded decisions

This mathematical rigor enables stable simulation of complex scenarios like articulated robots making contact with the environment, which often challenge traditional approaches.
