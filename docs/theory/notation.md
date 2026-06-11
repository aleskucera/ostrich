# Notation and Symbols

This page serves as a central glossary for the mathematical notation used throughout the Ostrich documentation. It defines the state variables, system parameters, and derivative matrices that form the foundation of the physics engine.

---

## 1. Dimension and Indexing Notation

This table defines the symbols used to represent the size of various vectors and matrices.

| Symbol | Description | Comment |
| :--- | :--- | :--- |
| \(n_q\) | **Configuration Space Dimension** | The dimension of the generalized coordinate vector \(\mathbf{q}\). |
| \(n_u\) | **Velocity Space Dimension (DOF)** | The dimension of the generalized velocity vector \(\mathbf{u}\). Equals total degrees of freedom. |
| \(n_b\) | **Number of Bilateral Constraints** | The number of active bilateral (joint) constraint equations. |
| \(n_n\) | **Number of Unilateral Constraints** | The number of active unilateral (contact) constraint equations. |
| \(n_f\) | **Number of Frictional Constraints** | The number of active frictional constraint equations (typically 2 per contact). |
| \(n_{\text{ctrl}}\) | **Number of Control Constraints** | The number of active control constraint equations. |
| \(n_c\) | **Total Number of Constraints** | The sum of all active constraints: \(n_c = n_b + n_n + n_f + n_{\text{ctrl}}\). |

**Note on \(n_q\) vs \(n_u\):** The relationship between configuration and velocity dimensions depends on the object representation:

**Rigid Bodies**: \(n_q = 7 \cdot n_{rb}\), \(n_u = 6 \cdot n_{rb}\)
  
- Configuration: \(\mathbf{q}_i = [\mathbf{x}_i, \boldsymbol{\theta}_i]\) where \(\mathbf{x}_i \in \mathbb{R}^3\) (position), \(\boldsymbol{\theta}_i \in \mathbb{R}^4\) (quaternion)
- Velocity: \(\mathbf{u}_i = [\mathbf{v}_i, \boldsymbol{\omega}_i]\) where \(\mathbf{v}_i \in \mathbb{R}^3\) (linear velocity), \(\boldsymbol{\omega}_i \in \mathbb{R}^3\) (angular velocity)

**Particles**: \(n_q = n_u = 3 \cdot n_{particles}\)
  
- Configuration/Velocity: \(\mathbf{q}_i = \mathbf{u}_i = \mathbf{x}_i \in \mathbb{R}^3\) (position/velocity only)

**Mixed Systems**: \(n_q\) and \(n_u\) computed based on the specific objects in the system

---

## 2. State Vectors and Their Components

These are the primary variables that the solver calculates at each time step. The full state is denoted by \(\mathbf{x} = [\mathbf{q}, \mathbf{u}, \boldsymbol{\lambda}]\).

| Symbol | Name / Description | Dimensions |
| :--- | :--- | :--- |
| \(\mathbf{q}\) | **Generalized Configuration** | \(\mathbb{R}^{n_q}\) |
| \(\mathbf{q}_i\) | Configuration of object \(i\) | Varies by object type (e.g., \(\mathbb{R}^7\) for rigid bodies, \(\mathbb{R}^3\) for particles) |
| \(\mathbf{u}\) | **Generalized Velocity** | \(\mathbb{R}^{n_u}\) |
| \(\mathbf{u}_i\) | Velocity of object \(i\) | Varies by object type (e.g., \(\mathbb{R}^6\) for rigid bodies, \(\mathbb{R}^3\) for particles) |
| \(\boldsymbol{\lambda}\) | **Constraint Forces (Lagrange Multipliers)** | \(\mathbb{R}^{n_c}\) |
| \(\boldsymbol{\lambda}_b\) | Forces for bilateral constraints (joints) | \(\mathbb{R}^{n_b}\) |
| \(\boldsymbol{\lambda}_n\) | Forces for unilateral constraints (contacts) | \(\mathbb{R}^{n_n}\) |
| \(\boldsymbol{\lambda}_f\) | Forces for frictional constraints | \(\mathbb{R}^{n_f}\) |
| \(\boldsymbol{\lambda}_{\text{ctrl}}\) | Forces for control constraints | \(\mathbb{R}^{n_{\text{ctrl}}}\) |

---

## 3. System Matrices and Parameters

These matrices and scalar parameters define the physical properties of the system.

| Symbol | Name / Description | Dimensions |
| :--- | :--- | :--- |
| \(\mathbf{M}\) | **Spatial Mass Matrix** | \(\mathbb{R}^{n_q \times n_q}\) |
| \(\mathbf{G}(\mathbf{q})\) | **Kinematic Mapping** | \(\mathbb{R}^{n_q \times n_u}\) |
| \(\mathbf{\tilde{M}}\) | **Generalized Mass Matrix** | \(\mathbb{R}^{n_u \times n_u}\) |

**Relationship between \(\mathbf{M}\), \(\mathbf{G}\), and \(\mathbf{\tilde{M}}\):**

- \(\mathbf{M}\) is the mass matrix that operates on configuration derivatives (\(\dot{\mathbf{q}}\)).
- \(\mathbf{G}(\mathbf{q})\) is the matrix that maps generalized velocities to configuration derivatives:

\[
\dot{\mathbf{q}} = \mathbf{G}(\mathbf{q}) \cdot \mathbf{u}.
\]

This is necessary to handle the 7D quaternion representation of orientation.

- \(\mathbf{\tilde{M}}\) is the mass matrix in the space of generalized velocities (\(\mathbf{u}\)). It is derived via the kinematic mapping:

\[
\mathbf{\tilde{M}} = \mathbf{G}(\mathbf{q})^\top \cdot \mathbf{M} \cdot \mathbf{G}(\mathbf{q})
\]

---

## 4. Constraint and Residual Functions

These functions define the physical laws and constraints that must be satisfied. The solver works by finding the root of the stacked residual vector \(\mathbf{h}\).

| Symbol | Name / Description | Dimensions |
| :--- | :--- | :--- |
| \(\mathbf{c}(\mathbf{q})\) | **Position-level Constraint Function** | \(\mathbb{R}^{n_c}\) |
| \(\mathbf{c}_b(\mathbf{q})\) | Function for bilateral (joint) constraints | \(\mathbb{R}^{n_b}\) |
| \(\mathbf{c}_n(\mathbf{q})\) | Function for unilateral (contact) constraints (gap function) | \(\mathbb{R}^{n_n}\) |
| \(\mathbf{c}_f(\mathbf{q})\) | Function for friction constraints (tangential distances) | \(\mathbb{R}^{n_f}\) |
| \(\mathbf{h}(\mathbf{x})\) | **Nonlinear Residual Function** (Root Function) | \(\mathbb{R}^{n_{sys}}\) |
| \(\mathbf{h}_{\text{dyn}}\) | Residual for the equations of motion (Dynamics) | \(\mathbb{R}^{n_u}\) |
| \(\mathbf{h}_{\text{kin}}\) | Residual for the time integration scheme (Kinematics) | \(\mathbb{R}^{n_q}\) |
| \(\mathbf{h}_b\) | Residual for bilateral constraints | \(\mathbb{R}^{n_b}\) |
| \(\mathbf{h}_n\) | Residual for contact constraints | \(\mathbb{R}^{n_n}\) |
| \(\mathbf{h}_f\) | Residual for friction constraints | \(\mathbb{R}^{n_f}\) |
| \(\mathbf{h}_{\text{ctrl}}\) | Residual for control constraints | \(\mathbb{R}^{n_{\text{ctrl}}}\) |

**Note:** The total system dimension is \(n_{sys} = n_u + n_q + n_c\), representing the vector

\[
\mathbf{h} = \begin{bmatrix}
        \mathbf{h}_{\text{dyn}} \\
        \mathbf{h}_{\text{kin}} \\
        \mathbf{h}_{\text{b}} \\
        \mathbf{h}_{\text{n}} \\
        \mathbf{h}_{\text{f}} \\
        \mathbf{h}_{\text{ctrl}} \\
    \end{bmatrix}.
\]

---

## 5. Jacobians and System Derivatives

Jacobians are matrices of partial derivatives that are essential for linearizing the system. The following table defines the key derivative matrices used in the system.

| Symbol | Name | Definition | Dimensions |
| :--- | :--- | :--- | :--- |
| \(\mathbf{J}\) | **Velocity Jacobian** | \(\mathbf{J} = \frac{\partial \mathbf{c}}{\partial \mathbf{q}} \cdot \mathbf{G}(\mathbf{q})\) | \(n_c \times n_u\) |
| \(\mathbf{J}_b\) | Velocity Jacobian for bilateral constraints | \(\mathbf{J}_b = \frac{\partial \mathbf{c}_b}{\partial \mathbf{q}} \cdot \mathbf{G}(\mathbf{q})\) | \(n_b \times n_u\) |
| \(\mathbf{J}_n\) | Velocity Jacobian for contact constraints | \(\mathbf{J}_n = \frac{\partial \mathbf{c}_n}{\partial \mathbf{q}} \cdot \mathbf{G}(\mathbf{q})\) | \(n_n \times n_u\) |
| \(\mathbf{J}_f\) | Velocity Jacobian for friction constraints | \(\mathbf{J}_f = \frac{\partial \mathbf{c}_f}{\partial \mathbf{q}} \cdot \mathbf{G}(\mathbf{q})\) | \(n_f \times n_u\) |
| \(\mathbf{J}_{\text{ctrl}}\) | Velocity Jacobian for control constraints | \(\mathbf{J}_{\text{ctrl}}\) (See [Control Constraints](./constraints.md#5-control-constraints)) | \(n_{\text{ctrl}} \times n_u\) |
| \(\hat{\mathbf{J}}\) | **System Jacobian Block** | \(\hat{\mathbf{J}} = \frac{\partial \mathbf{h}}{\partial \mathbf{u}}\) | \(n_c \times n_u\) |
| \(\hat{\mathbf{J}}_b\) | System Jacobian block for bilateral constraints | \(\hat{\mathbf{J}}_b = \frac{\partial \mathbf{h}_b}{\partial \mathbf{u}}\) | \(n_b \times n_u\) |
| \(\hat{\mathbf{J}}_n\) | System Jacobian block for contact constraints | \(\hat{\mathbf{J}}_n = \frac{\partial \mathbf{h}_n}{\partial \mathbf{u}}\) | \(n_n \times n_u\) |
| \(\hat{\mathbf{J}}_f\) | System Jacobian block for friction constraints | \(\hat{\mathbf{J}}_f = \frac{\partial \mathbf{h}_f}{\partial \mathbf{u}}\) | \(n_f \times n_u\) |
| \(\hat{\mathbf{J}}_{\text{ctrl}}\) | System Jacobian block for control constraints | \(\hat{\mathbf{J}}_{\text{ctrl}} = \frac{\partial \mathbf{h}_{\text{ctrl}}}{\partial \mathbf{u}}\) | \(n_{\text{ctrl}} \times n_u\) |
| \(\mathbf{C}\) | **Compliance Block** | \(\mathbf{C} = \frac{\partial \mathbf{h}_c}{\partial \boldsymbol{\lambda}}\) | \(n_c \times n_c\) |
| \(\mathbf{C}_b\) | Compliance block for bilateral constraints | \(\mathbf{C}_b = \frac{\partial \mathbf{h}_b}{\partial \boldsymbol{\lambda}_b}\) | \(n_b \times n_b\) |
| \(\mathbf{C}_n\) | Compliance block for contact constraints | \(\mathbf{C}_n = \frac{\partial \mathbf{h}_n}{\partial \boldsymbol{\lambda}_n}\) | \(n_n \times n_n\) |
| \(\mathbf{C}_f\) | Compliance block for friction constraints | \(\mathbf{C}_f = \frac{\partial \mathbf{h}_f}{\partial \boldsymbol{\lambda}_f}\) | \(n_f \times n_f\) |
| \(\mathbf{C}_{\text{ctrl}}\) | Compliance block for control constraints | \(\mathbf{C}_{\text{ctrl}} = \frac{\partial \mathbf{h}_{\text{ctrl}}}{\partial \boldsymbol{\lambda}_{\text{ctrl}}}\) | \(n_{\text{ctrl}} \times n_{\text{ctrl}}\) |

### Velocity Jacobians (\(\mathbf{J}\))

The velocity Jacobians map generalized velocities \(\mathbf{u}\) directly to constraint derivatives. They are formed by composing the geometric constraint Jacobian with the kinematic mapping:

\[
\dot{\mathbf{c}} = \mathbf{J} \cdot \mathbf{u}
\]

**Constraint-specific Jacobians:**

- **\(\mathbf{J}_b\)**: Maps body velocities to bilateral constraint violation rates (joint separation velocities)
- **\(\mathbf{J}_n\)**: Maps body velocities to contact constraint violation rates (normal approach velocities)
- **\(\mathbf{J}_f\)**: Maps body velocities to friction constraint violation rates (tangential slip velocities)

The full velocity Jacobian is:

\[
\mathbf{J} = \begin{bmatrix}
        \mathbf{J}_{\text{b}} \\
        \mathbf{J}_{\text{n}} \\
        \mathbf{J}_{\text{f}} \\
    \end{bmatrix}.
\]

### System Jacobian Blocks (\(\hat{\mathbf{J}}\))

The system Jacobian blocks are the actual matrices that appear in the linearized system solved by Newton's method. They represent \(\frac{\partial \mathbf{h}}{\partial \mathbf{u}}\) for each constraint type.

**Key differences from velocity Jacobians:**

- **For Bilateral Constraints**: (identical to velocity Jacobian)

\[
\hat{\mathbf{J}}_b = \mathbf{J}_b
\]

- **For Contact Constraints**:

\[
\hat{\mathbf{J}}_n = s_n \cdot \mathbf{J}_n
\]

- **For Friction Constraints**: (identical, by design for symmetry)

\[
\hat{\mathbf{J}}_f = \mathbf{J}_f
\]

The \(s_n\) is a state-dependent scaling factor from the Fischer-Burmeister NCP-function derivative. The scaling factor \(s_n\) for contacts arises because the contact residual \(\mathbf{h}_n\) is nonlinear in the velocities due to the NCP-function formulation.

### Compliance Blocks (\(\mathbf{C}\))

Compliance blocks represent the derivative of constraint residuals with respect to constraint impulses: \(\mathbf{C} = \frac{\partial \mathbf{h}_c}{\partial \boldsymbol{\lambda}}\). They appear on the diagonal of the constraint portion of the system matrix and control the "softness" of constraints.

**Bilateral Compliance \(\mathbf{C}_b\):** This is the physical compliance matrix \(\boldsymbol{\Sigma}\) from Baumgarte stabilization. It allows modeling of soft joints or provides numerical stabilization for rigid constraints. For perfectly rigid constraints, this matrix is zero. When non-zero, it represents the inverse stiffness of the joint, enabling compliant behavior. The matrix has dimensions \(n_b \times n_b\) and is typically diagonal, with each diagonal entry corresponding to the compliance of a specific bilateral constraint.

**Contact Compliance \(\mathbf{C}_n\):** This represents numerical compliance arising from the Fischer-Burmeister NCP-function formulation. Unlike bilateral compliance, this is not a physical property but rather a mathematical artifact of the complementarity problem transformation. The matrix is diagonal with dimensions \(n_n \times n_n\), where each entry is computed as \(c_n = \frac{\partial \phi_n}{\partial \lambda_n}\). These values are state-dependent and vary during Newton iterations as the contact conditions evolve.

**Friction Compliance \(\mathbf{C}_f\):** This is the fixed-point iteration matrix \(\mathbf{W}\) used to create a symmetric system for efficient solving. The matrix is diagonal with dimensions \(n_f \times n_f\) and is updated at each Newton iteration. It is carefully designed so that when the Newton method converges, the exact Coulomb friction conditions are satisfied. This formulation enables the use of symmetric solvers like Preconditioned Conjugate Residual (PCR) for computational efficiency.

The complete compliance block assembles these individual blocks in a block-diagonal structure: \(\mathbf{C} = \text{diag}(\mathbf{C}_b, \mathbf{C}_n, \mathbf{C}_f)\).

---

## 6. Differentiable Simulation

These symbols are used in the [Adjoint Method](./adjoint-method.md) section.

| Symbol | Name / Description | Dimensions |
| :--- | :--- | :--- |
| \(\mathbf{s}\) | **Full State Vector** \([\mathbf{q}, \mathbf{u}, \boldsymbol{\lambda}]\) | \(\mathbb{R}^{n_q + n_u + n_c}\) |
| \(\mathbf{s}^-\) | State at the current time step (known) | \(\mathbb{R}^{n_q + n_u + n_c}\) |
| \(\mathbf{s}^+\) | State at the next time step (solved for) | \(\mathbb{R}^{n_q + n_u + n_c}\) |
| \(\mathbf{a}\) | **Actuation Vector** (control targets: target positions / velocities) | \(\mathbb{R}^{n_\text{ctrl}}\) |
| \(\boldsymbol{\theta}\) | **World Parameters** (masses, inertias, friction coefficients, gains, etc.) | \(\mathbb{R}^{n_\theta}\) |
| \(\mathbf{R}\) | **Implicit Residual** \(\mathbf{R}(\mathbf{s}^+, \mathbf{s}^-, \mathbf{a}^-, \boldsymbol{\theta}) = \mathbf{0}\) | \(\mathbb{R}^{n_q + n_u + n_c}\) |
| \(\mathcal{L}\) | **Scalar Loss Function** evaluated on the simulation trajectory | \(\mathbb{R}\) |
| \(\mathbf{w}\) | **Adjoint Vector** \([\mathbf{w}_q, \mathbf{w}_u, \mathbf{w}_\lambda]\) | \(\mathbb{R}^{n_q + n_u + n_c}\) |
| \(\mathbf{w}_q\) | Adjoint variable for configuration | \(\mathbb{R}^{n_q}\) |
| \(\mathbf{w}_u\) | Adjoint variable for velocity | \(\mathbb{R}^{n_u}\) |
| \(\mathbf{w}_\lambda\) | Adjoint variable for constraint forces | \(\mathbb{R}^{n_c}\) |
| \(\nabla_{\mathbf{s}^+}\mathcal{L}\) | Gradient of the loss with respect to the next state | \(\mathbb{R}^{n_q + n_u + n_c}\) |
