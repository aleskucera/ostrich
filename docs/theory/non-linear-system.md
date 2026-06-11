# The Nonlinear System

This section details how the optimization problem from [Gauss's Principle of Least Constraint](./gauss-least-constraint.md), combined with the constraint laws from [Constraints Formulation](./constraints.md), is formally expressed as a large, simultaneous system of nonlinear equations. The goal is to find the root of this system, which represents the physically correct state of all bodies at the next time step.

This transformation from a constrained optimization problem to a root-finding problem is achieved by applying the Karush-Kuhn-Tucker (KKT) conditions. The resulting system can be expressed as a single function \(\mathbf{h}(\mathbf{x}^+) = \mathbf{0}\), where the unknown vector \(\mathbf{x}^+ = [\mathbf{q}^+, \mathbf{u}^+, \boldsymbol{\lambda}^+]\) contains the final configurations, velocities, and constraint forces.

The following sections will deconstruct the residual vector \(\mathbf{h}\) piece by piece, explaining the physical meaning and mathematical formulation of each component before assembling them into the final system.

---

## **Residual Functions**

### Dynamics: The Equations of Motion (\(\mathbf{h_\text{dyn}}\))

This residual represents the core equations of motion, a discrete-time-step version of Newton's second law expressed in generalized coordinates. It stems directly from Gauss's Principle and dictates how constraint impulses alter the system's velocity.

The equation is:

\[
\mathbf{h_\text{dyn}} = \mathbf{\tilde{M}} \cdot (\mathbf{u}^+ - \tilde{\mathbf{u}}) - h \mathbf{J}_b^T \boldsymbol{\lambda}_b^+ - h \mathbf{J}_n^T \boldsymbol{\lambda}_n^+ - h \mathbf{J}_f^T \boldsymbol{\lambda}_f^+ - h \mathbf{J}_{\text{ctrl}}^T \boldsymbol{\lambda}_{\text{ctrl}}^+ = \mathbf{0}
\]

Breaking this down:

- \(\mathbf{\tilde{M}} \cdot (\mathbf{u}^+ - \tilde{\mathbf{u}})\) is the change in the system's generalized momentum caused by constraints. The term \(\tilde{\mathbf{u}} = \mathbf{u}^- + h \mathbf{\tilde{M}}^{-1} \mathbf{f}_{\text{ext}}\) represents the predicted "unconstrained" velocity—what the velocity *would be* if only external forces like gravity were applied.
- \(h(\mathbf{J}_b^T \boldsymbol{\lambda}_b^+ + \mathbf{J}_n^T \boldsymbol{\lambda}_n^+ + \mathbf{J}_f^T \boldsymbol{\lambda}_f^+ + \mathbf{J}_{\text{ctrl}}^T \boldsymbol{\lambda}_{\text{ctrl}}^+)\) is the total impulse applied over the time step \(h\) by all constraint forces (bilateral, normal contact, friction, and control). The factor \(h\) converts the constraint forces to impulses, and the Jacobians \(\mathbf{J}^T\) map them from constraint space back into generalized coordinate space.

In essence, this equation states: "The change in momentum from the unconstrained state to the final state must be exactly equal to the total impulse \(h \mathbf{J}^T \boldsymbol{\lambda}\) applied by all constraint forces."

### Kinematics: Time Integration (\(\mathbf{h_\text{kin}}\))

This residual connects the system's final configuration \(\mathbf{q}^+\) to its final velocity \(\mathbf{u}^+\) through a time integration rule. Ostrich uses an implicit integration scheme for superior stability.

The equation is:

\[
\mathbf{h_\text{kin}} = \mathbf{q}^+ - \mathbf{q}^- - h \cdot \mathbf{G}(\mathbf{q}^+) \cdot \mathbf{u}^+ = \mathbf{0}
\]

Here:

- This is a **Backward Euler** integration step. It defines the final position \(\mathbf{q}^+\) based on the final velocity \(\mathbf{u}^+\). Because the quantity we are solving for (\(\mathbf{u}^+\)) is used to compute the final state, the method is *implicit*. This is crucial for stability in stiff systems, like those with many contacts and joints.
- The matrix \(\mathbf{G}(\mathbf{q}^+)\) is the **kinematic mapping** that transforms generalized velocities (which have \(n_u\) dimensions, e.g., 6 for a rigid body) into configuration derivatives (which have \(n_q\) dimensions, e.g., 7 for a position +quaternion-based orientation of a  rigid body).

This equation ensures that the final configuration and velocity are mutually consistent according to the laws of motion over the discrete time step \(h\).

### Bilateral Constraints (Joints)

Bilateral constraints enforce an exact geometric relationship (\(\mathbf{c}_b(\mathbf{q}) = \mathbf{0}\)). Ostrich can enforce this at either the position or velocity level.

- **Position-Level Formulation:** This is Ostrich's preferred method as it completely eliminates numerical drift. The constraint is enforced directly on the final configuration \(\mathbf{q}^+\). A compliance matrix \(\boldsymbol{\Sigma}\) can be introduced to model "soft" joints or improve numerical conditioning. The resulting residual equation is:

\[
\mathbf{h_b}^{(\text{pos})} = \mathbf{c}_b(\mathbf{q}^+) + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+ = \mathbf{0}
\]

- **Velocity-Level Formulation (Alternative):** This alternative enforces the constraint on velocities (\(\mathbf{J}_b \cdot \mathbf{u} = \mathbf{0}\)). To counteract the inevitable positional drift from numerical integration, **Baumgarte stabilization** adds a correction term that pushes the system back toward the valid configuration. The residual equation becomes:

\[
\mathbf{h_b}^{(\text{vel})} = \mathbf{J}_b \cdot \mathbf{u}^+ + \boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_b(\mathbf{q}^-)}{h} + \boldsymbol{\Sigma} \cdot \boldsymbol{\lambda}_b^+ = \mathbf{0}
\]

   where the term \(\boldsymbol{\Upsilon} \cdot \frac{\mathbf{c}_b(\mathbf{q}^-)}{h}\) introduces a velocity goal that attempts to correct a fraction of the existing position error \(\mathbf{c}_b(\mathbf{q}^-)\) over the current time step.

### Unilateral Constraints (Contacts)

Contact non-penetration (\(\mathbf{c_n}(\mathbf{q}) \geq 0\)) is governed by a complementarity condition: a repulsive impulse (\(\lambda_n \geq 0\)) can only exist upon contact (\(\mathbf{c_n}(\mathbf{q}) = 0\)).

- **Position-Level Formulation:** The core physical principle is the Nonlinear Complementarity Problem (NCP):

\[
0 \leq \lambda_n \perp \mathbf{c}_{n}(\mathbf{q}) \geq 0
\]

To integrate this into a Newton-based solver, we convert this non-smooth condition into a smooth equation using the **Fischer-Burmeister NCP-function**, \(\boldsymbol{\phi}\). The residual equation becomes:

\[
\mathbf{h_n}^{(\text{pos})} = \boldsymbol{\phi}(\mathbf{c_n}(\mathbf{q}^+), \boldsymbol{\lambda_n}^+) = \mathbf{c_n}(\mathbf{q}^+) + \boldsymbol{\lambda_n}^+ - \sqrt{(\mathbf{c_n}(\mathbf{q}^+))^2 + (\boldsymbol{\lambda}_n^+)^2} = \mathbf{0}
\]

- **Velocity-Level Formulation (Alternative):** At the velocity level, the complementarity applies to the post-collision relative normal velocity. To model bounce (via restitution \(e \in [0, 1] \)) and combat drift (Baumgarte stabilization), we define a target velocity:

\[
\mathbf{v_{n, \text{target}}}^+ = \mathbf{J_n} \cdot (\mathbf{u}^+ + e \cdot \mathbf{u}^- ) + \boldsymbol{\Upsilon} \cdot \frac{\mathbf{c_n}(\mathbf{q}^-)}{h}
\]

The complementarity problem is now \(0 \leq \lambda_n \perp \mathbf{v_{n, \text{target}}}^+ \geq 0\). Applying the Fischer-Burmeister function gives the final residual:

\[
\mathbf{h_n}^{(\text{vel})} = \boldsymbol{\phi}(\mathbf{v_{n, \text{target}}}^+, \boldsymbol{\lambda_n}^+) = \mathbf{v_{n, \text{target}}}^+ + \boldsymbol{\lambda_n}^+ - \sqrt{(\mathbf{v_{n, \text{target}}}^+)^2 + (\boldsymbol{\lambda}_n^+)^2} = \mathbf{0}
\]

### Friction Constraints

Friction resists tangential motion and is formulated at the velocity level using the **principle of maximal dissipation**, subject to the Coulomb friction law:

\[
\|\boldsymbol{\lambda}_t\| \leq \mu \cdot \lambda_n
\]

The resulting Karush-Kuhn-Tucker (KKT) conditions mathematically express the stick-slip behavior:

1. **Directional Constraint:** The friction impulse must oppose the direction of slip.

\[
\mathbf{J}_f^T \cdot \mathbf{u} + \frac{|\mathbf{J}_f^T \cdot \mathbf{u}|}{|\boldsymbol{\lambda}_f|} \cdot \boldsymbol{\lambda}_f = \mathbf{0}
\]

2. **Stick-Slip Switching (Complementarity):** Either the bodies are slipping and friction is maximal, or they are sticking and friction is sub-maximal.

\[
0 \leq |\mathbf{J}_f^T \cdot \mathbf{u}| \perp \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\| \geq 0
\]

To make this complex system solvable and efficient, Ostrich employs a two-step transformation. First, the complementarity condition is turned into a root-finding problem with an NCP function, \(\phi_f\). Second, a **fixed-point iteration** is introduced via a carefully constructed scalar compliance term \(W\):

\[
W = \frac{|\mathbf{J}_f^T \cdot \mathbf{u}| - \phi_f(|\mathbf{J}_f^T \cdot \mathbf{u}|, \mu \cdot \lambda_n - \|\boldsymbol{\lambda}_f\|)}{\|\boldsymbol{\lambda}_f\| + \phi_f(|\mathbf{J}_f^T \cdot \mathbf{u}|, \mu \cdot \lambda_n - \|\boldsymbol{\lambda_f}\|)}
\]

This allows the entire friction model to be distilled into a single, elegant residual equation that crucially leads to a symmetric system matrix:

\[
\mathbf{h_f} = \mathbf{J}_f^T \cdot \mathbf{u}^+ + \mathbf{W} \cdot \boldsymbol{\lambda}_f^+ = \mathbf{0}
\]

### Control Constraints

Control constraints drive the system towards a target state (position or velocity). They are formulated as linear equations with compliance, effectively acting as implicit motors.

\[
\mathbf{h}_{\text{ctrl}} = \mathbf{J}_{\text{ctrl}} \cdot \mathbf{u}^+ + \mathbf{C}_{\text{ctrl}} \cdot \boldsymbol{\lambda}_{\text{ctrl}}^+ + \mathbf{b}_{\text{ctrl}} = \mathbf{0}
\]

Here, \(\mathbf{J}_{\text{ctrl}}\) is the Jacobian relating joint velocities to the control error, \(\mathbf{C}_{\text{ctrl}}\) is the diagonal compliance matrix (containing the \(\alpha\) terms described in the [Constraints](./constraints.md) section), and \(\mathbf{b}_{\text{ctrl}}\) accounts for the target values and any position-level error correction (e.g., \((q - q_{\text{target}})/h\)).

---

## **The Complete Nonlinear System**

Assembling all the individual residual blocks yields the complete nonlinear system that Ostrich must solve at every time step.

The full residual vector is stacked as follows:

\[
\mathbf{h}(\mathbf{x}^+) = \begin{bmatrix} \mathbf{h_\text{dyn}} \\ \mathbf{h_\text{kin}} \\ \mathbf{h_b} \\ \mathbf{h_n} \\ \mathbf{h_f} \\ \mathbf{h}_{\text{ctrl}} \end{bmatrix} = \mathbf{0}
\]

Explicitly, the full system of equations is:

\[
\begin{align*}
\text{Dynamics:} \quad & \mathbf{\tilde{M}} \cdot (\mathbf{u}^+ - \tilde{\mathbf{u}}) - h \mathbf{J}_b^T \boldsymbol{\lambda}_b^+ - h \mathbf{J}_n^T \boldsymbol{\lambda}_n^+ - h \mathbf{J}_f^T \boldsymbol{\lambda}_f^+ - h \mathbf{J}_{\text{ctrl}}^T \boldsymbol{\lambda}_{\text{ctrl}}^+ = \mathbf{0} \\
\text{Kinematics:} \quad & \mathbf{q}^+ - \mathbf{q}^- - h \cdot \mathbf{G}(\mathbf{q}^+) \cdot \mathbf{u}^+ = \mathbf{0} \\
\text{Bilateral:} \quad & \mathbf{h_b}^{(\text{pos})} \quad \text{or} \quad \mathbf{h_b}^{(\text{vel})} = \mathbf{0} \\
\text{Contact:} \quad & \mathbf{h_n}^{(\text{pos})} \quad \text{or} \quad \mathbf{h_n}^{(\text{vel})} = \mathbf{0} \\
\text{Friction:} \quad & \mathbf{J}_f^T \cdot \mathbf{u}^+ + \mathbf{W} \cdot \boldsymbol{\lambda}_f^+ = \mathbf{0} \\
\text{Control:} \quad & \mathbf{J}_{\text{ctrl}} \cdot \mathbf{u}^+ + \mathbf{C}_{\text{ctrl}} \cdot \boldsymbol{\lambda}_{\text{ctrl}}^+ + \mathbf{b}_{\text{ctrl}} = \mathbf{0}
\end{align*}
\]

This unified system represents all physical laws acting simultaneously. At each iteration of the numerical solver, all matrices (\(\mathbf{J}\), \(\mathbf{G}\), \(\mathbf{W}\), etc.) are evaluated. The solution to \(\mathbf{h}(\mathbf{x}^+) = \mathbf{0}\) is a state vector \(\mathbf{x}^+\) that satisfies dynamics, kinematics, and all physical constraints to a high degree of precision, ready for the next simulation frame.

→ **Next**: [Numerical Solution](./numerical-solution.md)
