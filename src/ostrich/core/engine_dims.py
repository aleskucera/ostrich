from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class EngineDimensions:
    """
    Calculates and stores the dimensions and offsets for the physics simulation.
    This provides a single source of truth for the size and layout of all system
    vectors and matrices, with helper properties for easy slicing.
    """

    # --- Primary Inputs ---
    num_worlds: int
    body_count: int
    contact_count: int
    joint_count: int
    joint_dof_count: int
    linesearch_step_count: int
    joint_constraint_count: int
    control_constraint_count: int

    @cached_property
    def N_w(self) -> int:
        return self.num_worlds

    @cached_property
    def N_b(self) -> int:
        return self.body_count

    @cached_property
    def N_u(self) -> int:
        return 6 * self.body_count

    @cached_property
    def N_q(self) -> int:
        return 7 * self.body_count

    @cached_property
    def N_j(self) -> int:
        return self.joint_constraint_count

    @cached_property
    def N_ctrl(self) -> int:
        return self.control_constraint_count

    @cached_property
    def N_n(self) -> int:
        return self.contact_count

    @cached_property
    def N_f(self) -> int:
        return 2 * self.contact_count

    @cached_property
    def num_constraints(self) -> int:
        return self.joint_constraint_count + self.control_constraint_count + 3 * self.contact_count

    @cached_property
    def N_c(self) -> int:
        return self.N_j + self.N_ctrl + self.N_n + self.N_f

    @cached_property
    def N_alpha(self) -> int:
        return self.linesearch_step_count

    # --- Per-Constraint-Type Offsets ---
    @cached_property
    def offset_j(self) -> int:
        """Start offset of joint constraints block."""
        return 0

    @cached_property
    def offset_ctrl(self) -> int:
        """Start offset of control constraints block."""
        return self.N_j

    @cached_property
    def offset_n(self) -> int:
        """Start offset of normal contact constraints block."""
        return self.N_j + self.N_ctrl

    @cached_property
    def offset_f(self) -> int:
        """Start offset of friction constraints block."""
        return self.N_j + self.N_ctrl + self.N_n

    # --- Slicing Helper Properties ---
    @cached_property
    def slice_j(self) -> slice:
        """Returns a slice object for the joint-constraint block."""
        return slice(self.offset_j, self.offset_ctrl)

    @cached_property
    def slice_ctrl(self) -> slice:
        """Returns a slice object for the control-constraint block."""
        return slice(self.offset_ctrl, self.offset_n)

    @cached_property
    def slice_n(self) -> slice:
        """Returns a slice object for the normal-constraint block."""
        return slice(self.offset_n, self.offset_f)

    @cached_property
    def slice_f(self) -> slice:
        """Returns a slice object for the friction-constraint block."""
        return slice(self.offset_f, self.N_c)
