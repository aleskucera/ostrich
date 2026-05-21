"""Adjoint (reverse-mode autodiff) kernels for the Axion engine.

These kernels implement the implicit-function-theorem backward pass:
adjoint RHS assembly, body-adjoint initialization, constraint feedback
subtraction, and the friction-mode freezing logic that keeps the
adjoint linear system well-posed when contact / friction modes flip
during the forward pass.

Consumed by ``AxionEngineBase`` when the engine is constructed with
``differentiable_simulation=True``. The ``AdjointConfig`` sub-config
itself stays in ``axion.core.engine_config`` because it is part of the
engine config tree.
"""
from .adjoint_friction import adjoint_regularize_compliance_kernel
from .adjoint_friction import freeze_contact_mode_kernel
from .adjoint_friction import freeze_contact_mode_soft_kernel
from .adjoint_utils import compute_adjoint_rhs_kernel
from .adjoint_utils import compute_body_adjoint_init_kernel
from .adjoint_utils import subtract_constraint_feedback_kernel


__all__ = [
    "adjoint_regularize_compliance_kernel",
    "compute_adjoint_rhs_kernel",
    "compute_body_adjoint_init_kernel",
    "freeze_contact_mode_kernel",
    "freeze_contact_mode_soft_kernel",
    "subtract_constraint_feedback_kernel",
]
