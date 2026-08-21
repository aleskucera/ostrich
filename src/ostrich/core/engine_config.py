"""Engine config dataclasses.

`OstrichEngineConfig` is structured as a small set of sub-configs, one
per concern:

  nr            - Newton-Raphson outer loop control
  linear        - PCR linear solver + preconditioner
  compliance    - per-constraint-type compliance values
  relaxation    - structural relaxations (alternative constraint forms)
  linesearch    - opt-in line search (off by default)
  warm_start    - cross-step contact warm-start + cold-start heuristics
  contacts      - max contacts per world + per-pair reduction policy
  adjoint       - differentiable-simulation adjoint pass options
  profiling     - segment timing + event-based GPU profiler

Each sub-config validates itself in __post_init__; OstrichEngineConfig's
own __post_init__ coerces Hydra DictConfig overrides through each
sub-config's `coerce()` classmethod (so partial YAML overrides like
`engine.linesearch.enabled=true` work even when only one field is set).

The non-Ostrich engine configs (Featherstone, MuJoCo, XPBD, SemiImplicit)
live at the bottom of this module and remain flat — they wrap upstream
Newton solvers which keep their own conventions.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from ostrich.collision import ContactReductionConfig

from ostrich.profiling import VALID_MODES as _VALID_PROFILING_MODES


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _coerce(cls, value):
    """Coerce a Hydra DictConfig (or dict-like) into an instance of `cls`.

    Hydra leaves nested overrides as DictConfig at instantiation time
    rather than as a real dataclass instance. This walker re-instantiates
    by filtering to the dataclass's known fields and constructing fresh.
    Returns `value` unchanged if it's already an instance of `cls`.
    """
    if isinstance(value, cls):
        return value
    try:
        items = dict(value)
    except Exception:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in items.items() if k in valid})


# -----------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    """
    Base configuration class.
    Defines the factory interface for creating physics engines.
    """

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: Optional[bool] = None,
    ) -> Any:
        """
        Factory method to create the appropriate solver instance.

        For standard Newton solvers (Featherstone, MuJoCo, etc.), this
        automatically passes all configuration fields as kwargs.

        `differentiable_simulation`: explicit kwarg overrides the
        config's own field (if any). Pass None to use the config field.

        Note: We intentionally DO NOT pass 'logging_config' to generic solvers
        because they do not support this custom Ostrich logging system.
        """
        solver_cls = self._get_solver_class()
        return solver_cls(model, **vars(self))

    def _get_solver_class(self):
        """Subclasses must implement this to return their solver class."""
        raise NotImplementedError(
            f"Solver class resolution not implemented for {type(self).__name__}"
        )


# -----------------------------------------------------------------------
# Sub-configs for OstrichEngineConfig
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonRaphsonConfig:
    """Outer Newton-Raphson loop control."""

    max_iters: int = 16
    backtrack_min_iter: int = 2
    atol: float = 1e-3
    # Under-relaxation theta of the friction weight w across NR iterations:
    #   w_used = (1 - theta) * w_prev + theta * w_raw
    # with w_prev a persistent per-contact buffer zeroed at each step start
    # (EngineData.friction_w). Damps the w oscillation of sliding contacts
    # near the cone boundary at truncated NR iteration counts. 1.0 (default)
    # is bit-exact with the unrelaxed solver. Threaded to the kernels as a
    # plain float argument (CUDA-graph-safe, per-engine-instance). The
    # batched linesearch-candidate kernels ignore it (known gap; they always
    # run unrelaxed).
    w_relaxation: float = 1.0

    def __post_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"nr.max_iters must be >= 1, got {self.max_iters}")
        if not (0.0 < self.w_relaxation <= 1.0):
            raise ValueError(
                f"nr.w_relaxation must be in (0, 1], got {self.w_relaxation}"
            )
        if self.backtrack_min_iter < 0:
            raise ValueError(
                f"nr.backtrack_min_iter must be >= 0, got {self.backtrack_min_iter}"
            )
        if self.backtrack_min_iter >= self.max_iters:
            raise ValueError(
                f"nr.backtrack_min_iter must be < nr.max_iters, "
                f"got {self.backtrack_min_iter} and {self.max_iters}"
            )
        if self.atol < 0:
            raise ValueError(f"nr.atol must be >= 0, got {self.atol}")

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class LinearSolverConfig:
    """Inner PCR linear solver + preconditioner."""

    max_iters: int = 16
    tol: float = 1e-5
    atol: float = 1e-5
    preconditioner_type: str = "jacobi"  # "jacobi" | "per_body_pair"
    regularization: float = 1e-6

    def __post_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"linear.max_iters must be >= 1, got {self.max_iters}")
        if self.tol < 0:
            raise ValueError(f"linear.tol must be >= 0, got {self.tol}")
        if self.atol < 0:
            raise ValueError(f"linear.atol must be >= 0, got {self.atol}")
        if self.regularization < 0:
            raise ValueError(
                f"linear.regularization must be >= 0, got {self.regularization}"
            )
        if self.preconditioner_type not in ("jacobi", "per_body_pair"):
            raise ValueError(
                "linear.preconditioner_type must be 'jacobi' | 'per_body_pair', "
                f"got {self.preconditioner_type!r}"
            )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class ComplianceConfig:
    """Per-constraint-type compliance (regularization on the FB block)."""

    joint: float = 1e-5
    contact: float = 1e-4
    friction: float = 1e-6
    # Smoothing parameter inside the contact-FB norm:
    #   φ_ε(a, b) = α·a + β·b − √((α·a)² + (β·b)² + fb_smooth_eps_sq).
    # Default 1e-8 reproduces pre-existing behavior. Larger values
    # remove the corner degeneracy at (λ_n>0, signed_dist=0) — the
    # place seeded warm-starts land — at the cost of slightly soft
    # complementarity (λ·g ≈ fb_smooth_eps_sq at convergence).
    contact_fb_smooth_eps_sq: float = 1e-8

    def __post_init__(self):
        for name in ("joint", "contact", "friction"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"compliance.{name} must be >= 0, got {v}")

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class RelaxationConfig:
    """Structural physics relaxations — alternative constraint formulations.

    Each field names a *textbook assumption*, not a fudge factor. A rung
    removes unknowns and/or removes complementarity; it does not merely
    re-tune stiffness. See RELAXATION_PLAN.md.

        friction  "cone"      — Fisher-Burmeister on the Coulomb cone (full)
                  "bilateral" — no-slip imposed at EVERY contact point.
                                Measured to weld a multi-point wheel to the
                                terrain (|omega| -> 0); kept as the naive
                                formulation for the attribution study, not
                                as a usable rung. See
                                constraints/friction_constraint.py
                                :bilateral_row_is_skipped.
                  "bilateral_patch" — no-slip imposed once per contact PAIR
                                (the patch), which is the well-posed reading
                                of "rolling without slipping". Unbounded,
                                sign-free multiplier; no complementarity, no
                                active set, no dependence on lambda_n.

        contact   "complementarity" — Fisher-Burmeister on (signed_dist, lambda_n)
                  "equality"        — persistent contact: signed_dist = 0 is
                                imposed on the DETECTED set (detection itself
                                is unchanged). lambda_n becomes sign-free, and
                                lambda_n < 0 is the certificate "this contact
                                should have separated".

        gyro      True  — keep the gyroscopic term w x (I w)
                  False — drop it. Free rung with an exact certificate
                          ||w x I w||*dt / ||M du||.

    The modes are passed to the kernels as scalars, so every thread in a
    launch takes the same branch. Rungs are selected per engine build.
    """

    friction: str = "cone"
    contact: str = "complementarity"
    gyro: bool = True
    # Minimum mu*lambda_n for a contact to receive a friction row.
    #
    # Under the cone this only has to skip exactly-zero contacts: the budget
    # mu*lambda_n scales continuously with load, so a barely-loaded contact is
    # automatically barely-relevant and costs nothing to include. 1e-6 is right
    # there, and is the historical value.
    #
    # A bilateral rung has no such scaling. Any contact past this threshold is
    # enforced with an unbounded multiplier, so this single number IS the
    # active-set decision. Measured: at rigid_gap >= 0.2 the chassis generates
    # contacts carrying lambda_n ~ 1e-4 against ~270 for a real wheel contact;
    # they clear 1e-6, pin the chassis, and the robot stops. Set this between
    # the two populations (~1e-2) and the gap dependence goes away.
    friction_load_gate: float = 1e-6

    _FRICTION_MODES = ("cone", "bilateral", "bilateral_patch")
    _CONTACT_MODES = ("complementarity", "equality")

    def __post_init__(self):
        if self.friction not in self._FRICTION_MODES:
            raise ValueError(
                f"relaxation.friction must be one of {self._FRICTION_MODES}, "
                f"got {self.friction!r}"
            )
        if self.friction_load_gate < 0:
            raise ValueError(
                "relaxation.friction_load_gate must be >= 0, "
                f"got {self.friction_load_gate}"
            )
        if self.contact not in self._CONTACT_MODES:
            raise ValueError(
                f"relaxation.contact must be one of {self._CONTACT_MODES}, "
                f"got {self.contact!r}"
            )

    @property
    def friction_mode(self) -> int:
        return self._FRICTION_MODES.index(self.friction)

    @property
    def contact_mode(self) -> int:
        return self._CONTACT_MODES.index(self.contact)

    @property
    def gyro_scale(self) -> float:
        return 1.0 if self.gyro else 0.0

    @property
    def is_default(self) -> bool:
        """True when every axis is at the full (unrelaxed) formulation."""
        return (
            self.friction == "cone"
            and self.contact == "complementarity"
            and self.gyro
            and self.friction_load_gate == 1e-6
        )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class LinesearchConfig:
    """Backtracking line search (off by default)."""

    enabled: bool = False
    min_step: float = 1e-6
    conservative_step_count: int = 32
    conservative_upper_bound: float = 0.05
    optimistic_step_count: int = 32
    optimistic_window: float = 0.2

    @property
    def num_steps(self) -> int:
        """Total candidate step count = conservative + optimistic."""
        return self.conservative_step_count + self.optimistic_step_count

    def __post_init__(self):
        if not self.enabled:
            return  # Skip strict validation when off; defaults are inactive.
        if self.conservative_step_count < 1:
            raise ValueError(
                "linesearch.conservative_step_count must be >= 1, "
                f"got {self.conservative_step_count}"
            )
        if self.optimistic_step_count < 1:
            raise ValueError(
                "linesearch.optimistic_step_count must be >= 1, "
                f"got {self.optimistic_step_count}"
            )
        if self.min_step < 0:
            raise ValueError(f"linesearch.min_step must be >= 0, got {self.min_step}")
        if self.conservative_upper_bound <= self.min_step:
            raise ValueError(
                "linesearch.conservative_upper_bound must be > min_step, "
                f"got {self.conservative_upper_bound} and {self.min_step}"
            )
        if self.optimistic_window < 0:
            raise ValueError(
                f"linesearch.optimistic_window must be >= 0, got {self.optimistic_window}"
            )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class WarmStartConfig:
    """Cross-step contact warm-start + cold-start heuristics for unmatched contacts.

    See `ostrich/collision/warm_start.py` for the predicted-position
    matching algorithm and the three cold-start terms.
    """

    enabled: bool = True
    cold_gravity: bool = True       # alpha: per-body residual-gravity split
    cold_impact: bool = True        # beta: m_eff * (-v_n) / dt
    cold_friction_v_threshold: float = 0.1  # gamma; auto-off when ≤ 0
    # Matching method:
    #   "position_match"  — original per-contact predicted-position match.
    #                       Reliable in stationary regimes, fails for
    #                       moving contacts (mesh-vertex jitter, terrain-
    #                       triangle changes, rolling-wheel kinematics).
    #   "pair_aggregate"  — sum prev-step λ_n per body pair, distribute
    #                       uniformly over current contacts in the same
    #                       pair. Robust to contact identity drift.
    method: str = "position_match"
    # If True, seed the NR initial iterate ``_constr_force`` from
    # ``_constr_force_prev_iter`` after warm_starter.apply. Only the
    # contact-normal slots are populated by warm_starter; joint and
    # friction slots stay zero. Default False — pair_aggregate is the
    # first method that produces seeds reliable enough for this to be
    # worth wiring up.
    seed_iterate: bool = False

    def __post_init__(self):
        if self.cold_friction_v_threshold < 0:
            # Negative threshold is allowed as "off" — just normalize.
            pass
        if self.method not in ("position_match", "pair_aggregate"):
            raise ValueError(
                f"WarmStartConfig.method must be 'position_match' or "
                f"'pair_aggregate', got {self.method!r}"
            )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class ContactsConfig:
    """Contact-set sizing and per-pair reduction policy."""

    max_per_world: int = 128
    reduction: ContactReductionConfig = field(
        default_factory=ContactReductionConfig
    )

    def __post_init__(self):
        if self.max_per_world < 1:
            raise ValueError(
                f"contacts.max_per_world must be >= 1, got {self.max_per_world}"
            )
        coerced = ContactReductionConfig.coerce(self.reduction)
        if coerced is not self.reduction:
            object.__setattr__(self, "reduction", coerced)

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class AdjointConfig:
    """Adjoint backward-pass interventions for differentiable simulation."""

    # Complete pose pull-back (2026-08-14): one kernel-adjoint sweep through
    # the residual assembly supplies w^T dR/dxi for every input, restoring the
    # implicit branch of dq+/dq- that the hand-written kernels missed
    # (position-loss gradients were ~2x low without it). Env override for
    # comparison: OSTRICH_POSE_VJP=0/1.
    pose_vjp: bool = True

    # Friction linearization in the adjoint solve. "true" (default) uses the
    # FB-derived linearization assembled at the converged state — the frozen
    # stick/slip surrogate below turned out to be a workaround for
    # since-fixed bugs, and the true linearization validates to <=7% on the
    # full gradient suite. "frozen" restores the legacy mode-classified
    # surrogate (soft_blending then applies). Env override for comparison:
    # OSTRICH_FRICTION_ADJOINT=true/frozen.
    friction_linearization: str = "true"

    soft_blending: bool = True
    soft_blending_temperature: float = 0.05
    regularization: float = 0.0
    gradient_normalization: bool = False

    def __post_init__(self):
        if self.friction_linearization not in ("true", "frozen"):
            raise ValueError(
                "adjoint.friction_linearization must be 'true' or 'frozen', "
                f"got {self.friction_linearization!r}"
            )
        if self.soft_blending_temperature <= 0:
            raise ValueError(
                "adjoint.soft_blending_temperature must be > 0, "
                f"got {self.soft_blending_temperature}"
            )
        if self.regularization < 0:
            raise ValueError(
                "adjoint.regularization must be >= 0, "
                f"got {self.regularization}"
            )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass(frozen=True)
class ProfilingConfig:
    """Engine performance profiling.

    Two complementary mechanisms; both require use_cuda_graph=True and
    one engine.step per captured segment for valid output.

    `segment_timing`: coarse host-side timer. Wraps wp.capture_launch
        with synchronize + time.perf_counter and prints ms/step. Quick
        "is this version faster?" check during development.

    `mode`: event-based GPU profiler. Three modes:
        "off"            — no events recorded
        "end_to_end"     — times major step phases (collide, load_data,
                           nr_solve, backtracking, output_copy)
        "per_component"  — times each NR iter's sub-phases
                           (linear_system, preconditioner, cr_solve,
                           step_or_linesearch, convergence_check).
                           Replaces wp.capture_while with a fixed unroll
                           of length max_newton_iters; every step pays
                           max iters (no early exit).
    """

    segment_timing: bool = False
    mode: str = "off"  # "off" | "end_to_end" | "per_component"

    def __post_init__(self):
        if self.mode not in _VALID_PROFILING_MODES:
            raise ValueError(
                f"profiling.mode must be one of {_VALID_PROFILING_MODES}, "
                f"got {self.mode!r}"
            )

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


# -----------------------------------------------------------------------
# OstrichEngineConfig
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class OstrichEngineConfig(EngineConfig):
    """Configuration for the OstrichEngine solver, organized into sub-configs.

    Top-level fields:
        differentiable: enable adjoint backward-pass support (allocates
            gradient buffers). Can also be set via the
            ``differentiable_simulation`` kwarg on ``create_engine()``,
            which overrides this field if explicitly passed.

    Sub-configs:
        nr           - NewtonRaphsonConfig — outer NR loop control
        linear       - LinearSolverConfig  — PCR + preconditioner
        compliance   - ComplianceConfig    — joint/contact/friction
        relaxation   - RelaxationConfig    — structural relaxation rungs
        linesearch   - LinesearchConfig    — backtracking line search
        warm_start   - WarmStartConfig     — cross-step contact warm start
        contacts     - ContactsConfig      — max + per-pair reduction
        adjoint      - AdjointConfig       — adjoint pass options
        profiling    - ProfilingConfig     — timing + event profiler
    """

    differentiable: bool = False

    nr: NewtonRaphsonConfig = field(default_factory=NewtonRaphsonConfig)
    linear: LinearSolverConfig = field(default_factory=LinearSolverConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    relaxation: RelaxationConfig = field(default_factory=RelaxationConfig)
    linesearch: LinesearchConfig = field(default_factory=LinesearchConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    contacts: ContactsConfig = field(default_factory=ContactsConfig)
    adjoint: AdjointConfig = field(default_factory=AdjointConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: Optional[bool] = None,
    ):
        from ostrich.core.engine import OstrichEngine

        diff = (
            self.differentiable
            if differentiable_simulation is None
            else differentiable_simulation
        )
        if diff and not self.relaxation.is_default:
            raise NotImplementedError(
                "relaxation.friction='bilateral' has no adjoint yet: the backward "
                "pass freezes each contact's stick/slip mode against the Coulomb "
                "limit mu*lambda_n (adjoint/adjoint_friction.py), which is "
                "meaningless when lambda_f is unbounded — every contact would "
                "classify as sliding — and it is derived from the FB branch the "
                "contact-equality rung removes. Gradients would be plausible and "
                "wrong. Run relaxed rungs forward-only (differentiable=False)."
            )
        return OstrichEngine(
            model=model,
            sim_steps=sim_steps,
            config=self,
            logging_config=logging_config,
            differentiable_simulation=diff,
        )

    def __post_init__(self):
        # Coerce sub-configs in case Hydra left them as DictConfig.
        # Each sub-config's __post_init__ runs its own validation when
        # constructed via coerce().
        for name, cls in (
            ("nr", NewtonRaphsonConfig),
            ("linear", LinearSolverConfig),
            ("compliance", ComplianceConfig),
            ("relaxation", RelaxationConfig),
            ("linesearch", LinesearchConfig),
            ("warm_start", WarmStartConfig),
            ("contacts", ContactsConfig),
            ("adjoint", AdjointConfig),
            ("profiling", ProfilingConfig),
        ):
            cur = getattr(self, name)
            coerced = cls.coerce(cur)
            if coerced is not cur:
                object.__setattr__(self, name, coerced)


# -----------------------------------------------------------------------
# Other engine configs (upstream Newton solver wrappers; flat by design)
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class FeatherstoneEngineConfig(EngineConfig):
    angular_damping: float = 0.05
    update_mass_matrix_interval: int = 1
    friction_smoothing: float = 1.0
    use_tile_gemm: bool = False
    fuse_cholesky: bool = True

    def _get_solver_class(self):
        from newton.solvers import SolverFeatherstone

        return SolverFeatherstone


@dataclass(frozen=True)
class MuJoCoEngineConfig(EngineConfig):
    separate_worlds: bool | None = None
    njmax: int | None = None
    nconmax: int | None = None
    iterations: int = 20
    ls_iterations: int = 10
    solver: int | str = "cg"
    integrator: int | str = "euler"
    cone: int | str = "pyramidal"
    impratio: float = 1.0
    use_mujoco_cpu: bool = False
    disable_contacts: bool = False
    update_data_interval: int = 1
    save_to_mjcf: str | None = None
    ls_parallel: bool = False
    use_mujoco_contacts: bool = True

    def _get_solver_class(self):
        from newton.solvers import SolverMuJoCo

        return SolverMuJoCo


@dataclass(frozen=True)
class XPBDEngineConfig(EngineConfig):
    iterations: int = 2
    soft_body_relaxation: float = 0.9
    soft_contact_relaxation: float = 0.9
    joint_linear_relaxation: float = 0.7
    joint_angular_relaxation: float = 0.4
    rigid_contact_relaxation: float = 0.8
    joint_linear_compliance: float = 0.01
    joint_angular_compliance: float = 0.01
    rigid_contact_con_weighting: bool = True
    angular_damping: float = 0.0
    enable_restitution: bool = False

    def _get_solver_class(self):
        from newton.solvers import SolverXPBD

        return SolverXPBD


@dataclass(frozen=True)
class SemiImplicitEngineConfig(EngineConfig):
    """Configuration for Newton's Semi-Implicit Euler solver."""

    angular_damping: float = 0.05
    friction_smoothing: float = 1.0
    joint_attach_ke: float = 1.0e4
    joint_attach_kd: float = 1.0e2
    enable_tri_contact: bool = True

    def _get_solver_class(self):
        from newton.solvers import SolverSemiImplicit

        return SolverSemiImplicit
