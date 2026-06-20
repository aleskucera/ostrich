"""Engine config dataclasses.

`AxionEngineConfig` is structured as a small set of sub-configs, one
per concern:

  nr            - Newton-Raphson outer loop control
  linear        - PCR linear solver + preconditioner
  compliance    - per-constraint-type compliance values
  linesearch    - opt-in line search (off by default)
  warm_start    - cross-step contact warm-start + cold-start heuristics
  contacts      - max contacts per world + per-pair reduction policy
  adjoint       - differentiable-simulation adjoint pass options
  profiling     - segment timing + event-based GPU profiler

Each sub-config validates itself in __post_init__; AxionEngineConfig's
own __post_init__ coerces Hydra DictConfig overrides through each
sub-config's `coerce()` classmethod (so partial YAML overrides like
`engine.linesearch.enabled=true` work even when only one field is set).

The non-Axion engine configs (Featherstone, MuJoCo, XPBD, SemiImplicit)
live at the bottom of this module and remain flat — they wrap upstream
Newton solvers which keep their own conventions.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

from axion.collision import ContactReductionConfig

from axion.profiling import VALID_MODES as _VALID_PROFILING_MODES


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _cast_scalar(value, field_type):
    """Cast YAML/Hydra string scalars to the declared field type."""
    if not isinstance(value, str):
        return value
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is bool:
        lowered = value.lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return value


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
    field_map = {f.name: f for f in fields(cls)}
    kwargs = {}
    for k, v in items.items():
        if k not in field_map:
            continue
        f = field_map[k]
        ft = f.type
        v = _cast_scalar(v, ft)
        coerce_fn = getattr(ft, "coerce", None)
        if coerce_fn is not None and not isinstance(v, ft):
            v = coerce_fn(v)
        kwargs[k] = v
    return cls(**kwargs)


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
        because they do not support this custom Axion logging system.
        """
        solver_cls = self._get_solver_class()
        return solver_cls(model, **vars(self))

    def _get_solver_class(self):
        """Subclasses must implement this to return their solver class."""
        raise NotImplementedError(
            f"Solver class resolution not implemented for {type(self).__name__}"
        )


# -----------------------------------------------------------------------
# Sub-configs for AxionEngineConfig
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonRaphsonConfig:
    """Outer Newton-Raphson loop control."""

    max_iters: int = 16
    backtrack_min_iter: int = 2
    atol: float = 1e-3

    def __post_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"nr.max_iters must be >= 1, got {self.max_iters}")
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

    See `axion/collision/warm_start.py` for the predicted-position
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

    soft_blending: bool = True
    soft_blending_temperature: float = 0.05
    regularization: float = 0.0
    gradient_normalization: bool = False

    def __post_init__(self):
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

    Two complementary mechanisms. Event-based ``mode`` works with or without
    ``use_cuda_graph``; when cuda graph is disabled the simulator calls
    ``collect()`` after each physics step. HybridGPTEngine with the eager
    torch ``NeuralPredictor`` path automatically uses synced wall-clock
    timing for accurate PyTorch + Warp phase breakdowns.

    `segment_timing`: coarse host-side timer. Wraps wp.capture_launch
        with synchronize + time.perf_counter and prints ms/step. Quick
        "is this version faster?" check during development. Requires
        use_cuda_graph=True.

    `mode`: step profiler. Three modes:
        "off"            — no timing recorded
        "end_to_end"     — times major step phases (collide, load_data,
                           warm_start_copy, nr_solve, backtracking,
                           output_copy)
        "per_component"  — times each NR iter's sub-phases
                           (linear_system, preconditioner, cr_solve,
                           step_or_linesearch, convergence_check).
                           Replaces wp.capture_while with a fixed unroll
                           of length max_newton_iters; requires cuda graph.
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
# AxionEngineConfig
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class AxionEngineConfig(EngineConfig):
    """Configuration for the AxionEngine solver, organized into sub-configs.

    Top-level fields:
        differentiable: enable adjoint backward-pass support (allocates
            gradient buffers). Can also be set via the
            ``differentiable_simulation`` kwarg on ``create_engine()``,
            which overrides this field if explicitly passed.

    Sub-configs:
        nr           - NewtonRaphsonConfig — outer NR loop control
        linear       - LinearSolverConfig  — PCR + preconditioner
        compliance   - ComplianceConfig    — joint/contact/friction
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
        from axion.core.engine import AxionEngine

        diff = (
            self.differentiable
            if differentiable_simulation is None
            else differentiable_simulation
        )
        return AxionEngine(
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


@dataclass(frozen=True)
class GPTEngineConfig(EngineConfig):
    """
    Configuration for the GPTEngine solver (neural-network-based physics).
    """

    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def __post_init__(self):
        coerced = ProfilingConfig.coerce(self.profiling)
        if coerced is not self.profiling:
            object.__setattr__(self, "profiling", coerced)

    def _get_solver_class(self):
        from axion.core.gpt_engine import GPTEngine

        return GPTEngine

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.logging import HDF5Logger
        from axion.logging import NullLogger

        from axion.core.gpt_engine import GPTEngine

        if logging_config is not None and getattr(
            logging_config, "enable_hdf5_logging", False
        ):
            logger = HDF5Logger(filepath=logging_config.hdf5_log_file)
            logger.open()
        else:
            logger = NullLogger()

        return GPTEngine(
            model=model,
            logger=logger,
            config=self,
        )

@dataclass(frozen=True)
class HybridGPTEngineConfig(AxionEngineConfig):
    """
    Configuration for the HybridGPTEngine solver (neural-network-assisted Newton method).
    """

    # If True, compute initial contact/constraint forces from the neural prediction
    # (warm-start Newton-Raphson). If False, use a zero initial guess for forces.
    use_warm_start_forces: bool = False

    # If True (default), use the MSEModel's lambda predictions to warm-start
    # data._constr_force before the Newton solve. Set to False to disable and
    # fall back to use_warm_start_forces / zero init (useful for ablations).
    use_neural_lambda_init: bool = True

    def _get_solver_class(self):
        from axion.core.hybrid_gpt_engine import HybridGPTEngine

        return HybridGPTEngine

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.logging import HDF5Logger
        from axion.logging import NullLogger

        from axion.core.hybrid_gpt_engine import HybridGPTEngine

        if logging_config is not None and getattr(
            logging_config, "enable_hdf5_logging", False
        ):
            logger = HDF5Logger(filepath=logging_config.hdf5_log_file)
            logger.open()
            _logging_cfg = logging_config
        else:
            logger = NullLogger()
            _logging_cfg = logging_config

        return HybridGPTEngine(
            model=model,
            sim_steps=int(sim_steps or 0),
            config=self,
            logging_config=_logging_cfg,
            differentiable_simulation=differentiable_simulation,
        )


@dataclass(frozen=True)
class TeacherForcedGPTEngineConfig(AxionEngineConfig):
    """
    Configuration for TeacherForcedGPTEngine.

    The Axion Newton-Raphson solver advances a private trajectory every step
    and always provides the NN's input context (teacher forcing).  The NN's
    predictions always drive the simulation output (state_out).

    All standard AxionEngineConfig solver parameters apply to the internal
    Axion teacher trajectory.  TensorRT is not supported; use execution:
    no_graph.
    """

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.core.teacher_forced_gpt_engine import TeacherForcedGPTEngine

        return TeacherForcedGPTEngine(
            model=model,
            sim_steps=int(sim_steps or 0),
            config=self,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )


@dataclass(frozen=True)
class TeacherForcedMLPEngineConfig(AxionEngineConfig):
    """
    Configuration for TeacherForcedMLPEngine.

    Single-state variant of TeacherForcedGPTEngineConfig: the Axion
    Newton-Raphson solver advances a private trajectory every step and always
    provides the NN's input context (teacher forcing), while the single-step
    SimpleMlpModel's predictions always drive the simulation output (state_out).

    All standard AxionEngineConfig solver parameters apply to the internal
    Axion teacher trajectory.  TensorRT is not supported; use execution:
    no_graph.
    """

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.core.teacher_forced_mlp_engine import TeacherForcedMLPEngine

        return TeacherForcedMLPEngine(
            model=model,
            sim_steps=int(sim_steps or 0),
            config=self,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )


@dataclass(frozen=True)
class RepeatedAxionEngineConfig(AxionEngineConfig):
    """
    Configuration for the `RepeatedAxionEngine` backend.

    This uses the same Newton/constraint parameters as `AxionEngineConfig`,
    but performs two Newton solves per step (see `src/axion/core/repeated_engine.py`).
    """

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.core.repeated_engine import RepeatedAxionEngine

        return RepeatedAxionEngine(
            model=model,
            sim_steps=int(sim_steps or 0),
            config=self,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )


@dataclass(frozen=True)
class AxionEngineWithNeuralLambdasConfig(AxionEngineConfig):
    """
    Configuration for AxionEngineWithNeuralLambdas backend.

    Uses Axion's Newton solver flow, but injects neural lambda predictions
    before solve.
    """

    neural_model_dir: str | None = None
    neural_model_pt_path: str | None = None
    neural_model_cfg_path: str | None = None

    def create_engine(
        self,
        model: Any,
        sim_steps: Optional[int] = None,
        init_state_fn: Optional[Callable] = None,
        logging_config: Optional[Any] = None,
        differentiable_simulation: bool = False,
        **kwargs,
    ) -> Any:
        from axion.core.axion_engine_with_neural_lambdas import AxionEngineWithNeuralLambdas

        return AxionEngineWithNeuralLambdas(
            model=model,
            sim_steps=int(sim_steps or 0),
            config=self,
            logging_config=logging_config,
            differentiable_simulation=differentiable_simulation,
        )
