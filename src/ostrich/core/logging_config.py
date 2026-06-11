"""Logging config dataclasses.

`LoggingConfig` is structured as three independent sub-configs, one
per logging subsystem:

  hdf5     - HDF5LoggingConfig: full per-step simulation log (state +
             linear system + constraints) for offline diagnostic
  dataset  - DatasetLoggingConfig: state-only log for ML training
  adjoint  - AdjointLoggingConfig: gradient-mode adjoint trace

Each sub-config is opt-in via its own `enabled` flag. The host-side
"segment timing" coarse timer used to live here as `enable_timing`;
it has been moved to ``engine.profiling.segment_timing`` since
profiling and persistent logging are different concerns.
"""
from dataclasses import dataclass, field, fields


def _coerce(cls, value):
    """Coerce a Hydra DictConfig (or dict-like) into an instance of `cls`.
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


@dataclass
class HDF5LoggingConfig:
    """HDF5 simulation log (full per-step state + linear system +
    constraints). Used for offline diagnostics and the convergence
    dashboard.

    Buffer size is auto-derived from the simulator's
    ``total_sim_steps`` (which comes from ``simulation.duration_seconds``
    / ``target_timestep_seconds``). There is no separate
    ``max_steps`` knob here — single source of truth for run length.
    """

    enabled: bool = False
    file: str = "simulation.h5"
    # Sub-streams within the HDF5 log (each can be disabled independently
    # to shrink file size).
    log_dynamics_state: bool = True
    log_linear_system_data: bool = True
    log_constraint_data: bool = True

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass
class DatasetLoggingConfig:
    """Dataset log (state-only, intended for ML training pipelines).

    Buffer size is auto-derived from the simulator's
    ``total_sim_steps``; see HDF5LoggingConfig.
    """

    enabled: bool = False
    file: str = "dataset.h5"

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass
class AdjointLoggingConfig:
    """Adjoint trace log (gradient-mode backward-pass intermediates).

    Requires the engine to be constructed with differentiable=True
    (or the equivalent kwarg flow through create_engine).
    """

    enabled: bool = False
    file: str = "adjoint.h5"

    @classmethod
    def coerce(cls, obj):
        return _coerce(cls, obj)


@dataclass
class LoggingConfig:
    """Persistent-state logging configuration.

    Three independent subsystems. The host-side "segment timing" coarse
    timer lives on ``engine.profiling.segment_timing`` (not here) since
    profiling and persistent logging are different concerns.
    """

    hdf5: HDF5LoggingConfig = field(default_factory=HDF5LoggingConfig)
    dataset: DatasetLoggingConfig = field(default_factory=DatasetLoggingConfig)
    adjoint: AdjointLoggingConfig = field(default_factory=AdjointLoggingConfig)

    def __post_init__(self):
        # Coerce sub-configs in case Hydra left them as DictConfig.
        for name, cls in (
            ("hdf5", HDF5LoggingConfig),
            ("dataset", DatasetLoggingConfig),
            ("adjoint", AdjointLoggingConfig),
        ):
            cur = getattr(self, name)
            coerced = cls.coerce(cur)
            if coerced is not cur:
                object.__setattr__(self, name, coerced)
