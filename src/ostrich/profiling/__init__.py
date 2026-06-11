"""CUDA-event profiling for the Ostrich engine.

The profiler captures per-iteration / per-segment GPU-side timings via
``wp.Event`` records. It is consumed by ``OstrichEngineBase`` when the
engine config's ``ProfilingConfig`` selects a non-``off`` mode; the
config itself lives in ``ostrich.core.engine_config`` because it is part
of the engine config tree.
"""
from .engine_profiler import EngineProfiler
from .engine_profiler import VALID_MODES


__all__ = ["EngineProfiler", "VALID_MODES"]
