"""Profiling for the Axion engine.

``EngineProfiler`` supports CUDA-event timing (cuda-graph replay) and a
synced wall-clock backend for eager execution (HybridGPT torch path).
Configured via ``ProfilingConfig`` on the engine config tree.
"""
from .engine_profiler import EngineProfiler
from .engine_profiler import VALID_MODES


__all__ = ["EngineProfiler", "VALID_MODES"]
