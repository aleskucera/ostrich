"""Profiler for AxionEngine forward steps.

Two profiling modes:

- ``"end_to_end"`` — times major coarse phases of one ``engine.step``
  (``collide``, ``load_data``, warm-start copy, NR solve, backtracking,
  output copy).

- ``"per_component"`` — times each NR iter's sub-phases (linear system /
  preconditioner / CR solve / step-or-linesearch / convergence).

Two timing backends (``end_to_end`` only for wall-clock):

- **CUDA events** (default) — ``wp.Event`` records inside a captured
  graph. The graph must contain exactly **one** ``engine.step`` call per
  replay; call :meth:`collect` after each ``wp.capture_launch`` (or after
  each eager step when ``use_cuda_graph=False``).

- **Synced wall-clock** — :meth:`enable_wall_clock` switches to
  ``wp.synchronize`` + ``torch.cuda.synchronize`` + ``perf_counter`` at
  each boundary. Works without cuda graph; intended for eager HybridGPT
  runs that mix PyTorch and Warp. :meth:`collect` is a no-op in this mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import time
from typing import Literal
from typing import Optional

import numpy as np
import torch
import warp as wp


ProfilingMode = Literal["off", "end_to_end", "per_component"]
VALID_MODES = ("off", "end_to_end", "per_component")


# Phase tags used by each mode. Keep in sync with the record sites in
# AxionEngineBase.
END_TO_END_PHASES = (
    "collide",
    "load_data",
    "warm_start_copy",
    "nr_solve",
    "backtracking",
    "output_copy",
)

PER_COMPONENT_PHASES = (
    "linear_system",
    "preconditioner",
    "cr_solve",
    "step_or_linesearch",
    "convergence_check",
)


@dataclass
class _PhaseStats:
    samples_ms: list = field(default_factory=list)

    def add(self, ms: float):
        self.samples_ms.append(ms)

    def stats(self):
        if not self.samples_ms:
            return {"count": 0}
        a = np.asarray(self.samples_ms, dtype=np.float64)
        return {
            "count": int(a.size),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()),
            "max_ms": float(a.max()),
            "total_ms": float(a.sum()),
        }


class EngineProfiler:
    """Pre-allocated CUDA-event pools + per-step accumulation.

    Built around two patterns:

    - ``end_to_end``: one event slot per phase boundary.

    - ``per_component``: ``K_unroll + 1`` slots per phase boundary, indexed
      by the unrolled NR iteration.

    The caller is responsible for calling :meth:`record_boundary` at each
    phase boundary and :meth:`collect` after each step/replay (except in
    wall-clock mode, where samples are accumulated inside
    :meth:`record_boundary`).
    """

    def __init__(self, mode: ProfilingMode, max_newton_iters: int, device):
        if mode not in VALID_MODES:
            raise ValueError(
                f"profiling mode must be one of {VALID_MODES}, got {mode!r}"
            )
        self._mode = mode
        self.device = device
        self._enabled = mode != "off"
        # Number of unrolled NR iterations under per_component mode. Sized
        # to max_newton_iters so the unroll covers the worst case the
        # engine could ever run.
        self._n_iters = max_newton_iters
        self._wall_clock = False
        self._last_wall_t: Optional[float] = None

        if not self._enabled:
            self._events = {}
            self._stats = {}
            return

        if mode == "end_to_end":
            phases = END_TO_END_PHASES
            slots = 1
        else:  # per_component
            phases = PER_COMPONENT_PHASES
            slots = max_newton_iters

        # boundaries = phases + 1 (start of phase 0 ... end of last phase)
        self._phase_names = phases
        self._slots = slots
        self._events = [
            [
                wp.Event(device=device, enable_timing=True)
                for _ in range(slots)
            ]
            for _ in range(len(phases) + 1)
        ]
        # Latched at capture time: True iff record_boundary(i, ...) was
        # ever called. Phases whose start or end boundary is unrecorded
        # are skipped by collect() (and reported as "n/a" in summary).
        # This lets callers leave phases un-bracketed (e.g. the smoke
        # test exercises engine.step directly with no collide phase).
        self._boundary_recorded = [False] * (len(phases) + 1)
        self._stats = {p: _PhaseStats() for p in phases}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> ProfilingMode:
        return self._mode

    @property
    def n_iters(self) -> int:
        """Length of the unrolled NR loop under per_component mode."""
        return self._n_iters

    @property
    def wall_clock(self) -> bool:
        return self._wall_clock

    def enable_wall_clock(self):
        """Use synced wall-clock timing instead of CUDA events (e2e only)."""
        if not self._enabled:
            return
        if self._mode != "end_to_end":
            raise ValueError(
                "wall-clock backend is only supported for end_to_end mode"
            )
        self._wall_clock = True

    def _sync_gpu(self):
        wp.synchronize()
        if self.device.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def record_boundary(self, boundary_idx: int, slot_idx: int = 0):
        """Record a phase boundary (CUDA event or synced wall-clock sample).

        For ``end_to_end`` (slots=1), pass ``slot_idx=0``. For
        ``per_component``, pass the unrolled NR iteration index.
        """
        if not self._enabled:
            return
        if self._wall_clock:
            if slot_idx != 0:
                raise ValueError(
                    "wall-clock backend does not support per_component slot_idx"
                )
            self._sync_gpu()
            t = time.perf_counter()
            if boundary_idx > 0 and self._last_wall_t is not None:
                phase = self._phase_names[boundary_idx - 1]
                self._stats[phase].add((t - self._last_wall_t) * 1000.0)
            self._last_wall_t = t
            self._boundary_recorded[boundary_idx] = True
            return
        wp.record_event(self._events[boundary_idx][slot_idx])
        self._boundary_recorded[boundary_idx] = True

    def collect(self):
        """Accumulate one step/replay's per-phase elapsed times.

        CUDA-event path: sync boundary events and read elapsed times.
        Wall-clock path: no-op (samples already accumulated).
        """
        if not self._enabled:
            return
        if self._wall_clock:
            return
        # Sync the latest recorded event so the elapsed-time reads below
        # don't race the GPU. Walk back from the final boundary to find
        # the last boundary that was actually recorded.
        last_recorded = next(
            (i for i in range(len(self._boundary_recorded) - 1, -1, -1)
             if self._boundary_recorded[i]),
            None,
        )
        if last_recorded is None:
            return  # nothing was bracketed this window
        wp.synchronize_event(self._events[last_recorded][self._slots - 1])

        for p_idx, phase in enumerate(self._phase_names):
            if not (self._boundary_recorded[p_idx] and self._boundary_recorded[p_idx + 1]):
                continue
            stats = self._stats[phase]
            for s in range(self._slots):
                ms = wp.get_event_elapsed_time(
                    self._events[p_idx][s],
                    self._events[p_idx + 1][s],
                    synchronize=False,
                )
                stats.add(ms)

    def reset(self):
        if not self._enabled:
            return
        self._last_wall_t = None
        for p in self._phase_names:
            self._stats[p] = _PhaseStats()

    def summary(self) -> dict:
        """Return per-phase {count, mean/p50/p95/min/max/total ms}."""
        if not self._enabled:
            return {}
        return {p: self._stats[p].stats() for p in self._phase_names}

    def print_summary(self, header: Optional[str] = None):
        if not self._enabled:
            print("[EngineProfiler] disabled (mode='off')")
            return
        print("=" * 72)
        print(header or f"[EngineProfiler] mode={self._mode}")
        print("-" * 72)
        s = self.summary()
        total = sum(v.get("mean_ms", 0.0) for v in s.values())
        print(f"{'phase':<24}{'count':>8}{'mean':>10}{'p50':>10}{'p95':>10}{'share':>10}")
        for phase, v in s.items():
            if v["count"] == 0:
                print(f"{phase:<24}{0:>8}")
                continue
            share = v["mean_ms"] / total if total > 0 else 0.0
            print(
                f"{phase:<24}{v['count']:>8}{v['mean_ms']:>10.3f}"
                f"{v['p50_ms']:>10.3f}{v['p95_ms']:>10.3f}{share*100:>9.1f}%"
            )
        print("-" * 72)
        print(f"{'sum of means':<24}{'':>8}{total:>10.3f}")
        print("=" * 72)
