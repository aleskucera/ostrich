#!/usr/bin/env python3
"""
Grouped bar chart: per-phase mean times from Axion vs two hybrid warm-start EngineProfiler summaries.

Uses the six canonical profiler phases with multiline labels matching axion_timing_plot.py.

Run from anywhere: python plot_axion_vs_hybrid_e2e_warmstart.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

_TIME_PROFILES_DIR = pathlib.Path(__file__).resolve().parent

AXION_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "axion_contact.txt"
HYBRID_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "hybrid_gpt_contact_304.txt"
HYBRID_ENGINE_NO_CUDA_PROFILE_PATH = (
    _TIME_PROFILES_DIR / "hybrid_gpt_contact_304_no_cuda.txt"
)

PLOT_TITLE = "Per-phase time profiling: Axion vs Hybrid mean measurements"

# When True, use a logarithmic y-axis (requires all plotted means to be > 0).
USE_LOG_Y_AXIS = True

ENGINE_PROFILER_PHASE_LABELS: dict[str, str] = {
    "collide": "Collision\ndetection",
    "load_data": "Contact\npreprocessing",
    "warm_start_copy": "Newton's method\ninitial guess",
    "nr_solve": "Newton system\nsolving",
    "backtracking": "Best iterate\nbacktracking",
    "output_copy": "Output\ncopying",
}

_ENGINE_PROFILER_PHASE_KEYS_EXPECTED = frozenset(
    ENGINE_PROFILER_PHASE_LABELS.keys()
)

_PROFILE_TABLE_HEADER_RE = re.compile(
    r"^\s*phase\s+count\s+mean\s+p50\s+p95\s+share\s*$", re.I
)

_AXION_BAR_COLOR = "C0"
_HYBRID_BAR_COLOR = "C1"
_HYBRID_NO_CUDA_BAR_COLOR = "C3"

BASE_FONTSIZE = 11
XTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 1
YTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE + 1
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 3
TITLE_FONTSIZE = BASE_FONTSIZE + 2

BAR_WIDTH = 0.25


def _parse_phase_means(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """Parse phase key and mean (ms) for each profiling table row after the header."""
    phases: list[str] = []
    means: list[float] = []

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines) and _PROFILE_TABLE_HEADER_RE.match(lines[i]) is None:
        i += 1
    if i >= len(lines):
        raise ValueError(f"No profiling table header in {path}")
    i += 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("-"):
            break
        if not stripped:
            i += 1
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(
                f"Expected ≥5 whitespace-separated columns in table row {line!r} in {path}"
            )
        phase_key = parts[0]
        try:
            mean_val = float(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid mean {parts[2]!r} for phase {phase_key!r} in {path}"
            ) from exc
        phases.append(phase_key)
        means.append(mean_val)
        i += 1

    if not phases:
        raise ValueError(f"No phase rows found after header in {path}")
    return phases, np.array(means, dtype=float)


def _mean_by_phase(
    phases: list[str],
    means: np.ndarray,
) -> dict[str, float]:
    if len(phases) != len(means):
        raise ValueError("phases and means length mismatch")
    return {p: float(m) for p, m in zip(phases, means)}


def _labels_for_phases(phase_keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in phase_keys:
        if key in ENGINE_PROFILER_PHASE_LABELS:
            out.append(ENGINE_PROFILER_PHASE_LABELS[key])
        else:
            out.append(key.replace("_", "\n"))
    return out


def _require_engine_phases(mean_by_phase: dict[str, float], *, path: pathlib.Path) -> None:
    keys = frozenset(mean_by_phase)
    missing = sorted(_ENGINE_PROFILER_PHASE_KEYS_EXPECTED - keys)
    extra = sorted(keys - _ENGINE_PROFILER_PHASE_KEYS_EXPECTED)
    if missing or extra:
        raise ValueError(
            f"Unexpected phase keys in {path}: missing={missing!r}, extra={extra!r}"
        )


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": AXES_LABELS_FONTSIZE,
            "axes.titlesize": AXES_LABELS_FONTSIZE + 1,
            "xtick.labelsize": XTICK_LABEL_FONTSIZE,
            "ytick.labelsize": YTICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
        }
    )


def main() -> None:
    axion_path = pathlib.Path(AXION_ENGINE_PROFILE_PATH)
    hybrid_path = pathlib.Path(HYBRID_ENGINE_PROFILE_PATH)
    hybrid_no_cuda_path = pathlib.Path(HYBRID_ENGINE_NO_CUDA_PROFILE_PATH)

    phases_ax, means_ax_arr = _parse_phase_means(axion_path)
    phases_hybrid, means_hybrid_arr = _parse_phase_means(hybrid_path)
    phases_nc, means_nc_arr = _parse_phase_means(hybrid_no_cuda_path)

    ax_by_phase = _mean_by_phase(phases_ax, means_ax_arr)
    hybrid_by_phase = _mean_by_phase(phases_hybrid, means_hybrid_arr)
    no_cuda_by_phase = _mean_by_phase(phases_nc, means_nc_arr)

    _require_engine_phases(ax_by_phase, path=axion_path)
    _require_engine_phases(hybrid_by_phase, path=hybrid_path)
    _require_engine_phases(no_cuda_by_phase, path=hybrid_no_cuda_path)

    labels = _labels_for_phases(phases_ax)
    means_axion_plot = means_ax_arr
    means_hybrid_plot = np.array(
        [hybrid_by_phase[p] for p in phases_ax], dtype=float
    )
    means_no_cuda_plot = np.array(
        [no_cuda_by_phase[p] for p in phases_ax], dtype=float
    )

    n = len(labels)
    if not (
        n == len(means_axion_plot)
        == len(means_hybrid_plot)
        == len(means_no_cuda_plot)
    ):
        raise ValueError("Aligned series length mismatch")

    x = np.arange(n)
    w = BAR_WIDTH

    _apply_plot_style()
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.bar(
        x - w,
        means_axion_plot,
        w,
        label="Axion engine",
        color=_AXION_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.bar(
        x,
        means_no_cuda_plot,
        w,
        label="Hybrid engine (warm-start, no CUDA)",
        color=_HYBRID_NO_CUDA_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.bar(
        x + w,
        means_hybrid_plot,
        w,
        label="Hybrid engine (warm-start)",
        color=_HYBRID_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylabel("Time [ms]")
    stacked = np.concatenate(
        [means_axion_plot, means_no_cuda_plot, means_hybrid_plot], dtype=float
    )
    if USE_LOG_Y_AXIS:
        if np.any(stacked <= 0):
            raise ValueError(
                "USE_LOG_Y_AXIS requires strictly positive bar heights; "
                f"got min={stacked.min()!r}"
            )
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
    ax.set_title(PLOT_TITLE)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
