#!/usr/bin/env python3
"""
Grouped bar chart: aggregated mean latency categories from two EngineProfiler E2E contact summaries.

Raw phases are rolled into four x-axis buckets; aggregation rules differ for Axion vs hybrid.

Run from anywhere: python plot_axion_vs_hybrid_e2e.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

_TIME_PROFILES_DIR = pathlib.Path(__file__).resolve().parent

AXION_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "axion_contact.txt"
HYBRID_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "gpt_contact_sweep_model.txt"

PLOT_TITLE = "Per-phase time profiling: Axion vs Hybrid engine"

_ENGINE_PROFILER_PHASE_KEYS_EXPECTED = frozenset({
    "collide",
    "load_data",
    "warm_start_copy",
    "nr_solve",
    "backtracking",
    "output_copy",
})

_AGGREGATED_XTICK_LABELS: tuple[str, str, str, str] = (
    "Collision\ndetection",
    "Contact and input\nprocessing",
    "Engine\ncalculation",
    "Output\ncopying",
)

_PROFILE_TABLE_HEADER_RE = re.compile(
    r"^\s*phase\s+count\s+mean\s+p50\s+p95\s+share\s*$", re.I
)

_AXION_BAR_COLOR = "C0"
_HYBRID_BAR_COLOR = "C1"

BASE_FONTSIZE = 11
XTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 1
YTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE + 2
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 3
TITLE_FONTSIZE = BASE_FONTSIZE + 2

BAR_GROUP_WIDTH = 0.4


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


def _require_engine_phases(mean_by_phase: dict[str, float], *, path: pathlib.Path) -> None:
    keys = frozenset(mean_by_phase)
    missing = sorted(_ENGINE_PROFILER_PHASE_KEYS_EXPECTED - keys)
    extra = sorted(keys - _ENGINE_PROFILER_PHASE_KEYS_EXPECTED)
    if missing or extra:
        raise ValueError(
            f"Unexpected phase keys in {path}: missing={missing!r}, extra={extra!r}"
        )


def _aggregate_axion_means(mean_by_phase: dict[str, float]) -> np.ndarray:
    """Axion: contact/input = load_data; engine = warm_start + solve + backtrack."""
    ms = mean_by_phase
    return np.array(
        [
            ms["collide"],
            ms["load_data"],
            ms["warm_start_copy"] + ms["nr_solve"] + ms["backtracking"],
            ms["output_copy"],
        ],
        dtype=float,
    )


def _aggregate_hybrid_means(mean_by_phase: dict[str, float]) -> np.ndarray:
    """Hybrid: contact/input = load_data + warm_start; engine = solve + backtrack."""
    ms = mean_by_phase
    return np.array(
        [
            ms["collide"],
            ms["load_data"] + ms["warm_start_copy"],
            ms["nr_solve"] + ms["backtracking"],
            ms["output_copy"],
        ],
        dtype=float,
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

    phases_ax, means_ax_arr = _parse_phase_means(axion_path)
    phases_hybrid, means_hybrid_arr = _parse_phase_means(hybrid_path)

    ax_by_phase = _mean_by_phase(phases_ax, means_ax_arr)
    hybrid_by_phase = _mean_by_phase(phases_hybrid, means_hybrid_arr)

    _require_engine_phases(ax_by_phase, path=axion_path)
    _require_engine_phases(hybrid_by_phase, path=hybrid_path)

    labels = list(_AGGREGATED_XTICK_LABELS)
    means_axion_plot = _aggregate_axion_means(ax_by_phase)
    means_hybrid_plot = _aggregate_hybrid_means(hybrid_by_phase)

    n = len(labels)
    if n != len(means_axion_plot) or n != len(means_hybrid_plot):
        raise ValueError("Aggregated series length mismatch")

    x = np.arange(n)
    w = BAR_GROUP_WIDTH

    _apply_plot_style()
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.bar(
        x - w / 2,
        means_axion_plot,
        w,
        label="Axion engine",
        color=_AXION_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.bar(
        x + w / 2,
        means_hybrid_plot,
        w,
        label="Hybrid engine (surrogate use case)",
        color=_HYBRID_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylabel("Mean computation time [ms]")
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
    ax.set_title(PLOT_TITLE)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
