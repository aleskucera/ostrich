#!/usr/bin/env python3
"""
Axion vs hybrid surrogate: mean latency per profiler phase with p50–p95 whiskers.

Uses summary columns from EngineProfiler text (mean as dot; vertical segment from p50 to p95 with caps).

Run from anywhere: python plot_axion_vs_hybrid_e2e_surrogate_better_vis.py
"""

from __future__ import annotations

import pathlib
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np

_TIME_PROFILES_DIR = pathlib.Path(__file__).resolve().parent

AXION_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "axion_contact.txt"
HYBRID_ENGINE_PROFILE_PATH = _TIME_PROFILES_DIR / "gpt_contact_sweep_model.txt"

PLOT_TITLE = "Contact E2E: Axion vs hybrid surrogate (mean and p50–p95 span)"

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

_AXION_MARKER_COLOR = "C0"
_HYBRID_MARKER_COLOR = "C1"

BASE_FONTSIZE = 11
XTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 1
YTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE + 1
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 3
TITLE_FONTSIZE = BASE_FONTSIZE + 2

_X_DODGE = 0.15
_CAP_HALF_WIDTH_X = 0.055


def _plot_mean_and_quantile_span(
    plot_ax: plt.Axes,
    x_centers: np.ndarray,
    mean: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    color: str,
    *,
    cap_half_width: float,
) -> None:
    """Vertical segment from p50 to p95 with end caps; mean marker (may lie outside segment)."""
    for xc, m, lo, hi in zip(x_centers, mean, p50, p95):
        plot_ax.vlines(xc, lo, hi, colors=color, linewidths=2.0, zorder=2)
        plot_ax.plot(
            [xc - cap_half_width, xc + cap_half_width],
            [lo, lo],
            color=color,
            lw=1.0,
            zorder=2,
        )
        plot_ax.plot(
            [xc - cap_half_width, xc + cap_half_width],
            [hi, hi],
            color=color,
            lw=1.0,
            zorder=2,
        )
        plot_ax.plot(xc, m, "o", color=color, markersize=6, zorder=3)


def _parse_phase_mean_p50_p95(path: pathlib.Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Parse phase rows: mean, p50, p95 (ms). Validates ordering where possible."""
    phases: list[str] = []
    means: list[float] = []
    p50s: list[float] = []
    p95s: list[float] = []

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
        if len(parts) < 6:
            raise ValueError(
                f"Expected ≥6 whitespace-separated columns in table row {line!r} in {path}"
            )
        phase_key = parts[0]
        try:
            mean_val = float(parts[2])
            p50_val = float(parts[3])
            p95_val = float(parts[4])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric columns for phase {phase_key!r} in {path}: {line!r}"
            ) from exc

        if not (p50_val <= mean_val <= p95_val):
            warnings.warn(
                f"{path}: phase {phase_key!r} has mean outside [p50, p95] "
                f"(p50={p50_val}, mean={mean_val}, p95={p95_val}); "
                f"mean marker may sit outside the vertical p50–p95 segment.",
                stacklevel=2,
            )

        phases.append(phase_key)
        means.append(mean_val)
        p50s.append(p50_val)
        p95s.append(p95_val)
        i += 1

    if not phases:
        raise ValueError(f"No phase rows found after header in {path}")
    return (
        phases,
        np.array(means, dtype=float),
        np.array(p50s, dtype=float),
        np.array(p95s, dtype=float),
    )


def _mean_by_phase(phases: list[str], values: np.ndarray) -> dict[str, float]:
    if len(phases) != len(values):
        raise ValueError("phases and values length mismatch")
    return {p: float(v) for p, v in zip(phases, values)}


def _labels_for_phases(phase_keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in phase_keys:
        if key in ENGINE_PROFILER_PHASE_LABELS:
            out.append(ENGINE_PROFILER_PHASE_LABELS[key])
        else:
            out.append(key.replace("_", "\n"))
    return out


def _require_engine_phases(keys_present: frozenset[str], *, path: pathlib.Path) -> None:
    missing = sorted(_ENGINE_PROFILER_PHASE_KEYS_EXPECTED - keys_present)
    extra = sorted(keys_present - _ENGINE_PROFILER_PHASE_KEYS_EXPECTED)
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

    phases_ax, mean_ax, p50_ax, p95_ax = _parse_phase_mean_p50_p95(axion_path)
    phases_hy, mean_hy, p50_hy, p95_hy = _parse_phase_mean_p50_p95(hybrid_path)

    ax_mean_by_phase = _mean_by_phase(phases_ax, mean_ax)
    hy_mean_by_phase = _mean_by_phase(phases_hy, mean_hy)
    hy_p50_by_phase = _mean_by_phase(phases_hy, p50_hy)
    hy_p95_by_phase = _mean_by_phase(phases_hy, p95_hy)

    _require_engine_phases(frozenset(ax_mean_by_phase), path=axion_path)
    _require_engine_phases(frozenset(hy_mean_by_phase), path=hybrid_path)

    labels = _labels_for_phases(phases_ax)
    n = len(labels)

    mean_a = mean_ax
    p50_a = p50_ax
    p95_a = p95_ax

    mean_h = np.array([hy_mean_by_phase[p] for p in phases_ax], dtype=float)
    p50_h = np.array([hy_p50_by_phase[p] for p in phases_ax], dtype=float)
    p95_h = np.array([hy_p95_by_phase[p] for p in phases_ax], dtype=float)

    if not (
        n == len(mean_a) == len(mean_h) == len(p50_a) == len(p95_a) == len(p50_h) == len(p95_h)
    ):
        raise ValueError("Aligned series length mismatch")

    x = np.arange(n)
    dodge = _X_DODGE

    _apply_plot_style()
    fig, plot_ax = plt.subplots()
    plot_ax.set_axisbelow(True)

    # Legend handles (actual spans drawn below).
    plot_ax.plot([], [], "o", color=_AXION_MARKER_COLOR, markersize=6, label="Axion engine")
    plot_ax.plot(
        [],
        [],
        "o",
        color=_HYBRID_MARKER_COLOR,
        markersize=6,
        label="Hybrid engine (surrogate)",
    )

    _plot_mean_and_quantile_span(
        plot_ax,
        x - dodge,
        mean_a,
        p50_a,
        p95_a,
        _AXION_MARKER_COLOR,
        cap_half_width=_CAP_HALF_WIDTH_X,
    )
    _plot_mean_and_quantile_span(
        plot_ax,
        x + dodge,
        mean_h,
        p50_h,
        p95_h,
        _HYBRID_MARKER_COLOR,
        cap_half_width=_CAP_HALF_WIDTH_X,
    )

    plot_ax.set_xticks(x, labels, rotation=0, ha="center")
    plot_ax.set_xlim(-0.55, n - 0.45)
    plot_ax.set_ylabel("Time [ms]")
    plot_ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
    plot_ax.set_title(PLOT_TITLE)
    plot_ax.legend(
        title="Dots = mean; vertical bars span p50–p95 (not a CI)",
        loc="best",
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
