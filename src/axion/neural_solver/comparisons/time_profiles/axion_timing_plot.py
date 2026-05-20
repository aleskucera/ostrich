"""
Timing bar chart for EngineProfiler summaries (mean vs p95).

Default series are hard-coded. Set PROFILE_SUMMARY_PATH to a text file shaped
like `axion_contact.txt` to pull means and p95 from disk.
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

_TIME_PROFILES_DIR = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data source: leave None for defaults; set path to parse EngineProfiler text.
# Example (uncomment to use bundled sample output):
#
# PROFILE_SUMMARY_PATH = _TIME_PROFILES_DIR / "axion_contact.txt"

PROFILE_SUMMARY_PATH: pathlib.Path | None = None # _TIME_PROFILES_DIR / "axion_contact.txt"


# Canonical EngineProfiler phase keys → multiline xtick labels when loading from TXT.
ENGINE_PROFILER_PHASE_LABELS: dict[str, str] = {
    "collide": "Collision\ndetection",
    "load_data": "Contact\npreprocessing",
    "warm_start_copy": "Newton's method\ninitial guess",
    "nr_solve": "Newton system\nsolving",
    "backtracking": "Best iterate\nbacktracking",
    "output_copy": "Output\ncopying",
}

_PROFILE_TABLE_HEADER_RE = re.compile(
    r"^\s*phase\s+count\s+mean\s+p50\s+p95\s+share\s*$", re.I
)

DEFAULT_LABELS: list[str] = [
    "Collision\ndetection",
    "Contact\npreprocessing",
    "Newton's method\ninitial guess",
    "Newton system\nsolving",
    "Best iterate\nbacktracking",
    "Output\ncopying",
]
DEFAULT_MEANS = np.array([0.350, 1.272, 0.015, 8.734, 0.026, 0.020])
DEFAULT_P95S = np.array([0.369, 1.598, 0.014, 13.877, 0.023, 0.014])

BASE_FONTSIZE = 11
XTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 1
YTICK_LABEL_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE + 1
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2

# Match plot_time_profiles.py Axion blue (first tab color).
_MEAN_BAR_COLOR = "C0"
_P95_BAR_COLOR = "saddlebrown"


def _parse_engine_profiler_means_and_p95s(path: pathlib.Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse phase mean / p95 rows from an EngineProfiler text summary."""
    phases: list[str] = []
    means: list[float] = []
    p95_values: list[float] = []

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
        try:
            p95_val = float(parts[4])
        except ValueError as exc:
            raise ValueError(
                f"Invalid p95 {parts[4]!r} for phase {phase_key!r} in {path}"
            ) from exc
        phases.append(phase_key)
        means.append(mean_val)
        p95_values.append(p95_val)
        i += 1

    if not phases:
        raise ValueError(f"No phase rows found after header in {path}")
    return phases, np.array(means, dtype=float), np.array(p95_values, dtype=float)


def _labels_for_phases(phase_keys: list[str]) -> list[str]:
    out: list[str] = []
    for key in phase_keys:
        if key in ENGINE_PROFILER_PHASE_LABELS:
            out.append(ENGINE_PROFILER_PHASE_LABELS[key])
        else:
            out.append(key.replace("_", "\n"))
    return out


def _bar_series_for_plot() -> tuple[list[str], np.ndarray, np.ndarray]:
    if PROFILE_SUMMARY_PATH is None:
        return list(DEFAULT_LABELS), DEFAULT_MEANS.copy(), DEFAULT_P95S.copy()
    path = pathlib.Path(PROFILE_SUMMARY_PATH)
    phase_keys, means, p95s = _parse_engine_profiler_means_and_p95s(path)
    return _labels_for_phases(phase_keys), means, p95s


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
    labels, means, p95s = _bar_series_for_plot()
    if len(labels) != len(means) or len(means) != len(p95s):
        raise ValueError("labels, means, and p95s must have the same length")

    x = np.arange(len(labels))
    width = 0.4

    _apply_plot_style()
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.bar(
        x - width / 2,
        means,
        width,
        label="mean",
        color=_MEAN_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.bar(
        x + width / 2,
        p95s,
        width,
        label="p95",
        color=_P95_BAR_COLOR,
        edgecolor="black",
        alpha=1.0,
    )
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.set_ylabel("Mean computation time [ms]")
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="gray")
    ax.set_title("Axion simulator time profiling experiment: Helhest robot")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
