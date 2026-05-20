#!/usr/bin/env python3
"""
Bar chart of end-to-end engine timings from EngineProfiler text summaries.

Edit the user-configuration constants below (input paths, labels, output).
Run directly: python plot_time_profiles.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np

_TIME_PROFILES_DIR = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

PROFILE_PATHS = (
    _TIME_PROFILES_DIR / "axion_contact.txt",
    _TIME_PROFILES_DIR / "gpt_contact_sweep_model.txt",
    _TIME_PROFILES_DIR / "gpt_contact_sweep_model_no_cuda.txt",
)

BAR_LABELS = (
    "Axion Engine",
    "Hybrid Engine",
    "Hybrid Engine (no CUDA graph)",
)

# None → show interactively; set to save instead.
FIGURE_OUTPUT_PATH: pathlib.Path | None = None

# Bar colors per engine index (matplotlib default cycle C0..).
ENGINE_BAR_FACE_COLORS = ("C0", "C1", "C2")

BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2
GRID_ALPHA = 0.3
Y_LIM: tuple[float, float] | None = None

FIGURE_SIZE = (4.0, 4.8)
BAR_WIDTH = 0.88
XTICK_ROTATION = 25


def _multiline_xtick_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    """Break before parenthetical for narrow bar charts."""
    out: list[str] = []
    for lbl in labels:
        if "(" in lbl:
            left, right = lbl.split("(", 1)
            out.append(f"{left.strip()}\n({right}")
        else:
            out.append(lbl)
    return tuple(out)


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": BASE_FONTSIZE,
            "axes.labelsize": AXES_LABELS_FONTSIZE,
            "axes.titlesize": AXES_LABELS_FONTSIZE + 1,
            "xtick.labelsize": AXES_TICKS_FONTSIZE,
            "ytick.labelsize": AXES_TICKS_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.titlesize": TITLE_FONTSIZE,
        }
    )


_SUM_OF_MEANS_RE = re.compile(r"^sum of means\s+([\d.]+)\s*$", re.MULTILINE)


def _parse_sum_of_means_ms(path: pathlib.Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = _SUM_OF_MEANS_RE.search(text)
    if match is None:
        raise ValueError(f"No 'sum of means' line found in {path}")
    return float(match.group(1))


def main() -> None:
    if len(PROFILE_PATHS) != len(BAR_LABELS):
        raise ValueError("PROFILE_PATHS and BAR_LABELS must have the same length")
    if len(PROFILE_PATHS) != len(ENGINE_BAR_FACE_COLORS):
        raise ValueError(
            "PROFILE_PATHS and ENGINE_BAR_FACE_COLORS must have the same length"
        )

    timings_ms = [_parse_sum_of_means_ms(path) for path in PROFILE_PATHS]
    engine_count = len(timings_ms)

    _apply_plot_style()
    plt.figure(figsize=FIGURE_SIZE)
    x_pos = np.arange(engine_count)
    bar_colors = list(ENGINE_BAR_FACE_COLORS[:engine_count])
    bars = plt.bar(x_pos, timings_ms, width=BAR_WIDTH, color=bar_colors)
    plt.gca().bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(x_pos, _multiline_xtick_labels(BAR_LABELS), rotation=XTICK_ROTATION, ha="right")
    plt.ylabel("Time per step (ms)")
    plt.grid(True, axis="y", alpha=GRID_ALPHA)
    if Y_LIM is not None:
        plt.ylim(*Y_LIM)
    plt.xlim(-0.55, engine_count - 0.45)
    plt.tight_layout()

    if FIGURE_OUTPUT_PATH is not None:
        out_path = pathlib.Path(FIGURE_OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
