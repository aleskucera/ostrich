#!/usr/bin/env python3
"""
Analyze `data/engine_comparison*.hdf5`.

Computes, for each run, the total number of Newton iterations (sum of `iter_count`)
for selected engines and optionally plots them as a bar chart (Figure 1: first K runs).
Also plots a second figure with three bars: total iterations summed across all
trajectories per engine (shown via `plt.show()`).

When ``SAVE_ITER_COUNTS`` is true, saves per-run iter/step PDFs only for the first
``FIGURE_1_NUM_RUNS_TO_PLOT`` overlapping runs (same subset as Figure 1).

Additionally, for each saved run, creates a line plot where:
- x-axis = time step index
- y-axis = number of Newton iterations
- one line per engine

Also creates temporal plots for hardcoded slices of:
- converged lambda components
- converged state components (from `converged_body_vel` flattened)

Edit the user-configuration constants at the top of this file (HDF5 path, engine group
names, optional output directories).
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

CONTACT_FREE = "engine_comparison_20260427_104521.hdf5"
CONTACTS_INCLDUED = "engine_comparison_20260519_173216.hdf5"

INPUT_HDF5 = REPO_ROOT / "data" / CONTACTS_INCLDUED
AXION_GROUP = "axion_engine"
HYBRID_GROUP = "hybrid_gpt_engine_neural_warm_start_forces"
REPEATED_GROUP = "repeated_axion_engine"
# If set and Figure 1 is plotted (FIGURE_1_NUM_RUNS_TO_PLOT > 0), save it here.
FIGURE_1_OUTPUT_PATH: pathlib.Path | None = None
# None → <hdf5_dir>/iter_count_vs_step_per_run/<timestamp>/
ITER_PLOTS_DIR: pathlib.Path | None = None
# None → <hdf5_dir>/temporal_variables_per_run/<timestamp>/
TEMPORAL_PLOTS_DIR: pathlib.Path | None = None

ENGINE_LEGEND_LABELS = (
    "Axion engine",
    "Hybrid engine",
    "Axion engine (repeated)",
)
# Bar colors per engine index (matplotlib default cycle C0..); shared by Figure 1 and 2.
ENGINE_BAR_FACE_COLORS = ("C0", "C1", "C2")
# When True, write iter_count vs time-step PDFs for the first FIGURE_1_NUM_RUNS_TO_PLOT runs only
# (not every trajectory in the HDF5).
SAVE_ITER_COUNTS = True
# Figure 1: grouped bar chart of total Newton iterations per trajectory.
# Same N limits SAVE_ITER_COUNTS PDF output. 0 = skip Figure 1 and save no iter PDFs.
# N > 0 = first N runs (from overlapping run indices).
FIGURE_1_NUM_RUNS_TO_PLOT = 15
LAMBDA_SLICE = (0, 0)
STATE_SLICE = (0, 0)

BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE + 1
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE
LINEWIDTH = 2.5
GRID_ALPHA = 0.3
LEGEND_LOC = "upper right"
# Figure size (in) for saved iter_count vs time-step line plots; tick/label sizes use the
# constants above via _apply_plot_style() and explicit tick_params / legend fontsize below.
ITER_COUNT_LINE_PLOT_FIGSIZE = (12, 4.5)
# Optional y-axis limits for Figure 1 only (grouped bar chart). Not used for iter/step PDFs.
Y_LIM_FIGURE_1: tuple[float, float] | None = (0, 1000)


def _figure_2_xtick_labels() -> tuple[str, ...]:
    """Multi-line x tick strings for narrow Figure 2 (newline before parenthetical)."""
    out: list[str] = []
    for lbl in ENGINE_LEGEND_LABELS:
        if "(" in lbl:
            left, right = lbl.split("(", 1)
            out.append(f"{left.strip()}\n({right}")
        else:
            out.append(lbl)
    return tuple(out)


def _apply_plot_style() -> None:
    """Match rcParams from plot_states_from_multirollouts._apply_plot_style."""
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


def _sorted_run_keys(h5_group: h5py.Group) -> list[str]:
    keys = [k for k in h5_group.keys() if k.startswith("run_")]
    keys.sort()
    return keys


def _extract_run_index(run_key: str) -> int:
    # Expected format: run_000, run_001, ...
    try:
        return int(run_key.split("_", 1)[1])
    except Exception:
        return -1


def _iter_count_series_for_run(run_group: h5py.Group) -> np.ndarray:
    if "iter_count" not in run_group:
        raise KeyError(f"Missing dataset 'iter_count' in group {run_group.name}")
    iters = np.asarray(run_group["iter_count"][:])
    # Expected shape is (num_steps, 1) -> squeeze to (num_steps,)
    return iters.squeeze()


def _total_iters_for_series(iters: np.ndarray) -> int:
    return int(np.sum(iters))


def _validate_engine_group_exists(
    f: h5py.File, engine_group_name: str, h5_path: pathlib.Path
) -> h5py.Group:
    if engine_group_name not in f:
        available = ", ".join(sorted(f.keys()))
        raise KeyError(
            f"Engine group '{engine_group_name}' not found in {h5_path}. "
            f"Top-level groups: {available}"
        )
    return f[engine_group_name]


def _get_dataset_or_fallback(run_group: h5py.Group, preferred: str, fallback: str) -> np.ndarray:
    if preferred in run_group:
        return np.asarray(run_group[preferred][:])
    if fallback in run_group:
        return np.asarray(run_group[fallback][:])
    raise KeyError(
        f"Neither '{preferred}' nor '{fallback}' exists in {run_group.name}. "
        f"Available datasets: {sorted(run_group.keys())}"
    )


def _flatten_feature_series(arr: np.ndarray) -> np.ndarray:
    # Convert (T, ...features...) -> (T, F)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def main() -> None:
    _apply_plot_style()

    h5_path = pathlib.Path(INPUT_HDF5)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iter_plots_dir: pathlib.Path | None = None
    if SAVE_ITER_COUNTS:
        if ITER_PLOTS_DIR is not None:
            iter_plots_dir = pathlib.Path(ITER_PLOTS_DIR)
        else:
            iter_plots_dir = (
                h5_path.parent / "iter_count_vs_step_per_run" / run_stamp
            )
        iter_plots_dir.mkdir(parents=True, exist_ok=True)
    if TEMPORAL_PLOTS_DIR is not None:
        temporal_plots_dir = pathlib.Path(TEMPORAL_PLOTS_DIR)
    else:
        temporal_plots_dir = h5_path.parent / "temporal_variables_per_run" / run_stamp
    temporal_plots_dir.mkdir(parents=True, exist_ok=True)

    engine_groups = [
        (AXION_GROUP, "axion_engine"),
        (HYBRID_GROUP, "hybrid_gpt_engine_neural_warm_start_forces"),
        (REPEATED_GROUP, "repeated_axion_engine"),
    ]

    # Load iter_count series per run for each engine.
    # all_series[engine_group_name][run_idx] -> (num_steps,) ndarray
    all_series: dict[str, dict[int, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        for engine_group_name, _ in engine_groups:
            eng_grp = _validate_engine_group_exists(f, engine_group_name, h5_path)
            series_for_runs: dict[int, np.ndarray] = {}
            for run_key in _sorted_run_keys(eng_grp):
                run_idx = _extract_run_index(run_key)
                series_for_runs[run_idx] = _iter_count_series_for_run(
                    eng_grp[run_key]
                )
            all_series[engine_group_name] = series_for_runs

    run_sets = [set(s.keys()) for s in all_series.values()]
    run_indices = sorted(set.intersection(*run_sets)) if run_sets else []
    if not run_indices:
        available = {k: sorted(v.keys()) for k, v in all_series.items()}
        raise RuntimeError(
            "No overlapping run indices across selected engine groups. "
            f"Run indices per group: {available}"
        )

    # --- Bar plot of total iterations (Figure 1: subset of runs) ---
    totals_by_engine: dict[str, list[int]] = {}
    for engine_group_name, _ in engine_groups:
        totals_by_engine[engine_group_name] = [
            _total_iters_for_series(all_series[engine_group_name][i])
            for i in run_indices
        ]

    engine_count = len(engine_groups)
    width = 0.9 / engine_count

    title_top = "Contacts included"
    title_bottom = "Sum over 50 trajectories"
    fig_title_two_line = f"{title_top}\n{title_bottom}"

    n_fig1 = max(0, int(FIGURE_1_NUM_RUNS_TO_PLOT))
    run_indices_iter_save = run_indices[: min(n_fig1, len(run_indices))]
    if n_fig1 > 0:
        n_plot = min(n_fig1, len(run_indices))
        run_indices_fig1 = run_indices[:n_plot]
        x = np.arange(n_plot)
        # Narrower than default 12"; scale modestly with bar count; extra height for two-line title.
        fig1_w = max(4.5, min(9.0, 1.5 * n_plot + 3.5))
        plt.figure(figsize=(fig1_w, 5.2))
        offsets = np.linspace(-0.45 + width / 2, 0.45 - width / 2, engine_count)
        for idx, ((engine_group_name, _), off, legend_label) in enumerate(
            zip(engine_groups, offsets, ENGINE_LEGEND_LABELS)
        ):
            plt.bar(
                x + off,
                totals_by_engine[engine_group_name][:n_plot],
                width=width,
                label=legend_label,
                color=ENGINE_BAR_FACE_COLORS[idx],
            )
        plt.xticks(x, [str(i) for i in run_indices_fig1], rotation=0)
        plt.ylabel("Total number of Newton iterations")
        plt.xlabel("Trajectory number")
        traj_word_fig1 = "trajectory" if n_plot == 1 else "trajectories"
        plt.title(fig_title_two_line)
        if Y_LIM_FIGURE_1 is not None:
            plt.ylim(*Y_LIM_FIGURE_1)
        plt.grid(True, axis="y", alpha=GRID_ALPHA)
        plt.legend(loc=LEGEND_LOC)
        plt.tight_layout()

        if FIGURE_1_OUTPUT_PATH is not None:
            out_path = pathlib.Path(FIGURE_1_OUTPUT_PATH)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Saved plot to: {out_path}")
        else:
            plt.show()

    # --- Aggregate bar chart: total iterations summed over all trajectories ---
    grand_totals = [
        int(sum(totals_by_engine[engine_group_name]))
        for engine_group_name, _ in engine_groups
    ]
    plt.figure(figsize=(4.0, 4.8))
    x_pos = np.arange(engine_count)
    bar_colors_fig2 = list(ENGINE_BAR_FACE_COLORS[:engine_count])
    # Wide bars + tight x limits → minimal gap between columns.
    bars = plt.bar(x_pos, grand_totals, width=0.88, color=bar_colors_fig2)
    plt.gca().bar_label(bars, fmt="%d", padding=3)
    plt.xticks(x_pos, _figure_2_xtick_labels(), rotation=25, ha="right")
    plt.ylabel("Total Number of Newton iterations")
    plt.title(fig_title_two_line)
    plt.grid(True, axis="y", alpha=GRID_ALPHA)
    plt.xlim(-0.55, engine_count - 0.45)
    plt.tight_layout()
    plt.show()

    # --- Per-run iter_count vs time-step line plots (subset: first N runs only) ---
    if SAVE_ITER_COUNTS:
        assert iter_plots_dir is not None
        for run_idx in run_indices_iter_save:
            series_by_engine = {
                engine_group_name: np.ravel(
                    all_series[engine_group_name][run_idx]
                )
                for engine_group_name, _ in engine_groups
            }

            num_steps = int(
                min(arr.shape[0] for arr in series_by_engine.values())
            )
            steps = np.arange(num_steps)

            fig, ax = plt.subplots(figsize=ITER_COUNT_LINE_PLOT_FIGSIZE)
            for (engine_group_name, _), legend_label in zip(
                engine_groups, ENGINE_LEGEND_LABELS
            ):
                arr = series_by_engine[engine_group_name]
                ax.plot(
                    steps,
                    arr[:num_steps],
                    label=legend_label,
                    linewidth=LINEWIDTH,
                )
            ax.set_xlabel("Time step [-]", fontsize=AXES_LABELS_FONTSIZE)
            ax.set_ylabel(
                "Number of Newton iterations", fontsize=AXES_LABELS_FONTSIZE
            )
            ax.tick_params(axis="both", labelsize=AXES_TICKS_FONTSIZE)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, _pos: (
                        str(int(round(x))) if np.isfinite(x) else ""
                    )
                )
            )
            ax.grid(True, axis="both", alpha=GRID_ALPHA)
            ax.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONTSIZE)
            plt.tight_layout()

            out_path = (
                iter_plots_dir / f"iter_count_vs_step_run_{run_idx:03d}.pdf"
            )
            fig.savefig(out_path, format="pdf")
            plt.close(fig)
            print(f"Saved: {out_path}")

    # --- Per-run temporal plots for lambda/state slices ---
    with h5py.File(h5_path, "r") as f:
        axion_grp = _validate_engine_group_exists(f, AXION_GROUP, h5_path)
        hybrid_grp = _validate_engine_group_exists(f, HYBRID_GROUP, h5_path)
        repeated_grp = _validate_engine_group_exists(f, REPEATED_GROUP, h5_path)

        for run_idx in run_indices:
            run_key = f"run_{run_idx:03d}"
            axion_run = axion_grp[run_key]
            hybrid_run = hybrid_grp[run_key]
            repeated_run = repeated_grp[run_key]

            axion_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    axion_run, "converged_constr_force", "constr_force"
                )
            )
            hybrid_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    hybrid_run, "converged_constr_force", "constr_force"
                )
            )
            repeated_lambda = _flatten_feature_series(
                _get_dataset_or_fallback(
                    repeated_run, "converged_constr_force", "constr_force"
                )
            )
            hybrid_pred_lambda = None
            if "hybrid_predicted_next_lambdas" in hybrid_run:
                hybrid_pred_lambda = _flatten_feature_series(
                    np.asarray(hybrid_run["hybrid_predicted_next_lambdas"][:])
                )

            axion_state = _flatten_feature_series(
                _get_dataset_or_fallback(axion_run, "converged_body_vel", "body_vel")
            )
            hybrid_state = _flatten_feature_series(
                _get_dataset_or_fallback(hybrid_run, "converged_body_vel", "body_vel")
            )
            repeated_state = _flatten_feature_series(
                _get_dataset_or_fallback(
                    repeated_run, "converged_body_vel", "body_vel"
                )
            )
            # Hybrid neural state prediction in solver coordinates after FK warm start.
            hybrid_pred_state = _flatten_feature_series(
                _get_dataset_or_fallback(
                    hybrid_run, "hybrid_predicted_next_body_vel", "init_guess_body_vel"
                )
            )

            lambda_steps = min(
                axion_lambda.shape[0],
                hybrid_lambda.shape[0],
                repeated_lambda.shape[0],
                hybrid_pred_lambda.shape[0] if hybrid_pred_lambda is not None else np.inf,
            )
            state_steps = min(
                axion_state.shape[0],
                hybrid_state.shape[0],
                repeated_state.shape[0],
                hybrid_pred_state.shape[0],
            )
            lambda_hi = min(
                LAMBDA_SLICE[1],
                axion_lambda.shape[1],
                hybrid_lambda.shape[1],
                repeated_lambda.shape[1],
                hybrid_pred_lambda.shape[1] if hybrid_pred_lambda is not None else np.inf,
            )
            state_hi = min(
                STATE_SLICE[1],
                axion_state.shape[1],
                hybrid_state.shape[1],
                repeated_state.shape[1],
                hybrid_pred_state.shape[1],
            )
            line_styles = {
                "axion": {"alpha": 0.9, "linewidth": LINEWIDTH, "linestyle": "-"},
                "repeated": {
                    "alpha": 0.85,
                    "linewidth": LINEWIDTH,
                    "linestyle": "--",
                },
                "hybrid": {"alpha": 0.9, "linewidth": LINEWIDTH, "linestyle": "-."},
                "hybrid_pred": {
                    "alpha": 0.75,
                    "linewidth": LINEWIDTH * 0.72,
                    "linestyle": ":",
                },
            }

            steps_lambda = np.arange(int(lambda_steps))
            for idx in range(LAMBDA_SLICE[0], int(lambda_hi)):
                plt.figure(figsize=(12, 4.5))
                plt.plot(
                    steps_lambda,
                    axion_lambda[: int(lambda_steps), idx],
                    label=ENGINE_LEGEND_LABELS[0],
                    **line_styles["axion"],
                )
                plt.plot(
                    steps_lambda,
                    repeated_lambda[: int(lambda_steps), idx],
                    label=ENGINE_LEGEND_LABELS[2],
                    **line_styles["repeated"],
                )
                plt.plot(
                    steps_lambda,
                    hybrid_lambda[: int(lambda_steps), idx],
                    label=ENGINE_LEGEND_LABELS[1],
                    **line_styles["hybrid"],
                )
                if hybrid_pred_lambda is not None:
                    plt.plot(
                        steps_lambda,
                        hybrid_pred_lambda[: int(lambda_steps), idx],
                        label="Hybrid neural prediction",
                        **line_styles["hybrid_pred"],
                    )
                plt.xlabel("Simulation step")
                plt.ylabel(f"lambda[{idx}]")
                plt.title(f"Temporal lambda[{idx}] (run_{run_idx:03d})")
                plt.grid(True, axis="both", alpha=GRID_ALPHA)
                plt.legend(loc=LEGEND_LOC)
                plt.tight_layout()
                out_path = temporal_plots_dir / f"lambda_{idx:03d}_run_{run_idx:03d}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}")

            steps_state = np.arange(int(state_steps))
            for idx in range(STATE_SLICE[0], int(state_hi)):
                plt.figure(figsize=(12, 4.5))
                plt.plot(
                    steps_state,
                    axion_state[: int(state_steps), idx],
                    label=ENGINE_LEGEND_LABELS[0],
                    **line_styles["axion"],
                )
                plt.plot(
                    steps_state,
                    repeated_state[: int(state_steps), idx],
                    label=ENGINE_LEGEND_LABELS[2],
                    **line_styles["repeated"],
                )
                plt.plot(
                    steps_state,
                    hybrid_state[: int(state_steps), idx],
                    label=ENGINE_LEGEND_LABELS[1],
                    **line_styles["hybrid"],
                )
                plt.plot(
                    steps_state,
                    hybrid_pred_state[: int(state_steps), idx],
                    label="Hybrid neural prediction",
                    **line_styles["hybrid_pred"],
                )
                plt.xlabel("Simulation step")
                plt.ylabel(f"state_vel_component[{idx}]")
                plt.title(f"Temporal state component[{idx}] (run_{run_idx:03d})")
                plt.grid(True, axis="both", alpha=GRID_ALPHA)
                plt.legend(loc=LEGEND_LOC)
                plt.tight_layout()
                out_path = temporal_plots_dir / f"state_{idx:03d}_run_{run_idx:03d}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
