from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from axion.neural_solver.utils.pendulum_lambda_layout import expand_pendulum_engine_lambdas_numpy

NO_CONTACT_MODELS_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-24_14-01-52.h5"
]

MTL_JUMP_MODELS_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-06-11.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-06-39.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-07-02.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_10-07-23.h5",
]

CONTACT_MTL_MODELS_LAMBDA_REGR_ONLY_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-25-41.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-26-10.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-26-32.h5",
]

CONTACT_MTL_MODELS_CONDITIONED_LAMBDA_REGR_ONLY_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-53-39.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-04.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-26.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-49.h5",
]

MSE_STATE_JOINT_LAMBDA_MODELS_HDF5_LOG_FILE_NAMES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-26_16-19-48.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-26_16-20-57.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-26_16-21-58.h5",
]

MSE_32 = "AxioneEngineWithNeuralLambdas_example_2026-05-14_12-57-47.h5"
MODEL_299 = [
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_13-01-48.h5", # INITIAL_STATE = (0.5, -0.3, 1.0, -2.0)
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_13-09-27.h5", # INITIAL_STATE = (-3.1415/2, -0.1, 2.0, 3.0)
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_13-16-29.h5", # INITIAL_STATE = (-3.1415/3, -0.3, 1.0, -1.5)
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_13-21-25.h5"  # INITIAL_STATE = (-3.1415/3, -0.3, 0.5, -1.5) 
]

MODEL_304 = [
    "AxioneEngineWithNeuralLambdas_example_2026-05-18_10-01-20.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-05-18_10-07-00.h5"
] 

NO_CONTACT_MODELS_INFO = [
    "pure mse, w_state = 500",
    "mse with lambda in log space, w_state = 2",
    "residual pure",
    "residual + mse on last timestep, w_state = 500",
    "residual + mse on whole window T, w_state = 500"
]

MTL_JUMP_MODELS_MODEL_INFOS = [
    "with 0.01*(1-y_{cls}) * SmoothL1(jump_pred, 0) term",
    "y_{cls} * SmoothL1(jump_pred, jump_gt)",
    "y_{cls} * MSE(jump_pred, jump_gt)",
    "y_{cls} * SmoothL1(AxioneEngineWithNeuralLambdas_example_2026-05-18_09-45-20.h5jump_pred, jump_gt), 200 epoch",
]

CONTACT_MTL_MODELS_LAMBDA_REGR_ONLY_MODEL_INFOS = [
    "contact_mtl, asinh, mse",
    "contact_mtl, asinh + ouput normal, mse",
    "contact_mtl, no tranform, mse",
]

CONTACT_MTL_MODELS_CONDITIONED_LAMBDA_REGR_ONLY_MODEL_INFOS = [
    "contact_mtl, pure mse",
    "contact_mtl, asinh, mse",
    "contact_mtl, asinh + output normal, mse",
    "contact_mtl, asinh + output normal, mse, 1M dataset"
]

MSE_STATE_JOINT_LAMBDA_MODELS_MODEL_INFOS = [
    "mse, w_state = 500, 40k dataset",
    "mse, w_state = 500, 2M datadet, ||q||^2 term included in angle loss",
    "mse, w_state = 5e4, 2M datadet, ||q||^2 term NOT included in angle loss",
]

#--------------------------------------------------------
ACADEMIC_PLOTTING = True
BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE
TITLE_FONTSIZE = BASE_FONTSIZE + 2
LINEWIDTH = 2.25  # Used for every ax.plot linewidth in this script
GRID_ALPHA = 0.3
# Academic lambda panels use dimensionless [-] on the axis label (was SI Newtons [N]).
#--------------------------------------------------------
ID = 2
MODEL_INFO = "mse 32"
COMPARISON_CSV_PATH = None # Path(__file__).resolve().parent / "mse_state_and_joint_lambdas.csv" # None
DEFAULT_HDF5_PATH = Path(__file__).resolve().parents[4] / "data/logs" /MODEL_299[3]#MODEL_299[3] #AxioneEngineWithNeuralLambdas_example_2026-05-12_09-07-56.h5"
DEFAULT_LAMBDA_SLICE = slice(0,10) 
ANALYZE_INCOMPLETE_MTL = False
ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY = False
ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY = False
# Must match `jump_target_scale` / MTLModel.jump_target_scale for the checkpoint that produced the log.
DEFAULT_JUMP_TARGET_SCALE = 100.0
SIM_COLOR = "tab:blue"
PRED_COLOR = "tab:orange"
ACADEMIC_SMALL_LAMBDA_YLIM = 0.1
ACADEMIC_SMALL_LAMBDA_MAX_ABS_THRESHOLD = 0.05
# Inclination of time-step tick labels on academic lambda subplots (bottom row).
ACADEMIC_LAMBDA_SUBPLOT_XTICK_ROTATION_DEG = 32
# Endpoint labels for blue (simulator) state traces on the academic state panel.
# Offsets are per state component index (q_0, q_1, u_0, u_1).
ACADEMIC_STATE_TRACE_LABEL_XOFFSET_PTS = (-486, -457, -475, -440)
ACADEMIC_STATE_TRACE_LABEL_YOFFSET_PTS = (0, -13, +27, +100)
ACADEMIC_STATE_TRACE_LABEL_COLOR = "black"
ACADEMIC_STATE_TRACE_LABEL_FONTSIZE = BASE_FONTSIZE + 3


def _apply_academic_matplotlib_style() -> None:
    """Set matplotlib rcParams for publication-style typography when ACADEMIC_PLOTTING."""
    if not ACADEMIC_PLOTTING:
        return
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


def _format_x_label_for_plot(x_label: str) -> str:
    if not ACADEMIC_PLOTTING:
        return x_label
    if x_label == "Step":
        return r"Time step $t$ [-]"
    if x_label == "Time [s]":
        return r"Time $t$ [s]"
    return x_label


def _state_ylabel_academic(idx: int) -> str:
    """Y-axis label for the 4-DOF pendulum state vector (joint_q + joint_qd)."""
    return (
        r"$q_0$   [rad]",
        r"$q_1$   [rad]",
        r"$u_0$   [$\mathrm{rad}\cdot\mathrm{s}^{-1}$]",
        r"$u_1$   [$\mathrm{rad}\cdot\mathrm{s}^{-1}$]",
    )[idx]


def _state_component_math_symbol(idx: int) -> str:
    """Short LaTeX symbol for pendulum state component (matches `_state_ylabel_academic` order)."""
    return (r"$q_0$", r"$q_1$", r"$u_0$", r"$u_1$")[idx]


def _parse_optional_path(value: str) -> Path | None:
    if value.lower() in {"none", "null"}:
        return None
    return Path(value)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot real-vs-predicted next states and selected next lambdas "
            "from an AxioneEngineWithNeuralLambdas HDF5 log. "
            "If the file also contains lambda_activity and "
            "lambda_activity_ground_truth, a fourth figure compares predicted vs "
            "simulator activity labels (same lambda index slice as --lambda-start/--lambda-stop)."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=DEFAULT_HDF5_PATH,
        help="Path to AxioneEngineWithNeuralLambdas log (.h5).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional simulation time step. If omitted, x-axis is step index.",
    )
    parser.add_argument(
        "--lambda-start",
        type=int,
        default=DEFAULT_LAMBDA_SLICE.start,
        help="First lambda index to plot (inclusive).",
    )
    parser.add_argument(
        "--lambda-stop",
        type=int,
        default=DEFAULT_LAMBDA_SLICE.stop,
        help="Last lambda index to plot (exclusive).",
    )
    parser.add_argument(
        "--csv",
        type=_parse_optional_path,
        default=COMPARISON_CSV_PATH,
        help=(
            "Append one summary row to this CSV after a successful run "
            "(creates file with header if missing). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not append to the comparison CSV.",
    )
    return parser.parse_args()


def _load_and_validate(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    required = [
        "next_states",
        "next_lambdas",
    ]
    require_predicted_states = not (
        ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
        or ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
    )
    if require_predicted_states:
        required.append("predicted_next_states")

    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f:
            raise KeyError(f"Missing top-level group 'data' in {hdf5_path}")
        group = h5f["data"]
        missing = [key for key in required if key not in group]
        if missing:
            raise KeyError(
                f"Missing required dataset(s) in {hdf5_path}: {', '.join(missing)}"
            )

        next_states = np.asarray(group["next_states"][:]).squeeze()
        predicted_next_states = (
            np.asarray(group["predicted_next_states"][:]).squeeze()
            if "predicted_next_states" in group
            else None
        )
        next_lambdas = np.asarray(group["next_lambdas"][:]).squeeze()
        predicted_next_lambdas = (
            np.asarray(group["predicted_next_lambdas"][:]).squeeze()
            if "predicted_next_lambdas" in group
            else None
        )

    if next_states.ndim != 2:
        raise ValueError(
            f"next_states must become shape (T, 4) after squeezing; got {next_states.shape}"
        )
    if predicted_next_states is not None and predicted_next_states.ndim != 2:
        raise ValueError(
            "predicted_next_states must become shape (T, 4) after squeezing; "
            f"got {predicted_next_states.shape}"
        )
    if next_lambdas.ndim != 2:
        raise ValueError(
            "next_lambdas must become shape (T, N) after squeezing; "
            f"got {next_lambdas.shape}"
        )
    if predicted_next_lambdas is not None and predicted_next_lambdas.ndim != 2:
        raise ValueError(
            "predicted_next_lambdas must become shape (T, N) after squeezing; "
            f"got {predicted_next_lambdas.shape}"
        )
    if (
        predicted_next_lambdas is not None
        and next_lambdas.shape[1] < predicted_next_lambdas.shape[1]
    ):
        next_lambdas = expand_pendulum_engine_lambdas_numpy(
            next_lambdas, int(predicted_next_lambdas.shape[1])
        )
    if predicted_next_states is not None and next_states.shape != predicted_next_states.shape:
        raise ValueError(
            "next_states and predicted_next_states shape mismatch: "
            f"{next_states.shape} vs {predicted_next_states.shape}"
        )
    if predicted_next_lambdas is not None and next_lambdas.shape != predicted_next_lambdas.shape:
        raise ValueError(
            "next_lambdas and predicted_next_lambdas shape mismatch: "
            f"{next_lambdas.shape} vs {predicted_next_lambdas.shape}"
        )
    if next_states.shape[0] != next_lambdas.shape[0]:
        raise ValueError(
            "State and lambda sequence lengths differ: "
            f"{next_states.shape[0]} vs {next_lambdas.shape[0]}"
        )
    if next_states.shape[1] != 4:
        raise ValueError(f"Expected 4 state dimensions, got {next_states.shape[1]}")

    return next_states, predicted_next_states, next_lambdas, predicted_next_lambdas


def _try_load_lambda_activity_pair(
    hdf5_path: Path,
    t_len: int,
    reference_n_lambda: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load lambda_activity and lambda_activity_ground_truth if both exist under data/.
    Returns None if either dataset is missing. Validates shape (T, C) and T == t_len.
    When ``reference_n_lambda`` is set and C is narrower (e.g. engine λ without the
    control block), both arrays are expanded to canonical Pendulum width with zeros
    in the inserted slots so they match ``next_lambdas`` / predictions.
    """
    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f:
            return None
        group = h5f["data"]
        if "lambda_activity" not in group or "lambda_activity_ground_truth" not in group:
            return None
        pred = np.asarray(group["lambda_activity"][:]).squeeze()
        gt = np.asarray(group["lambda_activity_ground_truth"][:]).squeeze()

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            "lambda_activity and lambda_activity_ground_truth must be 2D (T, C) after squeeze; "
            f"got shapes {pred.shape} and {gt.shape}"
        )
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(
            "lambda_activity batch length mismatch: "
            f"{pred.shape[0]} vs {gt.shape[0]}"
        )
    w_pred, w_gt = pred.shape[1], gt.shape[1]
    if w_pred != w_gt:
        target_w = max(w_pred, w_gt)
        if w_pred < target_w:
            pred = expand_pendulum_engine_lambdas_numpy(pred, target_w)
        if w_gt < target_w:
            gt = expand_pendulum_engine_lambdas_numpy(gt, target_w)
    if pred.shape != gt.shape:
        raise ValueError(
            "lambda_activity and lambda_activity_ground_truth shape mismatch after alignment: "
            f"{pred.shape} vs {gt.shape}"
        )
    if pred.shape[0] != t_len:
        raise ValueError(
            "lambda_activity length does not match trajectory length: "
            f"{pred.shape[0]} vs {t_len}"
        )
    if (
        reference_n_lambda is not None
        and pred.shape[1] < reference_n_lambda
    ):
        pred = expand_pendulum_engine_lambdas_numpy(pred, reference_n_lambda)
        gt = expand_pendulum_engine_lambdas_numpy(gt, reference_n_lambda)
    return pred, gt


def _load_lambda_jump(hdf5_path: Path, t_len: int, n_lambda: int) -> np.ndarray:
    """Load data/lambda_jump and validate shape (T, N)."""
    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f or "lambda_jump" not in h5f["data"]:
            raise KeyError(
                f"Missing data/lambda_jump in {hdf5_path} (required for ANALYZE_INCOMPLETE_MTL)."
            )
        jump = np.asarray(h5f["data"]["lambda_jump"][:]).squeeze()

    if jump.ndim != 2:
        raise ValueError(
            "lambda_jump must be 2D (T, N) after squeeze; "
            f"got shape {jump.shape}"
        )
    if jump.shape[0] != t_len:
        raise ValueError(
            "lambda_jump shape mismatch: "
            f"got {jump.shape}, expected ({t_len}, {n_lambda})"
        )
    if jump.shape[1] != n_lambda:
        if jump.shape[1] < n_lambda:
            jump = expand_pendulum_engine_lambdas_numpy(jump, n_lambda)
        else:
            raise ValueError(
                "lambda_jump shape mismatch: "
                f"got {jump.shape}, expected ({t_len}, {n_lambda})"
            )
    return jump


def _reconstruct_incomplete_mtl_pred_lambdas(
    next_lambdas: np.ndarray,
    lambda_activity_pred: np.ndarray,
    lambda_jump: np.ndarray,
    jump_target_scale: float,
) -> np.ndarray:
    """
    Artificial next-lambda prediction for incomplete MTL logs:
    where predicted binary activity is true, use simulator next lambda + scaled neural jump;
    otherwise keep value as NaN so inactive predictions are not plotted.
    """
    if lambda_activity_pred.shape != next_lambdas.shape or lambda_jump.shape != next_lambdas.shape:
        raise ValueError(
            "Shape mismatch in incomplete-MTL reconstruction: "
            f"next_lambdas {next_lambdas.shape}, activity {lambda_activity_pred.shape}, "
            f"jump {lambda_jump.shape}"
        )
    active = (lambda_activity_pred >= 0.5)
    scaled_jump = lambda_jump.astype(np.float64, copy=False) * float(jump_target_scale)
    reconstructed = np.full_like(next_lambdas, np.nan, dtype=np.float64)
    reconstructed[active] = (
        next_lambdas.astype(np.float64, copy=False)[active] + scaled_jump[active]
    )
    return reconstructed


def _mask_predicted_lambdas_by_gt_activity(
    predicted_next_lambdas: np.ndarray,
    lambda_activity_ground_truth: np.ndarray,
) -> np.ndarray:
    """
    For conditioned lambda-regression analysis: keep predicted next lambda where the
    simulator GT activity label is active (>= 0.5); set to NaN elsewhere.
    """
    if predicted_next_lambdas.shape != lambda_activity_ground_truth.shape:
        raise ValueError(
            "Shape mismatch masking predictions by GT activity: "
            f"predicted_next_lambdas {predicted_next_lambdas.shape}, "
            f"lambda_activity_ground_truth {lambda_activity_ground_truth.shape}"
        )
    out = predicted_next_lambdas.astype(np.float64, copy=True)
    inactive = lambda_activity_ground_truth < 0.5
    out[inactive] = np.nan
    return out


def _build_time_axis(length: int, dt: float | None) -> tuple[np.ndarray, str]:
    if dt is None:
        return np.arange(length), "Step"
    if dt <= 0:
        raise ValueError(f"--dt must be positive when provided, got {dt}")
    return np.arange(length, dtype=float) * dt, "Time [s]"


def _plot_states(time_axis: np.ndarray, x_label: str, real: np.ndarray, pred: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Comparison of next state values: Axion ground truth vs neural network")

    x_disp = _format_x_label_for_plot(x_label)
    for idx, ax in enumerate(axes.flat):
        ax.plot(
            time_axis,
            real[:, idx],
            label="Axion simulator",
            linewidth=LINEWIDTH,
            color=SIM_COLOR,
        )
        ax.plot(
            time_axis,
            pred[:, idx],
            label="Neural prediction",
            linewidth=LINEWIDTH,
            linestyle="--",
            color=PRED_COLOR,
        )
        if ACADEMIC_PLOTTING and real.shape[1] == 4:
            ax.set_ylabel(_state_ylabel_academic(idx))
        else:
            ax.set_title(f"state[{idx}]")
            ax.set_ylabel("Value")
        ax.set_xlabel(x_disp)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc="best")

    fig.tight_layout()


def _plot_single_lambda_on_ax(
    ax: Axes,
    time_axis: np.ndarray,
    real: np.ndarray,
    pred: np.ndarray,
    lambda_idx: int,
    x_disp: str,
    *,
    jump_raw: np.ndarray | None = None,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    ax.plot(
        time_axis,
        real[:, lambda_idx],
        label="Simulator next lambda",
        linewidth=LINEWIDTH,
        color=SIM_COLOR,
    )
    pred_series = pred[:, lambda_idx]
    valid = np.isfinite(pred_series)
    if ANALYZE_INCOMPLETE_MTL:
        ax.scatter(
            time_axis[valid],
            pred_series[valid],
            label="Predicted next lambda (active only)",
            s=22.0,
            marker="o",
            color=PRED_COLOR,
        )
        if jump_raw is not None:
            for t_idx in np.where(valid)[0]:
                gt_lambda = real[t_idx, lambda_idx]
                scaled_jump = jump_raw[t_idx, lambda_idx] * DEFAULT_JUMP_TARGET_SCALE
                ax.annotate(
                    f"{gt_lambda:.2f}+{scaled_jump:.2f}={pred_series[t_idx]:.2f}",
                    (time_axis[t_idx], pred_series[t_idx]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="black",
                )
    elif (
        ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
        and not ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
        and not ANALYZE_INCOMPLETE_MTL
    ):
        ax.scatter(
            time_axis[valid],
            pred_series[valid],
            label="Predicted next lambda (GT-active only)",
            s=22.0,
            marker="o",
            color=PRED_COLOR,
        )
    else:
        ax.plot(
            time_axis,
            pred_series,
            label="Predicted next lambda",
            linewidth=LINEWIDTH,
            linestyle="--",
            color=PRED_COLOR,
        )
    if show_xlabel:
        ax.set_xlabel(x_disp)
    ax.set_ylabel(rf"$\lambda_{{{lambda_idx}}}$ [Nm]" if ACADEMIC_PLOTTING else "Value")
    if not ACADEMIC_PLOTTING:
        ax.set_title(f"lambda[{lambda_idx}]")
    ax.grid(True, alpha=GRID_ALPHA)
    if show_legend:
        ax.legend(loc="best")


def _maybe_apply_small_lambda_ylim(
    ax: Axes, real_series: np.ndarray, pred_series: np.ndarray
) -> None:
    vals = np.concatenate(
        [
            real_series[np.isfinite(real_series)],
            pred_series[np.isfinite(pred_series)],
        ]
    )
    if vals.size == 0:
        return
    if float(np.nanmax(np.abs(vals))) <= ACADEMIC_SMALL_LAMBDA_MAX_ABS_THRESHOLD:
        ax.set_ylim(-ACADEMIC_SMALL_LAMBDA_YLIM, ACADEMIC_SMALL_LAMBDA_YLIM)


def _plot_lambdas(
    time_axis: np.ndarray,
    x_label: str,
    real: np.ndarray,
    pred: np.ndarray,
    lambda_start: int,
    lambda_stop: int,
    jump_raw: np.ndarray | None = None,
) -> None:
    total_lambdas = real.shape[1]
    start = max(0, lambda_start)
    stop = min(total_lambdas, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda slice [{lambda_start}:{lambda_stop}] for total {total_lambdas} lambdas"
        )

    selected = list(range(start, stop))
    n = len(selected)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Next Lambdas: Simulator vs Neural Prediction (indices {start}:{stop})")

    x_disp = _format_x_label_for_plot(x_label)

    for local_idx, lambda_idx in enumerate(selected):
        ax = axes_arr[local_idx]
        _plot_single_lambda_on_ax(
            ax,
            time_axis,
            real,
            pred,
            lambda_idx,
            x_disp,
            jump_raw=jump_raw,
            show_xlabel=True,
        )

    for extra_idx in range(n, len(axes_arr)):
        axes_arr[extra_idx].set_visible(False)

    fig.tight_layout()


def _plot_academic_combined_dashboard(
    time_axis: np.ndarray,
    x_label: str,
    next_states: np.ndarray,
    predicted_next_states: np.ndarray,
    next_lambdas: np.ndarray,
    predicted_next_lambdas: np.ndarray,
    lambda_start: int,
    lambda_stop: int,
    jump_raw: np.ndarray | None = None,
) -> None:
    total_lambdas = next_lambdas.shape[1]
    start = max(0, lambda_start)
    stop = min(total_lambdas, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda slice [{lambda_start}:{lambda_stop}] for total {total_lambdas} lambdas"
        )
    n_lambda_selected = stop - start
    if n_lambda_selected != 10:
        raise ValueError(
            "Academic combined figure expects exactly 10 lambda indices "
            f"(slice length {stop}-{start}={n_lambda_selected}); "
            "adjust --lambda-start/--lambda-stop."
        )

    x_disp = _format_x_label_for_plot(x_label)
    selected = list(range(start, stop))

    fig = plt.figure(figsize=(11, 17))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[2.7, 5.4], hspace=0.25)
    ax_state = fig.add_subplot(outer_gs[0, 0])
    inner_gs = outer_gs[1].subgridspec(5, 2, hspace=0.35, wspace=0.4)

    for i in range(next_states.shape[1]):
        ax_state.plot(
            time_axis,
            next_states[:, i],
            linewidth=LINEWIDTH,
            color=SIM_COLOR,
            label=("Axion simulator" if i == 0 else "_sim_extra"),
        )
    for i in range(predicted_next_states.shape[1]):
        ax_state.plot(
            time_axis,
            predicted_next_states[:, i],
            linewidth=LINEWIDTH,
            linestyle="--",
            color=PRED_COLOR,
            label=("Neural prediction (teacher-forced)" if i == 0 else "_pred_extra"),
        )
    n_state = next_states.shape[1]
    for i in range(min(n_state, 4)):
        x_end = float(time_axis[-1])
        y_end = float(next_states[-1, i])
        ax_state.annotate(
            _state_component_math_symbol(i),
            (x_end, y_end),
            textcoords="offset points",
            xytext=(
                ACADEMIC_STATE_TRACE_LABEL_XOFFSET_PTS[i],
                ACADEMIC_STATE_TRACE_LABEL_YOFFSET_PTS[i],
            ),
            color=ACADEMIC_STATE_TRACE_LABEL_COLOR,
            fontsize=ACADEMIC_STATE_TRACE_LABEL_FONTSIZE,
            ha="left",
            va="center",
            clip_on=False,
        )
    ax_state.set_xlabel(x_disp)
    ax_state.set_ylabel("States")
    ax_state.grid(True, alpha=GRID_ALPHA)
    ax_state.legend(loc="best")

    for local_idx, lambda_idx in enumerate(selected):
        row, col = divmod(local_idx, 2)
        ax_l = fig.add_subplot(inner_gs[row, col], sharex=ax_state)
        bottom_row = row == 4
        _plot_single_lambda_on_ax(
            ax_l,
            time_axis,
            next_lambdas,
            predicted_next_lambdas,
            lambda_idx,
            x_disp,
            jump_raw=jump_raw,
            show_xlabel=bottom_row,
            show_legend=False,
        )
        _maybe_apply_small_lambda_ylim(
            ax_l, next_lambdas[:, lambda_idx], predicted_next_lambdas[:, lambda_idx]
        )
        if not bottom_row:
            plt.setp(ax_l.get_xticklabels(), visible=False)
        else:
            plt.setp(
                ax_l.get_xticklabels(),
                rotation=ACADEMIC_LAMBDA_SUBPLOT_XTICK_ROTATION_DEG,
                ha="right",
                rotation_mode="anchor",
            )

    # fig.suptitle(
    #     "Next states and constraint forces: simulator vs neural prediction "
    #     f"(indices $\\lambda_{{{start}}}\\ldots\\lambda_{{{stop - 1}}}$)"
    # )
    fig.tight_layout()


def _plot_lambda_activity_labels(
    time_axis: np.ndarray,
    x_label: str,
    pred: np.ndarray,
    gt: np.ndarray,
    lambda_start: int,
    lambda_stop: int,
) -> None:
    total_channels = pred.shape[1]
    start = max(0, lambda_start)
    stop = min(total_channels, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda activity slice [{lambda_start}:{lambda_stop}] "
            f"for total {total_channels} channels"
        )

    selected = list(range(start, stop))
    n = len(selected)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharex=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"Lambda activity: predicted vs ground truth (indices {start}:{stop})"
    )

    x_disp = _format_x_label_for_plot(x_label)
    activity_ylabel = "Activity [-]" if ACADEMIC_PLOTTING else "Label"

    for local_idx, ch_idx in enumerate(selected):
        ax = axes_arr[local_idx]
        ax.plot(
            time_axis,
            pred[:, ch_idx],
            label="Predicted activity",
            linewidth=LINEWIDTH,
            color=PRED_COLOR,
        )
        ax.plot(
            time_axis,
            gt[:, ch_idx],
            label="Simulator GT",
            linewidth=LINEWIDTH,
            linestyle="--",
            color=SIM_COLOR,
        )
        if not ACADEMIC_PLOTTING:
            ax.set_title(f"channel[{ch_idx}]")
        ax.set_xlabel(x_disp)
        ax.set_ylabel(activity_ylabel)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(loc="best")

    for extra_idx in range(n, len(axes_arr)):
        axes_arr[extra_idx].set_visible(False)

    fig.tight_layout()


def _compute_prediction_metrics(
    next_states: np.ndarray,
    predicted_next_states: np.ndarray | None,
    next_lambdas: np.ndarray,
    predicted_next_lambdas: np.ndarray | None,
) -> tuple[float, float, float, float, float, float]:
    """
    Returns (state_mae, lambda_mae, state_total_abs, lambda_total_abs,
             total_abs_error, total_squared_error).
    total_abs_error is sum of absolute errors over all state and lambda elements.
    total_squared_error is sum of squared errors over all state and lambda elements.
    When ``predicted_next_lambdas`` is None, lambda metrics are NaN / omitted from totals.
    """
    if predicted_next_states is not None:
        state_abs_err = np.abs(next_states - predicted_next_states)
        state_mae = float(np.mean(state_abs_err))
        state_total = float(np.sum(state_abs_err))
        state_sq = np.sum((next_states - predicted_next_states) ** 2)
    else:
        state_mae = float("nan")
        state_total = 0.0
        state_sq = 0.0
    if predicted_next_lambdas is None:
        lambda_mae = float("nan")
        lambda_total = float("nan")
        lambda_sq = 0.0
        total_abs_error = state_total
        total_squared_error = float(state_sq + lambda_sq)
        return (
            state_mae,
            lambda_mae,
            state_total,
            lambda_total,
            total_abs_error,
            total_squared_error,
        )
    lambda_diff = next_lambdas - predicted_next_lambdas
    lambda_valid = np.isfinite(lambda_diff)
    lambda_abs_err = np.abs(lambda_diff)

    if np.any(lambda_valid):
        lambda_mae = float(np.mean(lambda_abs_err[lambda_valid]))
        lambda_total = float(np.sum(lambda_abs_err[lambda_valid]))
        lambda_sq = np.sum((lambda_diff[lambda_valid]) ** 2)
    else:
        lambda_mae = float("nan")
        lambda_total = 0.0
        lambda_sq = 0.0
    total_abs_error = state_total + lambda_total

    total_squared_error = float(state_sq + lambda_sq)

    return (
        state_mae,
        lambda_mae,
        state_total,
        lambda_total,
        total_abs_error,
        total_squared_error,
    )


def _append_comparison_csv(
    csv_path: Path,
    log_filename: str,
    model_info: str,
    trajectory_timesteps: int,
    state_mae: float,
    lambda_mae: float,
    lambda_total_abs_error: float,
    total_abs_error: float,
    total_squared_error: float,
) -> None:
    fieldnames = (
        "log_file",
        "model_info",
        "trajectory_timesteps",
        "state_mae",
        "lambda_mae",
        "lambda_total_absolute_error",
        "total_absolute_error",
        "total_squared_error",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "log_file": log_filename,
                "model_info": model_info,
                "trajectory_timesteps": trajectory_timesteps,
                "state_mae": state_mae,
                "lambda_mae": lambda_mae,
                "lambda_total_absolute_error": lambda_total_abs_error,
                "total_absolute_error": total_abs_error,
                "total_squared_error": total_squared_error,
            }
        )


def _plot_mae_summary(
    state_mae: float,
    lambda_mae: float,
    state_total: float,
    lambda_total: float,
) -> None:
    fig, (ax_mae, ax_total) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Overall Prediction Error Summary")

    state_label = "State MAE" if not math.isnan(state_mae) else "State MAE (N/A)"
    lambda_label = "Lambda MAE" if not math.isnan(lambda_mae) else "Lambda MAE (N/A)"
    mae_labels = [state_label, lambda_label]
    mae_values = [state_mae, lambda_mae]
    mae_plot_heights = [0.0 if math.isnan(v) else v for v in mae_values]
    mae_bars = ax_mae.bar(mae_labels, mae_plot_heights, width=0.45)
    for bar, value in zip(mae_bars, mae_values):
        text = "N/A" if math.isnan(value) else f"{value:.6f}"
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
        )
    ax_mae.set_ylabel("Mean Absolute Error")
    ax_mae.set_title("MAE")
    ax_mae.grid(True, axis="y", alpha=GRID_ALPHA)

    total_labels = [
        "State total acc. abs. error",
        "Lambda total acc. abs. error"
        if not math.isnan(lambda_total) else "Lambda total (N/A)",
    ]
    total_values = [state_total, lambda_total]
    total_plot_heights = [0.0 if math.isnan(v) else v for v in total_values]
    total_bars = ax_total.bar(total_labels, total_plot_heights, width=0.45)
    for bar, value in zip(total_bars, total_values):
        text = "N/A" if math.isnan(value) else f"{value:.4f}"
        ax_total.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
        )
    ax_total.set_ylabel("Total Accumulated Absolute Error")
    ax_total.set_title("Total Accumulated Absolute Error")
    ax_total.grid(True, axis="y", alpha=GRID_ALPHA)

    fig.tight_layout()


def main() -> None:
    args = _parse_args()
    _apply_academic_matplotlib_style()
    next_states, predicted_next_states, next_lambdas, predicted_next_lambdas = _load_and_validate(
        args.hdf5
    )
    t_len = int(next_states.shape[0])
    activity_pair = _try_load_lambda_activity_pair(
        args.hdf5, t_len, reference_n_lambda=int(next_lambdas.shape[1])
    )
    lambda_jump_arr = None

    if ANALYZE_INCOMPLETE_MTL:
        if activity_pair is None:
            raise ValueError(
                "ANALYZE_INCOMPLETE_MTL is True but the HDF5 log is missing "
                "lambda_activity and/or lambda_activity_ground_truth under data/."
            )
        lambda_activity_pred, _ = activity_pair
        lambda_jump_arr = _load_lambda_jump(args.hdf5, t_len, next_lambdas.shape[1])
        predicted_next_lambdas = _reconstruct_incomplete_mtl_pred_lambdas(
            next_lambdas,
            lambda_activity_pred,
            lambda_jump_arr,
            DEFAULT_JUMP_TARGET_SCALE,
        )
    elif (
        ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY
        and not ANALYZE_CONTACT_MTL_LAMBDA_REGR_ONLY
        and not ANALYZE_INCOMPLETE_MTL
    ):
        if predicted_next_lambdas is None:
            raise ValueError(
                "ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY requires "
                "predicted_next_lambdas in the HDF5 log."
            )
        if activity_pair is None:
            raise ValueError(
                "ANALYZE_CONTACT_MTL_CONDITIONED_LAMBDA_REGR_ONLY requires "
                "lambda_activity and lambda_activity_ground_truth under data/."
            )
        _, lambda_activity_gt = activity_pair
        predicted_next_lambdas = _mask_predicted_lambdas_by_gt_activity(
            predicted_next_lambdas,
            lambda_activity_gt,
        )

    time_axis, x_label = _build_time_axis(next_states.shape[0], args.dt)

    academic_dashboard = (
        ACADEMIC_PLOTTING
        and predicted_next_states is not None
        and predicted_next_lambdas is not None
    )
    if ACADEMIC_PLOTTING and not academic_dashboard:
        raise ValueError(
            "ACADEMIC_PLOTTING requires predicted_next_states and predicted_next_lambdas "
            "for the combined figure; this log/configuration omits one of them."
        )

    if academic_dashboard:
        _plot_academic_combined_dashboard(
            time_axis,
            x_label,
            next_states,
            predicted_next_states,
            next_lambdas,
            predicted_next_lambdas,
            args.lambda_start,
            args.lambda_stop,
            jump_raw=lambda_jump_arr if ANALYZE_INCOMPLETE_MTL else None,
        )
    else:
        if predicted_next_states is not None:
            _plot_states(time_axis, x_label, next_states, predicted_next_states)
        if predicted_next_lambdas is not None:
            _plot_lambdas(
                time_axis=time_axis,
                x_label=x_label,
                real=next_lambdas,
                pred=predicted_next_lambdas,
                lambda_start=args.lambda_start,
                lambda_stop=args.lambda_stop,
                jump_raw=lambda_jump_arr if ANALYZE_INCOMPLETE_MTL else None,
            )

    (
        state_mae,
        lambda_mae,
        state_total_abs,
        lambda_total_abs,
        total_abs_error,
        total_squared_error,
    ) = _compute_prediction_metrics(
        next_states,
        predicted_next_states,
        next_lambdas,
        predicted_next_lambdas,
    )
    if not academic_dashboard:
        _plot_mae_summary(state_mae, lambda_mae, state_total_abs, lambda_total_abs)
    if not args.no_csv and args.csv is not None:
        _append_comparison_csv(
            args.csv,
            log_filename=args.hdf5.name,
            model_info=MODEL_INFO,
            trajectory_timesteps=int(next_states.shape[0]),
            state_mae=state_mae,
            lambda_mae=lambda_mae,
            lambda_total_abs_error=lambda_total_abs,
            total_abs_error=total_abs_error,
            total_squared_error=total_squared_error,
        )
    if not academic_dashboard and activity_pair is not None:
        lambda_activity, lambda_activity_gt = activity_pair
        _plot_lambda_activity_labels(
            time_axis=time_axis,
            x_label=x_label,
            pred=lambda_activity,
            gt=lambda_activity_gt,
            lambda_start=args.lambda_start,
            lambda_stop=args.lambda_stop,
        )
    plt.show()


if __name__ == "__main__":
    main()
