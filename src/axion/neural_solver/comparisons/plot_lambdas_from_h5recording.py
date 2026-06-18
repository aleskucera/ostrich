from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Example HDF5 filenames (repo `data/logs/`) — same style as plot_hdf5log_from_example.py
H5_FILES = [
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_17-25-41.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-04-23_23-54-49.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_13-21-25.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_17-34-35.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_17-37-21.h5",
    "AxioneEngineWithNeuralLambdas_example_2026-05-17_17-42-40.h5"
]

# --------------------------------------------------------
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "data/logs"

# Channels per figure (indices are inclusive ranges on λ index).
FIG1_LAMBDA_INDICES: tuple[int, ...] = (10, 11, 12, 13)
FIG2_LAMBDA_INDICES: tuple[int, ...] = tuple(range(14, 22))  # 14 … 22

# ---------- hardcoded knobs (edit here) ----------
HDF5_PATH = DEFAULT_LOG_DIR / H5_FILES[5]
# Simulation timestep [s]; `None` -> x-axis is step index ("Step").
DT: float | None = None
# Half-open slice along time `[STEP_START : STEP_STOP)`; `STEP_STOP=None` means end of traj.
STEP_START = 0
STEP_STOP = 250
STRIDE = 1
# "linear" | "symlog" | "log" (log masks non-positive values; prefer symlog for signed λ).
Y_SCALE = "symlog"
SYMLOG_LINTHRESH = 1e-3
# --------------------------------------------------------

LINEWIDTH = 1.5
GRID_ALPHA = 0.3

# Match typography in plot_hdf5log_from_example.py (_apply_academic_matplotlib_style).
BASE_FONTSIZE = 13
AXES_TICKS_FONTSIZE = BASE_FONTSIZE + 2
LEGEND_FONTSIZE = BASE_FONTSIZE
AXES_LABELS_FONTSIZE = BASE_FONTSIZE + 2
TITLE_FONTSIZE = BASE_FONTSIZE + 2


def _apply_matching_hdf5log_typography() -> None:
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


def _build_time_axis(length: int, dt: float | None) -> tuple[np.ndarray, str]:
    if dt is None:
        return np.arange(length), "Time step [-]"
    if dt <= 0:
        raise ValueError(f"DT must be positive when set, got {dt}")
    return np.arange(length, dtype=float) * dt, "Time [s]"


def _load_next_lambdas(hdf5_path: Path) -> np.ndarray:
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as h5f:
        if "data" not in h5f:
            raise KeyError(f"Missing top-level group 'data' in {hdf5_path}")
        group = h5f["data"]
        if "next_lambdas" not in group:
            raise KeyError(f"Missing data/next_lambdas in {hdf5_path}")
        arr = np.asarray(group["next_lambdas"][:]).squeeze()
    if arr.ndim != 2:
        raise ValueError(
            "next_lambdas must become shape (T, N) after squeezing; " f"got {arr.shape}"
        )
    return arr


def _resolve_channel_indices(
    n_lambda: int,
    lambda_indices: list[int] | None,
    lambda_start: int,
    lambda_stop: int,
) -> list[int]:
    if lambda_indices is not None:
        for i in lambda_indices:
            if i >= n_lambda:
                raise ValueError(
                    f"lambda index {i} out of range for {n_lambda} channels"
                )
        return lambda_indices
    start = max(0, lambda_start)
    stop = min(n_lambda, lambda_stop)
    if stop <= start:
        raise ValueError(
            f"Invalid lambda slice [{lambda_start}:{lambda_stop}] "
            f"for total {n_lambda} lambdas"
        )
    return list(range(start, stop))


def _apply_time_window_and_stride(
    time_axis: np.ndarray,
    lambdas: np.ndarray,
    *,
    step_start: int,
    step_stop: int | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_len = lambdas.shape[0]
    if step_start < 0 or step_start > t_len:
        raise ValueError(f"step_start must be in [0, {t_len}], got {step_start}")
    end = t_len if step_stop is None else step_stop
    if end < step_start or end > t_len:
        raise ValueError(
            f"Invalid step window [{step_start}:{end}) for trajectory length {t_len}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    sel = slice(step_start, end, stride)
    return time_axis[sel], lambdas[sel, :]


def plot_simulator_lambdas(
    hdf5_path: Path,
    *,
    dt: float | None = None,
    lambda_start: int = 0,
    lambda_stop: int = 0,
    lambda_indices: list[int] | None = None,
    step_start: int = 0,
    step_stop: int | None = None,
    stride: int = 1,
    y_scale: str = "linear",
    symlog_linthresh: float = 1e-3,
    lambdas_full: np.ndarray | None = None,
) -> None:
    if y_scale not in ("linear", "symlog", "log"):
        raise ValueError(f"y_scale must be 'linear', 'symlog', or 'log', got {y_scale!r}")
    arr = lambdas_full if lambdas_full is not None else _load_next_lambdas(hdf5_path)
    t_full, x_label = _build_time_axis(arr.shape[0], dt)

    indices = _resolve_channel_indices(arr.shape[1], lambda_indices, lambda_start, lambda_stop)
    time_plot, lambdas_plot = _apply_time_window_and_stride(
        t_full,
        arr,
        step_start=step_start,
        step_stop=step_stop,
        stride=stride,
    )

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(10, len(indices))))

    fig, ax = plt.subplots(figsize=(6, 5))
    if len(indices) <= 4:
        ch_str = ",".join(str(i) for i in indices)
        ax.set_title(rf"Normal contact $\lambda$ values (Axion simulator)")
    else:
        ax.set_title(rf"Friction $\lambda$ values (Axion simulator)")

    for k, chan in enumerate(indices):
        y = lambdas_plot[:, chan].astype(float, copy=False)
        plot_y = np.asarray(y)
        if y_scale == "log":
            bad = (~np.isfinite(plot_y)) | (plot_y <= 0)
            if np.any(bad):
                warnings.warn(
                    "y-scale=log masks non-positive and non-finite simulator λ values.",
                    stacklevel=1,
                )
                plot_y = np.ma.masked_where(bad, plot_y)

        ax.plot(
            time_plot,
            plot_y,
            linewidth=LINEWIDTH,
            color=colors[k % len(colors)],
            label=rf"$\lambda_{{{chan}}}$",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\lambda$ [N]")
    ax.grid(True, alpha=GRID_ALPHA)

    if y_scale == "symlog":
        if symlog_linthresh <= 0:
            raise ValueError("SYMLOG_LINTHRESH must be positive")
        ax.set_yscale("symlog", linthresh=symlog_linthresh)
    elif y_scale == "log":
        ax.set_yscale("log")

    ax.legend(loc="best", ncol=min(4, len(indices)))
    fig.tight_layout()


def main() -> None:
    _apply_matching_hdf5log_typography()
    lambdas_full = _load_next_lambdas(HDF5_PATH)
    common = dict(
        dt=DT,
        step_start=STEP_START,
        step_stop=STEP_STOP,
        stride=STRIDE,
        y_scale=Y_SCALE,
        symlog_linthresh=SYMLOG_LINTHRESH,
        lambdas_full=lambdas_full,
    )
    plot_simulator_lambdas(
        HDF5_PATH,
        lambda_indices=list(FIG1_LAMBDA_INDICES),
        **common,
    )
    plot_simulator_lambdas(
        HDF5_PATH,
        lambda_indices=list(FIG2_LAMBDA_INDICES),
        **common,
    )
    plt.show()


if __name__ == "__main__":
    main()
