import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "benchmarks/data/autoregressive"

SCENARIO_FILES = {
    "fall": DATA_DIR / "fall_benchmark.h5",
    "pendulum": DATA_DIR / "pendulum_benchmark.h5",
}
PLOT_FILES_POS = {
    "fall": DATA_DIR / "fall_error_translation.pdf",
    "pendulum": DATA_DIR / "pendulum_error_translation.pdf",
}
PLOT_FILES_ROT = {
    "fall": DATA_DIR / "fall_error_rotation.pdf",
    "pendulum": DATA_DIR / "pendulum_error_rotation.pdf",
}

GNN_KEYS = ["mae", "res"]
GNN_NAMES = {"mae": "MAE", "res": "Residual"}
GNN_COLORS = {"mae": "tab:blue", "res": "tab:orange"}


def _quat_geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12
    n2 = np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12
    dot = np.abs(np.sum((q1 / n1) * (q2 / n2), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def compute_stats(filepath: pathlib.Path) -> dict:
    with h5py.File(filepath, "r") as f:
        n_passes = int(f.attrs["n_passes"])
        T = int(f.attrs["num_steps"]) + 1
        dt = float(f.attrs["dt"])

        pass0 = f["pass_0"]
        present_keys = [k for k in GNN_KEYS if f"gnn_{k}" in pass0]

        errors_pos = {k: [[] for _ in range(T)] for k in present_keys}
        errors_rot = {k: [[] for _ in range(T)] for k in present_keys}

        for i in range(n_passes):
            gt_q = f[f"pass_{i}/axion/body_q"][:]  # (T, B, 7)
            gt_pos = gt_q[:, :, :3]
            gt_quat = gt_q[:, :, 3:]

            for key in present_keys:
                pred_q = f[f"pass_{i}/gnn_{key}/body_q"][:]  # (T, B, 7)
                pred_pos = pred_q[:, :, :3]
                pred_quat = pred_q[:, :, 3:]

                B = gt_q.shape[1]
                for t in range(T):
                    for b in range(B):
                        displacement = np.linalg.norm(gt_pos[t, b] - gt_pos[0, b])
                        pos_err = np.linalg.norm(pred_pos[t, b] - gt_pos[t, b])
                        errors_pos[key][t].append(pos_err / max(displacement, 1e-8) * 100.0)
                        errors_rot[key][t].append(
                            _quat_geodesic_deg(pred_quat[t, b], gt_quat[t, b])
                        )

    steps = np.arange(T)

    def _percentiles(lists):
        med = np.array([np.nanmedian(x) for x in lists])
        p25 = np.array([np.nanpercentile(x, 25) for x in lists])
        p75 = np.array([np.nanpercentile(x, 75) for x in lists])
        return med, p25, p75

    def _nan_report(lists, label):
        flat = np.array([v for step in lists for v in step], dtype=float)
        n_nan = int(np.isnan(flat).sum())
        if n_nan:
            print(f"  NaN in {label}: {n_nan} / {len(flat)} ({100.0 * n_nan / len(flat):.1f}%)")

    result = dict(steps=steps, n_passes=n_passes, dt=dt, present_keys=present_keys)
    for key in present_keys:
        name = GNN_NAMES.get(key, key)
        _nan_report(errors_pos[key], f"{name} pos")
        _nan_report(errors_rot[key], f"{name} rot")
        pm, p25, p75 = _percentiles(errors_pos[key])
        rm, rp25, rp75 = _percentiles(errors_rot[key])
        result[f"{key}_pos_median"] = pm
        result[f"{key}_pos_p25"] = p25
        result[f"{key}_pos_p75"] = p75
        result[f"{key}_rot_median"] = rm
        result[f"{key}_rot_p25"] = rp25
        result[f"{key}_rot_p75"] = rp75

    return result


class LineIQRHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        color = orig_handle.get_color()
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=color,
            alpha=0.3,
            edgecolor="none",
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + height * 0.5, ydescent + height * 0.5],
            color=color,
            linewidth=2.0,
            solid_capstyle="butt",
            transform=trans,
        )
        return [rect, line]


def _plot_band(ax, time, median, p25, p75, color, label):
    (line,) = ax.plot(time, median, color=color, linewidth=2.0, label=label)
    ax.fill_between(time, p25, p75, color=color, alpha=0.2)
    return line


def _save_single_plot(ax, lines, xlabel, ylabel, out_path):
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(handles=lines, handler_map={Line2D: LineIQRHandler()}, loc="upper left", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {out_path}")
    plt.close(ax.figure)


def plot_scenario(stats: dict, scenario: str, out_pos: pathlib.Path, out_rot: pathlib.Path) -> None:
    present_keys = stats["present_keys"]
    steps = stats["steps"]

    fig_pos, ax_pos = plt.subplots(figsize=(10, 4))
    fig_rot, ax_rot = plt.subplots(figsize=(10, 4))

    pos_lines, rot_lines = [], []
    for key in present_keys:
        color = GNN_COLORS.get(key, "tab:green")
        name = GNN_NAMES.get(key, key)
        pos_lines.append(
            _plot_band(
                ax_pos,
                steps,
                stats[f"{key}_pos_median"],
                stats[f"{key}_pos_p25"],
                stats[f"{key}_pos_p75"],
                color=color,
                label=name,
            )
        )
        rot_lines.append(
            _plot_band(
                ax_rot,
                steps,
                stats[f"{key}_rot_median"],
                stats[f"{key}_rot_p25"],
                stats[f"{key}_rot_p75"],
                color=color,
                label=name,
            )
        )

    _save_single_plot(ax_pos, pos_lines, "Simulation step [-]", "Translation error [%]", out_pos)
    _save_single_plot(ax_rot, rot_lines, "Simulation step [-]", "Rotation error [%]", out_rot)


def print_summary(stats: dict) -> None:
    present_keys = stats["present_keys"]
    T = len(stats["steps"])
    checkpoints = sorted({max(0, int(T * f) - 1) for f in [0.1, 0.25, 0.5, 0.75, 1.0]})

    for key in present_keys:
        name = GNN_NAMES.get(key, key)
        print(f"\n  [{name}]")
        print(
            f"  {'Step':>5}  {'Pos med(%)':>12}  "
            f"{'Pos IQR':>18}  {'Rot med(°)':>12}  {'Rot IQR':>18}"
        )
        print("  " + "-" * 72)
        for s in checkpoints:
            pos_iqr = f"[{stats[f'{key}_pos_p25'][s]:.2f}, {stats[f'{key}_pos_p75'][s]:.2f}]"
            rot_iqr = f"[{stats[f'{key}_rot_p25'][s]:.2f}, {stats[f'{key}_rot_p75'][s]:.2f}]"
            print(
                f"  {s:>5}  "
                f"{stats[f'{key}_pos_median'][s]:>12.4f}  {pos_iqr:>18}  "
                f"{stats[f'{key}_rot_median'][s]:>12.4f}  {rot_iqr:>18}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(description="GNN rollout error analysis")
    parser.add_argument("--scenario", choices=["fall", "pendulum", "both"], default="both")
    args = parser.parse_args()

    scenarios = (
        list(SCENARIO_FILES.items())
        if args.scenario == "both"
        else [(args.scenario, SCENARIO_FILES[args.scenario])]
    )

    for name, hdf5_path in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {name.upper()}")

        if not hdf5_path.exists():
            print(f"  HDF5 not found: {hdf5_path}")
            print(f"  Run gnn_benchmark_{name}.py first.")
            continue

        print(f"  Computing stats from {hdf5_path}…")
        stats = compute_stats(hdf5_path)
        print_summary(stats)
        plot_scenario(stats, name, PLOT_FILES_POS[name], PLOT_FILES_ROT[name])


if __name__ == "__main__":
    main()
