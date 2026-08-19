"""Error-phase decomposition for the best Stribeck config (results/ident_stribeck2.json).

Splits each run's scored samples into three phases by the robot's real
(aligned) x position relative to the pallet's near/far faces along the path:
  pre-pallet:  x < near_face - 0.3
  on-pallet:   near_face - 0.3 <= x <= far_face + 0.3
  post-pallet: x > far_face + 0.3
Near/far face x are the box's x-extent under its yaw (bounding-box projection
of the rotated rectangle onto the path axis), from box center/half_extents/yaw.

Reports, per phase, averaged over the 14 runs: position RMSE (3D), z RMSE,
yaw RMSE — showing WHERE the remaining sim-to-real error lives.

Replay itself uses the same subprocess-isolation worker pattern as
ident_stribeck2.py (_phase_worker.py) to avoid GPU memory accumulation across
the 14 runs; the phase split and RMSE aggregation are pure numpy, done here
in-process from the dumped trajectory arrays.

    .venv/bin/python experiments/1_sim_to_real_box/error_phases.py
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np

import eval_campaign2 as ec
from common_box import DATA_DIR, RESULTS_DIR, load_gt

HERE = pathlib.Path(__file__).parent
WORKER = HERE / "_phase_worker.py"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-kuceral4-projects-ostrich/"
    "6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

PALLET_MARGIN = 0.3  # m


def face_x(gt):
    cx = gt["box"]["center"][0]
    hx, hy = gt["box"]["half_extents"][0], gt["box"]["half_extents"][1]
    yaw = gt["box"].get("yaw", 0.0)
    extent = hx * abs(np.cos(yaw)) + hy * abs(np.sin(yaw))
    return cx - extent, cx + extent  # near, far


def run_worker(params, runs, cmd_scale, timeout):
    job_path = SCRATCH / "phase_job.json"
    out_path = SCRATCH / "phase_out.json"
    if out_path.exists():
        out_path.unlink()
    job_path.write_text(json.dumps({"params": params, "runs": runs,
                                    "cmd_scale": cmd_scale}))
    proc = subprocess.run([sys.executable, str(WORKER), str(job_path),
                          str(out_path)], cwd=str(HERE),
                          capture_output=True, text=True, timeout=timeout)
    if not out_path.exists():
        print(f"  ! worker crashed (rc={proc.returncode}): "
              f"{proc.stderr[-2000:]}")
        return {}
    return json.loads(out_path.read_text())


def phase_stats(gt, traj):
    """Recompute per-sample errors (same mask logic as common_box.score) and
    split into pre/on/post-pallet phases by real aligned x."""
    sim_rel = np.asarray(traj["sim_rel"])
    sta = np.asarray(traj["sim_t_aligned"])
    rtt = np.asarray(traj["real_t_used"])
    sim_yaw = np.asarray(traj["sim_yaw_rel_on_real_t"])

    rt = gt["real"]["t"]
    rx, ry, rz = gt["real"]["x"], gt["real"]["y"], gt["real"]["z"]
    ryaw = gt["real"]["yaw_rel"]

    m = (rt >= 0) & (rt <= min(rt.max(), sta.max()))
    assert len(rt[m]) == len(rtt), "mask recomputation mismatch"

    rx_m, ry_m, rz_m, ryaw_m = rx[m], ry[m], rz[m], ryaw[m]
    sx = np.interp(rtt, sta, sim_rel[:, 0])
    sy = np.interp(rtt, sta, sim_rel[:, 1])
    sz = np.interp(rtt, sta, sim_rel[:, 2])
    dx, dy, dz = sx - rx_m, sy - ry_m, sz - rz_m
    dyaw = sim_yaw - ryaw_m

    near_x, far_x = face_x(gt)
    phases = {
        "pre_pallet": rx_m < (near_x - PALLET_MARGIN),
        "on_pallet": (rx_m >= (near_x - PALLET_MARGIN)) & (rx_m <= (far_x + PALLET_MARGIN)),
        "post_pallet": rx_m > (far_x + PALLET_MARGIN),
    }
    out = {}
    for name, mask in phases.items():
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0, "pos_rmse": None, "z_rmse": None, "yaw_rmse_deg": None}
            continue
        pos_rmse = float(np.sqrt(np.mean(dx[mask]**2 + dy[mask]**2 + dz[mask]**2)))
        z_rmse = float(np.sqrt(np.mean(dz[mask]**2)))
        yaw_rmse_deg = float(np.degrees(np.sqrt(np.mean(dyaw[mask]**2))))
        out[name] = {"n": n, "pos_rmse": pos_rmse, "z_rmse": z_rmse,
                    "yaw_rmse_deg": yaw_rmse_deg}
    out["near_x"], out["far_x"] = float(near_x), float(far_x)
    return out


def main():
    best_path = RESULTS_DIR / "ident_stribeck2.json"
    d = json.load(open(best_path))
    best = d["best"]
    cmd_scale = d["cmd_scale_recalibration"]["cmd_scale_recalibrated"]
    params = dict(k_p=4000.0, mu_front=best["mu_lat"], mu_rear=best["mu_lat"],
                 mu_long_front=0.8, mu_long_rear=1.2,
                 mu_stiction_scale=best["mu_stiction_scale"],
                 v_stribeck=best["v_stribeck"])
    print(f"best config: {params}, cmd_scale={cmd_scale:.4f}")

    runs = ec.RUNS
    traj = run_worker(params, runs, cmd_scale, timeout=700)

    per_run = {}
    for n in runs:
        if n not in traj or "error" in traj[n]:
            print(f"  ! {n}: worker error {traj.get(n, {}).get('error')}")
            continue
        gt = load_gt(DATA_DIR / f"{n}.json")
        per_run[n] = phase_stats(gt, traj[n])
        p = per_run[n]
        print(f"{n}: pre={p['pre_pallet']['pos_rmse']} on={p['on_pallet']['pos_rmse']} "
              f"post={p['post_pallet']['pos_rmse']}")

    phase_names = ["pre_pallet", "on_pallet", "post_pallet"]
    avg = {}
    for ph in phase_names:
        pos = [per_run[n][ph]["pos_rmse"] for n in per_run if per_run[n][ph]["pos_rmse"] is not None]
        z = [per_run[n][ph]["z_rmse"] for n in per_run if per_run[n][ph]["z_rmse"] is not None]
        yaw = [per_run[n][ph]["yaw_rmse_deg"] for n in per_run if per_run[n][ph]["yaw_rmse_deg"] is not None]
        n_runs_with_phase = len(pos)
        avg[ph] = {
            "pos_rmse_mean": float(np.mean(pos)) if pos else None,
            "z_rmse_mean": float(np.mean(z)) if z else None,
            "yaw_rmse_deg_mean": float(np.mean(yaw)) if yaw else None,
            "n_runs": n_runs_with_phase,
        }

    out = {"best_params": params, "cmd_scale": cmd_scale,
           "pallet_margin_m": PALLET_MARGIN, "per_run": per_run, "average": avg}
    path = RESULTS_DIR / "error_phases.json"
    json.dump(out, open(path, "w"), indent=1)

    print("\nphase          pos_rmse(m)  z_rmse(m)  yaw_rmse(deg)  n_runs")
    for ph in phase_names:
        a = avg[ph]
        print(f"{ph:14s} {a['pos_rmse_mean']:.3f}        {a['z_rmse_mean']:.3f}      "
              f"{a['yaw_rmse_deg_mean']:.2f}          {a['n_runs']}")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
