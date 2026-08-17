"""Stribeck (velocity-dependent) lateral-friction identification for Ostrich.

Modeled on ident_motor_turn.py. That identification found a hard plateau at
constant-mu Coulomb lateral friction (train 0.164, TEST 0.136, all-4 0.152,
yaw 2.41 deg on the 4-run split) because the real per-run turn efficiency
varies with speed while constant mu realizes exactly one efficiency. This
script tests whether Stribeck-style velocity-dependent mu (mu *=
1 + (mu_stiction_scale - 1) * exp(-slip / v_stribeck), i.e. higher effective
mu at low slip speed / stiction, decaying to base mu at high slip) can beat
that plateau on the larger 14-run set (TRAIN=4, TEST=10).

Each config is scored in an isolated subprocess (_stribeck_worker.py):
a single long-lived process leaks ~35-40 MB of GPU memory per replay run,
which exhausts a 4 GB laptop GPU and poisons the CUDA context after ~90-100
runs (observed first-hand — the in-process version died partway through the
grid with "illegal memory access"/allocation failures). Isolating each
config (4 TRAIN runs) in its own process keeps every process well under
that budget.

    .venv/bin/python experiments/1_sim_to_real_box/ident_stribeck.py
"""
import itertools
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np

import eval_campaign2 as ec
from common_box import RESULTS_DIR

HERE = pathlib.Path(__file__).parent
WORKER = HERE / "_stribeck_worker.py"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-kuceral4-projects-ostrich/"
    "6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

TRAIN = ["ostrich0", "ostrich1", "ostrich10", "ostrich13"]
TEST = [f"ostrich{i}" for i in range(14) if f"ostrich{i}" not in TRAIN]
CMD_SCALE = 0.937
MU_LONG = (0.8, 1.2)       # (front, rear) — pinned

GRID_KP = (4000.0, 10000.0)
GRID_LAT = (0.4, 0.8)
GRID_STICTION = (1.5, 2.5, 4.0)
GRID_VSTRIBECK = (0.1, 0.3)

REFERENCE = {"k_p": 10000.0, "mu_lat": 0.8}  # constant-mu winner, no stribeck


def build_params(k_p, mu_lat, mu_stiction_scale=None, v_stribeck=None):
    p = dict(k_p=k_p, mu_front=mu_lat, mu_rear=mu_lat,
             mu_long_front=MU_LONG[0], mu_long_rear=MU_LONG[1])
    if mu_stiction_scale is not None:
        p["mu_stiction_scale"] = mu_stiction_scale
    if v_stribeck is not None:
        p["v_stribeck"] = v_stribeck
    return p


def run_worker(params, runs, timeout):
    job_path = SCRATCH / "stribeck_job.json"
    out_path = SCRATCH / "stribeck_out.json"
    if out_path.exists():
        out_path.unlink()
    job_path.write_text(json.dumps({"params": params, "runs": runs,
                                    "cmd_scale": CMD_SCALE}))
    proc = subprocess.run([sys.executable, str(WORKER), str(job_path),
                          str(out_path)], cwd=str(HERE),
                          capture_output=True, text=True, timeout=timeout)
    if not out_path.exists():
        print(f"  ! worker crashed (rc={proc.returncode}): "
              f"{proc.stderr[-2000:]}")
        return {n: {"combined_with_yaw": None, "yaw_rmse_deg": None}
                for n in runs}
    return json.loads(out_path.read_text())


def score_config(params, runs, timeout):
    scores = run_worker(params, runs, timeout)
    combined = [s["combined_with_yaw"] for s in scores.values()
                if s["combined_with_yaw"] is not None]
    yaw = [s["yaw_rmse_deg"] for s in scores.values()
           if s["yaw_rmse_deg"] is not None]
    mc = float(np.mean(combined)) if combined else float("nan")
    my = float(np.mean(yaw)) if yaw else float("nan")
    return scores, mc, my


def main():
    rows = []

    # Reference: constant-mu winner from the previous identification.
    ref_params = build_params(REFERENCE["k_p"], REFERENCE["mu_lat"])
    _, mc, my = score_config(ref_params, TRAIN, timeout=300)
    ref_row = {"label": "reference", "k_p": REFERENCE["k_p"],
               "mu_lat": REFERENCE["mu_lat"], "mu_stiction_scale": None,
               "v_stribeck": None, "train_combined": mc, "train_yaw_deg": my}
    print(f"[reference] k_p={REFERENCE['k_p']:6.0f} mu_lat={REFERENCE['mu_lat']:.1f} "
          f"no-stribeck: train comb={mc:.3f} yaw={my:.2f} deg", flush=True)

    for kp, lat, sc, vs in itertools.product(GRID_KP, GRID_LAT, GRID_STICTION,
                                              GRID_VSTRIBECK):
        params = build_params(kp, lat, mu_stiction_scale=sc, v_stribeck=vs)
        _, mc, my = score_config(params, TRAIN, timeout=300)
        rows.append({"k_p": kp, "mu_lat": lat, "mu_stiction_scale": sc,
                     "v_stribeck": vs, "train_combined": mc, "train_yaw_deg": my})
        print(f"k_p={kp:6.0f} mu_lat={lat:.1f} scale={sc:.1f} v_s={vs:.1f}: "
              f"train comb={mc:.3f} yaw={my:.2f} deg", flush=True)

    finite_rows = [r for r in rows if np.isfinite(r["train_combined"])]
    best = min(finite_rows, key=lambda r: r["train_combined"])
    best_params = build_params(best["k_p"], best["mu_lat"],
                               mu_stiction_scale=best["mu_stiction_scale"],
                               v_stribeck=best["v_stribeck"])

    te, te_mc, te_my = score_config(best_params, TEST, timeout=600)
    al, al_mc, al_my = score_config(best_params, ec.RUNS, timeout=700)

    out = {
        "protocol": f"train {TRAIN}, test {TEST}; contact params frozen at "
                    "campaign-1; longitudinal mu pinned (0.8, 1.2); "
                    f"cmd_scale={CMD_SCALE}",
        "reference": ref_row,
        "grid": rows,
        "best": best,
        "test": te,
        "test_mean": te_mc,
        "test_yaw_deg": te_my,
        "all14_mean": al_mc,
        "all14_yaw_deg": al_my,
    }
    path = RESULTS_DIR / "ident_stribeck.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nbest {best}")
    print(f"TEST mean {te_mc:.3f} ({te_my:.2f} deg), "
          f"all-14 {al_mc:.3f} ({al_my:.2f} deg) -> {path}")


if __name__ == "__main__":
    main()
