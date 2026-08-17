"""Boundary-extended Stribeck identification (round 2).

The first round (ident_stribeck.py -> results/ident_stribeck.json) found its
optimum at a corner of the searched grid: k_p=4000, mu_lat=0.4,
mu_stiction_scale=4.0, v_stribeck=0.3 (train comb=0.173, TEST 0.197,
all-14 0.189, yaw 5.65 deg). Since the winner sits on the boundary of
mu_lat, mu_stiction_scale and v_stribeck, this script extends the grid past
that corner (lower mu_lat, higher stiction scale, wider v_stribeck range)
with k_p fixed at 4000 to see whether the plateau continues to improve.

For the best config found, cmd_scale (motor command scale, previously
0.937) is also recalibrated on the pre-box flat cruise: eval_campaign2's
prebox_speed() is evaluated on the 4 TRAIN runs at the current cmd_scale,
and the median real/sim speed ratio rescales cmd_scale before the final
TEST / all-14 evaluation.

Same subprocess-isolation pattern as ident_stribeck.py: a single long-lived
process leaks ~35-40 MB of GPU memory per replay run, which exhausts a 4 GB
laptop GPU after ~90-100 runs. Each config (or evaluation set) is scored in
its own subprocess via _stribeck2_worker.py.

    .venv/bin/python experiments/1_sim_to_real_box/ident_stribeck2.py
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
from common_box import DATA_DIR, RESULTS_DIR, load_gt

HERE = pathlib.Path(__file__).parent
WORKER = HERE / "_stribeck2_worker.py"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-kuceral4-projects-ostrich/"
    "6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

TRAIN = ["ostrich0", "ostrich1", "ostrich10", "ostrich13"]
TEST = [f"ostrich{i}" for i in range(14) if f"ostrich{i}" not in TRAIN]
ALL14 = ec.RUNS
CMD_SCALE0 = 0.937
MU_LONG = (0.8, 1.2)       # (front, rear) — pinned
K_P = 4000.0               # pinned at round-1's winning value

# Focused grid around the round-1 corner (mu_lat=0.4, mu_stiction_scale=4.0,
# v_stribeck=0.3), extended past the boundary in every direction.
GRID_LAT = (0.3, 0.4, 0.6)
GRID_STICTION = (4.0, 6.0, 9.0)
GRID_VSTRIBECK = (0.2, 0.3, 0.5)


def build_params(mu_lat, mu_stiction_scale, v_stribeck):
    return dict(k_p=K_P, mu_front=mu_lat, mu_rear=mu_lat,
               mu_long_front=MU_LONG[0], mu_long_rear=MU_LONG[1],
               mu_stiction_scale=mu_stiction_scale, v_stribeck=v_stribeck,
               stribeck_lateral_only=1.0)


def run_worker(params, runs, cmd_scale, timeout):
    job_path = SCRATCH / "stribeck2_job.json"
    out_path = SCRATCH / "stribeck2_out.json"
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
        return {n: {"combined_with_yaw": None, "yaw_rmse_deg": None,
                    "sim_prebox_speed": None} for n in runs}
    return json.loads(out_path.read_text())


def score_config(params, runs, cmd_scale, timeout):
    scores = run_worker(params, runs, cmd_scale, timeout)
    combined = [s["combined_with_yaw"] for s in scores.values()
                if s["combined_with_yaw"] is not None]
    yaw = [s["yaw_rmse_deg"] for s in scores.values()
           if s["yaw_rmse_deg"] is not None]
    mc = float(np.mean(combined)) if combined else float("nan")
    my = float(np.mean(yaw)) if yaw else float("nan")
    prebox = {n: s["sim_prebox_speed"] for n, s in scores.items()}
    return scores, mc, my, prebox


def main():
    rows = []
    for lat, sc, vs in itertools.product(GRID_LAT, GRID_STICTION, GRID_VSTRIBECK):
        params = build_params(lat, sc, vs)
        scores, mc, my, prebox = score_config(params, TRAIN, CMD_SCALE0, timeout=300)
        rows.append({"mu_lat": lat, "mu_stiction_scale": sc, "v_stribeck": vs,
                     "train_combined": mc, "train_yaw_deg": my,
                     "train_prebox": prebox})
        print(f"mu_lat={lat:.1f} scale={sc:.1f} v_s={vs:.1f}: "
              f"train comb={mc:.3f} yaw={my:.2f} deg", flush=True)

    finite_rows = [r for r in rows if np.isfinite(r["train_combined"])]
    best = min(finite_rows, key=lambda r: r["train_combined"])
    best_params = build_params(best["mu_lat"], best["mu_stiction_scale"],
                               best["v_stribeck"])
    print(f"\ngrid best: {best}")

    # --- cmd_scale recalibration on the pre-box cruise (TRAIN runs only) ---
    train_gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in TRAIN}
    real_v = {n: ec.prebox_speed(gt["real"]["t"], gt["real"]["x"],
                                 gt["real"]["y"], gt)
              for n, gt in train_gts.items()}
    ratios = [real_v[n] / best["train_prebox"][n] for n in TRAIN
              if np.isfinite(real_v[n]) and best["train_prebox"][n]
              and np.isfinite(best["train_prebox"][n])
              and best["train_prebox"][n] > 1e-3]
    ratio_median = float(np.median(ratios)) if ratios else 1.0
    cmd_scale_new = CMD_SCALE0 * ratio_median
    recal = {"cmd_scale0": CMD_SCALE0, "real_v": real_v,
             "sim_v_at_cmd_scale0": best["train_prebox"],
             "ratios": ratios, "ratio_median": ratio_median,
             "cmd_scale_recalibrated": cmd_scale_new}
    print(f"\ncmd_scale recalibration: real/sim ratios={ratios} "
          f"median={ratio_median:.4f} -> cmd_scale {CMD_SCALE0:.4f} -> "
          f"{cmd_scale_new:.4f}")

    # --- re-evaluate best config at the recalibrated cmd_scale ---
    tr, tr_mc, tr_my, _ = score_config(best_params, TRAIN, cmd_scale_new, timeout=300)
    te, te_mc, te_my, _ = score_config(best_params, TEST, cmd_scale_new, timeout=600)
    al, al_mc, al_my, _ = score_config(best_params, ALL14, cmd_scale_new, timeout=700)

    out = {
        "protocol": f"train {TRAIN}, test {TEST}; contact params frozen at "
                    "campaign-1; longitudinal mu pinned (0.8, 1.2); "
                    f"k_p pinned at {K_P}; cmd_scale0={CMD_SCALE0}, "
                    f"recalibrated to {cmd_scale_new:.4f}",
        "grid": rows,
        "best": best,
        "cmd_scale_recalibration": recal,
        "train_recalibrated": tr,
        "train_mean_recalibrated": tr_mc,
        "train_yaw_deg_recalibrated": tr_my,
        "test": te,
        "test_mean": te_mc,
        "test_yaw_deg": te_my,
        "all14": al,
        "all14_mean": al_mc,
        "all14_yaw_deg": al_my,
        "previous_best_reference": {
            "note": "round-1 corner winner at cmd_scale=0.937 "
                    "(results/ident_stribeck.json)",
            "test_mean": 0.19746779687395255, "test_yaw_deg": 5.206359930554025,
            "all14_mean": 0.18926626087113443, "all14_yaw_deg": 5.650588480465967,
        },
    }
    path = RESULTS_DIR / "ident_stribeck3.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nbest {best}")
    print(f"recalibrated cmd_scale={cmd_scale_new:.4f}: "
          f"TRAIN mean {tr_mc:.3f} ({tr_my:.2f} deg)")
    print(f"TEST mean {te_mc:.3f} ({te_my:.2f} deg), "
          f"all-14 {al_mc:.3f} ({al_my:.2f} deg) -> {path}")


if __name__ == "__main__":
    main()
