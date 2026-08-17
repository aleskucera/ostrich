"""Stage-2 refinement on the round-1 Stribeck winner: surface-split friction
(box_mu: wood pallet vs grass ground) x speed-dependent LLC tracking model.

Base config (validated round-1 best, both-axes Stribeck): k_p=4000,
mu_lat=0.4, mu_stiction_scale=4.0, v_stribeck=0.3, mu_long (0.8, 1.2),
cmd_scale 0.937. Lateral-only variant is EXCLUDED: the A/B showed a 3.7x
degradation at identical parameters (velocity-dependent ellipse aspect
interacts with the FB transition region; needs a proper derivation first).

Grid: box_mu x (LLC deficit, omega0). LLC transforms the recorded commands
BEFORE cmd_scale (worker applies it to gt control): the real motor
regulators track poorly at low wheel speeds. Train ostrich0/1/10/13; the
best combo is evaluated on the 10 held-out runs + all 14, then a per-phase
error decomposition is written.

    .venv/bin/python experiments/1_sim_to_real_box/ident_stage2.py
"""
import itertools
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.resolve().parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np

RESULTS_DIR = HERE / "results"
SCRATCH = pathlib.Path("/tmp/claude-1000/-home-kuceral4-projects-ostrich/"
                       "6854194e-bd46-4738-b518-2ec169ef8b40/scratchpad")
WORKER = HERE / "_stribeck_worker.py"

TRAIN = ["ostrich0", "ostrich1", "ostrich10", "ostrich13"]
ALL = [f"ostrich{i}" for i in range(14)]
TEST = [r for r in ALL if r not in TRAIN]
CMD_SCALE = 0.937

BASE = dict(k_p=4000.0, mu_front=0.4, mu_rear=0.4,
            mu_long_front=0.8, mu_long_rear=1.2,
            mu_stiction_scale=4.0, v_stribeck=0.3)

GRID_BOX_MU = (0.4, 0.6, 0.8)
GRID_LLC = (None, (0.2, 1.0), (0.35, 1.0), (0.35, 2.0))


def run_worker(params, runs, llc):
    job_path = SCRATCH / "stage2_job.json"
    out_path = SCRATCH / "stage2_out.json"
    if out_path.exists():
        out_path.unlink()
    job = {"params": params, "runs": runs, "cmd_scale": CMD_SCALE}
    if llc is not None:
        job["llc"] = {"deficit": llc[0], "omega0": llc[1]}
    job_path.write_text(json.dumps(job))
    subprocess.run([sys.executable, str(WORKER), str(job_path),
                    str(out_path)], cwd=str(HERE), capture_output=True,
                   text=True, timeout=3600)
    return json.loads(out_path.read_text())


def mean_of(res, key):
    return float(np.mean([res[r][key] for r in res]))


def main():
    rows = []
    for box_mu, llc in itertools.product(GRID_BOX_MU, GRID_LLC):
        params = {**BASE, "box_mu": box_mu}
        res = run_worker(params, TRAIN, llc)
        mc = mean_of(res, "combined_with_yaw")
        my = mean_of(res, "yaw_rmse_deg")
        rows.append({"box_mu": box_mu, "llc": llc, "train_combined": mc,
                     "train_yaw_deg": my})
        print(f"box_mu={box_mu} llc={llc}: train={mc:.3f} yaw={my:.2f}",
              flush=True)

    best = min(rows, key=lambda r: r["train_combined"])
    params = {**BASE, "box_mu": best["box_mu"]}
    te = run_worker(params, TEST, best["llc"])
    al = run_worker(params, ALL, best["llc"])
    out = {
        "base": BASE, "cmd_scale": CMD_SCALE, "grid": rows, "best": best,
        "test_mean": mean_of(te, "combined_with_yaw"),
        "test_yaw_deg": mean_of(te, "yaw_rmse_deg"),
        "all14_mean": mean_of(al, "combined_with_yaw"),
        "all14_yaw_deg": mean_of(al, "yaw_rmse_deg"),
        "all14_per_run": {r: al[r] for r in al},
        "round1_reference": {"test_mean": 0.197, "all14_mean": 0.189,
                             "all14_yaw_deg": 5.65},
    }
    path = RESULTS_DIR / "ident_stage2.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nbest {best} -> TEST {out['test_mean']:.3f} "
          f"all14 {out['all14_mean']:.3f} yaw {out['all14_yaw_deg']:.2f} "
          f"-> {path}")


if __name__ == "__main__":
    main()
