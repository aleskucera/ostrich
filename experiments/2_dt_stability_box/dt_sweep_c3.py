"""Timestep-range sweep (Fig. 3) re-run on the campaign-2/3 dataset.

Forward-only accuracy vs. h for the three engines at their corrected-metric
identified configurations, on four held-out runs spanning the speed range
(ostrich3, 5, 9, 12; the latter two are runs where the Semi-Implicit
baseline is stable at its native h, so its curve is informative).
Divergence/NaN is recorded as such (plotted at ceiling).

    .venv/bin/python experiments/2_dt_stability_box/dt_sweep_c3.py --engine ostrich
    .venv/bin/python experiments/2_dt_stability_box/dt_sweep_c3.py --engine mujoco
    .venv/bin/python experiments/2_dt_stability_box/dt_sweep_c3.py --engine semi_implicit
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "1_sim_to_real_box"))

import numpy as np

RUNS = ["ostrich3", "ostrich5", "ostrich9", "ostrich12"]
GRIDS = {
    "ostrich": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
    "mujoco": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
    "semi_implicit": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
}
CMD = {"ostrich": 0.937, "mujoco": 0.9448, "semi_implicit": 0.8963}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=list(GRIDS))
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    import eval_campaign2 as ec
    from common_box import DATA_DIR, load_gt
    import examples.helhest_junior.replay_real as rr
    from plot_best14 import _run_mj_c3  # MuJoCo c3-identified runner

    # corrected-metric identified configs
    OST = dict(k_p=10000.0, mu_front=0.6, mu_rear=0.6,
               mu_long_front=0.8, mu_long_rear=1.2)
    _orig = rr.HelhestJuniorReplaySimulator.__init__

    def patched(self, *a, **kw):
        if args.engine == "ostrich":
            kw.update(OST)
        _orig(self, *a, **kw)

    rr.HelhestJuniorReplaySimulator.__init__ = patched

    gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in RUNS}
    rows = []
    for h in GRIDS[args.engine]:
        vals = {}
        for run, gt in gts.items():
            try:
                if args.engine == "ostrich":
                    ec.C1_OSTRICH["dt"] = h
                    s = ec._score_run(*ec.run_ostrich(gt, CMD["ostrich"]), gt)
                elif args.engine == "mujoco":
                    ec.C1_MUJOCO["dt"] = h
                    s = ec._score_run(*_run_mj_c3(gt), gt)
                else:
                    ec.C1_SI["dt"] = h
                    s = ec._score_run(
                        *ec.run_semi_implicit(gt, CMD["semi_implicit"]), gt)
                v = float(s["combined_with_yaw"])
                vals[run] = v if np.isfinite(v) else None
            except Exception as e:
                vals[run] = None
                print(f"    {run} @ h={h}: FAILED ({type(e).__name__})",
                      flush=True)
        finite = [v for v in vals.values() if v is not None]
        mean = float(np.mean(finite)) if len(finite) == len(RUNS) else None
        rows.append({"h": h, "per_run": vals, "mean_combined": mean,
                     "n_diverged": sum(v is None for v in vals.values())})
        print(f"h={h}: mean={mean if mean is None else round(mean, 3)} "
              f"diverged={rows[-1]['n_diverged']}/4", flush=True)

    out = args.save or (HERE / "results" / f"dt_sweep_c3_{args.engine}.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"engine": args.engine, "runs": RUNS,
               "cmd_scale": CMD[args.engine], "rows": rows},
              open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
