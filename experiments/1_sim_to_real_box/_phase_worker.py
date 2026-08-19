"""Worker: replay a set of GT runs at a fixed param config and dump the
per-sample trajectory arrays needed for post-hoc error-phase decomposition
(same subprocess-isolation pattern as _stribeck_worker.py / _stribeck2_worker.py).

Usage: python _phase_worker.py <job.json> <out.json>
  job.json: {"params": {...PARAMS...}, "runs": [...], "cmd_scale": float}
  out.json: {run_name: {"sim_rel": [[x,y,z],...], "sim_t_aligned": [...],
                        "real_t_used": [...],
                        "sim_yaw_rel_on_real_t": [...],
                        "combined_with_yaw": float, "yaw_rmse_deg": float}}
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import eval_campaign2 as ec
from common_box import DATA_DIR, load_gt
import examples.helhest_junior.replay_real as rr


def main():
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out_path = sys.argv[2]
    params, runs, cmd_scale = job["params"], job["runs"], job["cmd_scale"]

    _orig_init = rr.HelhestJuniorReplaySimulator.__init__

    def _patched_init(self, *a, **kw):
        kw.update(params)
        _orig_init(self, *a, **kw)

    rr.HelhestJuniorReplaySimulator.__init__ = _patched_init

    gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in runs}
    out = {}
    for n in runs:
        try:
            gt = gts[n]
            pose, dt = ec.run_ostrich(gt, cmd_scale)
            s = ec._score_run(pose, dt, gt)
            out[n] = {
                "sim_rel": s["sim_rel"].tolist(),
                "sim_t_aligned": s["sim_t_aligned"].tolist(),
                "real_t_used": s["real_t_used"].tolist(),
                "sim_yaw_rel_on_real_t": s["sim_yaw_rel_on_real_t"].tolist(),
                "combined_with_yaw": float(s["combined_with_yaw"]),
                "yaw_rmse_deg": float(s["yaw_rmse_deg"]),
            }
        except Exception as e:
            out[n] = {"error": str(e)}
    pathlib.Path(out_path).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
