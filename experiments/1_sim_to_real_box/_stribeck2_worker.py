"""Worker: score one Stribeck param config over a set of GT runs, in an
isolated process (same subprocess-isolation pattern as _stribeck_worker.py,
extended to also report the sim pre-box cruise speed for cmd_scale
recalibration).

Usage: python _stribeck2_worker.py <job.json> <out.json>
  job.json: {"params": {...PARAMS...}, "runs": [...], "cmd_scale": float}
  out.json: {run_name: {"combined_with_yaw": float|null,
                        "yaw_rmse_deg": float|null,
                        "sim_prebox_speed": float|null}}
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np

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
    scores = {}
    for n in runs:
        try:
            gt = gts[n]
            pose, dt = ec.run_ostrich(gt, cmd_scale)
            s = ec._score_run(pose, dt, gt)
            st = np.arange(pose.shape[0]) * dt
            sim_v = ec.prebox_speed(st, pose[:, 0], pose[:, 1], gt)
            scores[n] = {"combined_with_yaw": float(s["combined_with_yaw"]),
                         "yaw_rmse_deg": float(s["yaw_rmse_deg"]),
                         "sim_prebox_speed": float(sim_v)}
        except Exception as e:
            scores[n] = {"combined_with_yaw": None, "yaw_rmse_deg": None,
                        "sim_prebox_speed": None, "error": str(e)}
    pathlib.Path(out_path).write_text(json.dumps(scores))


if __name__ == "__main__":
    main()
