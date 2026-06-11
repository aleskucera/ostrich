"""Score the Brax forward poses (from forward_brax.py) with the exact §4.1
metric (common_box.score), so the combined position+yaw error is directly
comparable to Ostrich 0.062 / MuJoCo 0.054 / Semi-Implicit 0.110 m.

Run in the main axion venv:
    .venv/bin/python experiments/1_sim_to_real_box/score_brax.py
"""
import pathlib
import numpy as np
from collections import defaultdict

from common_box import load_gt, score

HERE = pathlib.Path(__file__).resolve().parent
NPZ = HERE / "results" / "forward_brax_poses.npz"


def main():
    data = np.load(NPZ)
    gts = {}
    # full score dict per (pipeline,wheel,kv,mu,dt) -> {run: score}
    per_cfg = defaultdict(dict)
    for key in data.files:
        run, pipe, wheel, kv, mu, dts = key.split("|")
        dt = float(dts[2:])  # "dt0.001" -> 0.001
        if run not in gts:
            gts[run] = load_gt(str(HERE / "data" / f"{run}.json"))
        pose = data[key]
        # A run only "tracks" if its pose stays physically bounded over the FULL
        # rollout. score() time-aligns to the overlap window and can miss a late
        # divergence (e.g. spring/capsule dt=5e-4 stays sane early, then blows up
        # to 1e7 after the window, scoring a deceptive 0.21). Guard against that.
        if not np.all(np.isfinite(pose)) or np.max(np.abs(pose[:, :3])) > 10.0:
            per_cfg[(pipe, wheel, kv, mu, dts)][run] = {"combined_with_yaw": np.inf,
                                                         "yaw_rmse_deg": np.nan}
            continue
        try:
            s = score(pose, dt, gts[run])
        except Exception as e:
            print(f"{key}: SCORE FAIL {str(e)[:80]}")
            continue
        per_cfg[(pipe, wheel, kv, mu, dts)][run] = s

    print(f"{'pipeline/wheel':24s} {'kv':>5s} {'mu':>4s} {'dt':>8s}  err(m)  [per-run]")
    rows = []
    for cfg, runs in per_cfg.items():
        errs = [v["combined_with_yaw"] for v in runs.values()]
        rows.append((float(np.mean(errs)), *cfg, runs))
    for mean, pipe, wheel, kv, mu, dts, runs in sorted(rows, key=lambda r: r[0]):
        es = " ".join(f"{v['combined_with_yaw']:.3f}" for v in runs.values())
        print(f"{pipe+'/'+wheel:24s} {kv:>5s} {mu:>4s} {dts:>8s}  {mean:.3f}   [{es}]")
    if rows:
        best = min(rows, key=lambda r: r[0])
        mean, pipe, wheel, kv, mu, dts, runs = best
        print(f"\nBEST Brax forward combined error: {mean:.3f} m "
              f"({pipe}/{wheel}, {kv}, {mu}, {dts})")
        print("compare: Ostrich 0.062 | MuJoCo 0.054 | Semi-Implicit 0.110")
        # emit sweep_brax.json in the plot_paper_panels bar-panel schema
        out = {
            "simulator": "Brax",
            "best_error": mean,
            "best_params": {"dt": float(dts[2:]), "pipeline": pipe, "wheel": wheel,
                            "kv": float(kv[2:]), "mu": float(mu[2:])},
            "best_per_run": {r: {"combined_with_yaw": v["combined_with_yaw"],
                                 "yaw_rmse_deg": v.get("yaw_rmse_deg", float("nan")),
                                 # trajectory for the xy/z panels (downsampled)
                                 "sim_rel": np.asarray(v["sim_rel"])[::5].tolist(),
                                 "sim_t_aligned": np.asarray(v["sim_t_aligned"])[::5].tolist()}
                             for r, v in runs.items()},
            "note": "best over dt in {1e-3,5e-4,2e-4} x kv in {50,150} x mu in {0.5,1.0}, "
                    "positional+sphere / spring+capsule",
        }
        import json as _json
        (HERE / "results" / "sweep_brax.json").write_text(_json.dumps(out, indent=2))
        print(f"wrote {HERE / 'results' / 'sweep_brax.json'}")


if __name__ == "__main__":
    main()
