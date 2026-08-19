"""Phase checks for an out-of-process engine runner (chrono | agx).

    .venv/bin/python -m experiments.7_engine_comparison.engines.verify_runner chrono

Runs, in one batch: (1) flat-ground smoke — all wheels 3 rad/s must give ~1.05 m/s
forward (+X) at ride height ~0.35; (2) tilt probe at 20° — with front/rear pair
friction 0.7/0.4 the robot must hold (needs mu ≈ tan20° = 0.36 on the loaded
wheels); with mu 0.05 it must slide — this validates that the engine's material
composition realizes the intended PAIR friction; (3) full replay of
fast_experiment1 at defaults, scored with the segmented window metric, with an
overlay plot saved to results/.
"""

import importlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
import common  # noqa: E402
from engines import bridge  # noqa: E402


def main(engine: str) -> None:
    run = {"chrono": bridge.run_chrono, "agx": bridge.run_agx}[engine]
    dt = 0.002 if engine == "chrono" else 0.01

    jobs = []
    smoke_cmds = np.full((int(5.0 / dt), 3), 3.0)
    jobs.append(bridge.make_job("smoke_fwd", smoke_cmds, dt))
    hold_cmds = np.zeros((int(3.0 / dt), 3))
    # AGX's default SPLIT friction creeps under sustained load (documented in the
    # AGX manual, measured here: ~0.65 m/s at 20 deg); the hold-check therefore
    # runs DIRECT friction for AGX, and the SPLIT creep is reported as a finding.
    hold_params = {"friction_solve_type": "direct"} if engine == "agx" else {}
    tilt = bridge.make_job("tilt_hold", hold_cmds, dt, params=hold_params)
    tilt["ground"]["tilt_deg"] = 20.0
    jobs.append(tilt)
    if engine == "agx":
        creep = bridge.make_job("tilt_creep_split", hold_cmds, dt)
        creep["ground"]["tilt_deg"] = 20.0
        jobs.append(creep)
    slide = bridge.make_job("tilt_slide", hold_cmds, dt,
                            params={"mu_front": 0.05, "mu_rear": 0.05})
    slide["ground"]["tilt_deg"] = 20.0
    jobs.append(slide)

    gt = common.load_gt("fast_experiment1")
    replay_cmds = common.prepare_commands(gt, dt)
    jobs.append(bridge.make_job("replay_fast1", replay_cmds, dt))

    results = {r["id"]: r for r in run(jobs)}

    # --- 1. smoke ---
    r = results["smoke_fwd"]
    pose = np.asarray(r["pose"])
    t = np.arange(pose.shape[0]) * dt
    m = t > 2.0  # steady window
    v = np.gradient(pose[:, 0], dt)[m].mean()
    zr = pose[m, 2].mean()
    ydrift = abs(pose[-1, 1] - pose[0, 1])
    ok_v = abs(v - 1.05) < 0.08
    ok_z = abs(zr - 0.35) < 0.01
    ok_y = ydrift < 0.15
    print(f"[smoke] fwd speed {v:.3f} m/s (want ~1.05) {'OK' if ok_v else 'FAIL'}; "
          f"ride height {zr:.3f} (want ~0.35) {'OK' if ok_z else 'FAIL'}; "
          f"y drift {ydrift:.3f} m {'OK' if ok_y else 'FAIL'}")

    # --- 2. tilt probe ---
    for jid, want_hold in (("tilt_hold", True), ("tilt_slide", False)):
        pose = np.asarray(results[jid]["pose"])
        slid = np.linalg.norm(pose[-1, :2] - pose[0, :2]) > 0.3
        ok = slid != want_hold
        print(f"[tilt] {jid}: displacement {np.linalg.norm(pose[-1,:2]-pose[0,:2]):.3f} m "
              f"-> {'slid' if slid else 'held'} {'OK' if ok else 'FAIL'}")
    if "tilt_creep_split" in results:
        pose = np.asarray(results["tilt_creep_split"]["pose"])
        d = np.linalg.norm(pose[-1, :2] - pose[0, :2])
        print(f"[tilt] FINDING: default SPLIT friction creeps {d:.2f} m in 3 s at 20 deg "
              f"({d/3.0:.2f} m/s) where DIRECT holds")

    # --- 3. replay + windows ---
    r = results["replay_fast1"]
    score = common.score_result(r, gt)
    print(f"[replay] stable={r['stable']} wall={r['wall_clock_s']:.1f}s "
          f"({r['n_steps']*dt/r['wall_clock_s']:.1f}x realtime)")
    print(f"[replay] windows={score['n_windows']} combined mean={score['combined_mean']:.3f} "
          f"median={score['combined_median']:.3f} m; yaw {score['yaw_rmse_deg_mean']:.1f} deg")

    # Overlay plot: real path vs re-anchored sim windows.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rp = gt["real"]["pos"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(rp[:, 0], rp[:, 1], "k-", lw=1.5, label="real (odin)")
    pose = np.asarray(r["pose"])
    st = np.arange(pose.shape[0]) * r["dt"]
    sim_yaw = np.unwrap(common.yaw_from_quat_xyzw(pose[:, 3:7]))
    ryaw = np.unwrap(common.yaw_from_quat_xyzw(gt["real"]["quat_xyzw"]))
    rt = gt["real"]["t"]
    for w in common.window_score(pose, r["dt"], gt)["windows"]:
        m = (rt >= w["t0"]) & (rt <= w["t0"] + common.WINDOW_S)
        rtt = rt[m]
        sx = np.interp(rtt, st, pose[:, 0]); sy = np.interp(rtt, st, pose[:, 1])
        syaw = np.interp(rtt, st, sim_yaw)
        d0 = ryaw[m][0] - syaw[0]
        c, s = np.cos(d0), np.sin(d0)
        px, py = sx - sx[0], sy - sy[0]
        ax.plot(rp[m][0, 0] + c * px - s * py, rp[m][0, 1] + s * px + c * py,
                lw=1, alpha=0.85)
    ax.set_title(f"{engine} replay fast_experiment1 — re-anchored {common.WINDOW_S:.0f}s windows\n"
                 f"combined mean {score['combined_mean']:.3f} m")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()
    common.RESULTS_DIR.mkdir(exist_ok=True)
    out = common.RESULTS_DIR / f"verify_{engine}_fast1.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[replay] overlay -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
