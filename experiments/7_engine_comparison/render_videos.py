"""Render comparison videos: 3-engine rock field + real-vs-sim replay.

    .venv/bin/python experiments/7_engine_comparison/render_videos.py rocks
    .venv/bin/python experiments/7_engine_comparison/render_videos.py replay

rocks   Re-runs the rock_field scenario in each engine (tuned best params) with
        scene-body logging and renders a synced top-down 3-panel animation:
        chassis + wheels + the nine rocks, with trails.
replay  Animates fast_experiment1: the real robot (odin odometry, black) driving
        with its trail, plus each engine's raw open-loop replay transformed into
        the real start frame. Open-loop divergence is expected and visible —
        that is the honest picture the window metric summarizes.

Sim runs are cached in results/render_log_<engine>_rocks.json; delete to re-run.
Videos land in results/*.mp4 (20 fps, ffmpeg).
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402
from engines import bridge  # noqa: E402

ENGINES = ["ostrich", "agx", "chrono"]
ENGINE_DT = {"chrono": 0.005, "agx": 0.01, "ostrich": 0.05}
FPS = 20

# chassis geometry (top-down), from common.py: two boxes + wheels
CHASSIS_BOXES = [(-0.13, 0.0, 0.48, 0.56), (-0.61, 0.0, 0.48, 0.24)]  # cx, cy, lx, ly
WHEELS = [(0.0, 0.365), (0.0, -0.365), (-0.75, 0.0)]
WHEEL_LX, WHEEL_LY = 0.7, 0.10
ROCK = 0.32


def best_params(engine):
    p = common.RESULTS_DIR / f"sweep_{engine}.json"
    return json.load(open(p))["best"]["params"] if p.exists() else {}


def rock_scene():
    return [{"type": "dynamic_box", "half_extents": [0.16, 0.16, 0.16],
             "pos": [1.5 + i * 0.9, (j - 1) * 0.9, 0.16], "mu": 0.5,
             "density": 400.0}
            for i in range(3) for j in range(3)]


def rock_job(engine):
    params = dict(best_params(engine))
    dt = params.pop("dt", ENGINE_DT[engine])
    T = int(10.0 / dt)
    cmds = np.full((T, 3), 3.0)
    cmds[: int(0.5 / dt)] = 0.0
    job = bridge.make_job(f"rocks_render_{engine}", cmds, dt, params=params,
                          scene=rock_scene())
    job["log_scene_bodies"] = True
    return job


def run_ostrich_rocks(job):
    """Custom loop logging the full body state (chassis + rocks)."""
    import warp as wp
    from engines.ostrich_runner import _make_sim, DEFAULT_PARAMS

    params = {**DEFAULT_PARAMS, **job.get("params", {})}
    dt = job["dt"]
    sim = _make_sim(job, params)
    setpoints = np.asarray(job["control"]["lrr"], dtype=np.float32)
    sim.target_velocities.zero_()
    for _ in range(max(60, int(round(1.0 / dt)))):
        sim._single_physics_step(0)
    pose, scene_pose = [], []
    for k in range(setpoints.shape[0]):
        wp.copy(sim.target_velocities,
                wp.array(setpoints[k], dtype=wp.float32, device=sim.model.device))
        sim._single_physics_step(k)
        bq = sim.current_state.body_q.numpy()  # robot bodies 0-3, rocks 4..12
        pose.append(bq[0].tolist())
        yaws = np.arctan2(2 * (bq[4:, 6] * bq[4:, 5] + bq[4:, 3] * bq[4:, 4]),
                          1 - 2 * (bq[4:, 4] ** 2 + bq[4:, 5] ** 2))
        scene_pose.append(np.column_stack([bq[4:, :3], yaws]).tolist())
    del sim
    return {"id": job["id"], "dt": dt, "pose": pose, "scene_body_pose": scene_pose,
            "stable": True}


def get_rock_logs():
    logs = {}
    for engine in ENGINES:
        cache = common.RESULTS_DIR / f"render_log_{engine}_rocks.json"
        if cache.exists():
            logs[engine] = json.load(open(cache))
            continue
        job = rock_job(engine)
        print(f"running rock scenario in {engine} ...", flush=True)
        if engine == "chrono":
            res = bridge.run_chrono([job])[0]
        elif engine == "agx":
            res = bridge.run_agx([job])[0]
        else:
            res = run_ostrich_rocks(job)
        with open(cache, "w") as f:
            json.dump(res, f)
        logs[engine] = res
    return logs


def draw_robot(ax, x, y, yaw, color):
    tr = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    arts = []
    for wx, wy in WHEELS:
        arts.append(ax.add_patch(Rectangle(
            (wx - WHEEL_LX / 2, wy - WHEEL_LY / 2), WHEEL_LX, WHEEL_LY,
            transform=tr, fc="0.15", ec="none", zorder=6)))
    for cx, cy, lx, ly in CHASSIS_BOXES:
        arts.append(ax.add_patch(Rectangle(
            (cx - lx / 2, cy - ly / 2), lx, ly, transform=tr,
            fc=color, ec="k", lw=0.8, alpha=0.9, zorder=7)))
    return arts


def draw_rocks(ax, rocks):
    arts = []
    for x, y, z, yaw in rocks:
        tr = Affine2D().rotate(yaw).translate(x, y) + ax.transData
        arts.append(ax.add_patch(Rectangle(
            (-ROCK / 2, -ROCK / 2), ROCK, ROCK, transform=tr,
            fc="#b09060", ec="#6b5233", lw=0.8, zorder=4)))
    return arts


def render_rocks(out):
    logs = get_rock_logs()
    colors = {"ostrich": "#d62728", "agx": "#1f77b4", "chrono": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.2))
    duration = 10.0
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    with writer.saving(fig, out, dpi=100):
        for fi in range(int(duration * FPS) + 1):
            t = fi / FPS
            arts = []
            for ax, engine in zip(axes, ENGINES):
                ax.clear()
                res = logs[engine]
                pose = np.asarray(res["pose"])
                rocks = np.asarray(res["scene_body_pose"])
                i = min(int(t / res["dt"]), pose.shape[0] - 1)
                yaw = common.yaw_from_quat_xyzw(pose[i:i + 1, 3:7])[0]
                ax.plot(pose[:i + 1, 0], pose[:i + 1, 1], color=colors[engine],
                        lw=1.2, alpha=0.7, zorder=5)
                draw_rocks(ax, rocks[i])
                draw_robot(ax, pose[i, 0], pose[i, 1], yaw, colors[engine])
                ax.set_xlim(-1.5, 10.5); ax.set_ylim(-3.0, 3.0)
                ax.set_aspect("equal"); ax.grid(alpha=0.25)
                ax.set_title(f"{engine}  (t={t:4.1f} s, x={pose[i,0]:5.2f} m)")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out}")


def variant_params(engine):
    """Best-variant config per engine: anisotropic friction where available
    (sweep_aniso results), SCM terrain for chrono. Falls back to the isotropic
    sweep best. Returns (params, label, runner_kind)."""
    if engine == "chrono":
        return {"terrain": "scm", "scm_phi": 20.0}, "chrono (SCM soil)", "chrono_scm"
    p = common.RESULTS_DIR / f"sweep_aniso_{engine}.json"
    if p.exists():
        best = dict(json.load(open(p))["best"]["params"])
        return best, f"{engine} (anisotropic)", engine
    return dict(best_params(engine)), engine, engine


def render_replay(out, bag="fast_experiment1", speedup=8.0):
    gt = common.load_gt(bag)
    rt = gt["real"]["t"]; rp = gt["real"]["pos"]
    ryaw = np.unwrap(common.yaw_from_quat_xyzw(gt["real"]["quat_xyzw"]))

    sims = {}
    labels = {}
    for engine in ENGINES:
        params, label, kind = variant_params(engine)
        labels[engine] = label
        cache = common.RESULTS_DIR / f"render_log_{engine}_{bag}_variant.json"
        if cache.exists():
            res = json.load(open(cache))
        else:
            gmu = params.pop("ground_mu", 0.8)
            dt = params.pop("dt", ENGINE_DT[engine])
            if kind == "chrono_scm":
                dt = 2e-3
            job = bridge.make_job(f"replay_{engine}", common.prepare_commands(gt, dt),
                                  dt, params=params, ground_mu=gmu)
            print(f"replaying {bag} in {label} ...", flush=True)
            if kind == "chrono_scm":
                res = bridge.run_chrono_scm([job])[0]
            elif kind == "chrono":
                res = bridge.run_chrono([job])[0]
            elif kind == "agx":
                res = bridge.run_agx([job])[0]
            else:
                from engines.ostrich_runner import run_ostrich
                res = run_ostrich([job])[0]
            res.pop("scene_body_pose", None)
            with open(cache, "w") as f:
                json.dump(res, f)
        pose = np.asarray(res["pose"])
        # Transform the sim track (starts at origin, heading +X) into the real
        # start frame: rotate by the real initial yaw, translate to real start.
        syaw = np.unwrap(common.yaw_from_quat_xyzw(pose[:, 3:7]))
        d0 = ryaw[0] - syaw[0]
        c, s = np.cos(d0), np.sin(d0)
        px, py = pose[:, 0] - pose[0, 0], pose[:, 1] - pose[0, 1]
        sims[engine] = {
            "t": np.arange(pose.shape[0]) * res["dt"],
            "x": rp[0, 0] + c * px - s * py,
            "y": rp[0, 1] + s * px + c * py,
            "yaw": syaw - syaw[0] + ryaw[0],
        }

    colors = {"ostrich": "#d62728", "agx": "#1f77b4", "chrono": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 9))
    duration = float(min(rt[-1], max(s["t"][-1] for s in sims.values())))
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    pad = 3.0
    xlim = (rp[:, 0].min() - pad, rp[:, 0].max() + pad)
    ylim = (rp[:, 1].min() - pad, rp[:, 1].max() + pad)
    with writer.saving(fig, out, dpi=100):
        n_frames = int(duration / speedup * FPS) + 1
        for fi in range(n_frames):
            t = fi * speedup / FPS
            ax.clear()
            i = np.searchsorted(rt, t)
            i = min(i, len(rt) - 1)
            ax.plot(rp[:i + 1, 0], rp[:i + 1, 1], "k-", lw=1.6, zorder=5,
                    label="real (odin)")
            draw_robot(ax, rp[i, 0], rp[i, 1], ryaw[i], "0.3")
            for engine in ENGINES:
                sdat = sims[engine]
                j = min(np.searchsorted(sdat["t"], t), len(sdat["t"]) - 1)
                ax.plot(sdat["x"][:j + 1], sdat["y"][:j + 1], color=colors[engine],
                        lw=1.1, alpha=0.75, zorder=4, label=labels[engine])
                draw_robot(ax, sdat["x"][j], sdat["y"][j], sdat["yaw"][j],
                           colors[engine])
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal"); ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title(f"{bag} — real vs open-loop replays (tuned params, "
                         f"{speedup:.0f}x speed)   t = {t:5.1f} s")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=("rocks", "replay", "both"))
    args = ap.parse_args()
    common.RESULTS_DIR.mkdir(exist_ok=True)
    if args.what in ("rocks", "both"):
        render_rocks(str(common.RESULTS_DIR / "rocks_compare.mp4"))
    if args.what in ("replay", "both"):
        render_replay(str(common.RESULTS_DIR / "replay_fast1_variants.mp4"))


if __name__ == "__main__":
    main()
