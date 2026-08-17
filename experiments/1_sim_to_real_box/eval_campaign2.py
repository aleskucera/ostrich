"""Campaign-2 generalization evaluation: replay the new real runs (ostrich0-3)
with the CAMPAIGN-1 CALIBRATED contact parameters — no retuning.

Each engine runs at its campaign-1 best config (results/sweep_*.json) over the
four campaign-2 runs, with the per-run lidar-fitted pallet pose (center + yaw,
0.8 x 1.2 x 0.144 m) and per-run duration. This tests whether the calibration
transfers to new trajectories, start poses, and a re-measured obstacle.

Motor tracking is the one thing re-fit, because the robot itself changed
between campaigns (LLC firmware fix 2026-07-27): the real robot covers ground
at a very consistent 88.4-88.9 % of commanded wheel speed x R across all four
runs and a 3x speed range — a proportional command-tracking / effective-radius
factor, not a contact property. We calibrate a single scalar command scale per
engine on the PRE-BOX flat cruise window (sim pre-box speed matched to real
pre-box speed, shared across runs), then score the full runs. Both passes
(scale=1 and calibrated) are saved so the effect is transparent.

Usage (main venv):
    .venv/bin/python experiments/1_sim_to_real_box/eval_campaign2.py \
        --engines ostrich mujoco semi_implicit
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import warp as wp

from common_box import DATA_DIR, RESULTS_DIR, load_gt, resample_setpoints, score

RUNS = ["ostrich0", "ostrich1", "ostrich2", "ostrich3"]

# Campaign-1 best parameters (results/sweep_*.json best_params) — FROZEN.
C1_OSTRICH = dict(dt=0.05, mu_front=0.8, mu_rear=1.2, mu_rolling=0.7,
                  compliance_contact=1e-7)
C1_MUJOCO = dict(dt=0.002, kv=1000, mu=1.2, tor=0.3, solref0=0.005,
                 condim=6, integrator="implicitfast", wheel_geom="capsule")
C1_SI = dict(dt=5e-4, mu=0.05, ke=8e4, kd=2e3, kf=1500.0, k_d_act=200.0,
             joint_attach_ke=1e6, joint_attach_kd=1e2)


def prebox_speed(t, x, y, gt):
    """Median ground speed on the settled pre-box cruise window. Identical
    smoothing pipeline for real and sim tracks: resample to 20 ms, 0.6 s
    boxcar (kills the finite-difference jitter inflation of the 14 Hz Odin
    poses), gradient, median over samples with a non-trivial command."""
    t, x, y = np.asarray(t, float), np.asarray(x, float), np.asarray(y, float)
    box_near = gt["box"]["center"][0] - gt["box"]["half_extents"][0]
    tg = np.arange(t[0], t[-1], 0.02)
    if len(tg) < 100:
        return float("nan")
    xg, yg = np.interp(tg, t, x), np.interp(tg, t, y)
    k = 31
    ker = np.ones(k) / k
    xs, ys = np.convolve(xg, ker, "same"), np.convolve(yg, ker, "same")
    spd = np.hypot(np.gradient(xs, 0.02), np.gradient(ys, 0.02))
    cmd = np.interp(tg, gt["control"]["t"],
                    gt["control"]["lrr"][:, :2].mean(axis=1))
    valid = ((xg < box_near - 0.4) & (tg > 1.5) & (tg < tg[-1] - 0.5)
             & (np.abs(cmd) > 0.3))
    if valid.sum() < 25:
        return float("nan")
    return float(np.median(spd[valid]))


def sim_box_center(gt):
    """Box center in the SIM world frame. The GT frame is anchored at the real
    base_link start pose, which sits prism_offset ahead of the sim chassis
    (front-axle) origin; initial heading is +X in both, so the scene shifts by
    the offset's xy."""
    c = gt["box"]["center"]
    off = gt["prism_offset"]
    return [c[0] + off[0], c[1] + off[1], c[2]]


def _score_run(pose, dt, gt):
    return score(pose, dt, gt, prism_offset=gt["prism_offset"])


# --- engine runners: (gt, cmd_scale) -> (pose [T,7] xyz+xyzw, dt) ---------

def run_ostrich(gt, cmd_scale):
    from ostrich import (OstrichEngineConfig, ComplianceConfig, ContactsConfig,
                       LinearSolverConfig, LinesearchConfig, LoggingConfig,
                       NewtonRaphsonConfig, RenderingConfig, SimulationConfig)
    from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator
    p = C1_OSTRICH
    dt, dur = p["dt"], float(gt["duration_s"])
    engine = OstrichEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=16, backtrack_min_iter=12, atol=1e-3),
        linear=LinearSolverConfig(max_iters=16, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=p["compliance_contact"], friction=1e-6),
        linesearch=LinesearchConfig(enabled=False),
        contacts=ContactsConfig(max_per_world=256),
    )
    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=dur, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=int(round(1 / dt))),
        engine, LoggingConfig(),
        control_mode="velocity",
        mu_front=p["mu_front"], mu_rear=p["mu_rear"], mu_rolling=p["mu_rolling"],
        box_center=sim_box_center(gt), box_half_extents=gt["box"]["half_extents"],
        box_yaw=gt["box"].get("yaw", 0.0),
    )
    setp = resample_setpoints(gt, dt, dur) * cmd_scale
    if wp.get_device().is_cuda:
        pose, _ = sim.replay_graph(setp)
    else:
        pose, _ = sim.replay(setp)
    del sim
    return pose, dt


def run_semi_implicit(gt, cmd_scale):
    from ostrich import (LoggingConfig, RenderingConfig, SemiImplicitEngineConfig,
                       SimulationConfig)
    from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator
    p = C1_SI
    dt, dur = p["dt"], float(gt["duration_s"])
    sim = HelhestJuniorReplaySimulator(
        SimulationConfig(duration_seconds=dur, target_timestep_seconds=dt,
                         num_worlds=1, use_cuda_graph=False),
        RenderingConfig(vis_type="null", target_fps=int(round(1 / dt))),
        SemiImplicitEngineConfig(angular_damping=0.05, friction_smoothing=0.1,
                                 joint_attach_ke=p["joint_attach_ke"],
                                 joint_attach_kd=p["joint_attach_kd"]),
        LoggingConfig(),
        control_mode="velocity",
        k_p=0.0, k_d=p["k_d_act"],
        mu_front=p["mu"], mu_rear=p["mu"], mu_rolling=0.7,
        ground_ke=p["ke"], ground_kd=p["kd"], ground_kf=p["kf"],
        box_ke=p["ke"], box_kd=p["kd"], box_kf=p["kf"],
        wheel_ke=p["ke"], wheel_kd=p["kd"], wheel_kf=p["kf"],
        box_center=sim_box_center(gt), box_half_extents=gt["box"]["half_extents"],
        box_yaw=gt["box"].get("yaw", 0.0),
    )
    setp = resample_setpoints(gt, dt, dur) * cmd_scale
    if wp.get_device().is_cuda:
        pose, _ = sim.replay_graph(setp)
    else:
        pose, _ = sim.replay(setp)
    del sim
    return pose, dt


def run_mujoco(gt, cmd_scale):
    import mujoco
    from sweep_mujoco import BASE_PARAMS, JUNIOR_BOX_XML, _patch_wheel_geom
    p = C1_MUJOCO
    dt, dur = p["dt"], float(gt["duration_s"])
    box = gt["box"]
    params = {**BASE_PARAMS, "dt": dt, "kv": p["kv"],
              "ground_friction": p["mu"], "box_friction": p["mu"],
              "front_friction": p["mu"], "rear_friction": p["mu"],
              "ground_torsional": p["tor"], "front_torsional": p["tor"],
              "rear_torsional": p["tor"],
              "solref0": p["solref0"], "condim": p["condim"],
              "integrator": p["integrator"],
              "box_x": sim_box_center(gt)[0], "box_y": sim_box_center(gt)[1],
              "box_z": box["center"][2],
              "box_hx": box["half_extents"][0], "box_hy": box["half_extents"][1],
              "box_hz": box["half_extents"][2]}
    xml = JUNIOR_BOX_XML.format(**params)
    # per-run pallet yaw (attribute order is free in MJCF)
    yaw_deg = np.degrees(box.get("yaw", 0.0))
    xml = xml.replace('<geom name="box" type="box"',
                      f'<geom name="box" type="box" euler="0 0 {yaw_deg:.3f}"')
    xml = _patch_wheel_geom(xml, p["wheel_geom"])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    T = int(dur / dt)
    pose = np.zeros((T, 7), dtype=np.float32)
    ts_t, cmd = gt["control"]["t"], gt["control"]["lrr"]
    for step in range(T):
        t = (step + 1) * dt
        for c in range(3):
            data.ctrl[c] = cmd_scale * np.interp(t, ts_t, cmd[:, c])
        mujoco.mj_step(model, data)
        q = data.qpos
        pose[step, 0:3] = q[0:3]
        pose[step, 3:6] = q[4:7]
        pose[step, 6] = q[3]
    return pose, dt


RUNNERS = {"ostrich": run_ostrich, "mujoco": run_mujoco,
           "semi_implicit": run_semi_implicit}


def eval_engine(engine, gts, cmd_scale_override=None):
    runner = RUNNERS[engine]

    def one_pass(scale):
        out = {}
        for name, gt in gts.items():
            t0 = time.perf_counter()
            pose, dt = runner(gt, scale)
            s = _score_run(pose, dt, gt)
            st = np.arange(pose.shape[0]) * dt
            s["sim_prebox_speed"] = prebox_speed(st, pose[:, 0], pose[:, 1], gt)
            out[name] = s
            print(f"    {name}: combined={s['combined']:.3f} m "
                  f"(+yaw {s['combined_with_yaw']:.3f}) z={s['z']:.3f} "
                  f"prebox v={s['sim_prebox_speed']:.3f} "
                  f"({time.perf_counter()-t0:.1f}s)")
        return out

    print(f"  pass 1 (cmd_scale=1.0):")
    pass1 = one_pass(1.0)
    if cmd_scale_override is not None:
        alpha = cmd_scale_override
    else:
        ratios = []
        for name, gt in gts.items():
            real_v = prebox_speed(gt["real"]["t"], gt["real"]["x"],
                                  gt["real"]["y"], gt)
            sim_v = pass1[name]["sim_prebox_speed"]
            if np.isfinite(real_v) and np.isfinite(sim_v) and sim_v > 1e-3:
                ratios.append(real_v / sim_v)
        alpha = float(np.median(ratios))
        # dead-band: within a few % the engine's own motor model already
        # matches the real tracking; don't churn the commands over noise
        if abs(alpha - 1.0) < 0.04:
            alpha = 1.0
    print(f"  motor calibration: cmd_scale={alpha:.4f}")
    if alpha == 1.0:
        print("  (within dead-band; pass 2 = pass 1)")
        return alpha, pass1, pass1
    print(f"  pass 2 (cmd_scale={alpha:.4f}):")
    pass2 = one_pass(alpha)
    return alpha, pass1, pass2


def _row(s):
    return {k: s[k] for k in ("combined", "combined_with_yaw", "xy", "z",
                              "yaw_rmse_deg", "shift", "sim_prebox_speed")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engines", nargs="+", default=["ostrich", "mujoco", "semi_implicit"],
                    choices=list(RUNNERS))
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--cmd-scale", type=float, default=None,
                    help="skip calibration, use this fixed command scale")
    ap.add_argument("--save", default=str(RESULTS_DIR / "eval_campaign2.json"))
    args = ap.parse_args()

    gts = {r: load_gt(DATA_DIR / f"{r}.json") for r in args.runs}
    for name, gt in gts.items():
        rv = prebox_speed(gt["real"]["t"], gt["real"]["x"], gt["real"]["y"], gt)
        print(f"{name}: dur={gt['duration_s']:.1f}s real prebox v={rv:.3f} m/s "
              f"box=({gt['box']['center'][0]:.2f},{gt['box']['center'][1]:.2f}) "
              f"yaw={np.degrees(gt['box'].get('yaw', 0)):.1f}deg")

    results = {}
    for engine in args.engines:
        print(f"\n=== {engine} (campaign-1 params, frozen) ===")
        alpha, pass1, pass2 = eval_engine(engine, gts, args.cmd_scale)
        mean1 = float(np.mean([s["combined_with_yaw"] for s in pass1.values()]))
        mean2 = float(np.mean([s["combined_with_yaw"] for s in pass2.values()]))
        print(f"  mean combined+yaw: {mean1:.3f} m -> {mean2:.3f} m (motor-calibrated)")
        results[engine] = {
            "campaign1_params": {"ostrich": C1_OSTRICH, "mujoco": C1_MUJOCO,
                                 "semi_implicit": C1_SI}[engine],
            "cmd_scale": alpha,
            "pass1": {n: _row(s) for n, s in pass1.items()},
            "pass2": {n: _row(s) for n, s in pass2.items()},
            "pass2_traj": {n: {"sim_rel": s["sim_rel"].tolist(),
                               "sim_t_aligned": s["sim_t_aligned"].tolist()}
                           for n, s in pass2.items()},
            "mean_combined_with_yaw": {"pass1": mean1, "pass2": mean2},
        }

    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f)
    print(f"\nSaved -> {args.save}")


if __name__ == "__main__":
    main()
