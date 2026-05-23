"""MuJoCo parameter sweep for the box sim-to-real benchmark.

Junior helhest geometry (matches examples/helhest_junior/common.py:HelhestJuniorConfig)
authored inline as MJCF, with the same static box obstacle as the Axion replay.
Drives the wheels with the GT command timeseries (sim-order, sign-flipped so
forward = positive) and scores the prism-tracked trajectory vs the real one.

Usage:
    python experiments/1_sim_to_real_box/sweep_mujoco.py \
        --gt data/run_2026_05_20-18_04_51.json data/run_2026_05_20-18_10_33.json \
        --dt 0.002 0.005 --kv 1000 4000 --mu 0.4 0.8 \
        --save results/sweep_mujoco.json
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from common_box import DATA_DIR, RESULTS_DIR, load_gt, score

DURATION = 12.0

# --- Junior MJCF template (mirrors HelhestJuniorConfig dimensions/masses) ---
#
# Frame: chassis body origin sits at wheel-axle height (z = wheel radius at
# rest). Wheels at local (0, ±0.365, 0) for front, (-0.75, 0, 0) for rear, all
# with hinge axis (0,1,0) — positive velocity → forward (same as Axion junior).
# Two chassis boxes are rigid geoms on the chassis body; their combined inertia
# about the chassis CoM is rolled into one <inertial>.
JUNIOR_BOX_XML = """<?xml version="1.0"?>
<mujoco model="helhest_junior_box">
  <option gravity="0 0 -9.81" timestep="{dt}"
          solver="{solver}" iterations="{iterations}" ls_iterations="{ls_iterations}"
          cone="{cone}" impratio="{impratio}" integrator="{integrator}"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="{ground_friction} {ground_torsional} {ground_rolling}"
          solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
          condim="{condim}"/>

    <geom name="box" type="box" pos="{box_x} {box_y} {box_z}"
          size="{box_hx} {box_hy} {box_hz}" rgba="0.6 0.4 0.2 1"
          friction="{box_friction} {ground_torsional} {ground_rolling}"
          solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
          condim="{condim}"/>

    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <!-- combined inertia of the two chassis boxes about chassis CoM -->
      <inertial mass="89.7" pos="-0.188 0 0" diaginertia="2.41 4.22 6.03"/>
      <!-- front chassis box: center (-0.13, 0, 0), size 0.48 x 0.56 x 0.20 -->
      <geom type="box" pos="-0.13 0 0" size="0.24 0.28 0.10"
            rgba="0.45 0.55 0.75 1" contype="0" conaffinity="0"/>
      <!-- rear chassis box: center (-0.61, 0, 0), size 0.48 x 0.24 x 0.20 -->
      <geom type="box" pos="-0.61 0 0" size="0.24 0.12 0.10"
            rgba="0.45 0.55 0.75 1" contype="0" conaffinity="0"/>

      <body name="left_wheel" pos="0 0.365 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.35"
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="right_wheel" pos="0 -0.365 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.35"
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="rear_wheel" pos="-0.75 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.173021 0.336875 0.173021"/>
        <geom type="cylinder" fromto="0 -0.05 0 0 0.05 0" size="0.35"
              friction="{rear_friction} {rear_torsional} {rear_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="left_act"  joint="left_wheel_j"  kv="{kv}"/>
    <velocity name="right_act" joint="right_wheel_j" kv="{kv}"/>
    <velocity name="rear_act"  joint="rear_wheel_j"  kv="{kv}"/>
  </actuator>
</mujoco>
"""

BASE_PARAMS = dict(
    solver="Newton", iterations=50, ls_iterations=50,
    cone="pyramidal", integrator="implicitfast", impratio=1.0,
    condim=3,
    ground_torsional=0.1, ground_rolling=0.01,
    front_torsional=0.1, front_rolling=0.01,
    rear_torsional=0.1, rear_rolling=0.01,
    solref0=0.005, solref1=1.0,
    solimp0=0.9, solimp1=0.95, solimp2=0.001,
)


def simulate(params, gt):
    """Run MuJoCo with GT wheel commands, return [N,7] chassis pose (x,y,z, qx,qy,qz,qw)."""
    xml = JUNIOR_BOX_XML.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = params["dt"]
    T = int(DURATION / dt)
    pose = np.zeros((T, 7), dtype=np.float32)

    ts_t = gt["control"]["t"]
    cmd = gt["control"]["lrr"]  # [N, 3] sim order L,R,rear; forward = positive

    for step in range(T):
        t = (step + 1) * dt
        data.ctrl[0] = np.interp(t, ts_t, cmd[:, 0])  # left
        data.ctrl[1] = np.interp(t, ts_t, cmd[:, 1])  # right
        data.ctrl[2] = np.interp(t, ts_t, cmd[:, 2])  # rear
        mujoco.mj_step(model, data)
        # qpos: [px,py,pz, qw,qx,qy,qz] → reorder to xyzw for prism_track
        q = data.qpos
        pose[step, 0:3] = q[0:3]
        pose[step, 3] = q[4]
        pose[step, 4] = q[5]
        pose[step, 5] = q[6]
        pose[step, 6] = q[3]
    return pose


def run_config(params, gts):
    out = {}
    for name, gt in gts.items():
        pose = simulate(params, gt)
        out[name] = score(pose, params["dt"], gt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", nargs="+", default=[
        str(DATA_DIR / "run_2026_05_20-18_04_51.json"),
        str(DATA_DIR / "run_2026_05_20-18_10_33.json")])
    ap.add_argument("--dt", type=float, nargs="+", default=[0.002, 0.005])
    ap.add_argument("--kv", type=float, nargs="+", default=[1000, 4000])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.4, 0.8],
                    help="friction (applied to ground, box, both wheel sets)")
    ap.add_argument("--save", default=str(RESULTS_DIR / "sweep_mujoco.json"))
    args = ap.parse_args()

    gts = {pathlib.Path(p).stem: load_gt(p) for p in args.gt}
    box = next(iter(gts.values()))["box"]
    box_geom = dict(box_x=box["center"][0], box_y=box["center"][1], box_z=box["center"][2],
                    box_hx=box["half_extents"][0], box_hy=box["half_extents"][1],
                    box_hz=box["half_extents"][2])

    configs = list(itertools.product(args.dt, args.kv, args.mu))
    print(f"MuJoCo box sweep: {len(configs)} configs x {len(gts)} runs")

    best, rows = None, []
    for dt, kv, mu in configs:
        params = {**BASE_PARAMS, **box_geom, "dt": dt, "kv": kv,
                  "ground_friction": mu, "box_friction": mu,
                  "front_friction": mu, "rear_friction": mu}
        t0 = time.perf_counter()
        scores = run_config(params, gts)
        combined = float(np.mean([s["combined"] for s in scores.values()]))
        rows.append({"dt": dt, "kv": kv, "mu": mu, "combined": combined,
                     "per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"]}
                                 for n, s in scores.items()}})
        print(f"  dt={dt} kv={kv} mu={mu}: combined={combined:.3f} m "
              f"({time.perf_counter()-t0:.1f}s)")
        if best is None or combined < best["combined"]:
            best = {"dt": dt, "kv": kv, "mu": mu, "combined": combined, "scores": scores}

    bp = {"dt": best["dt"], "kv": best["kv"], "mu": best["mu"]}
    out = {
        "simulator": "MuJoCo",
        "best_params": bp,
        "best_error": best["combined"],
        "best_per_run": {n: {"combined": s["combined"], "xy": s["xy"], "z": s["z"],
                             "sim_rel": s["sim_rel"].tolist(),
                             "sim_t_aligned": s["sim_t_aligned"].tolist()}
                         for n, s in best["scores"].items()},
        "grid": rows,
    }
    pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(out, f)
    print(f"\nBest: {bp}  combined={best['combined']:.3f} m  -> {args.save}")


if __name__ == "__main__":
    main()
