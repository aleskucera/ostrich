"""Plot sweep results vs real robot ground truth trajectory.

Re-simulates the best config for each simulator to get full trajectories.

Usage:
    python examples/comparison_accuracy/helhest/plot_sweep.py \
        --ground-truth results/helhest_2026_04_10-14_46_18.json \
        --sweep results/sweep_mujoco_14_46_18.json results/sweep_axion_14_46_18.json
"""
import argparse
import json
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import mujoco

os.environ["PYOPENGL_PLATFORM"] = "glx"

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

HELHEST_XML_TEMPLATE = """<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="{dt}"
          solver="{solver}" iterations="{iterations}" ls_iterations="{ls_iterations}"
          cone="{cone}" impratio="{impratio}" integrator="{integrator}"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1"
          friction="{ground_friction} {ground_torsional} {ground_rolling}"
          solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
          condim="{condim}"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" contype="0" conaffinity="0"/>
      <body name="battery" pos="-0.302 0.165 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.00768 0.0164 0.01208"/>
        <geom type="box" size="0.125 0.05 0.095" contype="0" conaffinity="0"/>
      </body>
      <body name="left_motor" pos="-0.09 0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="right_motor" pos="-0.09 -0.14 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="rear_motor" pos="-0.22 -0.04 0">
        <inertial mass="7.0" pos="0 0 0" diaginertia="0.0378 0.0084 0.0378"/>
        <geom type="box" size="0.0425 0.12 0.0425" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel_holder" pos="-0.477 0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wheel_holder" pos="-0.477 -0.095 0">
        <inertial mass="3.0" pos="0 0 0" diaginertia="0.0085 0.0981 0.0904"/>
        <geom type="box" size="0.3 0.02 0.09" contype="0" conaffinity="0"/>
      </body>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
              friction="{front_friction} {front_torsional} {front_rolling}"
              solref="{solref0} {solref1}" solimp="{solimp0} {solimp1} {solimp2} 0.5 2"
              condim="{condim}"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36"
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
</mujoco>"""


def simulate_mujoco(params, target_ctrl, duration):
    """Re-simulate MuJoCo with best params to get full trajectory."""
    xml = HELHEST_XML_TEMPLATE.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    dt = params["dt"]
    T = int(duration / dt)
    traj = []
    for step in range(T):
        data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)
        traj.append([float(data.qpos[0]), float(data.qpos[1])])
    return np.array(traj), dt


def simulate_axion(sweep_data, target_ctrl, duration):
    """Re-simulate Axion best config to get full trajectory."""
    import warp as wp
    from sweep_axion_fast import HelhestSweepSim
    from axion import AxionEngineConfig, ExecutionConfig, LoggingConfig, RenderingConfig, SimulationConfig

    fixed = sweep_data["fixed_params"]
    best = sweep_data["best_params"]
    dt = fixed["dt"]
    k_p = fixed["k_p"]

    sim_config = SimulationConfig(
        duration_seconds=duration,
        target_timestep_seconds=dt,
        num_worlds=1,
    )
    render_config = RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False)
    exec_config = ExecutionConfig(use_cuda_graph=False, headless_steps_per_segment=1)
    engine_config = AxionEngineConfig(
        max_newton_iters=16, max_linear_iters=16, backtrack_min_iter=12,
        newton_atol=1e-5, linear_atol=1e-5, linear_tol=1e-5,
        enable_linesearch=False,
        joint_compliance=6e-8, contact_compliance=1e-4,
        friction_compliance=best["friction_compliance"],
        regularization=1e-6,
        contact_fb_alpha=0.5, contact_fb_beta=1.0,
        friction_fb_alpha=1.0, friction_fb_beta=1.0,
        max_contacts_per_world=8,
    )
    logging_config = LoggingConfig(enable_timing=False, enable_hdf5_logging=False)

    sim = HelhestSweepSim(
        sim_config, render_config, exec_config, engine_config, logging_config,
        target_ctrl=target_ctrl, k_p=k_p, mu=best["mu"],
    )
    traj = sim.simulate_trajectory()
    return np.array(traj), dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, help="Path to extracted rosbag JSON")
    parser.add_argument("--sweep", nargs="+", required=True, help="Path(s) to sweep result JSON(s)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth) as f:
        gt = json.load(f)
    gt_xy = np.array([[p["x"], p["y"]] for p in gt["trajectory"]["points"]])
    gt_t = np.array([p["t"] for p in gt["trajectory"]["points"]])
    target_ctrl = gt["target_ctrl_rad_s"]
    duration = gt["trajectory"]["duration_s"]

    # Load and re-simulate each sweep
    SIM_COLORS = {
        "MuJoCo": "#E91E63",
        "Axion": "#2196F3",
        "Dojo": "#4CAF50",
    }

    sim_trajectories = []  # (name, color, traj_xy, traj_t, error)

    for path in args.sweep:
        with open(path) as f:
            sweep_data = json.load(f)
        sim_name = sweep_data["simulator"]
        color = SIM_COLORS.get(sim_name, "gray")

        # Get best error
        best_error = sweep_data["best_error"]

        # Get best params and re-simulate
        if sim_name == "MuJoCo":
            if "top_10_fine" in sweep_data:
                best_params = sweep_data["top_10_fine"][0]["params"]
            else:
                best_params = sweep_data["best_params"]
            print(f"Re-simulating {sim_name} (err={best_error:.4f})...")
            sim_traj, sim_dt = simulate_mujoco(best_params, target_ctrl, duration)
        elif sim_name == "Axion":
            print(f"Re-simulating {sim_name} (err={best_error:.4f})...")
            sim_traj, sim_dt = simulate_axion(sweep_data, target_ctrl, duration)
        else:
            print(f"  Skipping {sim_name} (re-simulation not implemented)")
            continue

        sim_t = np.arange(1, len(sim_traj) + 1) * sim_dt
        sim_trajectories.append((sim_name, color, sim_traj, sim_t, best_error))

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ax_xy, ax_x, ax_y = axes

    # Ground truth
    ax_xy.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2, label="Real robot", zorder=10)
    ax_xy.plot(gt_xy[0, 0], gt_xy[0, 1], "ko", ms=8, zorder=11)
    ax_xy.plot(gt_xy[-1, 0], gt_xy[-1, 1], "ks", ms=8, zorder=11)
    ax_x.plot(gt_t, gt_xy[:, 0], "k-", linewidth=2, label="Real robot")
    ax_y.plot(gt_t, gt_xy[:, 1], "k-", linewidth=2, label="Real robot")

    # Simulated trajectories
    for sim_name, color, sim_traj, sim_t, best_error in sim_trajectories:
        label = f"{sim_name} (err={best_error:.3f})"
        ax_xy.plot(sim_traj[:, 0], sim_traj[:, 1], color=color, linewidth=2, label=label)
        ax_xy.plot(sim_traj[-1, 0], sim_traj[-1, 1], "s", color=color, ms=6)
        ax_x.plot(sim_t, sim_traj[:, 0], color=color, linewidth=2, label=label)
        ax_y.plot(sim_t, sim_traj[:, 1], color=color, linewidth=2, label=label)

    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("XY trajectory (top-down)")
    ax_xy.set_aspect("equal")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    ax_x.set_xlabel("time (s)")
    ax_x.set_ylabel("x (m)")
    ax_x.set_title("X position vs time")
    ax_x.legend(fontsize=8)
    ax_x.grid(True, alpha=0.3)

    ax_y.set_xlabel("time (s)")
    ax_y.set_ylabel("y (m)")
    ax_y.set_title("Y position vs time")
    ax_y.legend(fontsize=8)
    ax_y.grid(True, alpha=0.3)

    bag_name = gt.get("bag_name", "unknown")
    fig.suptitle(f"Sweep results vs real robot — {bag_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
