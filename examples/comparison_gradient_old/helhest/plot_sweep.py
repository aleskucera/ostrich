"""Plot best sweep trajectories vs ground truth for all simulators.

Reads sweep result JSONs, re-simulates each best config, and plots
XY trajectory + X(t) + Y(t) comparison.

Usage:
    python examples/comparison_gradient/helhest/plot_sweep.py \
        --ground-truth results/helhest_chrono.json \
        --axion results/sweep_axion.json \
        --mujoco results/sweep_mujoco.json \
        --dojo results/sweep_dojo.json \
        --tinydiffsim results/sweep_tinydiffsim.json \
        -o results/sweep_all_vs_chrono.png
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

DURATION = 3.0


# ── Trajectory generators ──

def simulate_mujoco(params):
    sys.path.insert(0, os.path.dirname(__file__))
    from sweep_mujoco import HELHEST_XML_TEMPLATE, TARGET_CTRL
    import mujoco

    xml = HELHEST_XML_TEMPLATE.format(**params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T = int(DURATION / params["dt"])
    traj = [[0.0, 0.0]]
    for _ in range(T):
        data.ctrl[:] = TARGET_CTRL
        mujoco.mj_step(model, data)
        traj.append([float(data.qpos[0]), float(data.qpos[1])])
    return np.array(traj)


def simulate_tinydiffsim(params):
    sys.path.insert(0, os.path.dirname(__file__))
    from sweep_tinydiffsim import simulate
    return np.array(simulate(**params))


def simulate_axion(params):
    """Run in subprocess to isolate GPU memory."""
    tmp = tempfile.mktemp(suffix=".json")
    worker = f'''
import os, json, sys, numpy as np
os.environ["PYOPENGL_PLATFORM"] = "glx"
import newton, warp as wp
from axion import (
    AxionDifferentiableSimulator, AxionEngineConfig, ComplianceConfig,
    ContactsConfig, LinearSolverConfig, LinesearchConfig,
    LoggingConfig, NewtonRaphsonConfig, RenderingConfig, SimulationConfig,
)
from axion.simulation.sim_config import SyncMode
from examples.helhest.common import create_helhest_model, HelhestConfig

class Sim(AxionDifferentiableSimulator):
    def build_model(self):
        self.builder.rigid_gap = 0.1
        self.builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0))
        create_helhest_model(self.builder,
            xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            control_mode="velocity",
            k_p={params["k_p"]}, k_d=HelhestConfig.TARGET_KD,
            friction_left_right={params["friction_lr"]},
            friction_rear={params["friction_rear"]})
        return self.builder.finalize_replicated(num_worlds=1, requires_grad=False)
    def compute_loss(self): pass
    def update(self): pass

sim = Sim(
    SimulationConfig(duration_seconds=3.0, target_timestep_seconds={params["dt"]},
                     num_worlds=1, sync_mode=SyncMode.ALIGN_FPS_TO_DT, use_cuda_graph=False),
    RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False),
    AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=12, backtrack_min_iter=8, atol=1e-1),
        linear=LinearSolverConfig(max_iters=12, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-6, friction=1e-6),
        contacts=ContactsConfig(max_per_world=8),
        linesearch=LinesearchConfig(enabled=False),
    ),
    LoggingConfig())

model = sim.model
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.states[0])
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.target_states[0])
T = sim.clock.total_sim_steps
num_dofs = sim.trajectory.joint_target_vel.shape[-1]
for i in range(T):
    ctrl = np.zeros(num_dofs, dtype=np.float32)
    ctrl[6] = 1.0; ctrl[7] = 6.0; ctrl[8] = 0.0
    wp.copy(sim.target_controls[i].joint_target_vel,
            wp.array(ctrl, dtype=wp.float32, device=model.device))
sim.run_target_episode()
traj = []
for t in range(sim.trajectory.target_body_pose.shape[0]):
    bp = sim.trajectory.target_body_pose[t].numpy()[0, 0]
    traj.append([float(bp[0]), float(bp[1])])
import pathlib
pathlib.Path("{tmp}").write_text(json.dumps(traj))
'''
    result = subprocess.run([sys.executable, "-c", worker],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Axion subprocess failed: {result.stderr[-500:]}")
    with open(tmp) as f:
        traj = json.load(f)
    os.unlink(tmp)
    return np.array(traj)


def simulate_dojo(params):
    """Run in subprocess via Julia."""
    tmp_json = tempfile.mktemp(suffix=".json")
    tmp_jl = tempfile.mktemp(suffix=".jl")

    jl_code = f"""
using Dojo, JSON, LinearAlgebra

const WHEEL_RADIUS = 0.36
const WHEEL_WIDTH = 0.11
const WHEEL_MASS = 5.5

function build_helhest(; timestep, kv, friction_front, friction_rear)
    origin = Origin{{Float64}}()
    chassis_mass = 85.0 + 2.0 + 7.0 + 7.0 + 7.0 + 3.0 + 3.0
    chassis = Box(0.26, 0.6, 0.18, chassis_mass; color=RGBA(0.5, 0.5, 0.5), name=:chassis)
    wo = Quaternion(cos(pi/4), sin(pi/4), 0.0, 0.0)
    mk = (nm) -> Capsule(WHEEL_RADIUS, WHEEL_WIDTH, WHEEL_MASS; orientation_offset=wo, name=nm)
    lw, rw, rew = mk(:lw), mk(:rw), mk(:rew)
    j_base = JointConstraint(Floating(origin, chassis; spring=0.0, damper=0.0), name=:bj)
    j_l = JointConstraint(Revolute(chassis, lw, Y_AXIS; parent_vertex=[0,0.36,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jl)
    j_r = JointConstraint(Revolute(chassis, rw, Y_AXIS; parent_vertex=[0,-0.36,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jr)
    j_re = JointConstraint(Revolute(chassis, rew, Y_AXIS; parent_vertex=[-0.697,0,0], child_vertex=zeros(3), spring=0.0, damper=kv), name=:jre)
    n = [0.0, 0.0, 1.0]
    function wc(w, fr, nms)
        origins = [[0.0, WHEEL_WIDTH/2, 0.0], [0.0, -WHEEL_WIDTH/2, 0.0]]
        contact_constraint(w, fill(n, 2); friction_coefficients=fill(fr, 2), contact_origins=origins, contact_radii=fill(WHEEL_RADIUS, 2), contact_type=:nonlinear, names=nms)
    end
    cl = wc(lw, friction_front, [:lc1,:lc2])
    cr = wc(rw, friction_front, [:rc1,:rc2])
    cre = wc(rew, friction_rear, [:rec1,:rec2])
    Mechanism(origin, [chassis, lw, rw, rew], [j_base, j_l, j_r, j_re], [cl; cr; cre]; gravity=-9.81, timestep)
end

function initial_z()
    local z = zeros(52)
    for (i, pos) in enumerate([[0,0,WHEEL_RADIUS],[0,0.36,WHEEL_RADIUS],[0,-0.36,WHEEL_RADIUS],[-0.697,0,WHEEL_RADIUS]])
        idx = 13*(i-1)
        z[idx+1:idx+3] = pos
        z[idx+7:idx+10] = [1,0,0,0]
    end
    return z
end

function main()
    mech = build_helhest(timestep={params['dt']}, kv={params['kv']},
        friction_front={params['friction_front']}, friction_rear={params['friction_rear']})
    local z = initial_z()
    local T = Int(3.0 / {params['dt']})
    local kv = {params['kv']}
    local ctrl = [1.0, 6.0, 0.0]
    local traj = [[z[1], z[2]]]
    for t in 1:T
        local u = zeros(9)
        u[7:9] = kv .* ctrl
        z = step!(mech, z, u)
        push!(traj, [z[1], z[2]])
    end
    open("{tmp_json}", "w") do io
        JSON.print(io, traj)
    end
end

main()
"""
    with open(tmp_jl, "w") as f:
        f.write(jl_code)

    julia = os.path.expanduser("~/.juliaup/bin/julia")
    result = subprocess.run([julia, "+1.10", "--startup-file=no", tmp_jl],
                            capture_output=True, text=True, timeout=300)
    os.unlink(tmp_jl)
    if result.returncode != 0:
        raise RuntimeError(f"Dojo subprocess failed: {result.stderr[-500:]}")
    with open(tmp_json) as f:
        traj = json.load(f)
    os.unlink(tmp_json)
    return np.array(traj)


# ── Plotting ──

def load_trajectory_json(path):
    """Load a helhest_*.json that already contains target_trajectory + sweep error."""
    with open(path) as f:
        d = json.load(f)
    traj = np.array(d["target_trajectory"])
    err = d.get("best_error")
    dt = d.get("dt", "?")
    name = d.get("simulator", "?")
    return name, traj, err, dt


def main():
    parser = argparse.ArgumentParser(description="Plot sweep trajectories vs ground truth")
    parser.add_argument("--ground-truth", required=True, help="Chrono ground truth JSON")
    parser.add_argument("--axion", help="Axion sweep result JSON (or helhest_*.json with trajectory)")
    parser.add_argument("--mujoco", help="MuJoCo sweep result JSON (or helhest_*.json with trajectory)")
    parser.add_argument("--dojo", help="Dojo sweep result JSON (or helhest_*.json with trajectory)")
    parser.add_argument("--tinydiffsim", help="TinyDiffSim sweep result JSON (or helhest_*.json with trajectory)")
    parser.add_argument("-o", "--output", default="results/sweep_all_vs_chrono.png",
                        help="Output image path (default: results/sweep_all_vs_chrono.png)")
    args = parser.parse_args()

    # Ground truth
    with open(args.ground_truth) as f:
        gt = json.load(f)
    chrono_traj = np.array(gt["target_trajectory"])
    print(f"Ground truth: {gt.get('simulator', '?')}, "
          f"final=({chrono_traj[-1, 0]:.3f}, {chrono_traj[-1, 1]:.3f})")

    # Collect (name, traj, error, dt, color, linestyle, linewidth)
    sims = [("Chrono (GT)", chrono_traj, None, gt["dt"], "k", "-", 2.0)]

    simulator_args = [
        ("Axion", args.axion, simulate_axion, "#2196F3", "-", 1.5),
        ("Dojo", args.dojo, simulate_dojo, "#FF9800", "--", 1.5),
        ("MuJoCo", args.mujoco, simulate_mujoco, "#E91E63", "--", 1.3),
        ("TinyDiffSim", args.tinydiffsim, simulate_tinydiffsim, "#9C27B0", ":", 1.3),
    ]

    for name, sweep_path, sim_fn, color, ls, lw in simulator_args:
        if sweep_path is None:
            continue
        with open(sweep_path) as f:
            data = json.load(f)

        # If the JSON already has a trajectory, use it directly
        if "target_trajectory" in data:
            traj = np.array(data["target_trajectory"])
            # Try to find the matching sweep JSON for the error
            err = data.get("best_error")
            if err is None:
                sweep_name = name.lower().replace(" ", "")
                try:
                    with open(f"results/sweep_{sweep_name}.json") as sf:
                        err = json.load(sf).get("best_error")
                except FileNotFoundError:
                    pass
            print(f"Loaded {name}: err={err}, {len(traj)} pts, "
                  f"final=({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")
            sims.append((name, traj, err, data.get("dt", "?"), color, ls, lw))
            continue

        # Otherwise re-simulate from sweep params
        params = data["best_params"]
        err = data["best_error"]
        dt = params.get("dt", params.get("timestep", "?"))
        print(f"Simulating {name}: err={err:.4f}, dt={dt}, params={params}")
        try:
            traj = sim_fn(params)
            print(f"  -> {len(traj)} pts, final=({traj[-1, 0]:.3f}, {traj[-1, 1]:.3f})")
            sims.append((name, traj, err, dt, color, ls, lw))
        except Exception as e:
            print(f"  -> FAILED: {e}")

    # ── Paper-quality plot with TeX fonts ──
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    AXION_COLOR = "#2196F3"
    BAR_STYLES = {
        "Axion":       {"color": AXION_COLOR},
        "Dojo":        {"color": "#FF9800"},
        "MuJoCo":      {"color": "#E91E63"},
        "TinyDiffSim": {"color": "#9C27B0"},
    }
    BAR_LABELS = {
        "Axion":       r"\textbf{Axion}",
        "Dojo":        "Dojo",
        "MuJoCo":      "MuJoCo",
        "TinyDiffSim": "TinyDiffSim",
    }

    fig = plt.figure(figsize=(7.0, 3.5))
    ax_traj = fig.add_axes([0.07, 0.25, 0.32, 0.68])
    ax_bar = fig.add_axes([0.56, 0.25, 0.28, 0.65])

    # --- Left: xy trajectory ---
    for name, traj, err, dt, color, ls, lw in sims:
        ax_traj.plot(traj[:, 0], traj[:, 1], color=color, ls=ls, lw=lw, label=name)

    ax_traj.set_xlabel(r"$x$ (m)")
    ax_traj.set_ylabel(r"$y$ (m)")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.tick_params(direction="in", top=True, right=True)
    ax_traj.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                   ncol=3, frameon=True, framealpha=0.9, fontsize=9,
                   handlelength=1.5, columnspacing=0.6)

    # --- Right: horizontal bar chart (stacked_boxes style) ---
    # Sort by error ascending (best at top)
    bar_sims = [(name, err, BAR_STYLES.get(name, {"color": "gray"})["color"])
                for name, _, err, _, _, _, _ in sims if err is not None]
    bar_sims.sort(key=lambda x: x[1])

    bar_names = [BAR_LABELS.get(s[0], s[0]) for s in bar_sims]
    bar_errs = [s[1] for s in bar_sims]
    bar_colors = [s[2] for s in bar_sims]

    axion_err = next(e for n, e, _ in bar_sims if n == "Axion")

    y_pos = np.arange(len(bar_sims))
    bars = ax_bar.barh(y_pos, bar_errs, color=bar_colors, height=0.5, zorder=3)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_names)
    ax_bar.set_xlabel(r"Mean $L_2$ error (m)")
    ax_bar.grid(True, axis="x", alpha=0.25, zorder=0, linewidth=0.5)
    ax_bar.set_ylim(-0.6, len(bar_sims) - 0.4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    xmax = max(bar_errs)
    ax_bar.set_xlim(right=xmax * 1.8)

    right_xfm = blended_transform_factory(ax_bar.transAxes, ax_bar.transData)

    for i, (sim_name, val, _) in enumerate(bar_sims):
        cy = y_pos[i]
        ax_bar.text(val + 0.01, cy, f"{val:.2f}", va="center", ha="left", fontsize=10)

        if sim_name != "Axion":
            ratio = val / axion_err
            ratio_str = f"${ratio:.1f}\\times$"
            ax_bar.text(
                1.04, cy, ratio_str,
                va="center", ha="left", fontsize=10,
                color=AXION_COLOR, fontweight="bold",
                transform=right_xfm, clip_on=False,
            )

    ax_bar.text(
        1.04, 1.01, r"vs \textbf{Axion}",
        va="bottom", ha="left", fontsize=9,
        color="gray", transform=ax_bar.transAxes, clip_on=False,
    )

    # Manual positioning — no tight_layout needed

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=400, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
