"""Scalability benchmark: gradient computation time vs number of parallel environments.

Measures per-iteration wall-clock time (forward + backward) at increasing
batch sizes for GPU-batched simulators (Axion, MJX) and per-environment
time for CPU simulators (MuJoCo-FD, Dojo, TinyDiffSim).

Usage:
    python examples/comparison_gradient/helhest/benchmark_scalability.py \
        --save results/scalability_helhest.json \
        -o results/scalability_helhest.png
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DURATION = 3.0
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
WARMUP_ITERS = 3
BENCH_ITERS = 10


# ── Axion ──

def benchmark_axion(batch_sizes):
    """Run Axion gradient benchmark at each batch size in a subprocess."""
    results = {}
    for B in batch_sizes:
        tmp = tempfile.mktemp(suffix=".json")
        worker = f'''
import os, json, time, numpy as np
os.environ["PYOPENGL_PLATFORM"] = "glx"
import newton, warp as wp
from axion import (
    AxionDifferentiableSimulator, AxionEngineConfig, ComplianceConfig,
    ContactsConfig, LinearSolverConfig, LinesearchConfig,
    LoggingConfig, NewtonRaphsonConfig, RenderingConfig, SimulationConfig,
)
from axion.simulation.sim_config import SyncMode
from examples.helhest.common import create_helhest_model, HelhestConfig

K = 30
WHEEL_DOF_OFFSET = 6
NUM_WHEEL_DOFS = 3
TARGET_CTRL = (1.0, 6.0, 0.0)
INIT_CTRL = (2.0, 5.0, 0.0)

@wp.kernel
def loss_kernel(body_pose: wp.array(dtype=wp.transform, ndim=3),
                target_body_pose: wp.array(dtype=wp.transform, ndim=3),
                weight: float, loss: wp.array(dtype=wp.float32)):
    t = wp.tid()
    pos = wp.transform_get_translation(body_pose[t, 0, 0])
    target_pos = wp.transform_get_translation(target_body_pose[t, 0, 0])
    delta = pos - target_pos
    wp.atomic_add(loss, 0, weight * wp.dot(delta, delta))

class Sim(AxionDifferentiableSimulator):
    def build_model(self):
        self.builder.rigid_gap = 0.1
        self.builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, ke=50.0, kd=50.0, kf=50.0))
        create_helhest_model(self.builder,
            xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            control_mode="velocity", k_p=240.0, k_d=HelhestConfig.TARGET_KD,
            friction_left_right=1.05, friction_rear=0.17)
        return self.builder.finalize_replicated(num_worlds={B}, requires_grad=True)

    def compute_loss(self):
        num_steps = self.trajectory.body_pose.shape[0]
        wp.launch(kernel=loss_kernel, dim=num_steps,
            inputs=[self.trajectory.body_pose, self.trajectory.target_body_pose, 10.0 / num_steps],
            outputs=[self.loss], device=self.solver.model.device)

    def update(self):
        self.trajectory.joint_target_vel.grad.zero_()

sim = Sim(
    SimulationConfig(duration_seconds=3.0, target_timestep_seconds=0.05,
                     num_worlds={B}, sync_mode=SyncMode.ALIGN_FPS_TO_DT, use_cuda_graph=True),
    RenderingConfig(vis_type="null", target_fps=30, usd_file=None, start_paused=False),
    AxionEngineConfig(
        nr=NewtonRaphsonConfig(max_iters=12, backtrack_min_iter=8, atol=1e-1),
        linear=LinearSolverConfig(max_iters=12, tol=1e-3, atol=1e-3, regularization=1e-6),
        compliance=ComplianceConfig(joint=6e-8, contact=1e-6, friction=1e-6),
        contacts=ContactsConfig(max_per_world=8),
        linesearch=LinesearchConfig(enabled=False),
    ),
    LoggingConfig())

sim.loss = wp.zeros(1, dtype=float, requires_grad=True)
sim.track_body(body_idx=0, name="chassis", color=(0.0, 0.5, 1.0))
model = sim.model
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.states[0])
newton.eval_fk(model, model.joint_q, model.joint_qd, sim.target_states[0])

T = sim.clock.total_sim_steps
num_dofs = sim.trajectory.joint_target_vel.shape[-1]

# Set target controls
for i in range(T):
    ctrl = np.zeros(num_dofs, dtype=np.float32)
    ctrl[WHEEL_DOF_OFFSET + 0] = TARGET_CTRL[0]
    ctrl[WHEEL_DOF_OFFSET + 1] = TARGET_CTRL[1]
    ctrl[WHEEL_DOF_OFFSET + 2] = TARGET_CTRL[2]
    wp.copy(sim.target_controls[i].joint_target_vel,
            wp.array(ctrl, dtype=wp.float32, device=model.device))
sim.run_target_episode()

# Set init controls
W = np.zeros((T, K), dtype=np.float32)
for t in range(T):
    k_float = t * (K - 1) / max(T - 1, 1)
    k_low = int(k_float)
    k_high = min(k_low + 1, K - 1)
    alpha = k_float - k_low
    W[t, k_low] += 1.0 - alpha
    W[t, k_high] += alpha

params = np.array([[INIT_CTRL[0], INIT_CTRL[1], INIT_CTRL[2]]] * K, dtype=np.float64)
expanded = W @ params
vel_np = np.zeros((T, 1, num_dofs), dtype=np.float32)
vel_np[:, 0, WHEEL_DOF_OFFSET:WHEEL_DOF_OFFSET + NUM_WHEEL_DOFS] = expanded
wp.copy(sim.trajectory.joint_target_vel, wp.array(vel_np, dtype=wp.float32))
for i in range(T):
    wp.copy(sim.controls[i].joint_target_vel, sim.trajectory.joint_target_vel[i])

# Warmup
for _ in range({WARMUP_ITERS}):
    sim.diff_step()
    wp.synchronize()
    sim.tape.zero()
    sim.loss.zero_()

# Benchmark
times = []
for _ in range({BENCH_ITERS}):
    t0 = time.perf_counter()
    sim.diff_step()
    wp.synchronize()
    times.append(time.perf_counter() - t0)
    sim.tape.zero()
    sim.loss.zero_()

wp.synchronize()
mem_bytes = wp.get_device("cuda:0").context.runtime.core.cuda_context_get_memory_info(
    wp.get_device("cuda:0").context.handle)[1] if hasattr(wp.get_device("cuda:0"), "context") else 0
# Fallback: use nvidia-smi
import subprocess as _sp
try:
    _out = _sp.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], text=True)
    mem_mb = float(_out.strip().split("\\n")[0])
except Exception:
    mem_mb = 0.0

import pathlib
pathlib.Path("{tmp}").write_text(json.dumps({{"times_ms": [t * 1000 for t in times], "gpu_mem_mb": mem_mb}}))
'''
        print(f"  Axion B={B}...", end=" ", flush=True)
        result = subprocess.run([sys.executable, "-c", worker],
                                capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"FAILED")
            results[B] = None
        else:
            with open(tmp) as f:
                data = json.load(f)
            median_ms = float(np.median(data["times_ms"]))
            mem_mb = data.get("gpu_mem_mb", 0)
            print(f"{median_ms:.1f} ms, {mem_mb:.0f} MB GPU")
            results[B] = {"time_ms": median_ms, "gpu_mem_mb": mem_mb}
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return results


# ── MJX ──

def benchmark_mjx(batch_sizes):
    """Run MJX gradient benchmark at each batch size."""
    results = {}
    for B in batch_sizes:
        tmp = tempfile.mktemp(suffix=".json")
        worker = f'''
import os, json, time
os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)
import jax, jax.numpy as jnp, mujoco, mujoco.mjx as mjx, numpy as np

DT = 0.002
DURATION = 3.0
T = int(DURATION / DT)
K = 10
TARGET_CTRL = np.array([1.0, 6.0, 0.0], dtype=np.float32)
INIT_CTRL = np.array([2.0, 5.0, 0.0], dtype=np.float32)
TRAJECTORY_WEIGHT = 10.0
REGULARIZATION_WEIGHT = 1e-7

HELHEST_XML = """
<mujoco model="helhest">
  <option gravity="0 0 -9.81" timestep="0.002" iterations="10" ls_iterations="10"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1" friction="0.7 0.1 0.01"/>
    <body name="chassis" pos="0 0 0.37">
      <freejoint name="base_joint"/>
      <inertial mass="85.0" pos="-0.047 0 0" diaginertia="0.6213 0.1583 0.6770"/>
      <geom type="box" pos="-0.047 0 0" size="0.13 0.3 0.09" contype="0" conaffinity="0"/>
      <body name="left_wheel" pos="0 0.36 0">
        <joint name="left_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="right_wheel" pos="0 -0.36 0">
        <joint name="right_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.7 0.1 0.01"/>
      </body>
      <body name="rear_wheel" pos="-0.697 0 0">
        <joint name="rear_wheel_j" type="hinge" axis="0 1 0"/>
        <inertial mass="5.5" pos="0 0 0" diaginertia="0.20045 0.20045 0.3888"/>
        <geom type="cylinder" fromto="0 -0.055 0 0 0.055 0" size="0.36" friction="0.35 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_act" joint="left_wheel_j" kv="240"/>
    <velocity name="right_act" joint="right_wheel_j" kv="240"/>
    <velocity name="rear_act" joint="rear_wheel_j" kv="240"/>
  </actuator>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(HELHEST_XML)
mx = mjx.put_model(mj_model)

# Batched initial data
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)
dx0_single = mjx.put_data(mj_model, mj_data)
dx0 = jax.tree.map(lambda x: jnp.broadcast_to(x, ({B},) + x.shape).copy(), dx0_single)

# Interpolation matrix
W = np.zeros((T, K), dtype=np.float32)
for t in range(T):
    k_float = t * (K - 1) / max(T - 1, 1)
    k_low = int(k_float)
    k_high = min(k_low + 1, K - 1)
    alpha = k_float - k_low
    W[t, k_low] += 1.0 - alpha
    W[t, k_high] += alpha
W_jnp = jnp.array(W)

# Target trajectory (same for all envs)
target_ctrl_traj = jnp.tile(jnp.array(TARGET_CTRL), (T, 1))
def rollout_single(mx, dx0, ctrl_traj):
    def step_fn(carry, ctrl_t):
        d = carry.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, d.qpos[:2]
    _, xy = jax.lax.scan(step_fn, dx0, ctrl_traj)
    return xy
target_xy = jax.jit(rollout_single)(mx, dx0_single, target_ctrl_traj)
target_xy.block_until_ready()

# Batched rollout + loss
def batched_loss(params):
    ctrl_traj = W_jnp @ params  # (T, K) @ (K, 3) -> (T, 3)
    def single_loss(dx0_i):
        def step_fn(carry, ctrl_t):
            d = carry.replace(ctrl=ctrl_t)
            d = mjx.step(mx, d)
            return d, d.qpos[:2]
        _, xy = jax.lax.scan(step_fn, dx0_i, ctrl_traj)
        delta = xy - target_xy
        return TRAJECTORY_WEIGHT / T * jnp.sum(delta**2) + REGULARIZATION_WEIGHT * jnp.sum(ctrl_traj**2)
    losses = jax.vmap(single_loss)(dx0)
    return jnp.mean(losses)

grad_fn = jax.jit(jax.grad(batched_loss))
value_fn = jax.jit(batched_loss)

params = jnp.tile(jnp.array(INIT_CTRL), (K, 1))

# Compile
print("Compiling...", flush=True)
loss = value_fn(params)
grad = grad_fn(params)
loss.block_until_ready()
grad.block_until_ready()

# Warmup
for _ in range({WARMUP_ITERS}):
    loss = value_fn(params)
    grad = grad_fn(params)
    loss.block_until_ready()
    grad.block_until_ready()

# Benchmark
times = []
for _ in range({BENCH_ITERS}):
    t0 = time.perf_counter()
    loss = value_fn(params)
    grad = grad_fn(params)
    loss.block_until_ready()
    grad.block_until_ready()
    times.append(time.perf_counter() - t0)

import subprocess as _sp
try:
    _out = _sp.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], text=True)
    mem_mb = float(_out.strip().split("\\n")[0])
except Exception:
    mem_mb = 0.0

import pathlib
pathlib.Path("{tmp}").write_text(json.dumps({{"times_ms": [t * 1000 for t in times], "gpu_mem_mb": mem_mb}}))
'''
        print(f"  MJX B={B}...", end=" ", flush=True)
        result = subprocess.run([sys.executable, "-c", worker],
                                capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"FAILED")
            results[B] = None
        else:
            with open(tmp) as f:
                data = json.load(f)
            median_ms = float(np.median(data["times_ms"]))
            mem_mb = data.get("gpu_mem_mb", 0)
            print(f"{median_ms:.1f} ms, {mem_mb:.0f} MB GPU")
            results[B] = {"time_ms": median_ms, "gpu_mem_mb": mem_mb}
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return results


# ── Plot ──

def plot_scalability(all_results, output_path):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
    })

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7.16, 2.4))

    styles = {
        "Axion": {"color": "tab:blue", "marker": "o", "ls": "-", "lw": 1.5},
        "MJX": {"color": "tab:red", "marker": "s", "ls": "--", "lw": 1.5},
    }

    for name, results in all_results.items():
        if results is None:
            continue
        bs = sorted([b for b, v in results.items() if v is not None])
        times = [results[b]["time_ms"] for b in bs]
        mems = [results[b]["gpu_mem_mb"] for b in bs]
        s = styles.get(name, {"color": "gray", "marker": "^", "ls": ":", "lw": 1.0})

        ax_time.plot(bs, times, label=name, markersize=4, **s)
        ax_mem.plot(bs, mems, label=name, markersize=4, **s)

    for ax in [ax_time, ax_mem]:
        ax.set_xlabel(r"Batch size (parallel environments)")
        ax.set_xscale("log", base=2)
        used_bs = set()
        for results in all_results.values():
            if results:
                used_bs.update(b for b, v in results.items() if v is not None)
        used_bs = sorted(used_bs)
        ax.set_xticks(used_bs)
        ax.set_xticklabels([str(b) for b in used_bs])
        ax.grid(True, alpha=0.25, which="both")
        ax.tick_params(direction="in", top=True, right=True)

    ax_time.set_ylabel(r"Gradient time (ms)")
    ax_time.set_yscale("log")
    ax_time.legend(loc="upper left", framealpha=0.9)

    ax_mem.set_ylabel(r"GPU memory (MB)")
    ax_mem.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH", help="Save results JSON")
    parser.add_argument("-o", "--output", default="results/scalability_helhest.png",
                        help="Output plot path")
    parser.add_argument("--axion-only", action="store_true")
    parser.add_argument("--mjx-only", action="store_true")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,2,4,8,16,32,64,128)")
    args = parser.parse_args()

    global BATCH_SIZES
    if args.batch_sizes:
        BATCH_SIZES = [int(x) for x in args.batch_sizes.split(",")]

    all_results = {}

    if not args.mjx_only:
        print("=== Axion ===")
        all_results["Axion"] = benchmark_axion(BATCH_SIZES)

    if not args.axion_only:
        print("\n=== MJX (reverse-mode AD) ===")
        all_results["MJX"] = benchmark_mjx(BATCH_SIZES)

    if args.save:
        # Convert int keys to str for JSON
        save_data = {}
        for name, results in all_results.items():
            if results:
                save_data[name] = {str(k): v for k, v in results.items()}
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(save_data, indent=2))
        print(f"Saved results to {args.save}")

    plot_scalability(all_results, args.output)


if __name__ == "__main__":
    main()
