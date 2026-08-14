"""MJX box scalability benchmark WITH gradient checkpointing (Reviewer 2, point 1).

Same task/model/loss as mjx_sim.py, plus a --checkpoint flag:
  none  -- plain BPTT via jax.lax.scan (paper baseline, identical to mjx_sim.py)
  step  -- jax.checkpoint around each mjx.step: intra-step activations (contact
           solver inner iterations) are discarded and recomputed in the backward
           pass; the scan still stores one mjx.Data carry per step. ~2x backward
           compute.
  sqrt  -- two-level scan: T steps split into ~sqrt(T) blocks, jax.checkpoint on
           the block AND on each step inside it. Stores only block-boundary
           carries during the forward pass; each block's carries are
           rematerialized during backward. ~3x compute, minimal memory.

Usage:
    python experiments/4_scalability_box/mjx_sim_ckpt.py --num-worlds 4 \
        --checkpoint step --save experiments/4_scalability_box/results/mjx_ckpt_step_4.json
"""
import argparse
import json
import math
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "1_sim_to_real_box"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "3_gradient_quality_box"))

os.environ.setdefault("DISPLAY", ":1")
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # essential for NVML poller

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

# The params import chain (sweep_mujoco -> common_box -> replay_real) pulls in
# newton/warp, which are absent from the JAX env. common_box only imports three
# names from replay_real and none are used on this code path — stub the module.
try:
    import examples.helhest_junior.replay_real  # noqa: F401
except ModuleNotFoundError:
    import types
    for _name in ("examples", "examples.helhest_junior"):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    _stub = types.ModuleType("examples.helhest_junior.replay_real")
    _stub.PRISM_OFFSET = None
    _stub.best_time_shift = None
    _stub.prism_track = None
    sys.modules["examples.helhest_junior.replay_real"] = _stub

from sweep_mujoco import BASE_PARAMS, JUNIOR_BOX_XML  # noqa: E402
from optimize_mjx import MJX_PARAMS, _patch_wheels_for_mjx  # noqa: E402

K = 10
DT = 5e-3
DURATION = 6.0
T = int(DURATION / DT)
ITERATIONS = 5


def make_interp_matrix(T, K):
    W = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        x = t * (K - 1) / max(T - 1, 1)
        lo = int(x); hi = min(lo + 1, K - 1); a = x - lo
        W[t, lo] += 1.0 - a
        W[t, hi] += a
    return W


def build_mjx_model(gt_box):
    box = gt_box
    fmt = {**BASE_PARAMS, **MJX_PARAMS, "dt": DT,
           "ground_friction": MJX_PARAMS["mu"], "box_friction": MJX_PARAMS["mu"],
           "front_friction": MJX_PARAMS["mu"], "rear_friction": MJX_PARAMS["mu"],
           "ground_torsional": MJX_PARAMS["tor"], "front_torsional": MJX_PARAMS["tor"],
           "rear_torsional": MJX_PARAMS["tor"],
           "box_x": box["center"][0], "box_y": box["center"][1], "box_z": box["center"][2],
           "box_hx": box["half_extents"][0], "box_hy": box["half_extents"][1],
           "box_hz": box["half_extents"][2]}
    xml = JUNIOR_BOX_XML.format(**fmt)
    xml = _patch_wheels_for_mjx(xml)
    mj_model = mujoco.MjModel.from_xml_string(xml)
    return mjx.put_model(mj_model), mj_model


def _block_len(T):
    """Largest divisor of T closest to sqrt(T) (T=1200 -> 40)."""
    root = int(math.sqrt(T))
    for d in range(root, 0, -1):
        if T % d == 0:
            return T // d if T // d <= root * 2 else d
    return root


def make_rollout(mx, checkpoint):
    def step_fn(d, ctrl_t):
        d = d.replace(ctrl=ctrl_t)
        d = mjx.step(mx, d)
        return d, d.qpos[:2]

    if checkpoint == "none":
        def rollout(dx0, ctrl_traj):
            _, xy = jax.lax.scan(step_fn, dx0, ctrl_traj)
            return xy
    elif checkpoint == "step":
        ckpt_step = jax.checkpoint(step_fn)
        def rollout(dx0, ctrl_traj):
            _, xy = jax.lax.scan(ckpt_step, dx0, ctrl_traj)
            return xy
    elif checkpoint == "sqrt":
        L = _block_len(T)          # steps per block
        S = T // L                 # number of blocks
        ckpt_step = jax.checkpoint(step_fn)

        def block_fn(d, ctrl_block):
            d, xy = jax.lax.scan(ckpt_step, d, ctrl_block)
            return d, xy
        ckpt_block = jax.checkpoint(block_fn)

        def rollout(dx0, ctrl_traj):
            blocks = ctrl_traj.reshape(S, L, ctrl_traj.shape[-1])
            _, xy = jax.lax.scan(ckpt_block, dx0, blocks)
            return xy.reshape(T, 2)
        print(f"sqrt checkpointing: {S} blocks x {L} steps")
    else:
        raise ValueError(checkpoint)
    return rollout


def _nvml_poller():
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    baseline = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
    peak = [baseline]; stop = threading.Event()

    def _poll():
        while not stop.is_set():
            used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
            if used > peak[0]:
                peak[0] = used
            time.sleep(0.01)

    th = threading.Thread(target=_poll, daemon=True); th.start()

    def finalize():
        stop.set(); th.join(timeout=1.0)
        return float(peak[0]), float(peak[0] - baseline)

    return finalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-worlds", type=int, default=1)
    ap.add_argument("--checkpoint", choices=["none", "step", "sqrt"], default="step")
    ap.add_argument("--gt", default=str(pathlib.Path(__file__).resolve().parents[1]
                                         / "1_sim_to_real_box" / "data"
                                         / "run_2026_05_20-18_10_33.json"))
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    num_worlds = args.num_worlds

    with open(args.gt) as f:
        gt = json.load(f)
    real_t = np.asarray(gt["real"]["t"])
    real_xy = np.column_stack([gt["real"]["x"], gt["real"]["y"]])
    mask = (real_t >= 0) & (real_t <= DURATION)
    real_t = real_t[mask]; real_xy = real_xy[mask]
    print(f"T={T}, dt={DT}, K={K}, num_worlds={num_worlds}, "
          f"checkpoint={args.checkpoint}, horizon={DURATION}s")
    print(f"JAX devices: {jax.devices()}")

    mx, mj_model = build_mjx_model(gt["box"])
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    dx0 = mjx.put_data(mj_model, mj_data)

    sim_origin = np.asarray(dx0.qpos[:2])
    t_grid = np.arange(T) * DT
    target_xy = np.zeros((T, 2), dtype=np.float32)
    for c in range(2):
        target_xy[:, c] = np.interp(t_grid, real_t, real_xy[:, c])
    target_xy += sim_origin.astype(np.float32)
    target_xy = jnp.asarray(target_xy)

    W = jnp.asarray(make_interp_matrix(T, K))
    rollout_one = make_rollout(mx, args.checkpoint)

    dx_batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_worlds), dx0)
    rollout_batch = jax.vmap(rollout_one, in_axes=(0, None))

    def trajectory_loss(params):
        ctrl_traj = W @ params                              # [T, 3]
        xy_batch = rollout_batch(dx_batch, ctrl_traj)       # [W, T, 2]
        delta = xy_batch - target_xy[None, :, :]
        return jnp.mean(jnp.sum(delta**2, axis=-1))

    value_and_grad = jax.jit(jax.value_and_grad(trajectory_loss))
    params = jnp.tile(jnp.array([2.0, 2.0, 2.0]), (K, 1))

    finalize_nvml = _nvml_poller()

    print("Compiling...")
    t_compile = time.perf_counter()
    loss, grad = value_and_grad(params)
    loss.block_until_ready(); grad.block_until_ready()
    compile_s = time.perf_counter() - t_compile
    print(f"  compile: {compile_s:.1f}s")

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(params)
    device = jax.devices()[0]

    peak_mem_mb = 0.0
    time_ms_list = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        loss, grad = value_and_grad(params)
        loss.block_until_ready(); grad.block_until_ready()
        t_iter = (time.perf_counter() - t0) * 1000
        mem_stats = device.memory_stats()
        used_mb = mem_stats.get("peak_bytes_in_use", 0) / 1024**2
        peak_mem_mb = max(peak_mem_mb, used_mb)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"  iter {i:3d}: loss={float(loss):.4f} | t={t_iter:.0f}ms | mem={used_mb:.0f}MB",
              flush=True)
        time_ms_list.append(t_iter)

    nvml_abs, nvml_delta = (finalize_nvml() if finalize_nvml is not None else (None, None))

    results = {
        "simulator": f"MJX-grad-ckpt-{args.checkpoint}",
        "checkpoint": args.checkpoint,
        "num_worlds": num_worlds,
        "median_time_ms": (float(np.median(time_ms_list[3:]))
                           if len(time_ms_list) > 3 else float(np.median(time_ms_list))),
        "peak_gpu_mb": peak_mem_mb,
        "peak_gpu_mb_nvml_absolute": nvml_abs,
        "peak_gpu_mb_nvml": nvml_delta,
        "compile_s": compile_s,
        "time_ms": time_ms_list,
        "K": K, "dt": DT, "duration_s": DURATION, "iterations": ITERATIONS,
    }
    if nvml_abs is not None:
        print(f"NVML peak: {nvml_abs:.0f} MB (delta {nvml_delta:.0f} MB)")
    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
