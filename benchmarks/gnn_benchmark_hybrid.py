#!/usr/bin/env python3

import argparse
import math
import pathlib

import h5py
import newton
import numpy as np
import torch
import warp as wp
from tqdm import tqdm

from axion import AxionEngineConfig
from axion import GNNEngineConfig
from axion import LoggingConfig
from axion.core.engine import AxionEngine
from axion.core.gnn_engine import GNNEngine
from axion.core.model_builder import AxionModelBuilder
from axion.generation import RandomSceneGenerator
from axion.simulation.dataset_simulator import random_coords_kernel
from axion.simulation.dataset_simulator import random_velocities_kernel

ROOT = pathlib.Path(__file__).parent.parent

N_PASSES = 1000
DT = 0.05
DURATION = 5.0
NUM_STEPS = int(DURATION / DT)

MAX_NR_ITERS = 16


class _ResidualTracker:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_residuals: list[float] = []

    def _save_iter_to_history(self):
        super()._save_iter_to_history()
        res_sq = float(self.data.res_norm_sq.numpy()[0])
        self._step_residuals.append(float(np.sqrt(max(res_sq, 0.0))))


class TrackingAxionEngine(_ResidualTracker, AxionEngine):
    def step(self, state_in, state_out, control, contacts, dt):
        self._step_residuals = []
        super().step(state_in, state_out, control, contacts, dt)


class TrackingGNNEngine(_ResidualTracker, GNNEngine):
    def step(self, state_in, state_out, control, contacts, dt):
        self._step_residuals = []
        super().step(state_in, state_out, control, contacts, dt)


def _pad_residuals(residuals: list[float], max_iters: int) -> np.ndarray:
    arr = np.array(residuals, dtype=np.float32)
    if len(arr) < max_iters:
        arr = np.concatenate([arr, np.full(max_iters - len(arr), arr[-1])])
    return arr[:max_iters]


@wp.kernel
def _random_joint_velocities_kernel(
    joint_qd: wp.array(dtype=float),
    ang_vel_min: float,
    ang_vel_max: float,
    seed: int,
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    joint_qd[tid] = wp.randf(rng) * (ang_vel_max - ang_vel_min) + ang_vel_min


FALL_OUTPUT = ROOT / "benchmarks/data/fall_hybrid_benchmark.h5"
FALL_GNN_MODELS = {
    "mae": str(ROOT / "data/gnn_data/fall_dataset/models/model_mae.pt"),
    "res": str(ROOT / "data/gnn_data/fall_dataset/models/model_res.pt"),
}
FALL_LIN_VEL_MIN, FALL_LIN_VEL_MAX = -3.0, 3.0
FALL_ANG_VEL_MIN, FALL_ANG_VEL_MAX = -3.14, 3.14


def build_fall_model(seed: int) -> newton.Model:
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.2
    builder.add_ground_plane()
    gen = RandomSceneGenerator(builder, seed=seed)
    gen.generate_random_object(
        pos_bounds=((-1, -1, 0.5), (1, 1, 3)),
        density_bounds=(10.0, 100.0),
        size_bounds=(0.1, 0.3),
    )
    return builder.finalize_replicated(num_worlds=1)


def init_fall_state(model: newton.Model, seed: int, axion_engine, no_log) -> newton.State:
    state = model.state()
    next_state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    resolve_constraints(model, state, next_state, control, axion_engine, DT)
    wp.launch(
        kernel=random_velocities_kernel,
        dim=model.body_count,
        inputs=[
            state.body_qd,
            FALL_LIN_VEL_MIN,
            FALL_LIN_VEL_MAX,
            FALL_ANG_VEL_MIN,
            FALL_ANG_VEL_MAX,
            seed + 100,
        ],
        device=model.device,
    )
    init_state = model.state()
    copy_state(init_state, state)
    return init_state


PENDULUM_OUTPUT = ROOT / "benchmarks/data/pendulum_hybrid_benchmark.h5"
PENDULUM_GNN_MODELS = {
    "mae": str(ROOT / "data/gnn_data/pendulum_dataset/models/model_mae.pt"),
    "res": str(ROOT / "data/gnn_data/pendulum_dataset/models/model_res.pt"),
}

_ANCHOR_POS = wp.vec3(0.0, 0.0, 5.0)
_LINK_RADIUS = 0.03
_LINK_HALF_HEIGHT = 0.4
_LINK_DENSITY = 500.0
_PEND_ANG_VEL_MIN, _PEND_ANG_VEL_MAX = -5.0, 5.0


def build_pendulum_model(seed: int) -> newton.Model:
    builder = AxionModelBuilder()
    cfg = builder.ShapeConfig(density=_LINK_DENSITY)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    axis = wp.vec3(float(np.cos(theta)), float(np.sin(theta)), 0.0)

    link1_pos = _ANCHOR_POS - wp.vec3(0.0, 0.0, _LINK_HALF_HEIGHT)
    link1 = builder.add_link(xform=wp.transform(link1_pos, wp.quat_identity()))
    builder.add_shape_capsule(link1, radius=_LINK_RADIUS, half_height=_LINK_HALF_HEIGHT, cfg=cfg)
    j1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform(_ANCHOR_POS, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, _LINK_HALF_HEIGHT), wp.quat_identity()),
        axis=axis,
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )

    link2_pos = link1_pos - wp.vec3(0.0, 0.0, 2.0 * _LINK_HALF_HEIGHT)
    link2 = builder.add_link(xform=wp.transform(link2_pos, wp.quat_identity()))
    builder.add_shape_capsule(link2, radius=_LINK_RADIUS, half_height=_LINK_HALF_HEIGHT, cfg=cfg)
    j2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -_LINK_HALF_HEIGHT), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, _LINK_HALF_HEIGHT), wp.quat_identity()),
        axis=axis,
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )
    builder.add_articulation([j1, j2])
    return builder.finalize_replicated(num_worlds=1)


def init_pendulum_state(model: newton.Model, seed: int, axion_engine, no_log) -> newton.State:
    state = model.state()
    next_state = model.state()
    control = model.control()
    wp.launch(
        kernel=random_coords_kernel,
        dim=model.joint_count,
        inputs=[
            state.joint_q,
            model.joint_type,
            model.joint_q_start,
            model.joint_limit_lower,
            model.joint_limit_upper,
            wp.vec3(-1.0, -1.0, 0.0),
            wp.vec3(1.0, 1.0, 10.0),
            seed,
        ],
        device=model.device,
    )
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    resolve_constraints(model, state, next_state, control, axion_engine, DT)
    wp.launch(
        kernel=_random_joint_velocities_kernel,
        dim=model.joint_dof_count,
        inputs=[state.joint_qd, _PEND_ANG_VEL_MIN, _PEND_ANG_VEL_MAX, seed + 100],
        device=model.device,
    )
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    init_state = model.state()
    copy_state(init_state, state)
    return init_state


def wp_to_float32(arr: wp.array, cols: int) -> np.ndarray:
    raw = arr.numpy()
    if hasattr(raw.dtype, "names") and raw.dtype.names:
        return np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(-1, cols)
    return raw.astype(np.float32).reshape(-1, cols)


def copy_state(dst: newton.State, src: newton.State) -> None:
    wp.copy(dst.body_q, src.body_q)
    wp.copy(dst.body_qd, src.body_qd)
    wp.copy(dst.joint_q, src.joint_q)
    wp.copy(dst.joint_qd, src.joint_qd)


def resolve_constraints(model, state, next_state, control, engine, dt, iters=10):
    for _ in range(iters):
        contacts = model.collide(state)
        engine.step(state, next_state, control, contacts, dt)
        wp.copy(state.body_q, next_state.body_q)
        state.body_qd.zero_()


def collect_gt_trajectory(model, init_state, engine, dt, num_steps):
    state = model.state()
    next_state = model.state()
    control = model.control()
    copy_state(state, init_state)
    trajectory = [(wp_to_float32(state.body_q, 7), wp_to_float32(state.body_qd, 6))]
    for _ in range(num_steps):
        contacts = model.collide(state)
        engine.step(state, next_state, control, contacts, dt)
        wp.copy(state.body_q, next_state.body_q)
        wp.copy(state.body_qd, next_state.body_qd)
        wp.copy(state.joint_q, next_state.joint_q)
        wp.copy(state.joint_qd, next_state.joint_qd)
        trajectory.append((wp_to_float32(state.body_q, 7), wp_to_float32(state.body_qd, 6)))
    return trajectory


def run_step_and_get_iters(
    model, engine, q: np.ndarray, qd: np.ndarray, dt: float
) -> tuple[int, np.ndarray]:
    state = model.state()
    next_state = model.state()
    control = model.control()
    wp.copy(state.body_q, wp.array(q, dtype=wp.float32, device=state.body_q.device))
    wp.copy(state.body_qd, wp.array(qd, dtype=wp.float32, device=state.body_qd.device))
    contacts = model.collide(state)
    engine.step(state, next_state, control, contacts, dt)
    iters = int(engine.data.iter_count.numpy()[0])
    residuals = _pad_residuals(engine._step_residuals, MAX_NR_ITERS)
    return iters, residuals


def run_scenario(
    scenario_name: str,
    output_path: pathlib.Path,
    gnn_models: dict,
    gnn_weights: dict,
    build_model_fn,
    init_state_fn,
    no_log,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs.update(
            dict(
                dt=DT,
                num_steps=NUM_STEPS,
                duration=DURATION,
                n_passes=N_PASSES,
                max_nr_iters=MAX_NR_ITERS,
                scenario=scenario_name,
            )
        )

        for i in tqdm(range(N_PASSES), desc=f"{scenario_name} passes"):
            seed = i
            model = build_model_fn(seed)

            axion_engine = TrackingAxionEngine(
                model=model,
                sim_steps=NUM_STEPS,
                config=AxionEngineConfig(),
                logging_config=no_log,
            )
            init_state = init_state_fn(model, seed, axion_engine, no_log)
            trajectory = collect_gt_trajectory(model, init_state, axion_engine, DT, NUM_STEPS)

            hybrid_engines = {}
            for key, path in gnn_models.items():
                eng = TrackingGNNEngine(
                    model=model,
                    sim_steps=NUM_STEPS,
                    config=GNNEngineConfig(model_path=path),
                    logging_config=no_log,
                )
                eng.gnn_model = gnn_weights[key]
                hybrid_engines[key] = eng

            axion_iters = np.empty(NUM_STEPS, dtype=np.int32)
            axion_res = np.empty((NUM_STEPS, MAX_NR_ITERS), dtype=np.float32)
            hybrid_iters = {key: np.empty(NUM_STEPS, dtype=np.int32) for key in gnn_models}
            hybrid_res = {
                key: np.empty((NUM_STEPS, MAX_NR_ITERS), dtype=np.float32) for key in gnn_models
            }

            for step, (q, qd) in enumerate(trajectory[:-1]):
                axion_iters[step], axion_res[step] = run_step_and_get_iters(
                    model, axion_engine, q, qd, DT
                )
                for key in gnn_models:
                    hybrid_iters[key][step], hybrid_res[key][step] = run_step_and_get_iters(
                        model, hybrid_engines[key], q, qd, DT
                    )

            grp = f.create_group(f"pass_{i}")
            grp.create_dataset("axion_iters", data=axion_iters, compression="gzip")
            grp.create_dataset("axion_residuals", data=axion_res, compression="gzip")
            for key in gnn_models:
                grp.create_dataset(
                    f"hybrid_{key}_iters", data=hybrid_iters[key], compression="gzip"
                )
                grp.create_dataset(
                    f"hybrid_{key}_residuals", data=hybrid_res[key], compression="gzip"
                )

    print(f"  Saved {N_PASSES} passes ({NUM_STEPS} steps each) to {output_path}")


SCENARIOS = {
    "fall": (
        FALL_OUTPUT,
        FALL_GNN_MODELS,
        build_fall_model,
        init_fall_state,
        "single_falling_object",
    ),
    "pendulum": (
        PENDULUM_OUTPUT,
        PENDULUM_GNN_MODELS,
        build_pendulum_model,
        init_pendulum_state,
        "double_pendulum",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["fall", "pendulum", "both"], default="both")
    args = parser.parse_args()

    wp.init()
    no_log = LoggingConfig(hdf5=False)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_run = (
        list(SCENARIOS.items())
        if args.scenario == "both"
        else [(args.scenario, SCENARIOS[args.scenario])]
    )

    for name, (output, gnn_model_paths, build_fn, init_fn, scenario_label) in to_run:
        print(f"\n{'='*60}\nScenario: {name.upper()}")

        gnn_weights = {}
        for key, path in gnn_model_paths.items():
            print(f"  Loading GNN model [{key}] from {path}...")
            w = torch.load(path, map_location=torch_device, weights_only=False)
            w.eval()
            gnn_weights[key] = w

        run_scenario(
            scenario_name=scenario_label,
            output_path=output,
            gnn_models=gnn_model_paths,
            gnn_weights=gnn_weights,
            build_model_fn=build_fn,
            init_state_fn=init_fn,
            no_log=no_log,
        )


if __name__ == "__main__":
    main()
