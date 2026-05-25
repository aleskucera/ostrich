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
from axion.core.model_builder import AxionModelBuilder
from axion.generation import RandomSceneGenerator
from axion.simulation.dataset_simulator import random_velocities_kernel

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = ROOT / "benchmarks/data/fall_benchmark.h5"
GNN_MODELS = {
    "mae": str(ROOT / "data/gnn_data/fall_dataset/models/model_mae.pt"),
    "res": str(ROOT / "data/gnn_data/fall_dataset/models/model_res.pt"),
}

N_PASSES = 1000
DT = 0.05
DURATION = 5.0
NUM_STEPS = int(DURATION / DT)

LIN_VEL_MIN, LIN_VEL_MAX = -3.0, 3.0
ANG_VEL_MIN, ANG_VEL_MAX = -3.14, 3.14


def build_model(seed: int) -> newton.Model:
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


def run_sim(model, init_state, engine, dt, num_steps) -> tuple[np.ndarray, np.ndarray]:
    state = model.state()
    next_state = model.state()
    control = model.control()
    copy_state(state, init_state)

    n = model.body_count
    q_traj = np.empty((num_steps + 1, n, 7), dtype=np.float32)
    qd_traj = np.empty((num_steps + 1, n, 6), dtype=np.float32)

    q_traj[0] = wp_to_float32(state.body_q, 7)
    qd_traj[0] = wp_to_float32(state.body_qd, 6)

    for step in range(num_steps):
        contacts = model.collide(state)
        engine.step(state, next_state, control, contacts, dt)
        wp.copy(state.body_q, next_state.body_q)
        wp.copy(state.body_qd, next_state.body_qd)
        wp.copy(state.joint_q, next_state.joint_q)
        wp.copy(state.joint_qd, next_state.joint_qd)
        q_traj[step + 1] = wp_to_float32(state.body_q, 7)
        qd_traj[step + 1] = wp_to_float32(state.body_qd, 6)

    return q_traj, qd_traj


def main():
    wp.init()
    no_log = LoggingConfig(enable_hdf5_logging=False)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_weights = {}
    for key, path in GNN_MODELS.items():
        print(f"Loading GNN model [{key}] from {path}...")
        w = torch.load(path, map_location=torch_device, weights_only=False)
        w.eval()
        gnn_weights[key] = w

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT, "w") as f:
        f.attrs.update(
            dict(
                dt=DT,
                num_steps=NUM_STEPS,
                duration=DURATION,
                n_passes=N_PASSES,
                scenario="single_falling_object",
                body_q_layout="[px, py, pz, qx, qy, qz, qw]",
                body_qd_layout="[wx, wy, wz, vx, vy, vz]",
            )
        )

        for i in tqdm(range(N_PASSES), desc="Passes"):
            seed = i
            model = build_model(seed=seed)

            state = model.state()
            next_state = model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)

            axion_engine = AxionEngineConfig().create_engine(
                model, sim_steps=NUM_STEPS, logging_config=no_log
            )
            resolve_constraints(model, state, next_state, control, axion_engine, DT)
            wp.launch(
                kernel=random_velocities_kernel,
                dim=model.body_count,
                inputs=[
                    state.body_qd,
                    LIN_VEL_MIN,
                    LIN_VEL_MAX,
                    ANG_VEL_MIN,
                    ANG_VEL_MAX,
                    seed + 100,
                ],
                device=model.device,
            )

            init_state = model.state()
            copy_state(init_state, state)

            axion_q, axion_qd = run_sim(model, init_state, axion_engine, DT, NUM_STEPS)

            grp = f.create_group(f"pass_{i}")
            ag = grp.create_group("axion")
            ag.create_dataset("body_q", data=axion_q, compression="gzip")
            ag.create_dataset("body_qd", data=axion_qd, compression="gzip")

            for key, path in GNN_MODELS.items():
                gnn_engine = GNNEngineConfig(model_path=path).create_engine(
                    model, sim_steps=NUM_STEPS, logging_config=no_log
                )
                gnn_engine.gnn_model = gnn_weights[key]
                gnn_q, gnn_qd = run_sim(model, init_state, gnn_engine, DT, NUM_STEPS)
                gg = grp.create_group(f"gnn_{key}")
                gg.create_dataset("body_q", data=gnn_q, compression="gzip")
                gg.create_dataset("body_qd", data=gnn_qd, compression="gzip")

    print(f"\nSaved {N_PASSES} passes ({NUM_STEPS+1} steps each) to {OUTPUT}")


if __name__ == "__main__":
    main()
