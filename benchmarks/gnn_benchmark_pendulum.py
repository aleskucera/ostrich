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
from axion.core.model_builder import AxionModelBuilder
from axion.simulation.dataset_simulator import random_coords_kernel

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = ROOT / "benchmarks/data/pendulum_benchmark.h5"
GNN_MODELS = {
    "mae": str(ROOT / "data/gnn_data/pendulum_dataset/models/model_mae.pt"),
    "res": str(ROOT / "data/gnn_data/pendulum_dataset/models/model_res.pt"),
}

N_PASSES = 1000
DT = 0.05
DURATION = 5.0
NUM_STEPS = int(DURATION / DT)

ANCHOR_POS = wp.vec3(0.0, 0.0, 5.0)
LINK_RADIUS = 0.03
LINK_HALF_HEIGHT = 0.4
LINK_DENSITY = 500.0
ANG_VEL_MIN, ANG_VEL_MAX = -5.0, 5.0


@wp.kernel
def random_joint_velocities_kernel(
    joint_qd: wp.array(dtype=float),
    ang_vel_min: float,
    ang_vel_max: float,
    seed: int,
):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    joint_qd[tid] = wp.randf(rng) * (ang_vel_max - ang_vel_min) + ang_vel_min


def build_model(seed: int) -> newton.Model:
    builder = AxionModelBuilder()
    cfg = builder.ShapeConfig(density=LINK_DENSITY)

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    axis = wp.vec3(float(np.cos(theta)), float(np.sin(theta)), 0.0)

    link1_pos = ANCHOR_POS - wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT)
    link1 = builder.add_link(xform=wp.transform(link1_pos, wp.quat_identity()))
    builder.add_shape_capsule(link1, radius=LINK_RADIUS, half_height=LINK_HALF_HEIGHT, cfg=cfg)
    j1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform(ANCHOR_POS, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT), wp.quat_identity()),
        axis=axis,
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )

    link2_pos = link1_pos - wp.vec3(0.0, 0.0, 2.0 * LINK_HALF_HEIGHT)
    link2 = builder.add_link(xform=wp.transform(link2_pos, wp.quat_identity()))
    builder.add_shape_capsule(link2, radius=LINK_RADIUS, half_height=LINK_HALF_HEIGHT, cfg=cfg)
    j2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF_HEIGHT), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF_HEIGHT), wp.quat_identity()),
        axis=axis,
        limit_lower=-math.pi,
        limit_upper=math.pi,
    )

    builder.add_articulation([j1, j2])
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
                scenario="double_pendulum",
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

            axion_engine = AxionEngineConfig().create_engine(
                model, sim_steps=NUM_STEPS, logging_config=no_log
            )
            resolve_constraints(model, state, next_state, control, axion_engine, DT)

            wp.launch(
                kernel=random_joint_velocities_kernel,
                dim=model.joint_dof_count,
                inputs=[state.joint_qd, ANG_VEL_MIN, ANG_VEL_MAX, seed + 100],
                device=model.device,
            )
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

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
