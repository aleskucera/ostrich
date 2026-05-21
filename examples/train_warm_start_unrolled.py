"""Warm-start training with direct residual loss (Option B, fixed gradients).

Trains a WarmStartNet on multiple timesteps using ||residual||^2 loss
with exact gradients via AxionResidualAD (with corrected friction gradient).

Usage:
    python examples/train_warm_start_unrolled.py rendering=headless execution.use_cuda_graph=false
"""
import pathlib

import hydra
import newton
import numpy as np
import torch
import warp as wp
from axion import AxionEngine
from axion import EngineConfig
from axion import LoggingConfig
from axion import SimulationConfig
from axion.core.model_builder import AxionModelBuilder
from axion.generation import RandomSceneGenerator
from axion.learning.torch_residual_ad import AxionResidualAD
from axion.learning.warm_start_net import WarmStartNet
from omegaconf import DictConfig

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")

WARMUP_STEPS = 15
NUM_TRAIN_STEPS = 50
NUM_EPOCHS = 500
LR = 1e-2
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 3
GRAD_CLIP = 1.0
SEED = 42


def build_random_model(num_worlds: int = 1, seed: int = SEED) -> newton.Model:
    builder = AxionModelBuilder()
    builder.rigid_gap = 0.2
    builder.add_ground_plane()

    gen = RandomSceneGenerator(builder, seed=seed)
    np.random.seed(seed)
    num_objects = np.random.randint(3, 8)
    for _ in range(num_objects):
        gen.generate_random_object(
            pos_bounds=((-1, -1, 0.3), (1, 1, 2.0)),
            density_bounds=(10.0, 100.0),
            size_bounds=(0.1, 0.3),
        )

    return builder.finalize_replicated(num_worlds=num_worlds)


def get_state_tensor(engine: AxionEngine) -> torch.Tensor:
    pose = wp.to_torch(engine.data.body_pose_prev).clone()
    vel = wp.to_torch(engine.data.body_vel_prev).clone()
    pose_flat = pose.reshape(engine.dims.num_worlds, -1).float()
    vel_flat = vel.reshape(engine.dims.num_worlds, -1).float()
    return torch.cat([pose_flat, vel_flat], dim=-1)


def count_solver_iters(engine, dims, init_vel, init_cf):
    vel_wp = wp.from_torch(
        init_vel.detach().reshape(dims.num_worlds, dims.body_count, 6).contiguous(),
        dtype=wp.spatial_vector,
    )
    wp.copy(dest=engine.data.body_vel, src=vel_wp)

    cf_wp = wp.from_torch(init_cf.detach().contiguous())
    wp.copy(dest=engine.data._constr_force, src=cf_wp)
    wp.copy(dest=engine.data._constr_force_prev_iter, src=cf_wp)

    engine._solve()

    iters = engine.data.iter_count.numpy()[0]
    res_sq = wp.to_torch(engine.data.res_norm_sq).sum().item()
    return iters, res_sq


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def main(cfg: DictConfig):
    wp.init()

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    model = build_random_model(num_worlds=sim_config.num_worlds, seed=SEED)
    dt = sim_config.target_timestep_seconds

    engine = engine_config.create_engine(
        model=model,
        sim_steps=WARMUP_STEPS + NUM_TRAIN_STEPS + 1,
        logging_config=LoggingConfig(),
    )
    dims = engine.dims
    torch_device = wp.device_to_torch(engine.data.device)

    state_cur = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_cur)

    print(f"Scene: {dims.body_count} bodies, {dims.num_constraints} constraints")
    print(f"NN input: {7 * dims.body_count + 6 * dims.body_count}, "
          f"output: {dims.N_u + dims.num_constraints}")

    # === Phase 1: Warm up physics ===
    print(f"\nWarming up for {WARMUP_STEPS} steps...")
    for step in range(WARMUP_STEPS):
        contacts = model.collide(state_cur)
        engine.load_data(state_cur, control, contacts, dt)
        wp.copy(dest=engine.data.body_pose, src=state_cur.body_q)
        wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
        engine.data._constr_force.zero_()
        engine.data._constr_force_prev_iter.zero_()
        engine._solve()
        wp.copy(dest=state_next.body_q, src=engine.data.body_pose)
        wp.copy(dest=state_next.body_qd, src=engine.data.body_vel)
        state_cur, state_next = state_next, state_cur

    # === Phase 2: Collect timestep snapshots for training ===
    print(f"Collecting {NUM_TRAIN_STEPS} timesteps...")
    timestep_data = []

    for step in range(NUM_TRAIN_STEPS):
        contacts = model.collide(state_cur)
        engine.load_data(state_cur, control, contacts, dt)
        state_tensor = get_state_tensor(engine)

        wp.copy(dest=engine.data.body_pose, src=state_cur.body_q)
        wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
        engine.data._constr_force.zero_()
        engine.data._constr_force_prev_iter.zero_()
        engine._solve()
        default_iters = engine.data.iter_count.numpy()[0]
        default_res = wp.to_torch(engine.data.res_norm_sq).sum().item()

        timestep_data.append({
            "state_tensor": state_tensor,
            "contacts": contacts,
            "state_q": state_cur.body_q.numpy().copy(),
            "state_qd": state_cur.body_qd.numpy().copy(),
            "default_iters": default_iters,
            "default_res": default_res,
        })

        if step % 10 == 0:
            print(f"  step {step:3d} | default: {default_iters} iters, ||r||^2: {default_res:.4e}")

        wp.copy(dest=state_next.body_q, src=engine.data.body_pose)
        wp.copy(dest=state_next.body_qd, src=engine.data.body_vel)
        state_cur, state_next = state_next, state_cur

    avg_default = np.mean([t["default_iters"] for t in timestep_data])
    print(f"\nDefault warm-start: avg {avg_default:.1f} iters/step")

    # === Phase 3: Train NN ===
    net = WarmStartNet(dims, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS).to(torch_device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"\nNN: {NUM_HIDDEN_LAYERS} hidden layers, {HIDDEN_DIM} wide, {num_params:,} parameters")
    print(f"Training for {NUM_EPOCHS} epochs, Adam lr={LR}\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for t in timestep_data:
            wp.copy(dest=state_cur.body_q, src=wp.from_numpy(t["state_q"], dtype=wp.transform))
            wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(t["state_qd"], dtype=wp.spatial_vector))
            engine.load_data(state_cur, control, t["contacts"], dt)

            body_vel, constr_force = net(t["state_tensor"])
            residual = AxionResidualAD.apply(
                engine.axion_model, engine.axion_contacts, engine.data,
                engine.config, engine.dims, body_vel, constr_force,
            )
            loss = torch.sum(residual ** 2)
            loss.backward()
            epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % 20 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  epoch {epoch:4d} | avg loss: {epoch_loss / NUM_TRAIN_STEPS:.4e}")

    # === Phase 4: Evaluate ===
    print(f"\n{'='*60}")
    print("Evaluation: NN warm-start vs default (v=v_prev, cf=0)")
    print(f"{'='*60}")

    net.eval()
    nn_total_iters = 0

    for step_idx, t in enumerate(timestep_data):
        wp.copy(dest=state_cur.body_q, src=wp.from_numpy(t["state_q"], dtype=wp.transform))
        wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(t["state_qd"], dtype=wp.spatial_vector))
        engine.load_data(state_cur, control, t["contacts"], dt)

        with torch.no_grad():
            pred_vel, pred_cf = net(t["state_tensor"])

        nn_iters, nn_res = count_solver_iters(engine, dims, pred_vel, pred_cf)
        nn_total_iters += nn_iters

        if step_idx % 10 == 0:
            saved = t["default_iters"] - nn_iters
            print(f"  step {step_idx:3d} | default: {t['default_iters']} iters"
                  f" | NN: {nn_iters} iters (||r||^2={nn_res:.4e})"
                  f" | saved: {saved:+d}")

    avg_nn = nn_total_iters / NUM_TRAIN_STEPS
    print(f"\n  Average: default={avg_default:.1f}, NN={avg_nn:.1f}, "
          f"saved={avg_default - avg_nn:.1f} ({(avg_default - avg_nn) / avg_default * 100:.0f}%)")


if __name__ == "__main__":
    main()
