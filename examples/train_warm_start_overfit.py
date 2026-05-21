"""Overfit a WarmStartNet on a single timestep.

Phase 1: Supervised pretraining — MSE loss against converged (v*, λ*)
Phase 2: Fine-tune with ||residual||^2 loss

Usage:
    python examples/train_warm_start_overfit.py rendering=headless execution.use_cuda_graph=false
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
PRETRAIN_EPOCHS = 500
FINETUNE_EPOCHS = 500
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-3
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
        sim_steps=WARMUP_STEPS + 10,
        logging_config=LoggingConfig(),
    )
    dims = engine.dims
    torch_device = wp.device_to_torch(engine.data.device)

    state_cur = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_cur)

    print(f"Scene: {dims.body_count} bodies, {dims.num_constraints} constraints")

    # === Warm up physics ===
    print(f"Warming up for {WARMUP_STEPS} steps...")
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

    # === Snapshot one timestep ===
    contacts = model.collide(state_cur)
    engine.load_data(state_cur, control, contacts, dt)
    state_tensor = get_state_tensor(engine)
    saved_q = state_cur.body_q.numpy().copy()
    saved_qd = state_cur.body_qd.numpy().copy()

    # Solve to get target (v*, λ*)
    wp.copy(dest=engine.data.body_pose, src=state_cur.body_q)
    wp.copy(dest=engine.data.body_vel, src=state_cur.body_qd)
    engine.data._constr_force.zero_()
    engine.data._constr_force_prev_iter.zero_()
    engine._solve()
    default_iters = engine.data.iter_count.numpy()[0]
    default_res = wp.to_torch(engine.data.res_norm_sq).sum().item()

    target_vel = wp.to_torch(engine.data.body_vel).reshape(dims.num_worlds, dims.N_u).clone()
    target_cf = wp.to_torch(engine.data._constr_force).clone()
    print(f"Default solver: {default_iters} iters, ||r||^2 = {default_res:.4e}")
    print(f"  target vel norm: {torch.norm(target_vel).item():.4f}")
    print(f"  target cf norm:  {torch.norm(target_cf).item():.4f}")

    # === Build NN ===
    net = WarmStartNet(dims, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS).to(torch_device)
    num_params = sum(p.numel() for p in net.parameters())
    print(f"\nNN: {NUM_HIDDEN_LAYERS} hidden layers, {HIDDEN_DIM} wide, {num_params:,} parameters")

    # === Phase 1: Supervised pretraining ===
    print(f"\n--- Phase 1: Supervised pretraining ({PRETRAIN_EPOCHS} epochs, lr={PRETRAIN_LR}) ---\n")
    optimizer = torch.optim.Adam(net.parameters(), lr=PRETRAIN_LR)

    for epoch in range(PRETRAIN_EPOCHS):
        optimizer.zero_grad()
        pred_vel, pred_cf = net(state_tensor)
        loss_vel = torch.sum((pred_vel - target_vel) ** 2)
        loss_cf = torch.sum((pred_cf - target_cf) ** 2)
        loss = loss_vel + loss_cf
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            print(f"  epoch {epoch:4d} | MSE: {loss.item():.4e} "
                  f"(vel: {loss_vel.item():.4e}, cf: {loss_cf.item():.4e})")

    # Check residual after pretraining
    wp.copy(dest=state_cur.body_q, src=wp.from_numpy(saved_q, dtype=wp.transform))
    wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(saved_qd, dtype=wp.spatial_vector))
    engine.load_data(state_cur, control, contacts, dt)

    with torch.no_grad():
        pred_vel, pred_cf = net(state_tensor)
    pretrain_iters, pretrain_res = count_solver_iters(engine, dims, pred_vel, pred_cf)
    print(f"\n  After pretraining: {pretrain_iters} iters, ||r||^2 = {pretrain_res:.4e}")

    # === Phase 2: Fine-tune with residual loss ===
    print(f"\n--- Phase 2: Residual fine-tuning ({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR}) ---\n")
    optimizer = torch.optim.Adam(net.parameters(), lr=FINETUNE_LR)

    for epoch in range(FINETUNE_EPOCHS):
        wp.copy(dest=state_cur.body_q, src=wp.from_numpy(saved_q, dtype=wp.transform))
        wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(saved_qd, dtype=wp.spatial_vector))
        engine.load_data(state_cur, control, contacts, dt)

        optimizer.zero_grad()
        body_vel, constr_force = net(state_tensor)
        residual = AxionResidualAD.apply(
            engine.axion_model, engine.axion_contacts, engine.data,
            engine.config, engine.dims, body_vel, constr_force,
        )
        loss = torch.sum(residual ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()

        if epoch % 50 == 0 or epoch == FINETUNE_EPOCHS - 1:
            print(f"  epoch {epoch:4d} | residual loss: {loss.item():.4e}")

    # === Evaluate ===
    print(f"\n{'='*50}")
    print("Evaluation")
    print(f"{'='*50}")

    wp.copy(dest=state_cur.body_q, src=wp.from_numpy(saved_q, dtype=wp.transform))
    wp.copy(dest=state_cur.body_qd, src=wp.from_numpy(saved_qd, dtype=wp.spatial_vector))
    engine.load_data(state_cur, control, contacts, dt)

    net.eval()
    with torch.no_grad():
        pred_vel, pred_cf = net(state_tensor)

    nn_iters, nn_res = count_solver_iters(engine, dims, pred_vel, pred_cf)
    saved = default_iters - nn_iters

    print(f"  Default:       {default_iters} iters, ||r||^2 = {default_res:.4e}")
    print(f"  After pretrain:{pretrain_iters} iters, ||r||^2 = {pretrain_res:.4e}")
    print(f"  After finetune:{nn_iters} iters, ||r||^2 = {nn_res:.4e}")
    print(f"  Saved:         {saved:+d} iterations")


if __name__ == "__main__":
    main()
