"""Replay a trained PPO policy in the Axion GL viewer (or write to USD).

Loads a checkpoint from ``examples.helhest.balance_ppo.train`` and runs ONE
deterministic rollout (mean action, no exploration noise), then opens the
real Newton/Axion viewer to play it back. Use this to actually *see* what the
policy is doing — the matplotlib video logged to wandb is a crude side-view
sanity check, not a debugging tool.

Examples:
    python -m examples.helhest.balance_ppo.replay runs/eager-sponge-1/best.pt
    python -m examples.helhest.balance_ppo.replay runs/eager-sponge-1/best.pt --num-worlds 4 --loops 10
    python -m examples.helhest.balance_ppo.replay runs/eager-sponge-1/best.pt --vis usd --usd-path /tmp/replay.usd
"""
from __future__ import annotations

import argparse
import os
import pathlib

import newton
import torch
import warp as wp

from axion import (
    AxionEngineConfig,
    ComplianceConfig,
    ContactsConfig,
    LinearSolverConfig,
    LinesearchConfig,
    LoggingConfig,
    NewtonRaphsonConfig,
    RenderingConfig,
    SimulationConfig,
)
from examples.helhest.balance_ppo.ppo_trainer import PPOConfig
from examples.helhest.balance_ppo.train import (
    HelhestBalancePPO,
    _make_default_engine_config,
)

os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=str,
                    help="Path to a .pt file saved by train.py (best.pt / final.pt / latest.pt).")
    ap.add_argument("--num-worlds", type=int, default=1,
                    help="How many parallel rollouts to play back side-by-side.")
    ap.add_argument("--vis", choices=["gl", "usd"], default="gl")
    ap.add_argument("--usd-path", type=str, default="replay.usd",
                    help="Output USD path when --vis usd.")
    ap.add_argument("--loops", type=int, default=5,
                    help="How many times to replay the rollout in the viewer (GL only).")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (1.0 = realtime).")
    ap.add_argument("--hidden", type=int, default=64,
                    help="Must match training run's --hidden.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample from policy instead of using the mean (default deterministic).")
    ap.add_argument("--start-paused", action="store_true",
                    help="GL viewer starts paused — press space to play.")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    sim_config = SimulationConfig(
        duration_seconds=4.0,
        target_timestep_seconds=5e-2,
        num_worlds=args.num_worlds,
    )
    render_config = RenderingConfig(
        vis_type=args.vis,
        usd_file=(args.usd_path if args.vis == "usd" else None),
        target_fps=30,
        world_offset_x=4.0, world_offset_y=4.0,
    )

    sim = HelhestBalancePPO(
        sim_config, render_config,
        _make_default_engine_config(), LoggingConfig(),
        policy_hidden=args.hidden,
        ppo_cfg=PPOConfig(),
        wandb_run=None,
        video_every=0,
        checkpoint_dir=None,
    )

    ckpt = torch.load(ckpt_path, map_location=sim.torch_device, weights_only=False)
    sim.policy.load_state_dict(ckpt["policy_state_dict"])
    sim.policy.eval()
    iter_loaded = int(ckpt.get("iter", -1))
    best_alive = float(ckpt.get("best_alive_frac", float("nan")))
    print(f"[replay] loaded {ckpt_path.name}: iter={iter_loaded}, "
          f"best_alive_frac={best_alive:.3f}", flush=True)

    # --- Deterministic rollout (no autograd, no stochastic noise) ---
    newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd, sim.states[0])
    alive_count = torch.zeros(sim.N, device=sim.torch_device)

    with torch.no_grad():
        for t in range(sim.T):
            obs, _, _ = sim._state_features(sim.states[t])
            if args.stochastic:
                raw_action, _, _ = sim.policy.act(obs)
            else:
                raw_action = sim.policy.act_deterministic(obs)

            sim._write_action(sim.controls[t], raw_action)
            sim.collision_pipeline.collide(sim.states[t], sim.contacts)
            sim.solver.step(
                state_in=sim.states[t],
                state_out=sim.states[t + 1],
                control=sim.controls[t],
                contacts=sim.contacts,
                dt=sim.dt,
            )

            _, _, cos_err = sim._state_features(sim.states[t + 1])
            alive_count += (cos_err > sim.w.alive_threshold).float()

    wp.synchronize()
    alive_frac = float(alive_count.mean() / sim.T)
    print(f"[replay] rollout finished: alive_frac={alive_frac * 100:.1f}% "
          f"(over {sim.N} world{'s' if sim.N > 1 else ''}, {sim.T} steps)", flush=True)

    # --- Replay in viewer ---
    if args.vis == "gl":
        print(f"[replay] opening GL viewer (looping {args.loops}× at {args.speed}× speed)...",
              flush=True)
        sim.render_episode(
            iteration=iter_loaded,
            loop=True,
            loops_count=args.loops,
            playback_speed=args.speed,
            start_paused=args.start_paused,
        )
    else:
        print(f"[replay] writing USD to {args.usd_path}...", flush=True)
        sim.render_episode(iteration=iter_loaded, loop=False, loops_count=1,
                           playback_speed=args.speed)

    sim.close()


if __name__ == "__main__":
    main()
