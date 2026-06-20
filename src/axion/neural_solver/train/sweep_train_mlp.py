import argparse
from copy import deepcopy
from pathlib import Path

import wandb

from axion.neural_solver.algorithms.simple_mlp_trainer import SimpleMlpTrainer
from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.train.config_builder import build_cli_cfg, load_default_cfg
from axion.neural_solver.train.sweep_configurations import sweep_config_mlp_0
from axion.neural_solver.utils.python_utils import set_random_seed


def _parse_args():
    p = argparse.ArgumentParser(description="Run W&B sweep for Simple MLP architecture.")
    p.add_argument("--cfg", required=True, help="Path to base YAML config (defaults)")
    p.add_argument("--logdir", required=True, help="Base directory for checkpoints/logs")
    p.add_argument("--checkpoint", default=None, help="Checkpoint to restore")
    p.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0)")
    p.add_argument("--sweep_id", type=str, default=None, help="Existing sweep id to join")
    p.add_argument("--project", type=str, default="neural-solver-mlp", help="W&B project")
    p.add_argument("--count", type=int, default=None, help="Max runs for this agent")
    return p.parse_args()


def _apply_mlp_sweep_overrides(cfg: dict, sweep: dict) -> dict:
    out = deepcopy(cfg)
    network_cfg = out.setdefault("network", {})
    mlp_cfg = network_cfg.setdefault("model", {}).setdefault("mlp", {})

    if "depth" in sweep and sweep["depth"] is not None:
        network_cfg["depth"] = int(sweep["depth"])
    if "width" in sweep and sweep["width"] is not None:
        network_cfg["width"] = int(sweep["width"])
    if "layernorm" in sweep and sweep["layernorm"] is not None:
        mlp_cfg["layernorm"] = bool(sweep["layernorm"])
    if "activation" in sweep and sweep["activation"] is not None:
        mlp_cfg["activation"] = str(sweep["activation"])

    return out


def _make_sweep_train_fn(args):
    def _run_one_sweep_iteration():
        run = wandb.init()
        try:
            cfg = load_default_cfg(args.cfg)
            cfg = _apply_mlp_sweep_overrides(cfg, dict(wandb.config))

            seed = cfg["algorithm"].get("seed", 0)
            set_random_seed(seed)
            cfg["algorithm"]["seed"] = seed

            run_id = run.id if run is not None else "unknown-run"
            run_logdir = Path(args.logdir) / run_id
            cfg["cli"] = build_cli_cfg(
                logdir=run_logdir,
                train=True,
                cfg=cfg,
                skip_check_log_override=True,
            )

            print(f"Device = {args.device}")
            print(f"Sweep run logdir = {run_logdir}")
            neural_env = NnTrainingInterface(**cfg["env"], device=args.device)
            algo = SimpleMlpTrainer(
                neural_env=neural_env,
                model_checkpoint_path=args.checkpoint,
                cfg=cfg,
                device=args.device,
            )
            algo.train()
        except Exception as exc:
            print(f"Sweep run failed: {exc}")
            raise
        finally:
            if run is not None:
                run.finish()

    return _run_one_sweep_iteration


if __name__ == "__main__":
    args = _parse_args()
    wandb.login()

    project_name = args.project
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_config_mlp_0, project=project_name)
        print(f"\n=== Created NEW Sweep: {sweep_id} ===")
    else:
        sweep_id = args.sweep_id
        print(f"\n=== Joining EXISTING Sweep: {sweep_id} ===")

    wandb.agent(
        sweep_id,
        function=_make_sweep_train_fn(args),
        project=project_name,
        count=args.count,
    )
