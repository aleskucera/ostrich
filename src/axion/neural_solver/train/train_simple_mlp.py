import argparse
import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(base_dir)

import warp as wp

wp.config.verify_cuda = True

import wandb

from axion.neural_solver.algorithms.simple_mlp_trainer import SimpleMlpTrainer
from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.train.config_builder import build_cli_cfg, load_default_cfg
from axion.neural_solver.utils.python_utils import get_time_stamp, set_random_seed


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train simple MLP regression network from YAML config."
    )
    p.add_argument("--cfg", required=True, help="Path to config YAML")
    p.add_argument("--logdir", required=True, help="Directory for logs and checkpoints")
    p.add_argument("--checkpoint", default=None, help="Checkpoint to restore")
    p.add_argument(
        "--no-time-stamp",
        action="store_true",
        help="No timestamp subfolder under logdir",
    )
    p.add_argument("--device", default="cuda:0", help="Device (e.g. cuda:0)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    wandb.login()

    cfg = load_default_cfg(args.cfg)
    cfg.setdefault("network", {})
    cfg["network"]["model_impl"] = "simple_mlp"

    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    seed = cfg["algorithm"].get("seed", 0)
    set_random_seed(seed)
    cfg["algorithm"]["seed"] = seed

    cfg["cli"] = build_cli_cfg(
        logdir=args.logdir,
        train=True,
        cfg=cfg,
        skip_check_log_override=False,
    )

    print(f"Device = {args.device}")
    neural_env = NnTrainingInterface(**cfg["env"], device=args.device)

    algo = SimpleMlpTrainer(
        neural_env=neural_env,
        model_checkpoint_path=args.checkpoint,
        cfg=cfg,
        device=args.device,
    )

    print("Begin Simple MLP training")
    algo.train()
