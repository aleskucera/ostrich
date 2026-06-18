import os
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.models.simple_mlp_model import SimpleMlpModel
from axion.neural_solver.utils.datasets import TrajectoryDataset
from axion.neural_solver.utils.loss_utils import angular_prediction_loss
from axion.neural_solver.utils.logger import Logger
from axion.neural_solver.utils.python_utils import (
    set_random_seed,
    print_info, print_ok,
    format_value, format_dict,
)
from axion.neural_solver.utils.running_mean_std import RunningMeanStd
from axion.neural_solver.utils.time_report import TimeReport, TimeProfiler
from axion.neural_solver.utils.torch_utils import num_params_torch_model, grad_norm


class SimpleMlpTrainer:
    """
    Standalone trainer for single-step, state-only MLP regression.

    Predicts next_states from the current state and context (contacts, gravity).
    No lambda output, no sequence modelling, no transformer dependencies.

    Loss: angular 1-cos loss on joint positions q + MSE on joint velocities qd.
    """

    def __init__(
        self,
        neural_env: NnTrainingInterface,
        cfg: dict,
        model_checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        algo_cfg = cfg["algorithm"]
        cli_cfg = cfg["cli"]
        loss_cfg = algo_cfg.get("loss", {}) or {}

        self.device = device
        self.seed = int(algo_cfg.get("seed", 0))
        set_random_seed(self.seed)
        self.rng = np.random.default_rng(seed=self.seed)

        self.neural_env = neural_env
        self.utils_provider = neural_env.utils_provider

        self.utils_provider.set_expected_low_dim_keys(tuple(cfg["inputs"]["low_dim"]))

        # Loss weights
        self.state_loss_weight = float(loss_cfg.get("state_loss_weight", 1.0))
        self.angular_prediction_l2_weight = float(
            loss_cfg.get("angular_prediction_l2_weight", 0.0)
        )
        self._dof_q_per_env = int(self.utils_provider.dof_q_per_env)

        # --- Build or restore model ---
        if model_checkpoint_path is None:
            input_sample = self.utils_provider.get_neural_model_inputs()
            state_output_dim = int(
                self.neural_env.dof_q_per_env + self.neural_env.dof_qd_per_env
            )
            self.neural_model = SimpleMlpModel(
                input_sample=input_sample,
                state_output_dim=state_output_dim,
                input_cfg=cfg["inputs"],
                network_cfg=cfg["network"],
                device=self.device,
            )
        else:
            checkpoint = torch.load(
                model_checkpoint_path, map_location=self.device, weights_only=False
            )
            self.neural_model = checkpoint[0]
            self.neural_model.to(self.device)

        print("Model =\n", self.neural_model)
        print("# Model Parameters =", num_params_torch_model(self.neural_model))

        self.utils_provider.set_neural_model(self.neural_model)

        # --- Dataset ---
        self.batch_size = int(algo_cfg["batch_size"])
        self.num_valid_batches = int(algo_cfg.get("num_valid_batches", 50))
        self.dataset_max_capacity = algo_cfg["dataset"].get("max_capacity", 100_000_000)
        self.num_data_workers = algo_cfg["dataset"].get("num_data_workers", 4)
        self.sample_sequence_length = int(algo_cfg.get("sample_sequence_length", 1))

        train_dataset_path = algo_cfg["dataset"].get("train_dataset_path")
        valid_dataset_path = algo_cfg["dataset"].get("valid_dataset_path")
        self._build_datasets(train_dataset_path, valid_dataset_path)

        # --- Training parameters ---
        self.num_epochs = int(algo_cfg["num_epochs"])
        self.num_iters_per_epoch = int(algo_cfg.get("num_iters_per_epoch", -1))
        self.truncate_grad = bool(algo_cfg.get("truncate_grad", False))
        self.grad_norm_clip = float(algo_cfg.get("grad_norm", 1.0))

        lr_start = float(algo_cfg["optimizer"]["lr_start"])
        lr_end = float(algo_cfg["optimizer"].get("lr_end", 0.0))
        self.lr_schedule = algo_cfg["optimizer"]["lr_schedule"]
        self.lr_start = lr_start
        self.lr_end = lr_end
        weight_decay = float(algo_cfg["optimizer"].get("weight_decay", 0.0))
        optimizer_name = str(algo_cfg["optimizer"].get("name", "adamw")).lower()

        decay_params, no_decay_params = [], []
        for name, p in self.neural_model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or p.ndim == 1:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optim_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
        self.optimizer = optim_cls(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr_start,
        )

        # --- Logging / checkpointing ---
        self.log_dir = cli_cfg["logdir"]
        if os.path.exists(self.log_dir) and not cli_cfg.get("skip_check_log_override", False):
            ans = input(f"Logging directory {self.log_dir} exists, overwrite? [y/n] ")
            if ans.lower() != "y":
                raise SystemExit("Aborted by user.")
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_log_dir = os.path.join(self.log_dir, "nn")
        os.makedirs(self.model_log_dir, exist_ok=True)

        yaml.dump(cfg, open(os.path.join(self.log_dir, "cfg.yaml"), "w"))

        self.logger = Logger()
        wandb_cfg = dict(algo_cfg)
        wandb_cfg["env"] = cfg.get("env") or {}
        wandb_cfg["network"] = cfg.get("network") or {}
        self.logger.init_wandb(config=wandb_cfg)

        self.save_interval = int(cli_cfg.get("save_interval", 50))
        self.log_interval = int(cli_cfg.get("log_interval", 1))

        # --- Dataset statistics ---
        if algo_cfg.get("compute_dataset_statistics", True):
            print("Computing dataset statistics…")
            self._compute_dataset_statistics(self.train_dataset)
            print("Finished computing dataset statistics.")
            self.neural_model.set_input_rms(self.dataset_rms)
            self.neural_model.set_output_rms(self.dataset_rms.get("target"))
        else:
            if model_checkpoint_path is None:
                raise ValueError(
                    "model_checkpoint_path is required when compute_dataset_statistics=False"
                )
            print_info("Skipping dataset statistics computation.")

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _build_datasets(self, train_path, valid_path):
        def _make_dataset(path_spec):
            if isinstance(path_spec, (list, tuple)):
                parts = [
                    TrajectoryDataset(
                        sample_sequence_length=self.sample_sequence_length,
                        hdf5_dataset_path=p,
                        max_capacity=self.dataset_max_capacity,
                    )
                    for p in path_spec
                ]
                return ConcatDataset(parts)
            return TrajectoryDataset(
                sample_sequence_length=self.sample_sequence_length,
                hdf5_dataset_path=path_spec,
                max_capacity=self.dataset_max_capacity,
            )

        if train_path is None:
            raise ValueError("algorithm.dataset.train_dataset_path is required.")
        self.train_dataset = _make_dataset(train_path)
        self.valid_datasets = {}
        if valid_path is not None:
            self.valid_datasets["valid"] = _make_dataset(valid_path)

    # ------------------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def preprocess_data_batch(self, data: dict) -> dict:
        """Move to device, shape tensors to (B,1,D), compute prediction target."""
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                data[key] = val.to(self.device)

        self.utils_provider.process_neural_model_inputs(data)

        data["target"] = self.utils_provider.convert_next_states_to_prediction(
            states=data["states"],
            next_states=data["next_states"],
        )
        return data

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, data: dict, train: bool):
        del train
        prediction = self.neural_model(data)          # (B, 1, state_dim)
        target = data["target"]                        # (B, 1, state_prediction_dim)

        q_dim = self._dof_q_per_env
        q_loss = angular_prediction_loss(
            prediction[..., :q_dim],
            target[..., :q_dim],
            angular_prediction_l2_weight=self.angular_prediction_l2_weight,
        )
        qd_loss = F.mse_loss(prediction[..., q_dim:], target[..., q_dim:])
        state_loss = q_loss + qd_loss
        total_loss = self.state_loss_weight * state_loss

        with torch.no_grad():
            loss_itemized = {
                "q_loss": q_loss.detach(),
                "qd_loss": qd_loss.detach(),
                "state_loss": state_loss.detach(),
                "total_loss": total_loss.detach(),
            }
        return total_loss, loss_itemized

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_dataset_statistics(self, dataset):
        loader = DataLoader(
            dataset=dataset,
            batch_size=max(512, self.batch_size),
            shuffle=False,
            num_workers=self.num_data_workers,
            drop_last=True,
        )
        self.dataset_rms: dict = {}
        for data in loader:
            data = self.preprocess_data_batch(data)
            for key, val in data.items():
                if not isinstance(val, torch.Tensor):
                    continue
                if key not in self.dataset_rms:
                    self.dataset_rms[key] = RunningMeanStd(
                        shape=val.shape[2:], device=self.device
                    )
                self.dataset_rms[key].update(val, batch_dim=True, time_dim=True)

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------

    def _get_lr(self, epoch: int) -> float:
        if self.lr_schedule == "constant":
            return self.lr_start
        elif self.lr_schedule == "linear":
            ratio = epoch / self.num_epochs
            return self.lr_start * (1.0 - ratio) + self.lr_end * ratio
        elif self.lr_schedule == "cosine":
            coeff = 0.5 * (1.0 + np.cos(np.pi * epoch / self.num_epochs))
            return self.lr_end + coeff * (self.lr_start - self.lr_end)
        raise NotImplementedError(f"Unknown lr_schedule: {self.lr_schedule!r}")

    # ------------------------------------------------------------------
    # One epoch
    # ------------------------------------------------------------------

    def one_epoch(self, train: bool, dataloader, dataloader_iter, num_batches: int):
        if train:
            self.neural_model.train()
        else:
            self.neural_model.eval()

        sum_loss = 0.0
        sum_loss_itemized: dict = {}
        grad_info: dict = {"grad_norm_before_clip": 0.0}
        if self.truncate_grad:
            grad_info["grad_norm_after_clip"] = 0.0

        with torch.set_grad_enabled(train):
            for _ in tqdm(range(num_batches)):
                with TimeProfiler(self.time_report, "dataloader"):
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(dataloader)
                        data = next(dataloader_iter)
                    data = self.preprocess_data_batch(data)

                with TimeProfiler(self.time_report, "compute_loss"):
                    if train:
                        self.optimizer.zero_grad()
                    loss, loss_itemized = self.compute_loss(data, train)

                with TimeProfiler(self.time_report, "backward"):
                    if train:
                        loss.backward()
                        with torch.no_grad():
                            gn_before = grad_norm(self.neural_model.parameters())
                            grad_info["grad_norm_before_clip"] += gn_before
                            if self.truncate_grad:
                                clip_grad_norm_(
                                    self.neural_model.parameters(), self.grad_norm_clip
                                )
                                grad_info["grad_norm_after_clip"] += grad_norm(
                                    self.neural_model.parameters()
                                )
                        self.optimizer.step()

                sum_loss += loss.item()
                for key, val in loss_itemized.items():
                    if key in sum_loss_itemized:
                        sum_loss_itemized[key] += val
                    else:
                        sum_loss_itemized[key] = val

        avg_loss = sum_loss / num_batches
        avg_itemized = {k: v.cpu().item() / num_batches for k, v in sum_loss_itemized.items()}
        if train:
            grad_info["grad_norm_before_clip"] /= num_batches
            if self.truncate_grad:
                grad_info["grad_norm_after_clip"] /= num_batches

        return avg_loss, avg_itemized, grad_info

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        if self.train_dataset is None:
            raise ValueError("Training dataset is not set.")

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_workers,
            drop_last=True,
        )
        train_loader_iter = iter(train_loader)
        num_train_batches = (
            len(train_loader) if self.num_iters_per_epoch == -1
            else self.num_iters_per_epoch
        )

        valid_loaders = {}
        valid_loader_iters = {}
        best_valid_losses = {}
        for name, ds in self.valid_datasets.items():
            valid_loaders[name] = DataLoader(
                dataset=ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_data_workers,
                drop_last=True,
            )
            valid_loader_iters[name] = iter(valid_loaders[name])
            best_valid_losses[name] = np.inf

            fp = open(
                os.path.join(self.model_log_dir, f"saved_best_valid_{name}_model_epochs.txt"), "w"
            )
            fp.close()

        self.time_report = TimeReport(cuda_synchronize=False)
        self.time_report.add_timers(["epoch", "dataloader", "compute_loss", "backward"])

        for epoch in range(self.num_epochs):
            self.time_report.reset_timer()
            self.lr = self._get_lr(epoch)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr

            self.logger.init_epoch(epoch)

            with TimeProfiler(self.time_report, "epoch"):
                if epoch > 0:
                    avg_train_loss, avg_train_itemized, grad_info = self.one_epoch(
                        train=True,
                        dataloader=train_loader,
                        dataloader_iter=train_loader_iter,
                        num_batches=num_train_batches,
                    )

                avg_valid_losses = {}
                avg_valid_itemized = {}
                for name in self.valid_datasets:
                    n_valid = min(self.num_valid_batches, len(valid_loaders[name]))
                    avg_valid_losses[name], avg_valid_itemized[name], _ = self.one_epoch(
                        train=False,
                        dataloader=valid_loaders[name],
                        dataloader_iter=valid_loader_iters[name],
                        num_batches=n_valid,
                    )

            if epoch % self.log_interval == 0:
                time_summary = self.time_report.print(string_mode=True, in_second=True)
                print_info("-" * 80)
                print_info(f"Epoch {epoch}")
                if epoch > 0:
                    print_info(f"[Train] loss = {format_value(avg_train_loss, 8)}")
                    print_info(f"[Train] itemized: {format_dict(avg_train_itemized, 8)}")
                for name in self.valid_datasets:
                    print_info(
                        f"[Valid] {name}: loss = {format_value(avg_valid_losses[name], 8)}"
                    )
                    print_info(
                        f"[Valid] {name} itemized: {format_dict(avg_valid_itemized[name], 8)}"
                    )
                print_info(f"[Time] {time_summary}")
                if epoch > 0:
                    print_info(f"[Grad] {format_dict(grad_info, 3)}")

                self.logger.add_scalar("params/lr/epoch", self.lr, epoch)
                if epoch > 0:
                    self.logger.add_scalar("training/train_loss/epoch", avg_train_loss, epoch)
                    self.logger.add_scalar(
                        "training/grad_norm_before_clip/epoch",
                        grad_info["grad_norm_before_clip"], epoch,
                    )
                    if self.truncate_grad:
                        self.logger.add_scalar(
                            "training/grad_norm_after_clip/epoch",
                            grad_info["grad_norm_after_clip"], epoch,
                        )
                    for key, val in avg_train_itemized.items():
                        self.logger.add_scalar(f"training_info/{key}/epoch", val, epoch)

                for name in self.valid_datasets:
                    self.logger.add_scalar(
                        f"training/valid_{name}_loss/epoch", avg_valid_losses[name], epoch
                    )
                    for key, val in avg_valid_itemized[name].items():
                        self.logger.add_scalar(
                            f"validating_info/{key}_{name}/epoch", val, epoch
                        )

                self.logger.flush()

            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_model(f"model_epoch{epoch}")

            for name in self.valid_datasets:
                if avg_valid_losses[name] < best_valid_losses[name]:
                    best_valid_losses[name] = avg_valid_losses[name]
                    self.save_model(f"best_valid_{name}_model")
                    with open(
                        os.path.join(
                            self.model_log_dir,
                            f"saved_best_valid_{name}_model_epochs.txt",
                        ), "a"
                    ) as fp:
                        fp.write(f"{epoch}\n")
                    print_ok(
                        f"Saved best valid [{name}] model at epoch {epoch} "
                        f"with loss {format_value(avg_valid_losses[name], 8)}."
                    )

        self.save_model("final_model")
        self.logger.finish()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "best_model") -> None:
        torch.save(
            [self.neural_model, self.neural_env.robot_name],
            os.path.join(self.model_log_dir, f"{filename}.pt"),
        )
