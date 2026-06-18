# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(base_dir)

import time
import shutil
from typing import Optional

import warp as wp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.clip_grad import clip_grad_norm_
import yaml
import numpy as np
from tqdm import tqdm

from axion.neural_solver.envs.nn_training_interface import NnTrainingInterface
from axion.neural_solver.models.models import ModelMixedInput
from axion.neural_solver.models.lambda_models import LambdaClassificationModel
from axion.neural_solver.models.residual_model import ResidualModel
from axion.neural_solver.models.mtl_model import MTLModel
from axion.neural_solver.models.mse_model import MSEModel
from axion.neural_solver.models.contact_mtl_model import ContactMTLModel
from axion.neural_solver.utils.datasets import TrajectoryDataset
from axion.neural_solver.utils.evaluator import NeuralSimEvaluator
from axion.neural_solver.utils.python_utils import (
    set_random_seed, 
    print_info, print_ok, print_white, print_warning,
    format_dict, format_value
)
from axion.neural_solver.utils.torch_utils import num_params_torch_model, grad_norm
from axion.neural_solver.utils.running_mean_std import RunningMeanStd
from axion.neural_solver.utils.time_report import TimeReport, TimeProfiler
from axion.neural_solver.utils.logger import Logger

TIME_STEP_S = 0.01

class SequenceModelTrainer:
    def __init__(
        self, 
        neural_env: NnTrainingInterface, 
        cfg: dict, 
        model_checkpoint_path: Optional[str] = None, 
        device = 'cuda:0'
    ):
    
        algo_cfg = cfg['algorithm']
        cli_cfg = cfg['cli']

        self.seed = algo_cfg.get('seed', 0)
        self.device = device
        self.is_test_mode = not bool(cli_cfg.get('train', True))
        self.has_lambda_head = cfg['network'].get('enable_lambda_head', True)
        self.has_state_head = cfg['network'].get('enable_state_head', True)
        if not self.has_lambda_head and not self.has_state_head:
            raise ValueError(
                "At least one of network.enable_lambda_head or network.enable_state_head must be True."
            )
        self.use_energy_loss = bool(algo_cfg.get('use_energy_loss', False))
        self.lambda_loss_weight = float(algo_cfg.get('lambda_loss_weight', 1.0)) if self.has_lambda_head else 0.0
        loss_cfg = algo_cfg.get('loss', {}) or {}
        self.huber_delta = float(loss_cfg.get('huber_delta', 1.0))
        self.kinematics_loss_weight = float(loss_cfg.get('kinematics_loss_weight', 0.5))
        self.lambda_loss_type = str(loss_cfg.get('lambda_loss_type', 'mse')).lower()

        set_random_seed(self.seed)

        self.rng = np.random.default_rng(seed = self.seed)

        self.neural_env = neural_env
        self.utils_provider = neural_env.utils_provider

        # check if gravity_dir_body is included in input if using body frame
        if cfg['env']['utils_provider_cfg'].get('states_frame') == 'body':
            if 'gravity_dir' not in cfg['inputs']['low_dim']:
                cfg['inputs']['low_dim'].append('gravity_dir')
                print_warning("gravity_dir not included in low_dim inputs, "
                              "added it automatically.")

        self.utils_provider.set_expected_low_dim_keys(tuple(cfg["inputs"]["low_dim"]))

        # create neural sim model
        if model_checkpoint_path is None:
            input_sample = self.utils_provider.get_neural_model_inputs()
            model_impl = str(cfg['network'].get('model_impl', 'default')).lower()
            if model_impl == 'lambda_classification':

                # Wire classification-specific knobs from algorithm.loss into network_cfg
                # so LambdaClassificationModel can stay backward compatible.
                network_cfg = dict(cfg['network'])
                loss_cfg = algo_cfg.get('loss', {}) or {}
                if 'classification_num_classes' in loss_cfg and 'classification_num_classes' not in network_cfg:
                    network_cfg['classification_num_classes'] = loss_cfg['classification_num_classes']

                self.neural_model = LambdaClassificationModel(
                    input_sample=input_sample,
                    output_dim=self.utils_provider.lambda_prediction_dim,
                    input_cfg=cfg['inputs'],
                    network_cfg=network_cfg,
                    device=self.device,
                )
            elif model_impl in (
                'vel_and_lambda',
                'vel_and_lambda_model',
                'vel_lambda',
                'residual',
                'residual_model',
            ):
                engine_dims = self.neural_env.simulator_wrapper.engine.dims
                state_output_dim = int(self.neural_env.dof_q_per_env + self.neural_env.dof_qd_per_env)
                self.neural_model = ResidualModel(
                    input_sample=input_sample,
                    state_output_dim=state_output_dim,
                    lambda_output_dim=engine_dims.num_constraints,
                    input_cfg=cfg['inputs'],
                    network_cfg=cfg['network'],
                    device=self.device,
                )
            elif model_impl == 'mtl':
                engine_dims = self.neural_env.simulator_wrapper.engine.dims
                state_output_dim = int(self.neural_env.dof_q_per_env + self.neural_env.dof_qd_per_env)
                self.neural_model = MTLModel(
                    input_sample=input_sample,
                    state_output_dim=state_output_dim,
                    lambda_output_dim=engine_dims.num_constraints,
                    input_cfg=cfg['inputs'],
                    network_cfg=cfg['network'],
                    device=self.device,
                )
            elif model_impl == 'contact_mtl':
                engine_dims = self.neural_env.simulator_wrapper.engine.dims
                contact_lambda_start = int(cfg['network'].get('contact_lambda_start_index', 10))
                contact_lambda_dim = engine_dims.num_constraints - contact_lambda_start
                self.neural_model = ContactMTLModel(
                    input_sample=input_sample,
                    contact_lambda_dim=contact_lambda_dim,
                    input_cfg=cfg['inputs'],
                    network_cfg=cfg['network'],
                    device=self.device,
                )
            elif model_impl == 'mse':
                state_output_dim = int(self.neural_env.dof_q_per_env + self.neural_env.dof_qd_per_env)
                self.neural_model = MSEModel(
                    input_sample=input_sample,
                    state_output_dim=state_output_dim,
                    lambda_output_dim=self.utils_provider.lambda_dim,
                    input_cfg=cfg['inputs'],
                    network_cfg=cfg['network'],
                    device=self.device,
                    states_only=not self.has_lambda_head,
                )
            else:
                self.neural_model = ModelMixedInput(
                    input_sample = input_sample,
                    output_dim = self.utils_provider.state_prediction_dim,
                    lambda_output_dim = self.utils_provider.lambda_prediction_dim,
                    input_cfg = cfg['inputs'],
                    network_cfg = cfg['network'],
                    device = self.device
                )
        else:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device, weights_only= False)
            self.neural_model = checkpoint[0]
            self.neural_model.to(self.device)

            ckpt_has_state_head, ckpt_has_lambda_head = self._infer_checkpoint_heads(self.neural_model)
            self.neural_model.has_state_head = ckpt_has_state_head
            self.neural_model.has_lambda_head = ckpt_has_lambda_head

            # Keep config/provider target mode consistent with loaded checkpoint head size.
            state_head_dim = self._state_output_head_dim(self.neural_model)
            if state_head_dim is not None:
                if state_head_dim == self.utils_provider.state_dim:
                    self.utils_provider.prediction_quantity_type = "full_state"
                    self.utils_provider.state_prediction_dim = self.utils_provider.state_dim
                elif state_head_dim == self.utils_provider.dof_qd_per_env:
                    self.utils_provider.prediction_quantity_type = "velocities_only"
                    self.utils_provider.state_prediction_dim = self.utils_provider.dof_qd_per_env
                else:
                    raise ValueError(
                        "Checkpoint state head dim does not match expected full-state or velocity-only dims: "
                        f"{state_head_dim} vs state_dim={self.utils_provider.state_dim}, "
                        f"dof_qd={self.utils_provider.dof_qd_per_env}."
                    )
            elif self.has_state_head:
                raise ValueError(
                    "Checkpoint has no state head but config has enable_state_head: True."
                )

            if self.has_lambda_head and not ckpt_has_lambda_head:
                raise ValueError(
                    "Config enables lambda head but checkpoint has no lambda-capable head."
                )

            self.neural_model.has_state_head = self.has_state_head
            self.neural_model.has_lambda_head = self.has_lambda_head
            state_head_module = self._state_head_module(self.neural_model)
            if not self.has_state_head:
                if state_head_module is not None:
                    for p in state_head_module.parameters():
                        p.requires_grad = False
                elif hasattr(self.neural_model, "regression_head"):
                    raise ValueError(
                        "Config disables state head, but this checkpoint shares state/lambda "
                        "outputs in `regression_head`; selective state-head freezing is unsupported."
                    )
        
        print('Model = \n', self.neural_model)
        print('# Model Parameters = ', num_params_torch_model(self.neural_model))

        self.utils_provider.set_neural_model(self.neural_model)

        """ General parameters """
        self.batch_size = int(algo_cfg['batch_size'])
        self.num_valid_batches = int(algo_cfg.get('num_valid_batches', 50))
        self.dataset_max_capacity = algo_cfg['dataset'].get('max_capacity', 100000000)
        self.num_data_workers = algo_cfg['dataset'].get('num_data_workers', 4)
        self.sample_sequence_length = algo_cfg.get('sample_sequence_length', 1)
        train_dataset_path = algo_cfg['dataset'].get('train_dataset_path', None)
        valid_dataset_path = algo_cfg['dataset'].get('valid_dataset_path', None)
        legacy_valid_datasets_cfg = algo_cfg['dataset'].get('valid_datasets', None)

        self.train_dataset = None
        self.valid_datasets = {}
        self.collate_fn = None
        self.get_datasets(
            train_dataset_path,
            valid_dataset_path,
            legacy_valid_datasets_cfg,
            require_train_dataset=bool(cli_cfg.get('train', True)),
        )

        """ Parameters only used for training """
        if cli_cfg['train']:
            # load training general parameters
            self.num_epochs = int(algo_cfg['num_epochs'])
            self.num_iters_per_epoch = int(algo_cfg.get('num_iters_per_epoch', -1))

            # load learning rate params
            self.lr_start = float(algo_cfg['optimizer']['lr_start'])
            self.lr_end = float(algo_cfg['optimizer'].get('lr_end', 0.))
            self.lr_schedule = algo_cfg['optimizer']['lr_schedule']
            self.optimizer_name = str(algo_cfg['optimizer'].get('name', 'adamw')).lower()
            self.weight_decay = float(algo_cfg['optimizer'].get('weight_decay', 0.0))
            # Exclude biases and normalization parameters from weight decay.
            decay_params = []
            no_decay_params = []
            for name, p in self.neural_model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.endswith(".bias") or p.ndim == 1:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

            if self.optimizer_name in ("adamw",):
                optim_cls = torch.optim.AdamW
            elif self.optimizer_name in ("adam",):
                optim_cls = torch.optim.Adam
            else:
                raise ValueError(
                    f"Unknown optimizer '{self.optimizer_name}'. Expected 'adam' or 'adamw'."
                )

            self.optimizer = optim_cls(
                [
                    {"params": decay_params, "weight_decay": self.weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.lr_start,
            )

            # load gradient clipping params
            self.truncate_grad = algo_cfg.get('truncate_grad', False)
            self.grad_norm = algo_cfg.get('grad_norm', 1.0)

            # logging related
            self.log_dir = cli_cfg["logdir"]
            if os.path.exists(self.log_dir) and not cli_cfg["skip_check_log_override"]:
                ans = input(f"Logging Directory {self.log_dir} exist, overwrite? [y/n]")
                if ans == 'y':
                    shutil.rmtree(self.log_dir)
                else:
                    exit
                
            os.makedirs(self.log_dir, exist_ok = True)

            self.model_log_dir = os.path.join(self.log_dir, 'nn')
            os.makedirs(self.model_log_dir, exist_ok = True)

            # save config
            yaml.dump(cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))

            # create logger (include env + network alongside algorithm hyperparameters)
            self.logger = Logger()
            wandb_cfg = dict(algo_cfg)
            wandb_cfg["env"] = cfg.get("env") or {}
            wandb_cfg["network"] = cfg.get("network") or {}
            self.logger.init_wandb(config=wandb_cfg)
                
            # other logging params
            self.save_interval = cli_cfg.get("save_interval", 50)
            self.log_interval = cli_cfg.get("log_interval", 1)
            self.eval_interval = cli_cfg.get("eval_interval", 1)

            # do not need to compute dataset statistics if doing finetuning
            if algo_cfg.get("compute_dataset_statistics", True):
                # get dataset mean/std info
                print('Computing dataset statistics...')
                self.compute_dataset_statistics(self.train_dataset)
                print('Finished computing dataset statistics...')
                self.neural_model.set_input_rms(self.dataset_rms)
                self.neural_model.set_output_rms(
                    self.dataset_rms.get('target') if self.has_state_head else None,
                    self.dataset_rms.get('target_lambda') if self.has_lambda_head else None,
                )
            else:
                assert model_checkpoint_path is not None, \
                    "model_checkpoint_path is required to skip computing dataset statistics"
                print_info('Skip computing dataset statistics')
            
            # create logging files for saved best valid model
            for valid_dataset_name in self.valid_datasets.keys():
                fp = open(
                    os.path.join(
                        self.model_log_dir, 
                        f'saved_best_valid_{valid_dataset_name}_model_epochs.txt'
                    ), 'w'
                )
                fp.close()

            fp = open(
                os.path.join(
                    self.model_log_dir, 
                    "saved_best_eval_model_epochs.txt"
                ), 'w'
            )
            fp.close()

        # Create evaluator
        self.eval_mode = algo_cfg['eval'].get('mode', 'sampler')
        self.eval_horizon = algo_cfg['eval'].get("rollout_horizon", 5)
        self.num_eval_rollouts = algo_cfg['eval'].get("num_rollouts", self.neural_env.num_envs)
        self.eval_dataset_path = algo_cfg['eval'].get('dataset_path', None)
        if self.eval_dataset_path is None:
            _legacy = legacy_valid_datasets_cfg or {}
            if valid_dataset_path is not None:
                self.eval_dataset_path = valid_dataset_path
            elif 'valid' in _legacy:
                self.eval_dataset_path = _legacy['valid']
            elif _legacy:
                self.eval_dataset_path = next(iter(_legacy.values()))
        self.eval_passive = algo_cfg['eval'].get('passive', True)
        self.eval_render = cli_cfg['render']

        if self.eval_mode == 'dataset':
            assert self.eval_dataset_path is not None, (
                "If eval_mode is 'dataset', provide algorithm.eval.dataset_path, "
                "algorithm.dataset.valid_dataset_path, or algorithm.dataset.valid_datasets."
            )
            
        self.evaluator = NeuralSimEvaluator(
                            self.neural_env,
                            hdf5_dataset_path = self.eval_dataset_path if self.eval_mode == 'dataset' else None,
                            eval_horizon = self.eval_horizon,
                            device = self.device
                        )

        self.eval_primary_metric = algo_cfg.get('eval_primary_metric')
        if self.eval_primary_metric is None:
            self.eval_primary_metric = (
                'lambda_error(MSE)'
                if (not self.has_state_head and self.has_lambda_head)
                else 'error(MSE)'
            )

    def _state_output_head_dim(self, model) -> Optional[int]:
        """Infer the state-head output dimension for loaded checkpoints."""
        state_head = getattr(getattr(model, "model", None), "output_net", None)
        if state_head is not None:
            return int(state_head.out_features)

        regression_head = getattr(getattr(model, "regression_head", None), "output_net", None)
        if regression_head is None:
            return None

        state_output_dim = getattr(model, "state_output_dim", None)
        if state_output_dim is not None:
            return int(state_output_dim)
        return int(regression_head.out_features)

    def _infer_checkpoint_heads(self, model):
        """Return inferred state/lambda head availability for a checkpoint model."""
        has_state_head = getattr(model, "has_state_head", None)
        if has_state_head is None:
            state_dim = self._state_output_head_dim(model)
            has_state_head = state_dim is not None and state_dim > 0
        has_state_head = bool(has_state_head)

        has_lambda_head = getattr(model, "has_lambda_head", None)
        if has_lambda_head is None:
            lambda_model = getattr(model, "lambda_model", None)
            if lambda_model is not None:
                has_lambda_head = True
            else:
                lambda_output_dim = getattr(model, "lambda_output_dim", None)
                if lambda_output_dim is not None:
                    has_lambda_head = int(lambda_output_dim) > 0
                else:
                    has_lambda_head = getattr(model, "classification_head", None) is not None
        has_lambda_head = bool(has_lambda_head)
        return has_state_head, has_lambda_head

    def _state_head_module(self, model):
        """Return the trainable module used for state predictions when separable."""
        state_head_module = getattr(model, "model", None)
        if state_head_module is not None:
            return state_head_module
        return None
    
    def get_datasets(
        self,
        train_dataset_path,
        valid_dataset_path,
        legacy_valid_datasets_cfg,
        require_train_dataset: bool = True,
    ):
        # Training dataset is optional for `--test` runs.
        if train_dataset_path is None:
            if require_train_dataset:
                raise ValueError(
                    "Missing algorithm.dataset.train_dataset_path in config. "
                    "This is required for training runs."
                )
            self.train_dataset = None
        else:
            # Support single path (str) or multiple paths (list) for training
            if isinstance(train_dataset_path, (list, tuple)):
                train_datasets = [
                    TrajectoryDataset(
                        sample_sequence_length=self.sample_sequence_length,
                        hdf5_dataset_path=path,
                        max_capacity=self.dataset_max_capacity,
                    )
                    for path in train_dataset_path
                ]
                self.train_dataset = ConcatDataset(train_datasets)
            else:
                self.train_dataset = TrajectoryDataset(
                    sample_sequence_length=self.sample_sequence_length,
                    hdf5_dataset_path=train_dataset_path,
                    max_capacity=self.dataset_max_capacity,
                )

        def _build_validation_dataset(path_spec):
            """Scalar path or list/tuple of paths (concat), mirroring train_dataset_path."""
            if isinstance(path_spec, (list, tuple)):
                if not path_spec:
                    raise ValueError("Validation dataset path list must not be empty.")
                parts = [
                    TrajectoryDataset(
                        sample_sequence_length=self.sample_sequence_length,
                        hdf5_dataset_path=path,
                    )
                    for path in path_spec
                ]
                return ConcatDataset(parts)
            return TrajectoryDataset(
                sample_sequence_length=self.sample_sequence_length,
                hdf5_dataset_path=path_spec,
            )

        if valid_dataset_path is not None:
            self.valid_datasets["valid"] = _build_validation_dataset(valid_dataset_path)
        elif legacy_valid_datasets_cfg:
            valid_cfg = legacy_valid_datasets_cfg or {}
            for valid_dataset_name in valid_cfg.keys():
                self.valid_datasets[valid_dataset_name] = _build_validation_dataset(
                    valid_cfg[valid_dataset_name]
                )
        self.collate_fn = None

    def compute_dataset_statistics(self, dataset):
        # compute the mean and std of the input and output of the dataset
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = max(512, self.batch_size),
            collate_fn = self.collate_fn,
            shuffle = False,
            num_workers = self.num_data_workers,
            drop_last = True
        )
        dataloader_iter = iter(dataloader)
        self.dataset_rms = {}

        for _ in range(len(dataloader)):
            data = next(dataloader_iter)
            data = self.preprocess_data_batch(data)

            for key in data.keys():
                if not (key in self.dataset_rms):
                    self.dataset_rms[key] = RunningMeanStd(
                        shape = data[key].shape[2:],
                        device = self.device
                    )

                self.dataset_rms[key].update(
                    data[key], 
                    batch_dim = True, 
                    time_dim = True
                )

    def get_scheduled_learning_rate(self, iteration, total_iterations):
        if self.lr_schedule == 'constant':
            return self.lr_start
        elif self.lr_schedule == 'linear':
            ratio = iteration / total_iterations
            return self.lr_start * (1.0 - ratio) + self.lr_end * ratio
        elif self.lr_schedule == 'cosine':
            decay_ratio = iteration / total_iterations
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) # coeff ranges 0..1
            return self.lr_end + coeff * (self.lr_start - self.lr_end)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def preprocess_data_batch(self, data):
        # Move data to target device
        for key in data.keys():
            if type(data[key]) is dict:
                for sub_key in data[key].keys():
                    data[key][sub_key] = data[key][sub_key].to(self.device)
            else:
                data[key] = data[key].to(self.device)
        
        self.utils_provider.process_neural_model_inputs(data)

        # calculate prediction target from neural env
        if self.has_state_head:
            data['target'] = self.utils_provider.convert_next_states_to_prediction(
                states=data['states'],
                next_states=data['next_states'],
                dt=self.neural_env.frame_dt,
            )
        if self.has_lambda_head:
            data['target_lambda'] = self.utils_provider.convert_next_lambdas_to_prediction(
                    lambdas = data['lambdas'],
                    next_lambdas = data['next_lambdas'],
                )

        return data

    def _weighted_lambda_regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        wp = pred * weights
        wt = target * weights
        if self.lambda_loss_type == 'mse':
            return F.mse_loss(wp, wt)
        if self.lambda_loss_type == 'l1':
            return F.l1_loss(wp, wt)
        if self.lambda_loss_type == 'huber':
            return F.huber_loss(wp, wt, delta=self.huber_delta, reduction='mean')
        raise RuntimeError(f'Unhandled lambda_loss_type {self.lambda_loss_type!r}')

    def compute_loss(self, data, train):

        prediction = self.neural_model(data)
        state_prediction = prediction['state']
        lambda_prediction = prediction['lambda']

        if self.has_state_head:
            prediction_target = data['target']
            if self.neural_model.normalize_output and self.neural_model.output_rms is not None:
                state_loss_weights = 1. / torch.sqrt(self.neural_model.output_rms.var + 1e-5)
            else:
                state_loss_weights = torch.ones(
                    state_prediction.shape[-1], device=state_prediction.device
                )
            huber_loss = torch.nn.HuberLoss(delta=self.huber_delta)(
                state_prediction * state_loss_weights,
                prediction_target * state_loss_weights,
            )
            loss = huber_loss
        else:
            huber_loss = None
            loss = None

        if self.has_lambda_head:
            prediction_target_lambda = data['target_lambda']
            if self.neural_model.normalize_output and self.neural_model.lambda_output_rms is not None:
                lambda_loss_weights = 1. / torch.sqrt(
                    self.neural_model.lambda_output_rms.var + 1e-5
                )
            else:
                lambda_loss_weights = torch.ones(
                    lambda_prediction.shape[-1], device=lambda_prediction.device
                )
            lambda_loss = self._weighted_lambda_regression_loss(
                lambda_prediction,
                prediction_target_lambda,
                lambda_loss_weights,
            )
            if loss is None:
                loss = lambda_loss
            else:
                loss = loss + self.lambda_loss_weight * lambda_loss
        else:
            lambda_loss = torch.zeros((), device=data['states'].device, dtype=data['states'].dtype)

        if self.has_state_head:
            predicted_next_states = self.utils_provider.convert_prediction_to_next_states(
                states=data['states'],
                prediction=state_prediction,
            )
            if getattr(self.utils_provider, 'prediction_quantity_type', 'full_state') == 'full_state':
                self.utils_provider.wrap2PI(predicted_next_states)
        else:
            predicted_next_states = None

        predicted_next_lambdas = None
        if self.has_lambda_head:
            predicted_next_lambdas = self.utils_provider.convert_prediction_to_next_lambdas(
                lambdas=data['lambdas'],
                prediction=lambda_prediction,
            )

        with torch.no_grad():
            loss_itemized = self.compute_itemized_loss(
                huber_loss,
                lambda_loss,
                predicted_next_states,
                predicted_next_lambdas,
                data,
            )

        return loss, loss_itemized


    def compute_itemized_loss(self, state_loss, lambda_loss, predicted_next_states, predicted_next_lambdas, data):
        """
        Computes the itemized loss for a given prediction dimension.
        """
        loss_itemized = {}
        if self.has_lambda_head:
            loss_itemized['lambda_prediction_loss'] = lambda_loss.detach()
            loss_itemized['lambda_MSE'] = torch.nn.MSELoss()(
                predicted_next_lambdas,
                data['next_lambdas'],
            )
        if predicted_next_states is None:
            return loss_itemized

        if state_loss is not None:
            loss_itemized['state_prediction_MSE'] = state_loss.detach()
        if predicted_next_states.shape[-1] == 4:
            for i in range(predicted_next_states.shape[-1]):
                loss_itemized[f'state_{i}'] = ((
                    predicted_next_states[..., i] - data['next_states'][..., i]
                ) ** 2).mean()
            loss_itemized['position_MSE'] = torch.nn.MSELoss()(
                predicted_next_states[..., :self.utils_provider.dof_q_per_env],
                data['next_states'][..., :self.utils_provider.dof_q_per_env],
            )
            loss_itemized['velocity_MSE'] = torch.nn.MSELoss()(
                predicted_next_states[..., self.utils_provider.dof_q_per_env:],
                data['next_states'][..., self.utils_provider.dof_q_per_env:],
            )
            loss_itemized['q_error_norm'] = torch.norm(
                predicted_next_states[..., :self.utils_provider.dof_q_per_env]
                - data['next_states'][..., :self.utils_provider.dof_q_per_env],
                dim=-1,
            ).mean()
            loss_itemized['qd_error_norm'] = torch.norm(
                predicted_next_states[..., self.utils_provider.dof_q_per_env:]
                - data['next_states'][..., self.utils_provider.dof_q_per_env:],
                dim=-1,
            ).mean()
        else:
            for i in range(predicted_next_states.shape[-1]):
                loss_itemized[f'state_{i}'] = ((
                    predicted_next_states[..., i] - data['next_states'][..., i + 2]
                ) ** 2).mean()
            loss_itemized['qd_error_norm'] = torch.norm(
                predicted_next_states
                - data['next_states'][..., self.utils_provider.dof_q_per_env:],
                dim=-1,
            ).mean()

        return loss_itemized


    def compute_test_loss_reference(self, data):
        """Stable reference loss for `--test` runs: weighted state MSE, or lambda MSE if no state head.

        This is intentionally kept independent from `compute_loss()` so training-loss
        experiments don't change the reported test-time validation loss.
        """
        prediction = self.neural_model(data)
        state_prediction = prediction['state']
        lambda_prediction = prediction['lambda']

        if self.has_state_head:
            prediction_target = data['target']
            if self.neural_model.normalize_output and self.neural_model.output_rms is not None:
                state_loss_weights = 1. / torch.sqrt(self.neural_model.output_rms.var + 1e-5)
                state_loss_weights = state_loss_weights[..., : state_prediction.shape[-1]]
            else:
                state_loss_weights = torch.ones(
                    state_prediction.shape[-1], device=state_prediction.device
                )
            loss = torch.nn.MSELoss()(
                state_prediction * state_loss_weights,
                prediction_target * state_loss_weights,
            )
            with torch.no_grad():
                loss_itemized = {'state_prediction_MSE': loss.detach()}
            return loss, loss_itemized

        if not self.has_lambda_head:
            raise RuntimeError('compute_test_loss_reference requires a state or lambda head.')

        prediction_target_lambda = data['target_lambda']
        if self.neural_model.normalize_output and self.neural_model.lambda_output_rms is not None:
            lambda_loss_weights = 1. / torch.sqrt(
                self.neural_model.lambda_output_rms.var + 1e-5
            )
        else:
            lambda_loss_weights = torch.ones(
                lambda_prediction.shape[-1], device=lambda_prediction.device
            )
        loss = torch.nn.MSELoss()(
            lambda_prediction * lambda_loss_weights,
            prediction_target_lambda * lambda_loss_weights,
        )
        with torch.no_grad():
            loss_itemized = {'lambda_prediction_MSE': loss.detach()}
        return loss, loss_itemized
        
    def one_epoch(
        self, 
        train: bool, 
        dataloader, 
        dataloader_iter, 
        num_batches, 
        shuffle = False
    ):
        
        if train:
            self.neural_model.train()
        else:
            self.neural_model.eval()
            
        sum_loss = 0.
        sum_loss_itemized = {}
        if train:
            grad_info = {'grad_norm_before_clip': 0.}
            if self.truncate_grad:
                grad_info['grad_norm_after_clip'] = 0.
        else:
            grad_info = {}

        with torch.set_grad_enabled(train):
            for _ in tqdm(range(num_batches)):
                with TimeProfiler(self.time_report, 'dataloader'):
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        if shuffle and hasattr(self.train_dataset, 'shuffle'):
                            self.train_dataset.shuffle()
                        dataloader_iter = iter(dataloader)
                        data = next(dataloader_iter)

                    data = self.preprocess_data_batch(data)
                
                with TimeProfiler(self.time_report, 'compute_loss'):
                    if train:
                        self.optimizer.zero_grad()

                    if (not train) and self.is_test_mode:
                        loss, loss_itemized = self.compute_test_loss_reference(data)
                    else:
                        loss, loss_itemized = self.compute_loss(data, train)

                with TimeProfiler(self.time_report, 'backward'):
                    if train:
                        loss.backward()

                        # Truncate gradients
                        with torch.no_grad():
                            grad_norm_before_clip = grad_norm(
                                self.neural_model.parameters()
                            )
                            grad_info['grad_norm_before_clip'] += grad_norm_before_clip
                            if self.truncate_grad:
                                clip_grad_norm_(
                                    self.neural_model.parameters(), 
                                    self.grad_norm
                                )
                                grad_norm_after_clip = grad_norm(
                                    self.utils_provider.neural_model.parameters()
                                ) 
                                grad_info['grad_norm_after_clip'] += grad_norm_after_clip

                        self.optimizer.step()

                with TimeProfiler(self.time_report, 'other'):
                    sum_loss += loss

                    for key in loss_itemized.keys():
                        if key in sum_loss_itemized:
                            sum_loss_itemized[key] += loss_itemized[key]
                        else:
                            sum_loss_itemized[key] = loss_itemized[key]
        
        avg_loss = sum_loss.detach().cpu().item() / num_batches
        avg_loss_itemized = {}
        for key in sum_loss_itemized.keys():
            avg_loss_itemized[key] = sum_loss_itemized[key].cpu().item() / num_batches
        if train:
            grad_info['grad_norm_before_clip'] /= num_batches
            if self.truncate_grad:
                grad_info['grad_norm_after_clip'] /= num_batches

        return avg_loss, avg_loss_itemized, grad_info

    def _format_residual_breakdown(self, itemized: dict) -> Optional[str]:
        """Build a readable residual diagnostics summary when available."""
        if "residual_mse_d" not in itemized:
            return None

        block_labels = [
            ("dynamics (d)", "residual_mse_d"),
            ("joint constraints (j)", "residual_mse_j"),
            ("control constraints (ctrl)", "residual_mse_ctrl"),
            ("normal contact (n)", "residual_mse_n"),
            ("friction contact (f)", "residual_mse_f"),
        ]
        parts = []
        for label, mse_key in block_labels:
            if mse_key in itemized:
                parts.append(f"{label}: mse={format_value(itemized[mse_key], 6)}")
        if not parts:
            return None
        return "\n".join(f"  - {part}" for part in parts)

    def _format_itemized_multiline(self, itemized: dict, precision: int = 8) -> str:
        """
        Format itemized metrics one per line.

        For residual diagnostics readability, hide residual_sq_* when residual_mse_* is present.
        """
        display_itemized = dict(itemized)
        if any(key.startswith("residual_mse_") for key in display_itemized):
            display_itemized = {
                key: value
                for key, value in display_itemized.items()
                if not key.startswith("residual_sq_")
            }
        lines = [f"  - {key}: {format_value(value, precision)}" for key, value in display_itemized.items()]
        return "\n".join(lines) if lines else "  - (none)"
            
    def train(self):
        if self.train_dataset is None:
            raise ValueError(
                "Training dataset is not set. Please provide "
                "`algorithm.dataset.train_dataset_path` in the config."
            )
        train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            collate_fn = self.collate_fn,
            shuffle = True,
            num_workers = self.num_data_workers,
            drop_last = True
        )
        train_loader_iter = iter(train_loader)
        if self.num_iters_per_epoch == -1:
            self.num_train_batches = len(train_loader)
        else:
            self.num_train_batches = self.num_iters_per_epoch

        valid_loaders = {}
        valid_loader_iters = {}
        best_valid_losses = {}
        best_train_loss = np.inf
        for valid_dataset_name in self.valid_datasets.keys():
            valid_loaders[valid_dataset_name] = DataLoader(
                dataset = self.valid_datasets[valid_dataset_name],
                batch_size = self.batch_size,
                collate_fn = self.collate_fn,
                shuffle = True,
                num_workers = self.num_data_workers,
                drop_last = True
            )
            valid_loader_iters[valid_dataset_name] = iter(valid_loaders[valid_dataset_name])
            best_valid_losses[valid_dataset_name] = np.inf
            
        self.best_eval_error = np.inf

        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers(
            ['epoch', 'other', 'dataloader', 
             'compute_loss', 'backward', 'eval']
        )

        for epoch in range(self.num_epochs):
            self.time_report.reset_timer()
            
            with TimeProfiler(self.time_report, 'epoch'):
                # Learning rate schedule
                self.lr = self.get_scheduled_learning_rate(epoch, self.num_epochs)

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

                self.logger.init_epoch(epoch)
                
                # Train
                if epoch > 0:
                    avg_train_loss, avg_train_loss_itemized, grad_info = \
                        self.one_epoch(
                            train = True, 
                            dataloader = train_loader,
                            dataloader_iter = train_loader_iter, 
                            num_batches = self.num_train_batches,
                            shuffle = True
                        )

                # Valid
                avg_valid_losses, avg_valid_losses_itemized = {}, {}
                for valid_dataset_name in self.valid_datasets.keys():
                    avg_valid_losses[valid_dataset_name], avg_valid_losses_itemized[valid_dataset_name], _ = \
                        self.one_epoch(
                            train = False, 
                            dataloader = valid_loaders[valid_dataset_name],
                            dataloader_iter = valid_loader_iters[valid_dataset_name], 
                            num_batches = min(
                                self.num_valid_batches, 
                                len(valid_loaders[valid_dataset_name])
                            ),
                            shuffle = False
                        )
                
                # Eval: Rollout evaluation and visualization
                with TimeProfiler(self.time_report, 'eval'):
                    if self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                        self.eval(epoch)

            # Logging
            if epoch % self.log_interval == 0:
                # Print logs on screen
                time_summary = self.time_report.print(
                    string_mode = True, 
                    in_second = True
                )
                print_info("-"*100)
                print_info(f"Epoch {epoch}")
                if epoch > 0:
                    print_info("[Train] loss = {}".format(format_value(avg_train_loss, 8)))
                    print_info("[Train] itemized:\n{}".format(
                        self._format_itemized_multiline(avg_train_loss_itemized, precision=8)
                    ))
                for valid_dataset_name in self.valid_datasets.keys():
                    print_info("[Valid] dataset [{}]: loss = {}".format(
                        valid_dataset_name, 
                        format_value(avg_valid_losses[valid_dataset_name], 8),
                    ))
                    print_info("[Valid] dataset [{}] itemized:\n{}".format(
                        valid_dataset_name,
                        self._format_itemized_multiline(
                            avg_valid_losses_itemized[valid_dataset_name], precision=8
                        )
                    ))
                print_info("[Time Report] {}".format(time_summary))
                if epoch > 0:
                    print_info("[Grad Info] {}".format(format_dict(grad_info, 3)))
                
                # Logging to wandb
                self.logger.add_scalar(
                    'params/lr/epoch', self.lr, epoch)
                    
                if epoch > 0:
                    self.logger.add_scalar(
                        'training/train_loss/epoch', 
                        avg_train_loss, 
                        epoch
                    )
                    self.logger.add_scalar(
                        'training/gradients_before_clip/epoch', 
                        grad_info['grad_norm_before_clip'], 
                        epoch
                    )

                    if self.truncate_grad:
                        self.logger.add_scalar(
                            'training/gradients_after_clip/epoch', 
                            grad_info['grad_norm_after_clip'], 
                            epoch
                        )

                for valid_dataset_name in self.valid_datasets.keys():
                    self.logger.add_scalar(
                        f'training/valid_{valid_dataset_name}_loss/epoch', 
                        avg_valid_losses[valid_dataset_name], 
                        epoch
                    )

                if epoch > 0:
                    for key in avg_train_loss_itemized:
                        self.logger.add_scalar(
                            f'training_info/{key}/epoch', 
                            avg_train_loss_itemized[key], 
                            epoch
                        )
                            
                for valid_dataset_name in self.valid_datasets.keys():
                    for key in avg_valid_losses_itemized[valid_dataset_name]:
                        self.logger.add_scalar(
                            f'validating_info/{key}_{valid_dataset_name}/epoch', 
                            avg_valid_losses_itemized[valid_dataset_name][key], 
                            epoch
                        )

                self.logger.flush()
            
            # Saving model
            if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                self.save_model("model_epoch{}"
                                .format(epoch))
            
            for valid_dataset_name in self.valid_datasets.keys():
                if avg_valid_losses[valid_dataset_name] < best_valid_losses[valid_dataset_name]:
                    best_valid_losses[valid_dataset_name] = avg_valid_losses[valid_dataset_name]
                    self.save_model('best_valid_{}_model'.format(valid_dataset_name))
                    with open(os.path.join(
                            self.model_log_dir, 
                            f'saved_best_valid_{valid_dataset_name}_model_epochs.txt'
                        ), 'a') as fp:
                        fp.write(f"{epoch}\n")
                    print_ok('Save Best Valid {} Model at Epoch {} with loss {}.'.format(
                                valid_dataset_name, 
                                epoch, 
                                format_value(avg_valid_losses[valid_dataset_name], 8)
                            ))

            if epoch > 0 and avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                with open(
                    os.path.join(self.model_log_dir, 'saved_best_train_model_epochs.txt'),
                    'a',
                ) as fp:
                    fp.write(f"{epoch}\n")
                print_ok(
                    'Save Best Train Loss at Epoch {} with loss {}.'
                    .format(epoch, format_value(avg_train_loss, 8))
                )

        self.save_model("final_model")

        self.logger.finish()
            
    @torch.no_grad()
    def eval(self, epoch):
        self.neural_model.eval()
        print_info("-"*100)
        print('Evaluating')
        # eval_error in shape (T, N, state_dim)
        eval_error, eval_trajectories, error_stats = \
            self.evaluator.evaluate_action_mode(
                num_traj = self.num_eval_rollouts,
                eval_mode = 'rollout',
                env_mode = 'neural',
                trajectory_source = self.eval_mode,
                render = self.eval_render,
                passive = self.eval_passive,
            )

        # logging
        for error_metric_name in error_stats['overall'].keys():
            self.logger.add_scalar(
                f'eval_{self.eval_horizon}-steps/{error_metric_name}/epoch',
                error_stats['overall'][error_metric_name],
                epoch
            )
        for error_metric_name in error_stats['step-wise'].keys():
            for i in range(error_stats['step-wise'][error_metric_name].shape[0]):
                self.logger.add_scalar(
                    f'eval_details/{error_metric_name}_step_{i}/epoch',
                    error_stats['step-wise'][error_metric_name][i],
                    epoch
                )
        
        # Evaluator uses unit-aware aliases; older code used q_error(MSE) / qd_error(MSE).
        position_mse = error_stats['overall'].get(
            'position_error_MSE(rad^2)',
            error_stats['overall'].get('q_error(MSE)', float('nan')),
        )
        eval_msg = (
            "[Evaluate], Num Rollouts = {}, Rollout Length = {}, "
            "Rollout MSE Error = {}, Rollout MSE Error (joint_q) = {}".format(
                self.num_eval_rollouts,
                self.eval_horizon,
                format_value(error_stats['overall']['error(MSE)'], 8),
                format_value(position_mse, 8),
            )
        )
        if self.has_lambda_head:
            eval_msg += ", Rollout MSE Error (lambda) = {}".format(
                format_value(error_stats['overall'].get('lambda_error(MSE)', float('nan')), 8),
            )
        print_white(eval_msg)

        primary = float(error_stats['overall'][self.eval_primary_metric])
        if primary < self.best_eval_error:
            self.best_eval_error = primary
            self.save_model('best_eval_model')
            print_ok(
                'Save Best Eval Model at Epoch {} with {} = {}.'
                .format(epoch, self.eval_primary_metric, format_value(primary, 8))
            )
            with open(os.path.join(self.model_log_dir, 'saved_best_eval_model_epochs.txt'), 'a') as fp:
                fp.write(f"{epoch}\n")

    def test(self):
        self.time_report = TimeReport(cuda_synchronize = False)
        self.time_report.add_timers([
            'epoch', 'other', 'dataloader', 
            'compute_loss', 'backward', 'eval'
        ])

        self.neural_model.eval()
        
        valid_loaders = {}
        valid_loader_iters = {}
        best_valid_losses = {}
        for valid_dataset_name in self.valid_datasets.keys():
            valid_loaders[valid_dataset_name] = DataLoader(
                dataset = self.valid_datasets[valid_dataset_name],
                batch_size = self.batch_size,
                collate_fn = self.collate_fn,
                shuffle = True,
                num_workers = self.num_data_workers,
                drop_last = True
            )
            valid_loader_iters[valid_dataset_name] = iter(valid_loaders[valid_dataset_name])
            best_valid_losses[valid_dataset_name] = np.inf

        # Valid
        avg_valid_losses, avg_valid_losses_itemized = {}, {}
        for valid_dataset_name in self.valid_datasets.keys():
            num_valid_batches = len(valid_loaders[valid_dataset_name])
            avg_valid_losses[valid_dataset_name], avg_valid_losses_itemized[valid_dataset_name], _ = \
                self.one_epoch(train = False, 
                            dataloader = valid_loaders[valid_dataset_name],
                            dataloader_iter = valid_loader_iters[valid_dataset_name], 
                            num_batches = num_valid_batches,
                            shuffle = False)
            print_info("Valid dataset [{}]: loss = {}, itemized = {}".format(
                valid_dataset_name, 
                format_value(avg_valid_losses[valid_dataset_name], 8),
                format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)
            ))

        # Rollout Eval
        print('Evaluating')
        num_eval_rollouts = self.num_eval_rollouts
        eval_error, _, error_stats = self.evaluator.evaluate_action_mode(
            num_traj = num_eval_rollouts,
            eval_mode = 'rollout',
            env_mode = 'neural',
            trajectory_source = self.eval_mode,
            render = self.eval_render,
            passive = self.eval_passive
        )
        
        # Logging
        print_info("--------------------------------------------------")
        print_info(f"Test Summary:")
        for valid_dataset_name in self.valid_datasets.keys():
            print_info("Valid dataset [{}]: loss = {}, itemized = {}".format(
                valid_dataset_name, 
                format_value(avg_valid_losses[valid_dataset_name], 8),
                format_dict(avg_valid_losses_itemized[valid_dataset_name], 8)
            ))
        print_info("--------------------------------------------------")
        print_info("Eval ({} rollouts) Error: {}, Error per step: {}".format(
            num_eval_rollouts, 
            format_value((eval_error ** 2).mean(), 8), 
            format_value((eval_error ** 2).mean((-1, -2)), 8)
        ))
        position_mse = error_stats['overall'].get(
            'position_error_MSE(rad^2)',
            error_stats['overall'].get('q_error(MSE)', float('nan'))
        )
        velocity_mse = error_stats['overall'].get(
            'velocity_error_MSE((rad/s)^2)',
            error_stats['overall'].get('qd_error(MSE)', float('nan'))
        )
        print_info(
            "Eval ({} rollouts) Position MSE [rad^2]: {}".format(
                num_eval_rollouts,
                format_value(position_mse, 8),
            )
        )
        print_info(
            "Eval ({} rollouts) Velocity MSE [(rad/s)^2]: {}".format(
                num_eval_rollouts,
                format_value(velocity_mse, 8),
            )
        )
        if self.has_lambda_head and 'lambda_error(MSE)' in error_stats['overall']:
            print_info(
                "Eval ({} rollouts) Lambda Error: {}, Lambda Error per step: {}".format(
                    num_eval_rollouts,
                    format_value(error_stats['overall']['lambda_error(MSE)'], 8),
                    format_value(error_stats['step-wise']['lambda_error(MSE)'], 8)
                )
            )
        print_info("--------------------------------------------------")

        valid_loss_values = list(avg_valid_losses.values())
        valid_loss_total = float(np.mean(valid_loss_values)) if len(valid_loss_values) > 0 else float('nan')
        result = {
            'valid_loss_total': valid_loss_total,
            'state_eval_error_total': float(error_stats['overall']['error(MSE)']),
        }
        if self.has_lambda_head:
            result['lambda_eval_error_total'] = float(
                error_stats['overall'].get('lambda_error(MSE)', float('nan'))
            )
        return result

    def save_model(self, filename = None):
        if filename is None:
            filename = 'best_model'
        
        torch.save(
            [self.neural_model, self.neural_env.robot_name], 
            os.path.join(self.model_log_dir, '{}.pt'.format(filename))
        )