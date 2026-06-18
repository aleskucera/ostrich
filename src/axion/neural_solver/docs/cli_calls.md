## Create a Dataset 

### States + contacts (default non-passive, target-position control + additive FF gravity)
```
python src/axion/neural_solver/generate/generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --device cuda:0
```
### States + contacts fields, but **no tilted contact plane** (plane_coefficients are zeros; FF still active)
```
python src/axion/neural_solver/generate/generate_dataset_pendulum.py --env-name Pendulum --num-transitions 10000 --dataset-name pendulumDatasetName.hdf5 --trajectory-length 100 --num-envs 2 --seed 0 --device cuda:0 --without-contacts
```
Use `--passive` to run free/passive dynamics:
- sets `joint_dof_mode=JointMode.NONE` (no implicit target-position tracking),
- records `control_active` as zero and does not apply sampled targets to stepping,
- disables the additive FF gravity term.

Generated HDF5 includes:
- converted NN contact fields (`contact_normals`, `contact_points_0/1`, `contact_depths`, `contact_thicknesses`)
- pre-conversion batched Axion contact arrays under `data/axion_contacts/*` for reconstruction/debug.

## Training

### Begin Simple MLP training
```
python src/axion/neural_solver/train/train_simple_mlp.py \
  --cfg src/axion/neural_solver/train/cfg/Pendulum/simpleMLPNetwork.yaml \
  --logdir src/axion/neural_solver/train/trained_models/simple_mlp \
  --device cuda:0
```
Optional flags: `--no-time-stamp`, `--checkpoint /path/to/checkpoint.pt`

### Begin training (example)
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/
```
optionally: `--device cuda:0`, `--checkpoint /path/to/checkpoint.pt`, `--no-time-stamp`

### Begin lambda classifier training (example)
```
python src/axion/neural_solver/train/train_lambda_network.py --cfg src/axion/neural_solver/train/cfg/Pendulum/lambdaNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/lambda_classifier
```

### Begin residual training (example)
```
python src/axion/neural_solver/train/train_residual_network.py --cfg src/axion/neural_solver/train/cfg/Pendulum/residualNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/residual
```

### Begin MTL training (regression + lambda classification, example)
```
python src/axion/neural_solver/train/train_mtl.py --cfg src/axion/neural_solver/train/cfg/Pendulum/mtlNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/mtl
```

### Begin Contact MTL training (contact lambda classification + contact lambda regression, example)
```
python src/axion/neural_solver/train/train_contact_mtl.py --cfg src/axion/neural_solver/train/cfg/Pendulum/contactMtlNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/contact_mtl
```

### Begin MSE training (pure regression of [q, qd, lambda], example)
```
python src/axion/neural_solver/train/train_mse.py --cfg src/axion/neural_solver/train/cfg/Pendulum/mseNetwork.yaml --logdir src/axion/neural_solver/train/trained_models/mse
```

### Visualize training (wandb)
View runs at [wandb.ai](https://wandb.ai) in your project (e.g. `neural-solver-transformer`). 
Ensure `wandb.login()` runs before training (see `train.py`).

### W&B sweeps
Create a new sweep and start an agent:
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/sweep1 --project neural-solver-transformer
```

Join an existing sweep (replace `<SWEEP_ID>`):
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ --project neural-solver-transformer --sweep_id <SWEEP_ID>
```

Limit how many runs this agent executes:
```
python src/axion/neural_solver/train/sweep_train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --logdir src/axion/neural_solver/train/trained_models/ --project neural-solver-transformer --sweep_id <SWEEP_ID> --count 10
```

## Testing
Test a neural module that has already undergone training:
```
python src/axion/neural_solver/train/train.py --cfg src/axion/neural_solver/train/cfg/Pendulum/transformer.yaml --test --checkpoint src/axion/neural_solver/train/trained_models/03-01-2026-10-46-58/nn/final_model.pt --no-time-stamp --logdir ./eval_logs
```

Run a cli testing script that loads the model.pt into NeuralPredictor class and steps it:
```
python src/axion/neural_solver/standalone/test_trained_model_cli.py --model-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/nn/best_eval_model.pt --cfg-path src/axion/neural_solver/train/trained_models/02-23-2026-23-24-29/cfg.yaml 
```

## Dataset postprocessing
### Add lambda activity labels
```
python src/axion/neural_solver/utils/add_lambda_activity_labels.py --input src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1.hdf5 --output src/axion/neural_solver/datasets/Pendulum/pendulumLambdasValid500klen400envs250seed1_with_activity.hdf5
```

Optional multiclass labels (`0/1/2`) based on `abs(next_lambdas - lambdas)` with cutpoints `0.1` and `1000`:
```
python src/axion/neural_solver/utils/add_lambda_activity_labels.py --input <...>.hdf5 --output <...>_with_activity.hdf5 --multiclass
```

## Misc

### Copy from remote to local (called on local)
```
scp mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumTrainStatesOnly100kenvs100Seed0.hdf5 /home/maros/axion/src/axion/neural_solver/datasets/Pendulum
```

```
scp -r mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/train/trained_models/02-25-2026-13-07-56 /home/maros/axion/src/axion/neural_solver/train/trained_models
```
### Copy from to local to remote (called on local)
```
scp /home/maros/axion/src/axion/neural_solver/datasets/Pendulum/pendulumValidMtlNoContacts2000len500envs2seed1LabelsTh05.hdf5 mestemar@dasenka:/local/mestemar/axion/src/axion/neural_solver/datasets/Pendulum/pendulumValidMtlNoContacts2000len500envs2seed1LabelsTh05.hdf5
```

### Monitoring GPUs/CPU
one time:
```
nvidia-smi
```

continuous monitoring
```
watch -n 1 nvidia-smi
```

```
nvtop?
```

cpu continuous monitoring
```
top
```

### Produce tree structure for thesis
This results in disconnected tree calls:
```
tree examples/conf -I "control"
tree examples/double_pendulum -P "*.py|*.yaml" --prune
tree src/axion/core -P "engine_config.py|gpt_engine.py|hybrid_gpt_engine.py|repeated_engine.py" --prune
tree src/axion/neural_solver \
  -P "*.py|*.yaml" \
  -I "trained_models" \
  --prune
```

This makes a connected tree:
```
(
  echo examples
  find examples/conf -not -path "*/control/*"
  find examples/double_pendulum \( -name "*.py" -o -name "*.yaml" \)

  echo src
  echo src/axion
  find src/axion/core \( \
    -name "engine_config.py" -o \
    -name "gpt_engine.py" -o \
    -name "hybrid_gpt_engine.py" -o \
    -name "repeated_engine.py" \
  \)

  find src/axion/neural_solver \
    -not -path "*/trained_models/*" \
    \( -name "*.py" -o -name "*.yaml" \)
) | tree --fromfile --prune --noreport > src/axion/neural_solver/docs/tree_in_text.txt
```