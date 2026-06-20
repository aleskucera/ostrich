
"""
Config for sweeping training hyperparameters of 4-element state prediction from scratch.
"""
sweep_config_0 = {
    "method": "bayes",
    "name": "neural-solver-transformer-sweep-0",
    "metric": {"goal": "minimize", "name": "eval_10-steps/error(MSE)/epoch"},
    "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    "parameters": {
        "lr_start": {"max": 1e-2, "min": 1e-4},
        "lr_schedule": {"values": ["linear", "cosine", "constant"]},
        "dropout": {"values": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]},
        "batch_size": {"values": [512, 1024, 2048, 4096]},
        "normalize_input": {"values": [True, False]},
        "normalize_output": {"values": [True, False]},
        "optimizer": {"values": ["adam", "adamw"]},
        "weight_decay": {"max": 1e-3, "min": 1e-5},
        "huber_delta": {"max": 2.0, "min": 0.25},
        "kinematics_loss_weight": {"max": 1.5, "min": 0.0},
    },
}

"""
Config for sweeping lambda prediction training with state prediction already pre-trained.
"""
sweep_config_1 = {
    "method": "bayes",
    "name": "neural-solver-transformer-sweep-1",
    "metric": {"goal": "minimize", "name": "eval_10-steps/lambda_error(MSE)/epoch"},
    "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    "parameters": {
        "lambda_prediction_type": {"values": ["relative", "absolute"]},
        "skip_connection": {"values": [True, False]},
        "layer_norm": {"values": [True, False]},
        "lambda_head_size": {"values": [ [32], [64], [128], [256], [512], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512], [32, 32, 32], [64, 64, 64]]},
        "loss_type": {"values": ["l1", "mse", "huber"]},
        "huber_delta": {"max": 2.0, "min": 0.25},
        "dropout": {"max": 0.2, "min": 0.0},
        "normalize_input": {"values": [True, False]},
        "normalize_output": {"values": [True, False]},
    },
}

"""
Config for sweeping hyperparameters of Simple MLP network.
"""
sweep_config_mlp_0 = {
    "method": "bayes",
    "name": "neural-solver-mlp-sweep-0",
    "metric": {"goal": "minimize", "name": "training/valid_valid_loss/epoch"},
    "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
    "parameters": {
        "depth": {"values": [1,2,3,4,5]},
        "width": {"values":[16,32,64,128,256]},
        "layernorm": {"values": [True, False]},
        "activation": {"values": ["relu", "gelu", "tanh"]} 
    },
}