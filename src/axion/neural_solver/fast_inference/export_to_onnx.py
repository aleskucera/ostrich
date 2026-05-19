"""
Export a trained MSEModel checkpoint to ONNX.

Press F5 to run. Edit the four constants below to point at your checkpoint.
Output .onnx is written beside the checkpoint by default (OUTPUT = None).
"""

import sys
import warnings
from pathlib import Path

import onnx
import torch
import yaml

# ── configure here ────────────────────────────────────────────────────────────
MODEL_PT    = "src/axion/neural_solver/train/trained_models/mse/04-24-2026-17-02-15/nn/best_valid_valid_model.pt"
NN_MODEL_CFG = "src/axion/neural_solver/train/trained_models/mse/04-24-2026-17-02-15/cfg.yaml"
BATCH_SIZE  = 1          # number of robots/worlds processed in one forward pass
T_OVERRIDE  = None       # None → use yaml `env.utils_provider_cfg.num_states_history`,
                         #        falling back to transformer block_size if missing.
                         # NeuralPredictor caps history at num_states_history,
                         # so the engine T must equal that value (not block_size).
DEVICE      = "cuda:0"   # use "cpu" if no GPU is available
OUTPUT      = None       # None → <checkpoint_stem>.onnx beside the checkpoint
# ──────────────────────────────────────────────────────────────────────────────

try:
    from axion.neural_solver.models.fast_mse_model import FastMSEModel, make_fast_model
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from axion.neural_solver.models.fast_mse_model import FastMSEModel, make_fast_model


def _load_yaml(cfg_path: Path) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_checkpoint(checkpoint_path: Path, device: str):
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(payload, (list, tuple)):
        model = payload[0]
        robot_name = payload[1] if len(payload) > 1 else None
    else:
        model = payload
        robot_name = None
    return model, robot_name


def _cross_validate(fast_model: FastMSEModel, cfg: dict, yaml_path: Path) -> None:
    network = cfg.get("network", {})

    yaml_block_size = network.get("transformer", {}).get("block_size")
    model_block_size = fast_model.transformer_model.config.block_size
    if yaml_block_size is not None and yaml_block_size != model_block_size:
        warnings.warn(
            f"[cross-validate] block_size mismatch: yaml={yaml_block_size}, "
            f"checkpoint={model_block_size}. Using checkpoint value.",
            stacklevel=2,
        )

    yaml_lambda = network.get("enable_lambda_head", False)
    model_has_lambda = fast_model.lambda_output_dim > 0
    if yaml_lambda != model_has_lambda:
        warnings.warn(
            f"[cross-validate] enable_lambda_head mismatch: "
            f"yaml={yaml_lambda}, checkpoint implies lambda_output_dim="
            f"{fast_model.lambda_output_dim}. Checkpoint is authoritative.",
            stacklevel=2,
        )


def _print_summary(
    fast_model: FastMSEModel, low_dim_names: list[str], output_path: Path, T: int
) -> None:
    D = fast_model.total_low_dim
    print("\n--- Export summary ---")
    print(f"  input shape      : (B, T={T}, D={D})")
    print(f"  state_output_dim : {fast_model.state_output_dim}")
    print(f"  lambda_output_dim: {fast_model.lambda_output_dim}")
    if low_dim_names:
        print(f"  low_dim keys (concatenation order):")
        for name in low_dim_names:
            print(f"    {name}")
    else:
        print("  (no yaml cfg supplied — key order unknown)")
    print(f"  output file      : {output_path}")
    print("----------------------\n")


def export(
    checkpoint_path: str,
    cfg_path: str | None = None,
    output_path: str | None = None,
    batch_size: int = 1,
    device: str = "cuda:0",
    t_override: int | None = None,
) -> Path:
    """
    Convert a trained MSEModel checkpoint to ONNX.

    Returns the path to the written .onnx file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load yaml (optional)
    cfg = {}
    low_dim_names: list[str] = []
    if cfg_path is not None:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        low_dim_names = cfg.get("inputs", {}).get("low_dim", [])

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    mse_model, robot_name = _load_checkpoint(checkpoint_path, device)
    if robot_name:
        print(f"  robot: {robot_name}")
    mse_model.eval()

    # Convert to export-friendly model (handles both MSEModel and ModelMixedInput)
    print("Converting to FastMSEModel...")
    fast_model = make_fast_model(mse_model, device=device)
    fast_model.eval()

    # Cross-validate yaml vs checkpoint
    if cfg:
        _cross_validate(fast_model, cfg, cfg_path)

    # Resolve the static sequence length to bake into the ONNX. NeuralPredictor
    # caps history at `num_states_history`, so the engine T must equal that
    # number. block_size is only the transformer's maximum capacity (T ≤ block_size).
    block_size = int(fast_model.transformer_model.config.block_size)
    if t_override is not None:
        T = int(t_override)
    else:
        history_len = (
            cfg.get("env", {})
               .get("utils_provider_cfg", {})
               .get("num_states_history")
        )
        T = int(history_len) if history_len is not None else block_size
    if T > block_size:
        raise ValueError(
            f"T={T} exceeds transformer block_size={block_size}; "
            "lower T_OVERRIDE or retrain with a larger block_size."
        )

    _print_summary(fast_model, low_dim_names, output_path, T=T)

    # Build static dummy input (zeros; shape is all that matters for tracing)
    B = batch_size
    D = fast_model.total_low_dim
    dummy = torch.zeros(B, T, D, device=device)

    # Export
    print(f"Running torch.onnx.export  (opset 18, static shapes B={B} T={T} D={D}) ...")
    torch.onnx.export(
        fast_model,
        dummy,
        str(output_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,       # static shapes required for TRT + CUDA graph capture
        do_constant_folding=True,
    )

    # Validate
    print("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"\nONNX export successful: {output_path}")
    return output_path


if __name__ == "__main__":
    export(
        checkpoint_path=MODEL_PT,
        cfg_path=NN_MODEL_CFG,
        output_path=OUTPUT,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        t_override=T_OVERRIDE,
    )
