"""
Build a TensorRT engine (.plan) from a previously exported ONNX file.

Press F5 to run. Edit the constants below to point at the .onnx produced by
export_to_onnx.py and the matching trained .pt checkpoint (only used to
extract input RMS statistics + dims for the metadata sidecar).

Outputs (both written beside the .onnx by default):
  <stem>.plan              — serialized TensorRT engine, fixed (B, T, D) shapes
  <stem>.engine_meta.pt    — torch.save dict with low_dim_keys, dims, input_rms

Both files are consumed by TensorRTMSEEngine at inference time.

TensorRT 10.x API is used throughout:
  - builder.create_network()        (explicit batch is the default)
  - config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, ...)
  - builder.build_serialized_network(...)
  - tensor I/O via get_tensor_name / get_tensor_shape / TensorIOMode
"""

import sys
import warnings
from pathlib import Path

import tensorrt as trt
import torch
import yaml

# ── configure here ────────────────────────────────────────────────────────────
ONNX_PATH         = "src/axion/neural_solver/train/trained_models/mse/04-24-2026-17-02-15/nn/best_valid_valid_model.onnx"
MODEL_PT    = "src/axion/neural_solver/train/trained_models/mse/04-24-2026-17-02-15/nn/best_valid_valid_model.pt"
NN_MODEL_CFG = "src/axion/neural_solver/train/trained_models/mse/04-24-2026-17-02-15/cfg.yaml"
FP16              = False       # falls back to FP32 if the GPU has no fast FP16
WORKSPACE_GB      = 2          # tactic-search scratch memory, not runtime memory
OUTPUT_PLAN       = None       # None → <onnx_stem>.plan beside the ONNX
OUTPUT_META       = None       # None → <onnx_stem>.engine_meta.pt beside the ONNX
RUN_PARITY_CHECK  = True       # compare TRT vs FastMSEModel on a random tensor
DEVICE            = "cuda:0"
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


def _resolve_low_dim_keys(cfg: dict, mse_model) -> list[str]:
    """yaml is authoritative for the concat order; fall back to the trained model."""
    cfg_keys = cfg.get("inputs", {}).get("low_dim", []) if cfg else []
    if cfg_keys:
        return list(cfg_keys)
    model_keys = getattr(mse_model, "low_dim_input_names", None)
    if model_keys is None:
        raise RuntimeError(
            "Cannot determine low_dim key concatenation order: neither yaml "
            "inputs.low_dim nor mse_model.low_dim_input_names is available."
        )
    return list(model_keys)


def _extract_input_rms(mse_model, low_dim_keys: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """
    Pull per-key mean/var from the trained MSEModel's running_mean_std table.

    The trainer stores these in mse_model.input_rms (a dict-of-RunningMeanStd).
    For checkpoints trained with normalize_input: false this dict may be None,
    in which case we emit zero-mean / unit-var placeholders that act as a
    no-op at inference time.
    """
    rms: dict[str, dict[str, torch.Tensor]] = {}
    source = getattr(mse_model, "input_rms", None)
    for key in low_dim_keys:
        if source is not None and key in source:
            rms[key] = {
                "mean": source[key].mean.detach().cpu().clone(),
                "var":  source[key].var.detach().cpu().clone(),
            }
        else:
            # No-op normalization: (x - 0) * 1/sqrt(1 + eps) ≈ x.
            warnings.warn(
                f"[build_tensorrt_engine] input_rms entry for '{key}' is "
                "absent on the checkpoint. Emitting no-op normalization "
                "(mean=0, var=1) for this key.",
                stacklevel=2,
            )
            rms[key] = {
                "mean": torch.zeros(1, dtype=torch.float32),
                "var":  torch.ones(1, dtype=torch.float32),
            }
    return rms


def _io_tensor_info(engine):
    """TRT 10 tensor I/O introspection — returns lists of (name, shape, dtype)."""
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        target = inputs if mode == trt.TensorIOMode.INPUT else outputs
        target.append((name, shape, dtype))
    return inputs, outputs


def _build_engine(onnx_path: Path, fp16: bool, workspace_gb: int) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(parser.get_error(i).desc() for i in range(parser.num_errors))
        raise RuntimeError(f"OnnxParser failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1024 ** 3)
    )
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled (platform has fast fp16).")
        else:
            print("  FP16 requested but platform_has_fast_fp16 is False — building FP32.")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None (see logger output).")
    return bytes(serialized)


def _parity_check(
    plan_bytes: bytes,
    fast_model: FastMSEModel,
    device: str,
    fp16: bool,
) -> None:
    """One-shot TRT inference on a random tensor; compare against FastMSEModel."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    context = engine.create_execution_context()

    inputs, outputs = _io_tensor_info(engine)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"Expected 1 input / 1 output, got {len(inputs)} / {len(outputs)}."
        )
    in_name, in_shape, _ = inputs[0]
    out_name, out_shape, _ = outputs[0]

    torch.manual_seed(0)
    # Inputs at the post-normalization scale (the engine sees std≈1 data at
    # inference time). Using std=1 here matches the realistic distribution
    # and keeps activations inside FP16 range.
    x = torch.randn(*in_shape, device=device, dtype=torch.float32)
    in_buf = x.contiguous()
    out_buf = torch.empty(*out_shape, device=device, dtype=torch.float32)

    context.set_tensor_address(in_name, int(in_buf.data_ptr()))
    context.set_tensor_address(out_name, int(out_buf.data_ptr()))

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT execute_async_v3 returned False during parity check.")
    stream.synchronize()

    with torch.no_grad():
        torch_out = fast_model(x)

    diff = (torch_out - out_buf).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    torch_scale = float(torch_out.abs().max().item())
    trt_scale = float(out_buf.abs().max().item())
    denom = max(torch_scale, trt_scale, 1e-12)
    rel_max = max_err / denom
    tol_hint = "rel ~1e-3" if fp16 else "rel ~1e-5"
    print(
        f"  parity vs FastMSEModel: "
        f"max_abs={max_err:.3e}, mean_abs={mean_err:.3e}, "
        f"rel_max={rel_max:.3e}  "
        f"(|torch|_max={torch_scale:.3e}, |trt|_max={trt_scale:.3e}, expected {tol_hint})"
    )


def build(
    onnx_path: str,
    checkpoint_path: str,
    cfg_path: str | None,
    output_plan: str | None,
    output_meta: str | None,
    fp16: bool,
    workspace_gb: int,
    run_parity_check: bool,
    device: str,
) -> tuple[Path, Path]:
    onnx_path = Path(onnx_path)
    checkpoint_path = Path(checkpoint_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    plan_path = Path(output_plan) if output_plan else onnx_path.with_suffix(".plan")
    meta_path = Path(output_meta) if output_meta else onnx_path.with_suffix(".engine_meta.pt")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    if cfg_path is not None:
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = _load_yaml(cfg_path)

    print(f"Loading checkpoint (for metadata): {checkpoint_path}")
    mse_model, robot_name = _load_checkpoint(checkpoint_path, device)
    if robot_name:
        print(f"  robot: {robot_name}")
    mse_model.eval()

    low_dim_keys = _resolve_low_dim_keys(cfg, mse_model)
    print(f"  low_dim concatenation order: {low_dim_keys}")

    print(f"Building TensorRT engine from: {onnx_path}")
    plan_bytes = _build_engine(onnx_path, fp16=fp16, workspace_gb=workspace_gb)
    plan_path.write_bytes(plan_bytes)
    print(f"  wrote {plan_path}  ({len(plan_bytes) / (1024 * 1024):.2f} MB)")

    # Re-deserialize once to discover I/O shapes (single source of truth: the engine).
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_bytes)
    inputs_io, outputs_io = _io_tensor_info(engine)
    if len(inputs_io) != 1 or len(outputs_io) != 1:
        raise RuntimeError(
            f"Expected 1 input / 1 output, got {len(inputs_io)} / {len(outputs_io)}."
        )
    in_name, in_shape, _ = inputs_io[0]
    out_name, out_shape, _ = outputs_io[0]
    if any(d < 0 for d in in_shape) or any(d < 0 for d in out_shape):
        raise RuntimeError(
            f"Engine has dynamic dims (input={in_shape}, output={out_shape}); "
            "TRT + CUDA Graph requires fully static shapes. Re-export ONNX with dynamic_axes=None."
        )

    B, T, D = int(in_shape[0]), int(in_shape[1]), int(in_shape[2])
    Bo, To, R = int(out_shape[0]), int(out_shape[1]), int(out_shape[2])
    if (Bo, To) != (B, T):
        raise RuntimeError(
            f"Input/output batch+seq mismatch: input={(B, T)}, output={(Bo, To)}."
        )

    _lambda_head = getattr(mse_model, "lambda_model", None)
    state_output_dim = int(
        mse_model.state_output_dim
        if hasattr(mse_model, "state_output_dim")
        else mse_model.model.output_net.out_features
    )
    lambda_output_dim = int(
        mse_model.lambda_output_dim
        if hasattr(mse_model, "lambda_output_dim")
        else (_lambda_head.output_net.out_features if _lambda_head is not None else 0)
    )
    if state_output_dim + lambda_output_dim != R:
        raise RuntimeError(
            f"regression_output_dim mismatch: "
            f"state({state_output_dim}) + lambda({lambda_output_dim}) "
            f"!= engine output dim {R}."
        )

    block_size = int(mse_model.transformer_model.config.block_size)
    if block_size != T:
        warnings.warn(
            f"block_size on the checkpoint ({block_size}) does not match "
            f"engine T ({T}); engine T is authoritative.",
            stacklevel=2,
        )

    input_rms = _extract_input_rms(mse_model, low_dim_keys)
    low_dim_dims: dict[str, int] = {}
    rolling = 0
    for key in low_dim_keys:
        d_key = int(input_rms[key]["mean"].shape[-1])
        low_dim_dims[key] = d_key
        rolling += d_key
    if rolling != D:
        # Mean/var shape can be smaller than the actual feature dim (e.g. scalar
        # RMS over a flattened tensor). In that case we cannot infer per-key
        # dims from the RMS table — emit a warning but persist what we have.
        warnings.warn(
            f"Sum of low_dim_dims inferred from input_rms ({rolling}) != "
            f"engine total_low_dim ({D}). Persisting RMS as-is; the wrapper "
            "will broadcast.",
            stacklevel=2,
        )

    network_cfg = cfg.get("network", {}) if cfg else {}
    normalize_input = bool(
        network_cfg.get("normalize_input", getattr(mse_model, "normalize_input", False))
    )
    normalize_output = bool(
        network_cfg.get("normalize_output", getattr(mse_model, "normalize_output", False))
    )

    # ModelMixedInput stores it as `output_rms`; MSEModel as `regression_output_rms`.
    output_rms_meta = None
    if normalize_output:
        out_rms = getattr(mse_model, "regression_output_rms", None) or \
                  getattr(mse_model, "output_rms", None)
        if out_rms is not None:
            output_rms_meta = {
                "mean": out_rms.mean.detach().cpu().clone(),
                "var":  out_rms.var.detach().cpu().clone(),
            }
            print(f"  output_rms: mean shape={tuple(output_rms_meta['mean'].shape)}, "
                  f"var shape={tuple(output_rms_meta['var'].shape)}")
        else:
            warnings.warn(
                "[build_tensorrt_engine] normalize_output=True but output_rms "
                "absent on checkpoint; TRT path will skip output un-normalization.",
                stacklevel=2,
            )
    else:
        print("  output_rms: skipped (normalize_output=False)")

    meta = {
        "low_dim_keys": low_dim_keys,
        "low_dim_dims": low_dim_dims,
        "state_output_dim": state_output_dim,
        "lambda_output_dim": lambda_output_dim,
        "regression_output_dim": R,
        "total_low_dim": D,
        "block_size": T,
        "batch_size": B,
        "input_rms": input_rms,
        "output_rms": output_rms_meta,
        "normalize_input": normalize_input,
        "normalize_output": normalize_output,
        "engine_filename": plan_path.name,
        "fp16": bool(fp16),
    }
    torch.save(meta, str(meta_path))
    print(f"  wrote {meta_path}")
    print(f"  engine I/O: input{tuple(in_shape)} -> output{tuple(out_shape)} "
          f"(state={state_output_dim}, lambda={lambda_output_dim}, "
          f"normalize_input={normalize_input}, normalize_output={normalize_output})")

    if run_parity_check:
        print("Running parity check (FastMSEModel vs TRT, no normalization)...")
        fast_model = make_fast_model(mse_model, device=device)
        fast_model.eval()
        _parity_check(plan_bytes, fast_model, device=device, fp16=fp16)

    print(f"\nTensorRT engine build successful: {plan_path}")
    return plan_path, meta_path


if __name__ == "__main__":
    build(
        onnx_path=ONNX_PATH,
        checkpoint_path=MODEL_PT,
        cfg_path=NN_MODEL_CFG,
        output_plan=OUTPUT_PLAN,
        output_meta=OUTPUT_META,
        fp16=FP16,
        workspace_gb=WORKSPACE_GB,
        run_parity_check=RUN_PARITY_CHECK,
        device=DEVICE,
    )
