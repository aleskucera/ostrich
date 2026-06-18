import torch
import torch.nn as nn

from axion.neural_solver.models.base_models import MLPBase


class SimpleMlpModel(nn.Module):
    """
    Flat fully-connected network for single-step state prediction.

    Architecture: low_dim encoder → main MLP trunk → linear output head.

    ``depth`` and ``width`` from ``network_cfg`` derive the trunk's hidden
    layer sizes when ``network_cfg['model']['mlp']['layer_sizes']`` is empty
    (e.g. depth=3, width=256 → [256, 256, 256]).  Non-empty ``layer_sizes``
    always take precedence.

    Input tensors can be (B, D) or (B, T, D); MLPBase handles both shapes.
    """

    def __init__(
        self,
        input_sample,
        state_output_dim: int,
        input_cfg: dict,
        network_cfg: dict,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = device
        self.normalize_input: bool = bool(network_cfg.get("normalize_input", False))
        self.normalize_output: bool = bool(network_cfg.get("normalize_output", False))
        self.input_rms = None
        self.output_rms = None

        # --- Low-dim encoder ---
        self.low_dim_input_names = list(input_cfg.get("low_dim", []))
        if not self.low_dim_input_names:
            raise ValueError("SimpleMlpModel requires at least one low_dim input.")

        low_dim_size = sum(
            input_sample[name].shape[-1] for name in self.low_dim_input_names
        )
        encoder_low_dim_cfg = (
            network_cfg.get("encoder", {}).get("low_dim")
            or {"layer_sizes": [], "activation": "relu", "layernorm": False}
        )
        self.encoder = MLPBase(low_dim_size, encoder_low_dim_cfg, device=device)

        # --- Main MLP trunk (depth × width or explicit layer_sizes) ---
        trunk_cfg = dict(
            network_cfg.get("model", {}).get("mlp")
            or {"layer_sizes": [], "activation": "relu", "layernorm": False}
        )
        layer_sizes = list(trunk_cfg.get("layer_sizes") or [])
        if not layer_sizes:
            depth = int(network_cfg.get("depth", 3))
            width = int(network_cfg.get("width", 256))
            layer_sizes = [width] * depth
        trunk_cfg["layer_sizes"] = layer_sizes
        self.trunk = MLPBase(self.encoder.out_features, trunk_cfg, device=device)

        # --- Output head ---
        self.state_output_dim = int(state_output_dim)
        self.output_head = nn.Linear(self.trunk.out_features, self.state_output_dim).to(device)

    # ------------------------------------------------------------------
    # RMS setters (called by trainer after computing dataset statistics)
    # ------------------------------------------------------------------

    def set_input_rms(self, data_rms: dict) -> None:
        self.input_rms = {
            name: data_rms[name]
            for name in self.low_dim_input_names
            if name in data_rms
        }

    def set_output_rms(self, output_rms=None, lambda_output_rms=None) -> None:
        # lambda_output_rms is accepted for API compatibility but ignored.
        self.output_rms = output_rms

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict: dict) -> torch.Tensor:
        """Return state predictions with shape matching the input leading dims."""
        if self.normalize_input and self.input_rms is not None:
            input_dict = dict(input_dict)
            for name, rms in self.input_rms.items():
                if name in input_dict:
                    input_dict[name] = rms.normalize(input_dict[name])

        low_dim_parts = [input_dict[name] for name in self.low_dim_input_names]
        x = torch.cat(low_dim_parts, dim=-1)

        x = self.encoder(x)
        x = self.trunk(x)
        x = self.output_head(x)

        if self.normalize_output and self.output_rms is not None:
            x = self.output_rms.normalize(x, un_norm=True)

        return x

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.trunk.to(device)
        self.output_head.to(device)
        if self.input_rms is not None:
            for key in self.input_rms:
                self.input_rms[key] = self.input_rms[key].to(device)
        if self.output_rms is not None:
            self.output_rms = self.output_rms.to(device)
        return self
