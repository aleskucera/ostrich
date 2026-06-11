import torch
import torch.nn as nn
import warp as wp

from ostrich.core.engine_config import EngineConfig
from ostrich.core.engine_data import EngineData
from ostrich.core.engine_dims import EngineDimensions
from ostrich.core.model import OstrichModel
from ostrich.optim import SystemOperator

from ostrich.core.contacts import OstrichContacts
from .torch_residual import OstrichResidual


class WarmStartNet(nn.Module):
    """Neural network that predicts initial (body_vel, constr_force) for the solver.

    Input:  previous body poses (7 * body_count) + previous body velocities (6 * body_count)
    Output: body_vel (N_u) + constr_force (num_constraints)
    """

    def __init__(self, dims: EngineDimensions, hidden_dim: int = 256, num_hidden_layers: int = 2):
        super().__init__()
        # Input: body_pose_prev (7 per body) + body_vel_prev (6 per body)
        input_dim = 7 * dims.body_count + 6 * dims.body_count
        output_dim = dims.N_u + dims.num_constraints

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.N_u = dims.N_u

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (num_worlds, input_dim) flattened body_pose_prev + body_vel_prev

        Returns:
            body_vel:     (num_worlds, N_u)
            constr_force: (num_worlds, num_constraints)
        """
        out = self.net(state)
        body_vel = out[:, : self.N_u]
        constr_force = out[:, self.N_u :]
        return body_vel, constr_force


class WarmStartTrainer:
    """Trains WarmStartNet to minimize the residual norm at predicted initial states."""

    def __init__(
        self,
        net: WarmStartNet,
        model: OstrichModel,
        contacts: OstrichContacts,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        A_op: SystemOperator,
        lr: float = 1e-3,
    ):
        self.net = net
        self.model = model
        self.contacts = contacts
        self.data = data
        self.config = config
        self.dims = dims
        self.A_op = A_op
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def get_state_tensor(self) -> torch.Tensor:
        """Extract current state from engine data as a flat torch tensor."""
        body_pose_prev = wp.to_torch(self.data.body_pose_prev).clone()
        body_vel_prev = wp.to_torch(self.data.body_vel_prev).clone()

        # Flatten per-body data: (num_worlds, body_count, 7) -> (num_worlds, 7*body_count)
        body_pose_flat = body_pose_prev.reshape(self.dims.num_worlds, -1)
        body_vel_flat = body_vel_prev.reshape(self.dims.num_worlds, -1)

        return torch.cat([body_pose_flat, body_vel_flat], dim=-1)

    def train_step(self) -> float:
        """Run one training step: predict initial guess, compute residual loss, backprop.

        Call this after load_data() but before _solve(), so that body_pose_prev
        and body_vel_prev are set for the current timestep.

        Returns:
            loss value (float)
        """
        self.optimizer.zero_grad()

        state = self.get_state_tensor()
        body_vel, constr_force = self.net(state)

        residual = OstrichResidual.apply(
            self.model,
            self.contacts,
            self.data,
            self.config,
            self.dims,
            self.A_op,
            body_vel,
            constr_force,
        )

        loss = torch.sum(residual**2)
        loss.backward()
        self.optimizer.step()

        return loss.item()
