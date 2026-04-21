from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResidualAdapterConfig:
    state_dim: int = 26
    task_dim: int = 12
    action_dim: int = 6
    hidden_dim: int = 64
    residual_scale: float = 0.05


class ResidualACTAdapter(nn.Module):
    """A tiny residual head that learns action corrections on top of ACT."""

    def __init__(self, config: ResidualAdapterConfig):
        super().__init__()
        self.config = config
        input_dim = config.state_dim + config.task_dim + config.action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, state: Tensor, task_vector: Tensor, base_action: Tensor) -> Tensor:
        x = torch.cat([state, task_vector, base_action], dim=-1)
        residual = torch.tanh(self.network(x))
        return residual * self.config.residual_scale

    def parameter_vector(self) -> Tensor:
        return nn.utils.parameters_to_vector(self.parameters()).detach().clone()

    def load_parameter_vector(self, vector: Tensor) -> None:
        nn.utils.vector_to_parameters(vector.to(self.parameter_vector().device), self.parameters())


def save_adapter_checkpoint(
    path: str | Path,
    adapter: ResidualACTAdapter,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "config": asdict(adapter.config),
        "state_dict": adapter.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_adapter_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ResidualACTAdapter, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device)
    config = ResidualAdapterConfig(**checkpoint["config"])
    adapter = ResidualACTAdapter(config)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.to(device)
    return adapter, checkpoint.get("metadata", {})
