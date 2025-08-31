from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ding.utils import MODEL_REGISTRY


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, last_linear_layer_init_zero: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        if last_linear_layer_init_zero and isinstance(self.net[-1], nn.Linear):
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODEL_REGISTRY.register('SplendorAlphaZeroModel')
class SplendorAlphaZeroModel(nn.Module):
    """
    MLP-based AlphaZero model for Splendor's vector observation.
    Expects input state of shape (B, C, H, W) or (B, N). Internally flattens.
    """

    def __init__(
        self,
        observation_shape: Sequence[int] = (1, 15, 15),
        action_space_size: int = 45,
        policy_hidden_sizes: Sequence[int] = (512, 256),
        value_hidden_sizes: Sequence[int] = (512, 256),
        last_linear_layer_init_zero: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.action_space_size = int(action_space_size)
        # Infer flatten size
        if len(observation_shape) == 1:
            self.flatten_dim = int(observation_shape[0])
        else:
            c, h, w = observation_shape
            self.flatten_dim = int(c) * int(h) * int(w)

        self.policy_mlp = MLP(self.flatten_dim, list(policy_hidden_sizes), self.action_space_size, last_linear_layer_init_zero)
        self.value_mlp = MLP(self.flatten_dim, list(value_hidden_sizes), 1, last_linear_layer_init_zero)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = state_batch
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        logits = self.policy_mlp(x)
        value = self.value_mlp(x)
        return logits, value

    def compute_policy_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state_batch)
        probs = F.softmax(logits, dim=-1)
        return probs, value


