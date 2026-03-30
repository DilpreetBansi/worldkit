"""DreamerV3 backend stub.

Registered in the backend registry to signal that multi-architecture support
is real and community contributions are welcome. All methods raise
``NotImplementedError`` until a full implementation is provided.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from worldkit.core.config import ModelConfig

from .base import BaseWorldModelBackend

_MSG = (
    "DreamerV3 backend coming soon. "
    "See github.com/worldkit-ai/worldkit/issues for progress."
)


class DreamerV3Backend(BaseWorldModelBackend):
    """Stub backend for the DreamerV3 architecture."""

    def build(self, config: ModelConfig) -> nn.Module:
        raise NotImplementedError(_MSG)

    def encode(self, model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(_MSG)

    def predict(
        self, model: nn.Module, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(_MSG)

    def rollout(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        action_sequence: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        raise NotImplementedError(_MSG)

    def get_cost(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        goal: torch.Tensor,
        candidates: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        raise NotImplementedError(_MSG)

    def training_step(
        self, model: nn.Module, batch: tuple, config: ModelConfig
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError(_MSG)
