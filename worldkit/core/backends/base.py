"""Base class for world model backends.

Each backend encapsulates a specific world model architecture (LeWM, DreamerV3,
TD-MPC2, etc.) behind a uniform interface so that WorldModel can delegate to
any architecture without changing its public API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from worldkit.core.config import ModelConfig


class BaseWorldModelBackend(ABC):
    """Abstract base class for world model backends.

    A backend knows how to build, encode, predict, rollout, compute costs,
    and perform training steps for a specific world model architecture.
    WorldModel delegates all architecture-specific logic to the backend.
    """

    @abstractmethod
    def build(self, config: ModelConfig) -> nn.Module:
        """Build the underlying neural network module from config.

        Args:
            config: Model configuration.

        Returns:
            The constructed nn.Module.
        """

    @abstractmethod
    def encode(self, model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations into latent representations.

        Args:
            model: The neural network module built by ``build()``.
            pixels: Pixel tensor, shape varies by architecture.

        Returns:
            Latent embeddings tensor.
        """

    @abstractmethod
    def predict(
        self, model: nn.Module, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent states from current state and actions.

        Args:
            model: The neural network module.
            state: Current latent state.
            actions: Raw action tensor (not yet encoded).

        Returns:
            Predicted next latent states.
        """

    @abstractmethod
    def rollout(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        action_sequence: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Perform autoregressive rollout for planning.

        Args:
            model: The neural network module.
            pixels: Context observation frames.
            actions: Context actions.
            action_sequence: Planned action sequences to evaluate.
            context_length: Number of context frames.

        Returns:
            Predicted latent trajectory.
        """

    @abstractmethod
    def get_cost(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        goal: torch.Tensor,
        candidates: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Compute costs for action candidates (used by CEM planner).

        Args:
            model: The neural network module.
            pixels: Context observation frames.
            actions: Context actions.
            goal: Goal observation pixels.
            candidates: Candidate action sequences.
            context_length: Number of context frames.

        Returns:
            Cost tensor — lower is better (closer to goal).
        """

    @abstractmethod
    def training_step(
        self, model: nn.Module, batch: tuple, config: ModelConfig
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one forward pass and compute the loss.

        Does NOT call backward or step the optimizer — the caller handles that.

        Args:
            model: The neural network module.
            batch: Tuple of (pixels, actions) tensors, already on device.
            config: Model configuration (for hyperparameters like context_length).

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict maps metric names
            to float values for logging.
        """
