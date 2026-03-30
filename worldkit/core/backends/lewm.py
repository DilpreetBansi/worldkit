"""LeWM (Le World Model) backend — the default WorldKit architecture.

Wraps the existing JEPA encoder + AR predictor + SIGReg pipeline so that
WorldModel can treat it as one pluggable backend among many.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from worldkit.core.config import ModelConfig
from worldkit.core.jepa import JEPA
from worldkit.core.losses import SIGReg, worldkit_loss

from .base import BaseWorldModelBackend


class LeWMBackend(BaseWorldModelBackend):
    """Backend for the LeWorldModel JEPA architecture.

    This is the default backend. It delegates to the existing JEPA class
    for all neural-network operations and owns the SIGReg regularizer
    used during training.
    """

    def __init__(self) -> None:
        self._sigreg: SIGReg | None = None

    def build(self, config: ModelConfig) -> nn.Module:
        """Build a JEPA module from config."""
        return JEPA.from_config(config)

    def encode(self, model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels via the JEPA encoder."""
        return model.encode(pixels)

    def predict(
        self, model: nn.Module, state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """One-step prediction: encode actions, then predict next states."""
        act_emb = model.encode_actions(actions)
        return model.predict(state, act_emb)

    def rollout(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        action_sequence: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Autoregressive rollout through the JEPA predictor."""
        return model.rollout(pixels, actions, action_sequence, context_length)

    def get_cost(
        self,
        model: nn.Module,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        goal: torch.Tensor,
        candidates: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """Compute goal-distance costs via JEPA's get_cost."""
        return model.get_cost(pixels, actions, goal, candidates, context_length)

    def training_step(
        self, model: nn.Module, batch: tuple, config: ModelConfig
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass + loss for one batch of the LeWM training loop.

        Computes the prediction loss plus SIGReg regularization.
        """
        batch_pixels, batch_actions = batch
        device = batch_pixels.device

        ctx_len = config.context_length
        context_pixels = batch_pixels[:, :ctx_len]  # (B, T_ctx, C, H, W)
        target_pixels = batch_pixels[:, ctx_len:]  # (B, T_tgt, C, H, W)

        context_emb = model.encode(context_pixels)  # (B, T_ctx, D)
        target_emb = model.encode(target_pixels).detach()  # (B, T_tgt, D)
        action_emb = model.encode_actions(batch_actions)  # (B, T, D)
        pred_emb = model.predict(context_emb, action_emb[:, :ctx_len])  # (B, T_pred, D)

        min_len = min(pred_emb.shape[1], target_emb.shape[1])
        pred_for_loss = pred_emb[:, :min_len]
        target_for_loss = target_emb[:, :min_len]

        sigreg = self._ensure_sigreg(config, device)

        total_loss, loss_dict = worldkit_loss(
            predicted=pred_for_loss,
            target=target_for_loss,
            latent_z=context_emb,
            lambda_reg=config.lambda_reg,
            sigreg=sigreg,
        )
        return total_loss, loss_dict

    # ── internal helpers ───────────────────────────────────

    def _ensure_sigreg(self, config: ModelConfig, device: torch.device | str) -> SIGReg:
        """Lazily create and cache the SIGReg module on the right device."""
        if self._sigreg is None:
            self._sigreg = SIGReg(
                knots=config.sigreg_knots,
                num_proj=config.sigreg_num_proj,
            )
        self._sigreg = self._sigreg.to(device)
        return self._sigreg
