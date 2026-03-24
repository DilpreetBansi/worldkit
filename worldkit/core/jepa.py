from __future__ import annotations

"""JEPA (Joint-Embedding Predictive Architecture) world model.

Combines encoder, predictor, and action encoder into a unified model
that can encode observations, predict future states, and compute costs
for planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .config import ModelConfig
from .encoder import ViTEncoder
from .predictor import ARPredictor


def detach_clone(v):
    """Detach and clone a tensor, or return as-is if not a tensor."""
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    return v


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture for world modeling.

    Encodes pixel observations into a compact latent space, then predicts
    future latent states conditioned on actions.
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        predictor: ARPredictor,
        action_encoder: ActionEncoder,
        proj_encoder: nn.Module | None = None,
        proj_predictor: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proj_encoder = proj_encoder or nn.Identity()
        self.proj_predictor = proj_predictor or nn.Identity()

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations into latent representations."""
        return self.encoder(pixels)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions into embedding space."""
        return self.action_encoder(actions)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next latent states from current embeddings and action embeddings."""
        return self.predictor(emb, act_emb)

    def rollout(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        action_sequence: torch.Tensor,
        context_length: int = 3,
    ) -> torch.Tensor:
        """Perform autoregressive rollout for planning.

        Args:
            pixels: Initial observation frames (B, T_ctx, C, H, W)
            actions: Context actions (B, T_ctx, action_dim)
            action_sequence: Planned actions to evaluate (B, S, T_plan, action_dim)
                where S = number of candidate action sequences
            context_length: Number of context frames to use.

        Returns:
            Predicted latent trajectory: (B, S, T_plan, D)
        """
        B = pixels.shape[0]

        context_emb = self.encode(pixels[:, :context_length])
        context_act_emb = self.encode_actions(actions[:, :context_length])

        if action_sequence.dim() == 4:
            S = action_sequence.shape[1]
            T_plan = action_sequence.shape[2]

            context_emb = context_emb.unsqueeze(1).expand(-1, S, -1, -1)
            context_emb = context_emb.reshape(B * S, context_length, -1)

            context_act_emb = context_act_emb.unsqueeze(1).expand(-1, S, -1, -1)
            context_act_emb = context_act_emb.reshape(B * S, context_length, -1)

            action_sequence = action_sequence.reshape(B * S, T_plan, -1)
        else:
            S = 1
            T_plan = action_sequence.shape[1]

        plan_act_emb = self.encode_actions(action_sequence)
        full_act_emb = torch.cat([context_act_emb, plan_act_emb], dim=1)

        current_emb = context_emb
        predictions = []

        for t in range(T_plan):
            act_emb_step = full_act_emb[:, : current_emb.shape[1] + 1]
            pred = self.predict(current_emb, act_emb_step)
            next_state = pred[:, -1:]
            predictions.append(next_state)
            current_emb = torch.cat([current_emb, next_state], dim=1)

            if current_emb.shape[1] > context_length + t + 1:
                current_emb = current_emb[:, 1:]

        trajectory = torch.cat(predictions, dim=1)

        if S > 1:
            trajectory = trajectory.reshape(B, S, T_plan, -1)

        return trajectory

    def get_cost(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        goal_pixels: torch.Tensor,
        action_candidates: torch.Tensor,
        context_length: int = 3,
    ) -> torch.Tensor:
        """Compute costs for action candidates (used by CEM planner).

        Returns:
            Costs: (B, S) — lower is better (closer to goal).
        """
        if goal_pixels.dim() == 4:
            goal_pixels = goal_pixels.unsqueeze(1)
        goal_emb = self.encode(goal_pixels).squeeze(1)

        trajectory = self.rollout(pixels, actions, action_candidates, context_length)

        final_states = trajectory[:, :, -1, :]
        goal_expanded = goal_emb.unsqueeze(1).expand_as(final_states)

        costs = F.mse_loss(final_states, goal_expanded, reduction="none").mean(dim=-1)
        return costs

    @classmethod
    def from_config(cls, config: ModelConfig) -> "JEPA":
        """Build a JEPA model from a ModelConfig."""
        encoder = ViTEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.encoder_embed_dim,
            latent_dim=config.latent_dim,
            pretrained_name=None,
        )

        predictor = ARPredictor(
            num_frames=config.sequence_length,
            depth=config.pred_depth,
            heads=config.pred_heads,
            mlp_dim=config.pred_mlp_dim,
            input_dim=config.latent_dim,
            hidden_dim=config.latent_dim,
            output_dim=config.latent_dim,
            dim_head=config.pred_dim_head,
            dropout=config.pred_dropout,
            emb_dropout=config.pred_emb_dropout,
        )

        action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            embed_dim=config.latent_dim,
            continuous=True,
        )

        proj_encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(config.proj_hidden_dim, config.proj_output_dim),
        )

        proj_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(config.proj_hidden_dim, config.proj_output_dim),
        )

        return cls(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            proj_encoder=proj_encoder,
            proj_predictor=proj_predictor,
        )
