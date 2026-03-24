from __future__ import annotations

"""Action encoder module.

Embeds discrete or continuous actions into the same latent space
as the state representations, for conditioning the predictor.
"""

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Encode actions into embedding space for predictor conditioning.

    Supports both continuous (float vector) and discrete (integer) actions.

    Args:
        action_dim: Dimension of continuous action space, or number of discrete actions.
        embed_dim: Output embedding dimension (should match predictor hidden dim).
        continuous: Whether actions are continuous (True) or discrete (False).
    """

    def __init__(
        self,
        action_dim: int = 2,
        embed_dim: int = 192,
        continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous
        self.embed_dim = embed_dim

        if continuous:
            self.encoder = nn.Sequential(
                nn.Linear(action_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.encoder = nn.Embedding(action_dim, embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions.

        Args:
            actions: (B, T, action_dim) for continuous, (B, T) for discrete.

        Returns:
            Action embeddings: (B, T, embed_dim)
        """
        return self.encoder(actions)
