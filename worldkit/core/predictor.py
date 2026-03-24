"""Autoregressive predictor for latent dynamics.

Given a sequence of latent states and actions, predicts future latent states.
Based on the ARPredictor from LeWM which uses a conditional transformer with
AdaLN-Zero (Adaptive Layer Normalization with zero initialization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-Zero modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Feed-forward network with LayerNorm and GELU."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class Attention(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, self.dim_head).transpose(1, 2) for t in qkv]

        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    Uses adaptive layer normalization to condition on action embeddings.
    Produces 6 modulation parameters: shift/scale/gate for both attention and MLP paths.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass with action conditioning.

        Args:
            x: Input sequence (B, T, D)
            c: Conditioning signal from actions (B, T, D) or (B, 1, D)
        """
        mods = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(
            6, dim=-1
        )

        x_mod = modulate(x, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_mod)

        x_mod = modulate(x, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ff(x_mod)

        return x


class ARPredictor(nn.Module):
    """Autoregressive predictor for latent state dynamics.

    Given a context of past latent states and a sequence of actions,
    predicts future latent states autoregressively.
    """

    def __init__(
        self,
        num_frames: int = 16,
        depth: int = 3,
        heads: int = 4,
        mlp_dim: int = 384,
        input_dim: int = 192,
        hidden_dim: int = 384,
        output_dim: int = 192,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, hidden_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    dim=hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next latent states.

        Args:
            z: Context latent states (B, T_ctx, D)
            action_emb: Action embeddings (B, T, D)

        Returns:
            Predicted latent states (B, T_pred, D)
        """
        B, T, D = z.shape

        x = self.input_proj(z)
        if T <= self.num_frames:
            x = x + self.pos_embed[:, :T]
        else:
            # Interpolate position embeddings for sequences longer than num_frames
            pos = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=T, mode="linear", align_corners=False
            ).transpose(1, 2)
            x = x + pos
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, action_emb[:, :T])

        return self.output_proj(x)
