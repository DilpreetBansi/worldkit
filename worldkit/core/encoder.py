from __future__ import annotations

"""Vision Transformer encoder for WorldKit.

Converts raw pixel observations (RGB images) into compact latent representations
using a ViT backbone with CLS token pooling.
"""

import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    """Vision Transformer encoder that maps pixels to latent vectors.

    Uses a ViT backbone and extracts the CLS token as a compact representation.
    ~200x more compact than patch-level approaches like DINO-WM.
    """

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        embed_dim: int = 384,
        latent_dim: int = 192,
        pretrained_name: str | None = None,
        num_channels: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        actual_embed_dim = embed_dim

        if pretrained_name is not None:
            try:
                from transformers import ViTConfig, ViTModel

                config = ViTConfig.from_pretrained(pretrained_name)
                config.image_size = image_size
                self.vit = ViTModel(config)
                actual_embed_dim = config.hidden_size
            except ImportError:
                pretrained_name = None

        if pretrained_name is None:
            # Pick a head count that divides embed_dim
            for h in [6, 4, 8, 3, 2, 1]:
                if embed_dim % h == 0:
                    n_heads = h
                    break
            self.vit = SimpleViT(
                image_size=image_size,
                patch_size=patch_size,
                dim=embed_dim,
                depth=6,
                heads=n_heads,
                mlp_dim=embed_dim * 4,
                channels=num_channels,
            )
            actual_embed_dim = embed_dim

        self.projection = nn.Sequential(
            nn.LayerNorm(actual_embed_dim),
            nn.Linear(actual_embed_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel observations to latent vectors.

        Args:
            x: Input images, shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            Latent vectors, shape (B, D) or (B, T, D).
        """
        has_time = x.dim() == 5
        if has_time:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)

        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        if hasattr(self.vit, "forward") and hasattr(self.vit, "config"):
            outputs = self.vit(pixel_values=x)
            cls_token = outputs.last_hidden_state[:, 0]
        else:
            cls_token = self.vit(x)

        z = self.projection(cls_token)

        if has_time:
            z = z.reshape(B, T, -1)

        return z


class SimpleViT(nn.Module):
    """Lightweight Vision Transformer for training from scratch."""

    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_dim: int = 1536,
        channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).reshape(B, -1, C * p * p)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]

        for block in self.transformer:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
