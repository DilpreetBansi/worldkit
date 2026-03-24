"""Tests for the ViT encoder."""

import torch

from worldkit.core.encoder import SimpleViT, ViTEncoder


def test_simple_vit():
    vit = SimpleViT(image_size=96, patch_size=16, dim=192, depth=2, heads=4, mlp_dim=384)
    x = torch.randn(2, 3, 96, 96)
    out = vit(x)
    assert out.shape == (2, 192)


def test_vit_encoder():
    encoder = ViTEncoder(image_size=96, patch_size=16, embed_dim=192, latent_dim=128)
    x = torch.randn(2, 3, 96, 96)
    z = encoder(x)
    assert z.shape == (2, 128)


def test_vit_encoder_with_time():
    encoder = ViTEncoder(image_size=96, patch_size=16, embed_dim=192, latent_dim=128)
    x = torch.randn(2, 5, 3, 96, 96)  # (B, T, C, H, W)
    z = encoder(x)
    assert z.shape == (2, 5, 128)
