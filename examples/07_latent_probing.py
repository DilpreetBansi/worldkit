"""Explore the latent space of a trained world model."""

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
import numpy as np
import torch

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

print("Encoding observations...")
latents = []
for i in range(20):
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(obs)
    latents.append(z)

latents = torch.stack(latents)
print(f"Latent matrix shape: {latents.shape}")

print(f"\nLatent space statistics:")
print(f"  Mean:   {latents.mean().item():.4f}")
print(f"  Std:    {latents.std().item():.4f}")
print(f"  Min:    {latents.min().item():.4f}")
print(f"  Max:    {latents.max().item():.4f}")

dists = torch.cdist(latents.unsqueeze(0), latents.unsqueeze(0)).squeeze()
print(f"\nPairwise latent distances:")
print(f"  Mean distance: {dists.mean().item():.4f}")
print(f"  Min distance:  {dists[dists > 0].min().item():.4f}")
print(f"  Max distance:  {dists.max().item():.4f}")
