"""Data loading utilities for WorldKit.

Supports HDF5 and video data formats.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Load training data from HDF5 files.

    Expected HDF5 structure:
        - pixels/observations/obs: (N, T, H, W, C) or (N, T, C, H, W)
        - actions/action: (N, T, action_dim)
    """

    def __init__(
        self,
        path: str | Path,
        sequence_length: int = 16,
        transform=None,
    ):
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.transform = transform

        with h5py.File(self.path, "r") as f:
            for key in ["pixels", "observations", "obs", "images"]:
                if key in f:
                    self.pixel_key = key
                    self.n_episodes = f[key].shape[0]
                    self.episode_length = f[key].shape[1]
                    break
            else:
                raise KeyError(f"No pixel data found. Keys: {list(f.keys())}")

            for key in ["actions", "action"]:
                if key in f:
                    self.action_key = key
                    break
            else:
                raise KeyError(f"No action data found. Keys: {list(f.keys())}")

        self.windows_per_episode = max(1, self.episode_length - self.sequence_length + 1)
        self.total_windows = self.n_episodes * self.windows_per_episode

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int):
        episode_idx = idx // self.windows_per_episode
        window_start = idx % self.windows_per_episode

        with h5py.File(self.path, "r") as f:
            pixels = f[self.pixel_key][
                episode_idx, window_start : window_start + self.sequence_length
            ]
            actions = f[self.action_key][
                episode_idx, window_start : window_start + self.sequence_length
            ]

        pixels = torch.tensor(pixels, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        if pixels.shape[-1] in (1, 3):
            pixels = pixels.permute(0, 3, 1, 2)

        if self.transform:
            pixels = self.transform(pixels)

        return pixels, actions
