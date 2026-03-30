"""Multi-environment dataset for training across heterogeneous data sources.

Wraps multiple HDF5Datasets, handling different action dimensions via
zero-padding and sampling proportionally to dataset sizes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import HDF5Dataset


class MultiEnvironmentDataset(Dataset):
    """Dataset that interleaves samples from multiple HDF5 datasets.

    Handles different action dimensions by zero-padding smaller ones
    to match the maximum action dimension across all datasets.

    Args:
        paths: List of HDF5 file paths.
        sequence_length: Number of timesteps per sample.
        transform: Optional transform applied to pixel tensors.
    """

    def __init__(
        self,
        paths: list[str | Path],
        sequence_length: int = 16,
        transform=None,
    ):
        if not paths:
            raise ValueError("Must provide at least one data path.")

        self.datasets: list[HDF5Dataset] = []
        self.action_dims: list[int] = []

        for p in paths:
            ds = HDF5Dataset(p, sequence_length=sequence_length, transform=transform)
            self.datasets.append(ds)
            # Probe action dim from first sample
            _, actions = ds[0]
            self.action_dims.append(actions.shape[-1])

        self.max_action_dim = max(self.action_dims)

        # Build cumulative length index for O(1) lookup
        self._cumulative_lengths: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cumulative_lengths.append(total)

    def __len__(self) -> int:
        return self._cumulative_lengths[-1] if self._cumulative_lengths else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample, padding actions to max_action_dim if needed.

        Returns:
            (pixels, actions) where actions is zero-padded to max_action_dim.
        """
        # Find which dataset this index belongs to
        ds_idx = 0
        offset = 0
        for i, cum_len in enumerate(self._cumulative_lengths):
            if idx < cum_len:
                ds_idx = i
                break
            offset = cum_len

        local_idx = idx - offset
        pixels, actions = self.datasets[ds_idx][local_idx]

        # Zero-pad actions if this dataset has fewer action dims
        # actions shape: (T, action_dim)
        if actions.shape[-1] < self.max_action_dim:
            pad_size = self.max_action_dim - actions.shape[-1]
            padding = torch.zeros(*actions.shape[:-1], pad_size)
            actions = torch.cat([actions, padding], dim=-1)

        return pixels, actions
