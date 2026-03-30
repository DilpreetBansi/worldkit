"""AutoConfig — automatic model configuration selection.

Samples data, runs quick training trials with different configs, and
recommends the best configuration given time and device constraints.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, get_config

# Device constraints: maps target device to allowed config names (ascending size).
_DEVICE_CAPS: dict[str, list[str]] = {
    "jetson": ["nano", "base"],
    "browser": ["nano"],
    "mobile": ["nano"],
    "cpu": ["nano", "base"],
}

# All configs ordered by ascending model size.
_CONFIG_ORDER = ["nano", "base", "large", "xl"]

# Minimum number of epochs considered useful for training.
_MIN_USEFUL_EPOCHS = 50


def _parse_time_limit(max_training_time: str) -> float:
    """Parse a human-readable time string into seconds.

    Supports: ``"2h"``, ``"30m"``, ``"1.5h"``, ``"3600s"``.
    """
    time_str = max_training_time.strip().lower()
    if time_str.endswith("h"):
        return float(time_str[:-1]) * 3600
    elif time_str.endswith("m"):
        return float(time_str[:-1]) * 60
    elif time_str.endswith("s"):
        return float(time_str[:-1])
    else:
        raise ValueError(
            f"Cannot parse time limit '{max_training_time}'. "
            "Use format like '2h', '30m', or '3600s'."
        )


def _load_data_sample(
    data_path: Path,
    max_samples: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load a sample of data from an HDF5 file.

    Returns:
        Tuple of (pixels, actions, total_dataset_size).
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        # Find pixel key
        pixel_key: str | None = None
        for key in ("pixels", "observations", "obs", "images"):
            if key in f:
                pixel_key = key
                break
        if pixel_key is None:
            keys = list(f.keys())
            raise KeyError(
                f"No pixel data found. Expected 'pixels', 'observations', "
                f"'obs', or 'images'. Found: {keys}"
            )

        total_samples = f[pixel_key].shape[0]
        # Sample first 10% or max_samples, whichever is smaller
        n_samples = min(max_samples, max(1, int(total_samples * 0.1)))
        pixels = torch.tensor(
            np.array(f[pixel_key][:n_samples]), dtype=torch.float32
        )

        # Find action key
        action_key: str | None = None
        for key in ("actions", "action"):
            if key in f:
                action_key = key
                break
        if action_key is None:
            keys = list(f.keys())
            raise KeyError(
                f"No action data found. Expected 'actions' or 'action'. "
                f"Found: {keys}"
            )
        actions = torch.tensor(
            np.array(f[action_key][:n_samples]), dtype=torch.float32
        )

    # Normalize pixels
    if pixels.max() > 1.0:
        pixels = pixels / 255.0

    # Ensure (N, T, C, H, W)
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(1)
    if pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(0, 1, 4, 2, 3)

    return pixels, actions, total_samples


def _trial_train(
    config_name: str,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    epochs: int = 5,
    device: str = "cpu",
) -> tuple[float, float]:
    """Run a quick training trial.

    Returns:
        Tuple of (final_loss, seconds_per_epoch).  Both ``inf`` on failure.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from .backends import backend_registry

    action_dim = actions.shape[-1] if actions.dim() >= 2 else 1
    cfg = get_config(config_name, action_dim=action_dim)

    backend_cls = backend_registry.get(cfg.backend)
    backend = backend_cls()
    model = backend.build(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    dataset = TensorDataset(pixels, actions)
    loader = DataLoader(
        dataset, batch_size=min(16, len(dataset)), shuffle=True, num_workers=0
    )

    epoch_times: list[float] = []
    last_loss = float("inf")

    for _epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        steps = 0

        for batch_px, batch_act in loader:
            batch_px = batch_px.to(device)
            batch_act = batch_act.to(device)

            optimizer.zero_grad()
            try:
                loss, _ = backend.training_step(model, (batch_px, batch_act), cfg)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
            except RuntimeError:
                # OOM or shape mismatch — this config is too large
                return float("inf"), float("inf")

        epoch_times.append(time.time() - epoch_start)
        last_loss = epoch_loss / max(steps, 1)

    avg_epoch_time = (
        sum(epoch_times) / len(epoch_times) if epoch_times else float("inf")
    )
    return last_loss, avg_epoch_time


def auto_config(
    data: str | Path,
    max_training_time: str = "2h",
    target_device: str | None = None,
    trial_epochs: int = 5,
    device: str = "cpu",
) -> tuple[ModelConfig, str]:
    """Recommend a model configuration based on data and constraints.

    Samples the data, runs quick training trials with different configs,
    and picks the best one that fits within time and device constraints.

    Args:
        data: Path to HDF5 training data.
        max_training_time: Maximum training time budget (e.g. ``"2h"``, ``"30m"``).
        target_device: Deployment target. Caps config size.
            Options: ``"jetson"``, ``"browser"``, ``"mobile"``, ``"cpu"``,
            or ``None`` (no cap).
        trial_epochs: Epochs per trial run (default 5).
        device: Device for running trials.

    Returns:
        Tuple of (recommended ModelConfig, human-readable explanation).
    """
    data_path = Path(data)
    time_budget_s = _parse_time_limit(max_training_time)

    # Determine allowed configs for target device
    if target_device and target_device in _DEVICE_CAPS:
        allowed_configs = _DEVICE_CAPS[target_device]
    else:
        allowed_configs = list(_CONFIG_ORDER)

    # Load data sample and total size
    pixels, actions, total_samples = _load_data_sample(data_path)
    action_dim = actions.shape[-1] if actions.dim() >= 2 else 1
    n_samples = pixels.shape[0]

    # Run training trials
    # name -> (loss, trial_sec_per_epoch, estimated_full_sec_per_epoch)
    results: dict[str, tuple[float, float, float]] = {}

    for config_name in allowed_configs:
        loss, sec_per_epoch = _trial_train(
            config_name,
            pixels,
            actions,
            epochs=trial_epochs,
            device=device,
        )
        if sec_per_epoch == float("inf"):
            continue

        # Extrapolate to full dataset
        scale_factor = total_samples / max(n_samples, 1)
        estimated_epoch_time = sec_per_epoch * scale_factor
        results[config_name] = (loss, sec_per_epoch, estimated_epoch_time)

    if not results:
        cfg = get_config("nano", action_dim=action_dim)
        explanation = (
            "AutoConfig: No config completed trials successfully. "
            "Recommending 'nano' as the safest default."
        )
        return cfg, explanation

    # Pick the largest config that fits >= _MIN_USEFUL_EPOCHS in the budget
    best_name: str | None = None
    best_loss = float("inf")

    for config_name in reversed(_CONFIG_ORDER):
        if config_name not in results:
            continue
        loss, _, est_epoch_time = results[config_name]
        max_epochs = int(time_budget_s / max(est_epoch_time, 0.001))

        if max_epochs >= _MIN_USEFUL_EPOCHS and loss < best_loss:
            best_name = config_name
            best_loss = loss

    # Fallback: pick the config that allows the most epochs
    if best_name is None:
        most_epochs = 0
        for config_name, (_loss, _, est_epoch_time) in results.items():
            max_epochs = int(time_budget_s / max(est_epoch_time, 0.001))
            if max_epochs > most_epochs:
                most_epochs = max_epochs
                best_name = config_name

    if best_name is None:
        best_name = "nano"

    cfg = get_config(best_name, action_dim=action_dim)

    # Build explanation
    loss, _sec_epoch, est_epoch = results.get(best_name, (0.0, 0.0, 0.001))
    max_epochs = int(time_budget_s / max(est_epoch, 0.001))
    parts = [
        f"AutoConfig recommends '{best_name}' config.",
        f"Data: {total_samples} samples, action_dim={action_dim}.",
        f"Trial loss ({trial_epochs} epochs): {loss:.4f}.",
        f"Estimated {est_epoch:.1f}s/epoch on full data"
        f" -> ~{max_epochs} epochs in {max_training_time}.",
    ]
    if target_device:
        parts.append(
            f"Device constraint '{target_device}'"
            f" limits configs to: {allowed_configs}."
        )

    return cfg, " ".join(parts)
