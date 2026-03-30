"""WKFormat — versioned ZIP archive format for WorldKit models.

A .wk file is a ZIP archive containing:
    config.json          — Full ModelConfig serialized as JSON
    weights.safetensors  — Model weights in safetensors format
    metadata.json        — Training info: dataset, epochs, loss, timing, version
    action_space.json    — Action space definition: dim, bounds, type
    model_card.yaml      — Human-readable model card

Format version: 2 (version 1 was raw torch.save pickle).
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

import worldkit
from worldkit.core.config import ModelConfig

WK_FORMAT_VERSION = 2
_REQUIRED_ENTRIES = {"config.json", "weights.safetensors", "metadata.json"}


class WKFormat:
    """Read/write the .wk ZIP archive format."""

    @staticmethod
    def save(
        path: str | Path,
        model_state_dict: dict[str, torch.Tensor],
        config: ModelConfig,
        metadata: dict[str, Any] | None = None,
        action_space: dict[str, Any] | None = None,
        model_card: dict[str, Any] | None = None,
    ) -> Path:
        """Save a model to the .wk ZIP archive format.

        Args:
            path: Destination file path (should end with .wk).
            model_state_dict: PyTorch state dict of the JEPA model.
            config: The ModelConfig used to build the model.
            metadata: Optional training metadata (dataset, epochs, loss, etc.).
            action_space: Optional action space definition.
            model_card: Optional model card dict (written as YAML).

        Returns:
            The path the file was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata with defaults
        meta = {
            "worldkit_version": worldkit.__version__,
            "format_version": WK_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            meta.update(metadata)

        # Build action space with defaults from config
        act = {
            "dim": config.action_dim,
            "type": "continuous",
            "low": -1.0,
            "high": 1.0,
        }
        if action_space:
            act.update(action_space)

        # Build model card with defaults
        card = {
            "name": config.name,
            "description": f"WorldKit {config.name} model",
            "architecture": "JEPA + SIGReg",
            "parameters": sum(p.numel() for p in model_state_dict.values()),
            "latent_dim": config.latent_dim,
            "worldkit_version": worldkit.__version__,
        }
        if model_card:
            card.update(model_card)

        # Serialize weights to safetensors via a temp file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
        safetensors_save_file(model_state_dict, tmp_path)
        weights_bytes = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)

        # Write the ZIP archive
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("config.json", json.dumps(dataclasses.asdict(config), indent=2))
            zf.writestr("weights.safetensors", weights_bytes)
            zf.writestr("metadata.json", json.dumps(meta, indent=2))
            zf.writestr("action_space.json", json.dumps(act, indent=2))
            zf.writestr("model_card.yaml", yaml.dump(card, default_flow_style=False))

        return path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load all components from a .wk ZIP archive.

        Args:
            path: Path to the .wk file.

        Returns:
            Dict with keys: config (ModelConfig), model_state_dict, metadata,
            action_space, model_card.

        Raises:
            ValueError: If the file is not a valid .wk archive.
        """
        path = Path(path)

        if not zipfile.is_zipfile(path):
            raise ValueError(
                f"Not a valid .wk archive: {path}. "
                "This may be an old-format file — use WorldModel.load() for backward compatibility."
            )

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            missing = _REQUIRED_ENTRIES - names
            if missing:
                raise ValueError(
                    f"Invalid .wk archive: missing entries {missing}. Found: {sorted(names)}"
                )

            config_dict = json.loads(zf.read("config.json"))
            config = ModelConfig(**config_dict)

            # Load safetensors weights via temp file
            with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
                tmp_path = tmp.name
            Path(tmp_path).write_bytes(zf.read("weights.safetensors"))
            model_state_dict = safetensors_load_file(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

            metadata = json.loads(zf.read("metadata.json"))

            action_space = None
            if "action_space.json" in names:
                action_space = json.loads(zf.read("action_space.json"))

            model_card = None
            if "model_card.yaml" in names:
                model_card = yaml.safe_load(zf.read("model_card.yaml"))

        return {
            "config": config,
            "model_state_dict": model_state_dict,
            "metadata": metadata,
            "action_space": action_space,
            "model_card": model_card,
        }

    @staticmethod
    def validate(path: str | Path) -> bool:
        """Check whether a .wk file has a valid structure.

        Args:
            path: Path to the .wk file.

        Returns:
            True if the archive is structurally valid.

        Raises:
            ValueError: With a description of what's wrong.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        if not zipfile.is_zipfile(path):
            raise ValueError(f"Not a ZIP archive: {path}")

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            missing = _REQUIRED_ENTRIES - names
            if missing:
                raise ValueError(f"Missing required entries: {missing}")

            # Validate config.json is parseable
            try:
                config_dict = json.loads(zf.read("config.json"))
                ModelConfig(**config_dict)
            except Exception as e:
                raise ValueError(f"Invalid config.json: {e}") from e

            # Validate metadata.json is parseable
            try:
                meta = json.loads(zf.read("metadata.json"))
                if "format_version" not in meta:
                    raise ValueError("metadata.json missing 'format_version' key")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid metadata.json: {e}") from e

            # Check weights.safetensors is non-empty
            info = zf.getinfo("weights.safetensors")
            if info.file_size == 0:
                raise ValueError("weights.safetensors is empty")

        return True

    @staticmethod
    def inspect(path: str | Path) -> dict[str, Any]:
        """Read metadata and config from a .wk file without loading weights.

        Args:
            path: Path to the .wk file.

        Returns:
            Dict with keys: config, metadata, action_space, model_card,
            weights_size_bytes.
        """
        path = Path(path)

        if not zipfile.is_zipfile(path):
            raise ValueError(f"Not a valid .wk archive: {path}")

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())

            config_dict = json.loads(zf.read("config.json"))
            config = ModelConfig(**config_dict)

            metadata = json.loads(zf.read("metadata.json"))

            action_space = None
            if "action_space.json" in names:
                action_space = json.loads(zf.read("action_space.json"))

            model_card = None
            if "model_card.yaml" in names:
                model_card = yaml.safe_load(zf.read("model_card.yaml"))

            weights_size = zf.getinfo("weights.safetensors").file_size

        return {
            "config": config,
            "metadata": metadata,
            "action_space": action_space,
            "model_card": model_card,
            "weights_size_bytes": weights_size,
        }

    @staticmethod
    def is_new_format(path: str | Path) -> bool:
        """Check if a file uses the new ZIP-based .wk format (vs old torch.save).

        Both formats may be ZIP files (PyTorch 2.x also uses ZIP), so we check
        for the presence of our required entries inside the archive.
        """
        path = Path(path)
        if not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = set(zf.namelist())
                return _REQUIRED_ENTRIES.issubset(names)
        except (zipfile.BadZipFile, OSError):
            return False
