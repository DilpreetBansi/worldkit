"""Tests for the .wk ZIP archive format (F-018)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.format import WKFormat
from worldkit.core.jepa import JEPA


@pytest.fixture
def nano_model():
    """Create a nano model for testing."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


@pytest.fixture
def saved_wk(nano_model, tmp_path):
    """Save a nano model and return the path."""
    path = tmp_path / "test.wk"
    nano_model.save(path)
    return path


# ─── Roundtrip ──────────────────────────────────────


def test_save_load_roundtrip(nano_model, tmp_path):
    """Save a model, load it back, verify weights and config match."""
    path = tmp_path / "roundtrip.wk"
    nano_model.save(path)

    loaded = WorldModel.load(path, device="cpu")
    assert loaded.config.name == nano_model.config.name
    assert loaded.config.latent_dim == nano_model.config.latent_dim
    assert loaded.config.action_dim == nano_model.config.action_dim
    assert loaded.num_params == nano_model.num_params

    # Verify weights produce the same encoding
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    z_orig = nano_model.encode(obs)
    z_loaded = loaded.encode(obs)
    assert torch.allclose(z_orig, z_loaded, atol=1e-5)


# ─── Validate ───────────────────────────────────────


def test_validate_good_file(saved_wk):
    """A properly saved .wk file should validate."""
    assert WKFormat.validate(saved_wk) is True


def test_validate_bad_file_not_zip(tmp_path):
    """A non-ZIP file should fail validation."""
    bad = tmp_path / "bad.wk"
    bad.write_text("not a zip file")
    with pytest.raises(ValueError, match="Not a ZIP archive"):
        WKFormat.validate(bad)


def test_validate_bad_file_missing_entries(tmp_path):
    """A ZIP missing required entries should fail validation."""
    bad = tmp_path / "incomplete.wk"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("config.json", "{}")
    with pytest.raises(ValueError, match="Missing required entries"):
        WKFormat.validate(bad)


def test_validate_bad_config(tmp_path):
    """A ZIP with invalid config.json should fail validation."""
    bad = tmp_path / "badconfig.wk"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("config.json", "not json")
        zf.writestr("weights.safetensors", b"\x00" * 10)
        zf.writestr("metadata.json", '{"format_version": 2}')
    with pytest.raises(ValueError, match="Invalid config.json"):
        WKFormat.validate(bad)


def test_validate_nonexistent(tmp_path):
    """Validating a nonexistent file should fail."""
    with pytest.raises(ValueError, match="File not found"):
        WKFormat.validate(tmp_path / "nope.wk")


# ─── Inspect ────────────────────────────────────────


def test_inspect_metadata(saved_wk):
    """Inspect should return config and metadata without loading weights."""
    info = WKFormat.inspect(saved_wk)

    assert info["config"].name == "nano"
    assert info["config"].latent_dim == 128
    assert info["metadata"]["format_version"] == 2
    assert "worldkit_version" in info["metadata"]
    assert "created_at" in info["metadata"]
    assert info["weights_size_bytes"] > 0

    # Action space defaults
    assert info["action_space"]["dim"] == 2
    assert info["action_space"]["type"] == "continuous"


def test_inspect_with_custom_metadata(nano_model, tmp_path):
    """Inspect should include custom metadata passed at save time."""
    path = tmp_path / "custom.wk"
    nano_model.save(
        path,
        metadata={"dataset": "pusht", "epochs": 50, "final_val_loss": 0.05},
        action_space={"dim": 2, "type": "continuous", "low": -2.0, "high": 2.0},
    )

    info = WKFormat.inspect(path)
    assert info["metadata"]["dataset"] == "pusht"
    assert info["metadata"]["epochs"] == 50
    assert info["action_space"]["low"] == -2.0


# ─── Backward Compatibility ────────────────────────


def test_backward_compatibility(nano_model, tmp_path):
    """Loading an old-format (torch.save) .wk file should work with a warning."""
    legacy_path = tmp_path / "legacy.wk"

    # Save in the old format directly
    torch.save(
        {
            "config": nano_model.config,
            "model_state_dict": nano_model._model.state_dict(),
            "worldkit_version": "0.1.0",
        },
        legacy_path,
    )

    with pytest.warns(DeprecationWarning, match="legacy .wk file"):
        loaded = WorldModel.load(legacy_path, device="cpu")

    assert loaded.config.name == nano_model.config.name
    assert loaded.num_params == nano_model.num_params


def test_is_new_format_detection(nano_model, tmp_path):
    """is_new_format should distinguish ZIP from legacy files."""
    new_path = tmp_path / "new.wk"
    nano_model.save(new_path)
    assert WKFormat.is_new_format(new_path) is True

    legacy_path = tmp_path / "legacy.wk"
    torch.save({"config": nano_model.config}, legacy_path)
    assert WKFormat.is_new_format(legacy_path) is False


# ─── ZIP Structure ──────────────────────────────────


def test_zip_contains_expected_entries(saved_wk):
    """The saved file should be a valid ZIP with the expected entries."""
    assert zipfile.is_zipfile(saved_wk)
    with zipfile.ZipFile(saved_wk, "r") as zf:
        names = set(zf.namelist())
        assert "config.json" in names
        assert "weights.safetensors" in names
        assert "metadata.json" in names
        assert "action_space.json" in names
        assert "model_card.yaml" in names


def test_config_json_is_valid(saved_wk):
    """config.json should be parseable and reconstruct a valid ModelConfig."""
    with zipfile.ZipFile(saved_wk, "r") as zf:
        config_dict = json.loads(zf.read("config.json"))
    assert config_dict["name"] == "nano"
    assert config_dict["latent_dim"] == 128

    # Should be usable to construct a ModelConfig
    config = get_config("nano", action_dim=config_dict["action_dim"])
    assert config.latent_dim == config_dict["latent_dim"]
