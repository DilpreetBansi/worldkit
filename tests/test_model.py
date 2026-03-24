"""Tests for the WorldModel class."""


import numpy as np
import pytest

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA


@pytest.fixture
def model():
    """Create a test model (nano config for speed)."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


def test_model_creation(model):
    assert model.num_params > 0
    assert model.latent_dim == 128  # nano config
    assert model.device == "cpu"


def test_encode(model):
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(obs)
    assert z.shape == (128,)  # nano latent dim


def test_predict(model):
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    actions = [np.array([0.1, 0.2])] * 5
    result = model.predict(obs, actions)
    assert result.steps == 5
    assert result.latent_trajectory.shape[0] == 5


def test_plausibility(model):
    frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(10)]
    score = model.plausibility(frames)
    assert 0.0 <= score <= 1.0


def test_save_load(model, tmp_path):
    save_path = tmp_path / "test_model.wk"
    model.save(save_path)
    loaded = WorldModel.load(save_path, device="cpu")
    assert loaded.num_params == model.num_params
    assert loaded.latent_dim == model.latent_dim


def test_configs():
    for name in ["nano", "base", "large", "xl"]:
        config = get_config(name)
        jepa = JEPA.from_config(config)
        model = WorldModel(jepa, config, device="cpu")
        assert model.num_params > 0


def test_export_torchscript(model, tmp_path):
    path = model.export(format="torchscript", output=tmp_path)
    assert path.exists()
