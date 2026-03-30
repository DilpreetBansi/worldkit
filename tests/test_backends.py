"""Tests for the multi-architecture backend system."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldkit import WorldModel
from worldkit.core.backends import (
    BackendRegistry,
    DreamerV3Backend,
    LeWMBackend,
    backend_registry,
)
from worldkit.core.config import get_config


@pytest.fixture
def nano_config():
    """Return a nano config for fast tests."""
    return get_config("nano", action_dim=2)


@pytest.fixture
def lewm_backend():
    """Create a LeWM backend instance."""
    return LeWMBackend()


@pytest.fixture
def nano_model(lewm_backend, nano_config):
    """Build a nano JEPA model via the LeWM backend."""
    return lewm_backend.build(nano_config)


# ─── Backend Registry ──────────────────────────────────


def test_backend_registry():
    """Test that the global registry has built-in backends registered."""
    names = backend_registry.list()
    assert "lewm" in names
    assert "dreamerv3" in names


def test_backend_registry_get():
    """Test retrieving backends by name."""
    cls = backend_registry.get("lewm")
    assert cls is LeWMBackend

    cls = backend_registry.get("dreamerv3")
    assert cls is DreamerV3Backend


def test_backend_registry_unknown():
    """Test that requesting an unknown backend raises KeyError."""
    with pytest.raises(KeyError, match="Unknown backend"):
        backend_registry.get("nonexistent")


def test_custom_backend_registration():
    """Test registering and retrieving a custom backend."""
    registry = BackendRegistry()
    registry.register("lewm", LeWMBackend)
    assert registry.list() == ["lewm"]
    assert registry.get("lewm") is LeWMBackend


# ─── LeWM Backend ──────────────────────────────────────


def test_lewm_backend_build(lewm_backend, nano_config):
    """Test that LeWMBackend.build() produces a valid nn.Module."""
    model = lewm_backend.build(nano_config)
    assert isinstance(model, torch.nn.Module)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count > 0


def test_lewm_backend_encode(lewm_backend, nano_model, nano_config):
    """Test encoding pixels through the LeWM backend."""
    nano_model.eval()
    # (B, T, C, H, W) — single frame batch
    pixels = torch.randn(1, 1, 3, nano_config.image_size, nano_config.image_size)
    with torch.no_grad():
        z = lewm_backend.encode(nano_model, pixels)
    # Encoder returns (B, T, D)
    assert z.shape == (1, 1, nano_config.latent_dim)


def test_lewm_backend_predict(lewm_backend, nano_model, nano_config):
    """Test one-step prediction through the LeWM backend."""
    nano_model.eval()
    B, T = 1, 3
    state = torch.randn(B, T, nano_config.latent_dim)
    actions = torch.randn(B, T, nano_config.action_dim)
    with torch.no_grad():
        pred = lewm_backend.predict(nano_model, state, actions)
    # Predictor outputs (B, T, D)
    assert pred.shape[0] == B
    assert pred.shape[2] == nano_config.latent_dim


def test_lewm_backend_rollout(lewm_backend, nano_model, nano_config):
    """Test autoregressive rollout through the LeWM backend."""
    nano_model.eval()
    B, T_ctx, T_plan = 1, 1, 3
    pixels = torch.randn(B, T_ctx, 3, nano_config.image_size, nano_config.image_size)
    actions = torch.randn(B, T_ctx, nano_config.action_dim)
    action_seq = torch.randn(B, T_plan, nano_config.action_dim)
    with torch.no_grad():
        traj = lewm_backend.rollout(nano_model, pixels, actions, action_seq, context_length=T_ctx)
    assert traj.shape[0] == B
    assert traj.shape[1] == T_plan
    assert traj.shape[2] == nano_config.latent_dim


def test_lewm_backend_training_step(lewm_backend, nano_model, nano_config):
    """Test that training_step returns a loss and loss dict."""
    nano_model.train()
    B = 2
    seq_len = nano_config.sequence_length
    pixels = torch.randn(B, seq_len, 3, nano_config.image_size, nano_config.image_size)
    actions = torch.randn(B, seq_len, nano_config.action_dim)
    loss, loss_dict = lewm_backend.training_step(nano_model, (pixels, actions), nano_config)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.requires_grad
    assert "loss/total" in loss_dict
    assert "loss/prediction" in loss_dict
    assert "loss/sigreg" in loss_dict


# ─── DreamerV3 Stub ────────────────────────────────────


def test_dreamer_stub_raises():
    """Test that every DreamerV3Backend method raises NotImplementedError."""
    stub = DreamerV3Backend()
    config = get_config("nano")
    dummy = torch.randn(1)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.build(config)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.encode(dummy, dummy)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.predict(dummy, dummy, dummy)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.rollout(dummy, dummy, dummy, dummy, 1)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.get_cost(dummy, dummy, dummy, dummy, dummy, 1)

    with pytest.raises(NotImplementedError, match="DreamerV3 backend coming soon"):
        stub.training_step(dummy, (dummy, dummy), config)


# ─── WorldModel with LeWM Backend (end-to-end) ─────────


def test_worldmodel_with_lewm_backend():
    """End-to-end: WorldModel should work identically with the backend system."""
    config = get_config("nano", action_dim=2)
    assert config.backend == "lewm"

    backend = LeWMBackend()
    model_module = backend.build(config)
    model = WorldModel(model_module, config, device="cpu")

    assert model.num_params > 0
    assert model.latent_dim == 128  # nano config

    # Encode
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(obs)
    assert z.shape == (128,)

    # Predict
    actions = [np.array([0.1, 0.2])] * 5
    result = model.predict(obs, actions)
    assert result.steps == 5
    assert result.latent_trajectory.shape[0] == 5

    # Plausibility
    frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(5)]
    score = model.plausibility(frames)
    assert 0.0 <= score <= 1.0


def test_worldmodel_save_load_with_backend(tmp_path):
    """Test that save/load round-trips correctly with backend metadata."""
    config = get_config("nano", action_dim=2)
    backend = LeWMBackend()
    model_module = backend.build(config)
    model = WorldModel(model_module, config, device="cpu")

    save_path = tmp_path / "test_backend.wk"
    model.save(save_path)
    loaded = WorldModel.load(save_path, device="cpu")

    assert loaded.num_params == model.num_params
    assert loaded.latent_dim == model.latent_dim
    assert loaded.config.backend == "lewm"


def test_worldmodel_default_backend_from_config():
    """Test that WorldModel auto-resolves the backend from config.backend."""
    config = get_config("nano", action_dim=2)
    # Build model the old way (pass nn.Module directly) — backend resolved from config
    from worldkit.core.jepa import JEPA

    jepa = JEPA.from_config(config)
    model = WorldModel(jepa, config, device="cpu")

    # Should have auto-resolved LeWM backend
    assert isinstance(model._backend, LeWMBackend)
    assert model.num_params > 0
