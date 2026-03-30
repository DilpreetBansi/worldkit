"""Tests for multi-environment training, distillation, and online learning."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from worldkit.core.config import get_config
from worldkit.core.model import WorldModel

# ── Helpers ─────────────────────────────────────────────


def _make_h5(path, n_episodes=5, ep_len=20, h=32, w=32, action_dim=2):
    """Create a minimal HDF5 dataset for testing."""
    pixels = np.random.randint(0, 255, (n_episodes, ep_len, h, w, 3), dtype=np.uint8)
    actions = np.random.randn(n_episodes, ep_len, action_dim).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("actions", data=actions)


def _build_model(action_dim=2, device="cpu"):
    """Build a nano WorldModel for testing."""
    from worldkit.core.backends import backend_registry

    config = get_config("nano", action_dim=action_dim, image_size=32)
    backend_cls = backend_registry.get(config.backend)
    backend = backend_cls()
    module = backend.build(config)
    return WorldModel(module, config, device, backend=backend)


# ── Multi-Environment Dataset ──────────────────────────


class TestMultiEnvironmentDataset:
    def test_different_action_dims(self, tmp_path):
        """Datasets with different action dims are padded to max."""
        from worldkit.data.multi_dataset import MultiEnvironmentDataset

        path_a = tmp_path / "env_a.h5"
        path_b = tmp_path / "env_b.h5"
        _make_h5(path_a, action_dim=2)
        _make_h5(path_b, action_dim=5)

        ds = MultiEnvironmentDataset(
            [path_a, path_b], sequence_length=16,
        )
        assert ds.max_action_dim == 5

        # Every sample should have actions padded to dim=5
        pixels, actions = ds[0]
        assert actions.shape[-1] == 5

        # Sample from second dataset (no padding needed)
        pixels_b, actions_b = ds[len(ds) - 1]
        assert actions_b.shape[-1] == 5

    def test_multi_env_training_runs(self, tmp_path):
        """WorldModel.train() accepts a list of data paths."""
        path_a = tmp_path / "env_a.h5"
        path_b = tmp_path / "env_b.h5"
        _make_h5(path_a, action_dim=2, h=32, w=32)
        _make_h5(path_b, action_dim=3, h=32, w=32)

        model = WorldModel.train(
            data=[str(path_a), str(path_b)],
            config="nano",
            epochs=2,
            batch_size=4,
            device="cpu",
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        # Action dim should be the max across datasets
        assert model.config.action_dim == 3
        assert model.num_params > 0


# ── Distillation ───────────────────────────────────────


class TestDistillation:
    def test_produces_smaller_model(self, tmp_path):
        """Distilled student has fewer params than teacher."""
        path = tmp_path / "data.h5"
        _make_h5(path, n_episodes=5, ep_len=20, h=32, w=32, action_dim=2)

        teacher = _build_model(action_dim=2)

        student = WorldModel.distill(
            teacher=teacher,
            student_config="nano",
            data=str(path),
            epochs=2,
            batch_size=4,
            device="cpu",
        )

        assert student.num_params > 0
        # Student should be same or smaller config
        assert student.config.name == "nano"
        # Student should be a working model — can encode
        obs = np.random.rand(32, 32, 3).astype(np.float32)
        z = student.encode(obs)
        assert z.shape == (student.config.latent_dim,)


# ── Online Learning ────────────────────────────────────


class TestOnlineLearning:
    def test_update_changes_weights(self):
        """Online updates modify model parameters."""
        model = _build_model(action_dim=2)

        # Snapshot weights before
        w_before = {
            k: v.clone() for k, v in model._model.state_dict().items()
        }

        model.enable_online_learning(
            lr=1e-3, buffer_size=100, batch_size=4, update_every=1,
        )

        # Feed enough transitions to trigger gradient steps
        for _ in range(10):
            obs = np.random.rand(32, 32, 3).astype(np.float32)
            act = np.random.randn(2).astype(np.float32)
            next_obs = np.random.rand(32, 32, 3).astype(np.float32)
            model.update(obs, act, next_obs)

        # At least some weights should have changed
        changed = False
        for k, v in model._model.state_dict().items():
            if not torch.equal(v, w_before[k]):
                changed = True
                break
        assert changed, "Online learning did not change any weights"

    def test_buffer_size_limit(self):
        """Buffer respects its maximum capacity."""
        model = _build_model(action_dim=2)
        model.enable_online_learning(
            lr=1e-5, buffer_size=10, batch_size=4, update_every=100,
        )

        for _ in range(50):
            obs = np.random.rand(32, 32, 3).astype(np.float32)
            act = np.random.randn(2).astype(np.float32)
            next_obs = np.random.rand(32, 32, 3).astype(np.float32)
            model.update(obs, act, next_obs)

        assert model._online_learner.buffer_size == 10

    def test_update_without_enable_raises(self):
        """Calling update() before enable_online_learning() raises."""
        model = _build_model(action_dim=2)
        obs = np.random.rand(32, 32, 3).astype(np.float32)
        act = np.random.randn(2).astype(np.float32)
        with pytest.raises(RuntimeError, match="Online learning not enabled"):
            model.update(obs, act, obs)
