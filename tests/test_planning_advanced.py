"""Tests for hierarchical planning (F-040) and AutoConfig (F-043)."""

from __future__ import annotations

import numpy as np
import torch

from worldkit.core.config import ModelConfig, get_config
from worldkit.core.hierarchical_planner import HierarchicalPlanResult
from worldkit.core.model import WorldModel

# ── Helpers ──────────────────────────────────────────────────────

def _make_model(action_dim: int = 2) -> WorldModel:
    """Build a nano WorldModel on CPU for testing."""
    from worldkit.core.backends import backend_registry

    config = get_config("nano", action_dim=action_dim)
    backend_cls = backend_registry.get(config.backend)
    backend = backend_cls()
    model_module = backend.build(config)
    return WorldModel(model_module, config, device="cpu", backend=backend)


def _make_h5(tmp_path, n_samples: int = 20, seq_len: int = 6):
    """Create a minimal HDF5 dataset for testing."""
    import h5py

    path = tmp_path / "test_data.h5"
    with h5py.File(path, "w") as f:
        # (N, T, H, W, C) uint8 pixels
        f.create_dataset(
            "pixels",
            data=np.random.randint(0, 255, (n_samples, seq_len, 96, 96, 3), dtype=np.uint8),
        )
        # (N, T, action_dim) actions
        f.create_dataset(
            "actions",
            data=np.random.randn(n_samples, seq_len, 2).astype(np.float32),
        )
    return path


# ── Hierarchical Planning Tests ──────────────────────────────────


def test_hierarchical_plan_structure():
    """HierarchicalPlanResult has the expected fields and types."""
    model = _make_model()
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    goal = np.random.rand(96, 96, 3).astype(np.float32)

    result = model.hierarchical_plan(
        current_state=obs,
        goal_state=goal,
        max_subgoals=2,
        steps_per_subgoal=5,
        n_candidates=10,
        n_elite=3,
        n_iterations=2,
    )

    assert isinstance(result, HierarchicalPlanResult)
    assert isinstance(result.actions, list)
    assert len(result.actions) > 0
    assert isinstance(result.actions[0], np.ndarray)
    assert isinstance(result.subgoals, list)
    assert all(isinstance(s, torch.Tensor) for s in result.subgoals)
    assert isinstance(result.segment_plans, list)
    assert len(result.segment_plans) > 0
    assert result.total_planning_time_ms > 0


def test_hierarchical_plan_has_correct_subgoal_count():
    """Number of subgoals = max_subgoals + 2 (start + intermediates + goal)."""
    model = _make_model()
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    goal = np.random.rand(96, 96, 3).astype(np.float32)

    for n_sub in (1, 3, 5):
        result = model.hierarchical_plan(
            current_state=obs,
            goal_state=goal,
            max_subgoals=n_sub,
            steps_per_subgoal=5,
            n_candidates=10,
            n_elite=3,
            n_iterations=2,
        )
        # n_sub + 2 subgoal latents (start + intermediates + goal)
        assert len(result.subgoals) == n_sub + 2
        # n_sub + 1 segments (between consecutive pairs)
        assert len(result.segment_plans) == n_sub + 1
        # Total actions = segments * steps_per_subgoal
        assert len(result.actions) == (n_sub + 1) * 5


# ── AutoConfig Tests ─────────────────────────────────────────────


def test_auto_config_returns_valid_config(tmp_path):
    """auto_config returns a valid ModelConfig and explanation string."""
    h5_path = _make_h5(tmp_path)

    config, explanation = WorldModel.auto_config(
        data=str(h5_path),
        max_training_time="2h",
        trial_epochs=1,
        device="cpu",
    )

    assert isinstance(config, ModelConfig)
    assert config.name in ("nano", "base", "large", "xl")
    assert config.action_dim == 2
    assert isinstance(explanation, str)
    assert "AutoConfig" in explanation


def test_auto_config_respects_time_limit(tmp_path):
    """A very short time budget should not pick large/xl configs."""
    h5_path = _make_h5(tmp_path)

    config, explanation = WorldModel.auto_config(
        data=str(h5_path),
        max_training_time="1s",
        trial_epochs=1,
        device="cpu",
    )

    assert isinstance(config, ModelConfig)
    # With only 1 second, the recommendation should be small
    assert config.name in ("nano", "base")


def test_auto_config_respects_device(tmp_path):
    """target_device='browser' should cap at nano."""
    h5_path = _make_h5(tmp_path)

    config, explanation = WorldModel.auto_config(
        data=str(h5_path),
        max_training_time="10h",
        target_device="browser",
        trial_epochs=1,
        device="cpu",
    )

    assert config.name == "nano"
    assert "browser" in explanation
