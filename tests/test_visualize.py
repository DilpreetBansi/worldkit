"""Tests for visualization tools: LatentVisualizer, RolloutGIFGenerator, ModelComparator."""

from __future__ import annotations

import numpy as np
import pytest

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
from worldkit.eval.comparison import ComparisonResult, ModelComparator
from worldkit.eval.rollout_gif import RolloutGIFGenerator
from worldkit.eval.visualize import LatentVisualizer


@pytest.fixture
def model():
    """Create a test model (nano config for speed)."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


@pytest.fixture
def dummy_h5(tmp_path):
    """Create a small HDF5 file with random data for testing."""
    import h5py

    path = tmp_path / "test_data.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "pixels",
            data=np.random.randint(0, 256, (3, 10, 96, 96, 3), dtype=np.uint8),
        )
        f.create_dataset(
            "actions",
            data=np.random.randn(3, 10, 2).astype(np.float32),
        )
    return path


# ─── LatentVisualizer tests ────────────────────────


def test_pca_plot_saves_file(model, dummy_h5, tmp_path):
    save_path = tmp_path / "pca.png"
    viz = LatentVisualizer(model)
    fig = viz.plot_pca(dummy_h5, save_to=save_path, max_frames=30)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert fig is not None


def test_tsne_plot_saves_file(model, dummy_h5, tmp_path):
    save_path = tmp_path / "tsne.png"
    viz = LatentVisualizer(model)
    fig = viz.plot_tsne(dummy_h5, save_to=save_path, max_frames=30)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    assert fig is not None


# ─── RolloutGIFGenerator tests ─────────────────────


def test_rollout_gif_saves_file(model, tmp_path):
    from PIL import Image

    obs = np.random.rand(96, 96, 3).astype(np.float32)
    actions = [np.array([0.1, 0.2])] * 5
    save_path = tmp_path / "rollout.gif"

    gen = RolloutGIFGenerator(model)
    result_path = gen.generate(obs, actions, save_to=save_path)

    assert result_path == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 0

    # Verify it's a valid GIF with multiple frames
    img = Image.open(save_path)
    assert img.format == "GIF"
    assert img.n_frames == 5


# ─── ModelComparator tests ─────────────────────────


def test_comparison_result_structure(dummy_h5):
    config_a = get_config("nano", action_dim=2)
    config_b = get_config("nano", action_dim=2)
    jepa_a = JEPA.from_config(config_a)
    jepa_b = JEPA.from_config(config_b)
    model_a = WorldModel(jepa_a, config_a, device="cpu")
    model_b = WorldModel(jepa_b, config_b, device="cpu")

    comparator = ModelComparator({"model_a": model_a, "model_b": model_b})
    result = comparator.compare(dummy_h5, episodes=2)

    assert isinstance(result, ComparisonResult)
    assert len(result.model_names) == 2
    assert "model_a" in result.model_names
    assert "model_b" in result.model_names
    assert result.best_model in result.model_names

    for name in result.model_names:
        assert name in result.metrics
        m = result.metrics[name]
        assert "prediction_error" in m
        assert "plausibility_score" in m
        assert "encoding_time_ms" in m
        assert "latent_variance" in m
        assert "num_params" in m
        assert "latent_dim" in m


def test_comparison_report_saves_file(dummy_h5, tmp_path):
    config_a = get_config("nano", action_dim=2)
    config_b = get_config("nano", action_dim=2)
    jepa_a = JEPA.from_config(config_a)
    jepa_b = JEPA.from_config(config_b)
    model_a = WorldModel(jepa_a, config_a, device="cpu")
    model_b = WorldModel(jepa_b, config_b, device="cpu")

    comparator = ModelComparator({"model_a": model_a, "model_b": model_b})
    result = comparator.compare(dummy_h5, episodes=2)

    report_path = tmp_path / "report.html"
    saved = comparator.report(result, save_to=report_path)

    assert saved == report_path
    assert report_path.exists()

    content = report_path.read_text()
    assert "model_a" in content
    assert "model_b" in content
    assert "WorldKit Model Comparison" in content
    assert result.best_model in content
