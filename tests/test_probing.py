"""Tests for the linear probing evaluation suite."""

from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
from worldkit.eval.probing import LinearProbe, ProbeResult


@pytest.fixture
def model():
    """Create a test model (nano config for speed)."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


@pytest.fixture
def probe_data(tmp_path, model):
    """Create synthetic HDF5 data and CSV labels for probing tests.

    Generates 50 random 96x96 frames with correlated labels so that
    a Ridge probe can fit something non-degenerate.
    """
    n_samples = 50
    h, w, c = 96, 96, 3

    # Random pixel observations
    pixels = np.random.randint(0, 256, (n_samples, h, w, c), dtype=np.uint8)

    data_path = tmp_path / "probe_data.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("pixels", data=pixels)

    # Generate labels (random — we only test shapes, not R² quality)
    labels_path = tmp_path / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_x", "agent_y", "block_angle"])
        writer.writeheader()
        for _ in range(n_samples):
            writer.writerow(
                {
                    "agent_x": float(np.random.uniform(0, 512)),
                    "agent_y": float(np.random.uniform(0, 512)),
                    "block_angle": float(np.random.uniform(0, 2 * np.pi)),
                }
            )

    return data_path, labels_path


def test_probe_with_random_data(model, probe_data):
    """Verify probing runs end-to-end and returns correct types/shapes."""
    data_path, labels_path = probe_data
    properties = ["agent_x", "agent_y", "block_angle"]

    result = model.probe(
        data=data_path,
        properties=properties,
        labels=labels_path,
        seed=42,
    )

    assert isinstance(result, ProbeResult)
    assert set(result.property_scores.keys()) == set(properties)
    assert set(result.mse_scores.keys()) == set(properties)
    assert set(result.probes.keys()) == set(properties)

    for prop in properties:
        # R² is a float (can be negative for bad fits on random data)
        assert isinstance(result.property_scores[prop], float)
        # MSE is non-negative
        assert isinstance(result.mse_scores[prop], float)
        assert result.mse_scores[prop] >= 0.0


def test_probe_result_structure(model, probe_data):
    """Verify ProbeResult has all expected fields with correct types."""
    data_path, labels_path = probe_data

    result = model.probe(
        data=data_path,
        properties=["agent_x"],
        labels=labels_path,
    )

    assert hasattr(result, "property_scores")
    assert hasattr(result, "mse_scores")
    assert hasattr(result, "probes")
    assert hasattr(result, "summary")

    assert isinstance(result.property_scores, dict)
    assert isinstance(result.mse_scores, dict)
    assert isinstance(result.probes, dict)
    assert isinstance(result.summary, str)


def test_probe_summary_string(model, probe_data):
    """Verify the summary string is well-formed."""
    data_path, labels_path = probe_data
    properties = ["agent_x", "agent_y", "block_angle"]

    result = model.probe(
        data=data_path,
        properties=properties,
        labels=labels_path,
    )

    assert "Linear Probe Results" in result.summary
    assert "Average R²" in result.summary
    for prop in properties:
        assert prop in result.summary
        assert "R²=" in result.summary
        assert "MSE=" in result.summary


def test_probe_predict_single_observation(model, probe_data):
    """Verify predict() returns a dict of floats for one frame."""
    data_path, labels_path = probe_data

    result = model.probe(
        data=data_path,
        properties=["agent_x", "agent_y"],
        labels=labels_path,
    )

    prober = LinearProbe(model)
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    preds = prober.predict(obs, result.probes)

    assert isinstance(preds, dict)
    assert set(preds.keys()) == {"agent_x", "agent_y"}
    for v in preds.values():
        assert isinstance(v, float)


def test_probe_with_hdf5_labels(model, tmp_path):
    """Verify probing works with HDF5-format labels."""
    n_samples = 30
    h, w, c = 96, 96, 3

    pixels = np.random.randint(0, 256, (n_samples, h, w, c), dtype=np.uint8)
    data_path = tmp_path / "data.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("pixels", data=pixels)

    labels_path = tmp_path / "labels.h5"
    with h5py.File(labels_path, "w") as f:
        f.create_dataset("agent_x", data=np.random.rand(n_samples))
        f.create_dataset("agent_y", data=np.random.rand(n_samples))

    result = model.probe(
        data=data_path,
        properties=["agent_x", "agent_y"],
        labels=labels_path,
    )

    assert isinstance(result, ProbeResult)
    assert "agent_x" in result.property_scores
    assert "agent_y" in result.property_scores


def test_probe_missing_property_raises(model, probe_data):
    """Verify KeyError when requesting a property not in labels."""
    data_path, labels_path = probe_data

    with pytest.raises(KeyError, match="velocity"):
        model.probe(
            data=data_path,
            properties=["velocity"],
            labels=labels_path,
        )


def test_probe_multi_episode_hdf5(model, tmp_path):
    """Verify probing handles multi-episode (5D) observation layout."""
    n_eps, t_steps, h, w, c = 3, 10, 96, 96, 3
    n_total = n_eps * t_steps

    pixels = np.random.randint(0, 256, (n_eps, t_steps, h, w, c), dtype=np.uint8)
    data_path = tmp_path / "multi_ep.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("pixels", data=pixels)

    labels_path = tmp_path / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pos_x"])
        writer.writeheader()
        for _ in range(n_total):
            writer.writerow({"pos_x": float(np.random.uniform(0, 1))})

    result = model.probe(
        data=data_path,
        properties=["pos_x"],
        labels=labels_path,
    )

    assert isinstance(result, ProbeResult)
    assert "pos_x" in result.property_scores
