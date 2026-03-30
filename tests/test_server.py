"""Tests for the WorldKit FastAPI server (F-024)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
from worldkit.server.app import _metrics, _models, app


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset global server state between tests."""
    _models.clear()
    _metrics["request_count"].clear()
    _metrics["total_latency_ms"].clear()
    yield
    _models.clear()
    _metrics["request_count"].clear()
    _metrics["total_latency_ms"].clear()


@pytest.fixture
def model():
    """Create a nano test model."""
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


@pytest.fixture
def client(model):
    """TestClient with a model pre-loaded."""
    import worldkit.server.app as srv

    srv._models["test_model"] = model
    srv._default_model = "test_model"
    return TestClient(app)


def _make_image_bytes(h: int = 96, w: int = 96) -> io.BytesIO:
    """Create a dummy RGB image as bytes."""
    from PIL import Image

    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ── Health ──────────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.0"
    assert data["models_loaded"] == 1


# ── Models list ─────────────────────────────────────────


def test_models_list(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    info = data["models"][0]
    assert info["name"] == "test_model"
    assert info["config_name"] == "nano"
    assert info["latent_dim"] == 128
    assert info["num_params"] > 0
    assert data["default"] == "test_model"


# ── Encode ──────────────────────────────────────────────


def test_encode(client):
    img_bytes = _make_image_bytes()
    resp = client.post("/encode", files={"observation": ("img.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["dim"] == 128
    assert len(data["latent"]) == 128


def test_encode_with_model_param(client, model):
    import worldkit.server.app as srv

    srv._models["second"] = model
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/encode?model=second", files={"observation": ("img.png", img_bytes, "image/png")}
    )
    assert resp.status_code == 200
    assert resp.json()["dim"] == 128


# ── Predict ─────────────────────────────────────────────


def test_predict(client):
    img_bytes = _make_image_bytes()
    actions = [[0.1, 0.2]] * 3
    resp = client.post(
        "/predict",
        files={"observation": ("img.png", img_bytes, "image/png")},
        data={"actions": json.dumps(actions)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["steps"] == 3
    assert len(data["latent_trajectory"]) == 3


# ── Plan ────────────────────────────────────────────────


def test_plan(client):
    current_bytes = _make_image_bytes()
    goal_bytes = _make_image_bytes()
    resp = client.post(
        "/plan",
        files={
            "current": ("current.png", current_bytes, "image/png"),
            "goal": ("goal.png", goal_bytes, "image/png"),
        },
        data={"max_steps": "10", "n_candidates": "20"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "actions" in data
    assert "expected_cost" in data
    assert "planning_time_ms" in data


# ── Batch encode ────────────────────────────────────────


def test_batch_encode(client):
    obs1 = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    obs2 = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    resp = client.post("/batch/encode", json={"observations": [obs1, obs2]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["dim"] == 128
    assert len(data["latents"]) == 2
    assert len(data["latents"][0]) == 128


# ── Batch predict ───────────────────────────────────────


def test_batch_predict(client):
    obs = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    actions = [[0.1, 0.2]] * 3
    resp = client.post(
        "/batch/predict",
        json={"observations": [obs], "actions": [actions]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["predictions"][0]["steps"] == 3


def test_batch_predict_mismatched_lengths(client):
    obs = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    actions = [[0.1, 0.2]] * 3
    resp = client.post(
        "/batch/predict",
        json={"observations": [obs, obs], "actions": [actions]},
    )
    assert resp.status_code == 400
    assert "must match" in resp.json()["detail"]


# ── WebSocket predict ───────────────────────────────────


def test_websocket_predict(client):
    obs = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    with client.websocket_connect("/ws/predict") as ws:
        ws.send_json({"observation": obs})
        data = ws.receive_json()
        assert "latent" in data
        assert len(data["latent"]) == 128


def test_websocket_predict_with_action(client):
    obs = np.random.rand(96, 96, 3).astype(np.float32).tolist()
    actions = [[0.1, 0.2]] * 3
    with client.websocket_connect("/ws/predict") as ws:
        ws.send_json({"observation": obs, "action": actions})
        data = ws.receive_json()
        assert "latent" in data
        assert "latent_trajectory" in data
        assert data["steps"] == 3


def test_websocket_missing_observation(client):
    with client.websocket_connect("/ws/predict") as ws:
        ws.send_json({"action": [[0.1, 0.2]]})
        data = ws.receive_json()
        assert "error" in data
        assert "observation" in data["error"].lower()


# ── Metrics ─────────────────────────────────────────────


def test_metrics(client):
    # Make a few requests first
    client.get("/health")
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] == 1
    assert "/health" in data["request_count"]
    assert data["request_count"]["/health"] >= 2


# ── Error handling ──────────────────────────────────────


def test_unknown_model_returns_error(client):
    img_bytes = _make_image_bytes()
    resp = client.post(
        "/encode?model=nonexistent",
        files={"observation": ("img.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 404
    assert "nonexistent" in resp.json()["detail"]


# ── CORS ────────────────────────────────────────────────


def test_cors_headers(client):
    resp = client.options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" in resp.headers
