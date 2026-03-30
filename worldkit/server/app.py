"""FastAPI inference server for WorldKit.

Supports multi-model loading, batch prediction, WebSocket streaming,
request metrics, and proper error handling (F-024).
"""

from __future__ import annotations

import io
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    _load_models()
    yield


app = FastAPI(title="WorldKit API", version="0.2.0", lifespan=lifespan)

# ── CORS ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ───────────────────────────────────────────────
_models: dict[str, object] = {}
_default_model: str | None = None

# Metrics counters
_metrics: dict[str, dict] = {
    "request_count": defaultdict(int),
    "total_latency_ms": defaultdict(float),
}


# ── Pydantic models ────────────────────────────────────
# Use typing.Optional for Pydantic fields — Pydantic evaluates annotations
# at runtime, so PEP 604 union syntax requires Python 3.10+.


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int


class ModelInfo(BaseModel):
    name: str
    config_name: str
    latent_dim: int
    num_params: int
    device: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default: Optional[str] = None


class EncodeResponse(BaseModel):
    latent: List[float]
    dim: int


class PredictResponse(BaseModel):
    latent_trajectory: List[List[float]]
    confidence: float
    steps: int


class PlanResponse(BaseModel):
    actions: List[List[float]]
    expected_cost: float
    success_probability: float
    planning_time_ms: float


class PlausibilityResponse(BaseModel):
    plausibility_score: float
    anomaly_detected: bool
    num_frames: int


class BatchEncodeRequest(BaseModel):
    observations: List[List[List[List[float]]]] = Field(
        ..., description="List of observations, each shaped (H, W, C)"
    )
    model: Optional[str] = None


class BatchEncodeResponse(BaseModel):
    latents: List[List[float]]
    dim: int
    count: int


class BatchPredictRequest(BaseModel):
    observations: List[List[List[List[float]]]] = Field(
        ..., description="List of observations, each shaped (H, W, C)"
    )
    actions: List[List[List[float]]] = Field(
        ..., description="List of action sequences, one per observation"
    )
    model: Optional[str] = None


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int


class MetricsResponse(BaseModel):
    request_count: Dict[str, int]
    average_latency_ms: Dict[str, float]
    models_loaded: int
    model_info: List[ModelInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: str


# ── Error handlers ──────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": "Bad Request", "detail": str(exc)})


@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"error": "Not Found", "detail": str(exc)})


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ── Middleware for metrics ──────────────────────────────


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    path = request.url.path
    _metrics["request_count"][path] += 1
    _metrics["total_latency_ms"][path] += elapsed_ms
    return response


# ── Model loading ───────────────────────────────────────


def _load_models() -> None:
    """Load models from environment configuration.

    Reads WORLDKIT_MODELS (comma-separated paths) or falls back to
    WORLDKIT_MODEL_PATH (single path).
    """
    global _default_model

    models_env = os.environ.get("WORLDKIT_MODELS", "")
    single_env = os.environ.get("WORLDKIT_MODEL_PATH", "")

    paths: list[str] = []
    if models_env:
        paths = [p.strip() for p in models_env.split(",") if p.strip()]
    elif single_env:
        paths = [single_env]

    if not paths:
        return

    from worldkit import WorldModel

    for path_str in paths:
        path = Path(path_str)
        name = path.stem
        _models[name] = WorldModel.load(path)
        if _default_model is None:
            _default_model = name


def _resolve_model(model_name: str | None = None):
    """Resolve a model by name, falling back to the default."""
    if not _models:
        _load_models()

    if not _models:
        raise ValueError(
            "No models loaded. Set WORLDKIT_MODELS or WORLDKIT_MODEL_PATH "
            "environment variable."
        )

    name = model_name or _default_model
    if name not in _models:
        available = list(_models.keys())
        raise KeyError(f"Model '{name}' not found. Available models: {available}")
    return _models[name]


def _model_info(name: str, model: object) -> ModelInfo:
    """Build a ModelInfo from a WorldModel instance."""
    return ModelInfo(
        name=name,
        config_name=model.config.name,
        latent_dim=model.latent_dim,
        num_params=model.num_params,
        device=model.device,
    )


# ── Endpoints ───────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.2.0",
        models_loaded=len(_models),
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """List all loaded models with their configs and parameters."""
    infos = [_model_info(name, m) for name, m in _models.items()]
    return ModelsResponse(models=infos, default=_default_model)


@app.post("/encode", response_model=EncodeResponse)
async def encode(
    observation: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model name"),
):
    """Encode an observation image into a latent vector."""
    from PIL import Image

    wm = _resolve_model(model)
    img = Image.open(io.BytesIO(await observation.read())).convert("RGB")
    obs = np.array(img)
    z = wm.encode(obs)
    return EncodeResponse(latent=z.tolist(), dim=len(z))


@app.post("/predict", response_model=PredictResponse)
async def predict(
    observation: UploadFile = File(...),
    actions: str = Form(...),
    model: Optional[str] = Query(None, description="Model name"),
):
    """Predict future states from observation and action sequence."""
    import json

    from PIL import Image

    wm = _resolve_model(model)
    img = Image.open(io.BytesIO(await observation.read())).convert("RGB")
    obs = np.array(img)
    action_list = json.loads(actions)

    result = wm.predict(obs, action_list)
    return PredictResponse(
        latent_trajectory=result.latent_trajectory.tolist(),
        confidence=result.confidence,
        steps=result.steps,
    )


@app.post("/plan", response_model=PlanResponse)
async def plan(
    current: UploadFile = File(...),
    goal: UploadFile = File(...),
    max_steps: int = Form(50),
    n_candidates: int = Form(200),
    model: Optional[str] = Query(None, description="Model name"),
):
    """Plan an optimal action sequence to reach a goal state."""
    from PIL import Image

    wm = _resolve_model(model)
    current_img = np.array(Image.open(io.BytesIO(await current.read())).convert("RGB"))
    goal_img = np.array(Image.open(io.BytesIO(await goal.read())).convert("RGB"))

    result = wm.plan(current_img, goal_img, max_steps=max_steps, n_candidates=n_candidates)
    return PlanResponse(
        actions=[a.tolist() for a in result.actions],
        expected_cost=result.expected_cost,
        success_probability=result.success_probability,
        planning_time_ms=result.planning_time_ms,
    )


@app.post("/plausibility", response_model=PlausibilityResponse)
async def plausibility(
    video: UploadFile = File(...),
    model: Optional[str] = Query(None, description="Model name"),
):
    """Score how physically plausible a video sequence is."""
    import tempfile

    import cv2

    wm = _resolve_model(model)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        os.unlink(tmp_path)

    score = wm.plausibility(frames)
    return PlausibilityResponse(
        plausibility_score=score,
        anomaly_detected=score < 0.3,
        num_frames=len(frames),
    )


# ── Batch endpoints ─────────────────────────────────────


@app.post("/batch/encode", response_model=BatchEncodeResponse)
async def batch_encode(req: BatchEncodeRequest):
    """Encode a batch of observations into latent vectors."""
    wm = _resolve_model(req.model)
    latents = []
    for obs_data in req.observations:
        obs = np.array(obs_data, dtype=np.float32)
        z = wm.encode(obs)
        latents.append(z.tolist())
    return BatchEncodeResponse(
        latents=latents,
        dim=wm.latent_dim,
        count=len(latents),
    )


@app.post("/batch/predict", response_model=BatchPredictResponse)
async def batch_predict(req: BatchPredictRequest):
    """Predict future states for a batch of observations + actions."""
    wm = _resolve_model(req.model)
    if len(req.observations) != len(req.actions):
        raise ValueError(
            f"Observation count ({len(req.observations)}) must match "
            f"action count ({len(req.actions)})."
        )

    predictions = []
    for obs_data, act_data in zip(req.observations, req.actions):
        obs = np.array(obs_data, dtype=np.float32)
        action_list = [np.array(a, dtype=np.float32) for a in act_data]
        result = wm.predict(obs, action_list)
        predictions.append(
            PredictResponse(
                latent_trajectory=result.latent_trajectory.tolist(),
                confidence=result.confidence,
                steps=result.steps,
            )
        )
    return BatchPredictResponse(predictions=predictions, count=len(predictions))


# ── WebSocket streaming ─────────────────────────────────


@app.websocket("/ws/predict")
async def ws_predict(
    websocket: WebSocket,
    model: Optional[str] = Query(None),
):
    """WebSocket endpoint for real-time frame-by-frame predictions.

    Client sends JSON: {"observation": [...], "action": [...]}
    Server responds JSON: {"latent": [...], "confidence": float}
    """
    await websocket.accept()
    try:
        wm = _resolve_model(model)
    except (ValueError, KeyError) as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1008)
        return

    try:
        while True:
            data = await websocket.receive_json()

            if "observation" not in data:
                await websocket.send_json({"error": "Missing 'observation' field"})
                continue

            obs = np.array(data["observation"], dtype=np.float32)
            z = wm.encode(obs)

            response: dict = {"latent": z.tolist()}

            if "action" in data:
                action_list = [np.array(a, dtype=np.float32) for a in data["action"]]
                result = wm.predict(obs, action_list)
                response["latent_trajectory"] = result.latent_trajectory.tolist()
                response["confidence"] = result.confidence
                response["steps"] = result.steps
            else:
                response["confidence"] = 1.0

            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Metrics ─────────────────────────────────────────────


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """Server metrics: request counts and average latencies."""
    avg_latency = {}
    for path, total_ms in _metrics["total_latency_ms"].items():
        count = _metrics["request_count"].get(path, 1)
        avg_latency[path] = round(total_ms / count, 2)

    model_infos = [_model_info(name, m) for name, m in _models.items()]
    return MetricsResponse(
        request_count=dict(_metrics["request_count"]),
        average_latency_ms=avg_latency,
        models_loaded=len(_models),
        model_info=model_infos,
    )
