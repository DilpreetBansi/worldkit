"""FastAPI inference server for WorldKit."""

from __future__ import annotations

import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

app = FastAPI(title="WorldKit API", version="0.1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        from worldkit import WorldModel

        path = os.environ.get("WORLDKIT_MODEL_PATH", "./model.wk")
        _model = WorldModel.load(path)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/encode")
async def encode(observation: UploadFile = File(...)):
    model = get_model()
    img = Image.open(io.BytesIO(await observation.read())).convert("RGB")
    obs = np.array(img)
    z = model.encode(obs)
    return {"latent": z.tolist(), "dim": len(z)}


@app.post("/predict")
async def predict(
    observation: UploadFile = File(...),
    actions: str = Form(...),
):
    import json

    model = get_model()
    img = Image.open(io.BytesIO(await observation.read())).convert("RGB")
    obs = np.array(img)
    action_list = json.loads(actions)

    result = model.predict(obs, action_list)
    return {
        "latent_trajectory": result.latent_trajectory.tolist(),
        "confidence": result.confidence,
        "steps": result.steps,
    }


@app.post("/plan")
async def plan(
    current: UploadFile = File(...),
    goal: UploadFile = File(...),
    max_steps: int = Form(50),
    n_candidates: int = Form(200),
):
    model = get_model()
    current_img = np.array(
        Image.open(io.BytesIO(await current.read())).convert("RGB")
    )
    goal_img = np.array(Image.open(io.BytesIO(await goal.read())).convert("RGB"))

    result = model.plan(
        current_img, goal_img, max_steps=max_steps, n_candidates=n_candidates
    )
    return {
        "actions": [a.tolist() for a in result.actions],
        "expected_cost": result.expected_cost,
        "success_probability": result.success_probability,
        "planning_time_ms": result.planning_time_ms,
    }


@app.post("/plausibility")
async def plausibility(video: UploadFile = File(...)):
    import tempfile

    import cv2

    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    os.unlink(tmp_path)

    score = model.plausibility(frames)
    return {
        "plausibility_score": score,
        "anomaly_detected": score < 0.3,
        "num_frames": len(frames),
    }
