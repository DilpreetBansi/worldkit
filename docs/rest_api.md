# REST API Reference

WorldKit includes a FastAPI inference server for serving models over HTTP.

## Start the server

```bash
worldkit serve --model my_model.wk --port 8000 --host 0.0.0.0
```

Or with environment variables for multi-model serving:

```bash
WORLDKIT_MODELS="pusht:pusht.wk,cartpole:cartpole.wk" worldkit serve --port 8000
```

## Endpoints

### `GET /health`

Server health check.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": 1
}
```

### `GET /models`

List loaded models.

**Response:**

```json
{
  "models": [
    {
      "name": "default",
      "config_name": "base",
      "latent_dim": 192,
      "num_params": 13000000,
      "device": "cpu"
    }
  ],
  "default_model": "default"
}
```

### `POST /encode`

Encode an observation image to a latent vector.

**Request:**
- `observation` (file): Image file (PNG, JPEG)
- `model` (query, optional): Model name

**Response:**

```json
{
  "latent": [0.123, -0.456, 0.789, ...],
  "dim": 192
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/encode \
  -F "observation=@frame.png"
```

### `POST /predict`

Predict future latent states from an observation and action sequence.

**Request:**
- `observation` (file): Current observation image
- `actions` (form): JSON-encoded list of action arrays
- `model` (query, optional): Model name

**Response:**

```json
{
  "latent_trajectory": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "confidence": 0.8,
  "steps": 10
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "observation=@frame.png" \
  -F 'actions=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]'
```

### `POST /plan`

Plan an optimal action sequence to reach a goal state.

**Request:**
- `current` (file): Current observation image
- `goal` (file): Goal observation image
- `max_steps` (form, optional): Planning horizon (default: 50)
- `n_candidates` (form, optional): CEM candidates (default: 200)
- `model` (query, optional): Model name

**Response:**

```json
{
  "actions": [[0.1, 0.2], [0.3, 0.4], ...],
  "expected_cost": 0.0234,
  "success_probability": 0.85,
  "planning_time_ms": 150.3
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/plan \
  -F "current=@current.png" \
  -F "goal=@goal.png" \
  -F "max_steps=50"
```

### `POST /plausibility`

Score the physical plausibility of a video.

**Request:**
- `video` (file): MP4 video file
- `model` (query, optional): Model name

**Response:**

```json
{
  "plausibility_score": 0.87,
  "anomaly_detected": false,
  "num_frames": 100
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/plausibility \
  -F "video=@test.mp4"
```

### `POST /batch/encode`

Batch encode multiple observations.

**Request body:**

```json
{
  "observations": [
    [[[0.1, 0.2, 0.3], ...], ...],
    [[[0.4, 0.5, 0.6], ...], ...]
  ],
  "model": "default"
}
```

**Response:**

```json
{
  "latents": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "dim": 192,
  "count": 2
}
```

### `POST /batch/predict`

Batch predict for multiple observations.

**Request body:**

```json
{
  "observations": [...],
  "actions": [...],
  "model": "default"
}
```

### `WS /ws/predict`

WebSocket endpoint for streaming frame-by-frame predictions.

**Connect:**

```
ws://localhost:8000/ws/predict?model=default
```

**Send:**

```json
{
  "observation": [[[0.1, 0.2, 0.3], ...], ...],
  "action": [0.1, 0.2]
}
```

**Receive:**

```json
{
  "latent": [0.123, -0.456, ...],
  "confidence": 0.8,
  "latent_trajectory": [[0.1, 0.2, ...], ...],
  "steps": 1
}
```

**Python client example:**

```python
import asyncio
import websockets
import json
import numpy as np

async def stream():
    uri = "ws://localhost:8000/ws/predict"
    async with websockets.connect(uri) as ws:
        obs = np.random.rand(96, 96, 3).tolist()
        action = [0.1, 0.2]

        await ws.send(json.dumps({
            "observation": obs,
            "action": action,
        }))

        response = json.loads(await ws.recv())
        print(f"Latent dim: {len(response['latent'])}")

asyncio.run(stream())
```

### `GET /metrics`

Server performance metrics.

**Response:**

```json
{
  "request_count": {
    "encode": 150,
    "predict": 89,
    "plan": 23,
    "plausibility": 12
  },
  "average_latency_ms": {
    "encode": 12.3,
    "predict": 45.6,
    "plan": 156.7,
    "plausibility": 234.5
  },
  "models_loaded": 1,
  "model_info": [
    {
      "name": "default",
      "config_name": "base",
      "latent_dim": 192,
      "num_params": 13000000,
      "device": "cpu"
    }
  ]
}
```

## Error responses

All errors return a consistent format:

```json
{
  "error": "ValueError",
  "detail": "No pixel data found in uploaded image"
}
```

| Status code | When |
|-------------|------|
| 400 | Invalid input (bad image, malformed JSON) |
| 404 | Model not found |
| 500 | Internal server error |

## Python client

```python
import requests

base_url = "http://localhost:8000"

# Health check
resp = requests.get(f"{base_url}/health")
print(resp.json())

# Encode
with open("frame.png", "rb") as f:
    resp = requests.post(f"{base_url}/encode", files={"observation": f})
latent = resp.json()["latent"]

# Plan
with open("current.png", "rb") as c, open("goal.png", "rb") as g:
    resp = requests.post(
        f"{base_url}/plan",
        files={"current": c, "goal": g},
        data={"max_steps": 50},
    )
plan = resp.json()
print(f"Actions: {len(plan['actions'])}")

# Plausibility
with open("video.mp4", "rb") as f:
    resp = requests.post(f"{base_url}/plausibility", files={"video": f})
print(f"Score: {resp.json()['plausibility_score']:.3f}")
```

## Related

- [Export and deploy tutorial](tutorials/export_deploy.md) — full deployment walkthrough
- [CLI reference](cli.md) — `worldkit serve` command
