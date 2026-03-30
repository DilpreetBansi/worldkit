# Export and deploy

This tutorial covers exporting WorldKit models for production deployment — ONNX, TorchScript, TensorRT, CoreML — and serving via REST API.

## Prerequisites

```bash
pip install worldkit[export,serve]
```

## Save and load

The `.wk` format is WorldKit's native model format — a ZIP archive containing weights (safetensors), config (JSON), and metadata.

```python
from worldkit import WorldModel

model = WorldModel.train(data="data.h5", config="base", epochs=100)

# Save with metadata
model.save(
    "my_model.wk",
    metadata={"dataset": "pusht", "epochs": 100},
    action_space={"dim": 2, "type": "continuous", "low": -1.0, "high": 1.0},
)

# Load
model = WorldModel.load("my_model.wk", device="cpu")
```

## Export to ONNX

ONNX is the most portable format. It runs on any platform with ONNX Runtime.

```python
path = model.export(format="onnx", output="./deploy/")
print(f"Exported to {path}")
# Exported to ./deploy/worldkit_encoder.onnx
```

### Use the ONNX model

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("./deploy/worldkit_encoder.onnx")

obs = np.random.rand(1, 3, 96, 96).astype(np.float32)
result = session.run(None, {"input": obs})
latent = result[0]
print(f"Latent: {latent.shape}")  # (1, 192)
```

## Export to TorchScript

TorchScript models run in C++ without Python. Good for embedded systems.

```python
path = model.export(format="torchscript", output="./deploy/")
print(f"Exported to {path}")
```

### Use the TorchScript model

```python
import torch

scripted = torch.jit.load("./deploy/worldkit_encoder.pt")
obs = torch.randn(1, 3, 96, 96)
latent = scripted(obs)
print(f"Latent: {latent.shape}")  # (1, 192)
```

## Export to TensorRT

TensorRT optimizes models for NVIDIA GPUs. Requires `pip install worldkit[tensorrt]`.

```python
path = model.export(
    format="tensorrt",
    output="./deploy/",
    fp16=True,   # FP16 precision (default)
    int8=False,  # INT8 quantization
)
```

### Benchmark TensorRT performance

```python
from worldkit.export import benchmark_tensorrt

results = benchmark_tensorrt(
    "./deploy/worldkit_encoder.engine",
    input_shape=(1, 3, 96, 96),
    n_runs=100,
)
print(f"Avg latency: {results['avg_latency_ms']:.2f}ms")
print(f"Throughput: {results['throughput_fps']:.0f} FPS")
```

## Export to CoreML

CoreML runs on Apple devices (iOS, macOS). Requires `pip install worldkit[coreml]`.

```python
path = model.export(format="coreml", output="./deploy/")
# Creates ./deploy/worldkit_encoder.mlpackage
```

## CLI export

```bash
# ONNX (default)
worldkit export --model my_model.wk --format onnx --output ./deploy/

# TorchScript
worldkit export --model my_model.wk --format torchscript --output ./deploy/

# TensorRT with FP16
worldkit export --model my_model.wk --format tensorrt --output ./deploy/ --fp16

# TensorRT with INT8
worldkit export --model my_model.wk --format tensorrt --output ./deploy/ --int8

# ROS2 package
worldkit export --model my_model.wk --format ros2 --output ./deploy/ --node-name my_node
```

## REST API server

Serve a model as a REST API with FastAPI:

```bash
worldkit serve --model my_model.wk --port 8000 --host 0.0.0.0
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/models` | GET | List loaded models |
| `/encode` | POST | Encode observation to latent |
| `/predict` | POST | Predict future states |
| `/plan` | POST | Plan action sequence |
| `/plausibility` | POST | Score plausibility |
| `/batch/encode` | POST | Batch encoding |
| `/batch/predict` | POST | Batch prediction |
| `/ws/predict` | WebSocket | Streaming prediction |
| `/metrics` | GET | Performance metrics |

### Query the API

```bash
# Health check
curl http://localhost:8000/health

# Encode an image
curl -X POST http://localhost:8000/encode \
  -F "observation=@frame.png"

# Plan from current to goal
curl -X POST http://localhost:8000/plan \
  -F "current=@current.png" \
  -F "goal=@goal.png" \
  -F "max_steps=50"

# Score a video
curl -X POST http://localhost:8000/plausibility \
  -F "video=@test.mp4"
```

### Python client

```python
import requests
import numpy as np

url = "http://localhost:8000"

# Encode
with open("frame.png", "rb") as f:
    resp = requests.post(f"{url}/encode", files={"observation": f})
latent = resp.json()["latent"]

# Plan
with open("current.png", "rb") as c, open("goal.png", "rb") as g:
    resp = requests.post(
        f"{url}/plan",
        files={"current": c, "goal": g},
        data={"max_steps": 50},
    )
plan = resp.json()
print(f"Actions: {len(plan['actions'])}")
print(f"Cost: {plan['expected_cost']:.4f}")
```

### Multi-model serving

Set the `WORLDKIT_MODELS` environment variable to load multiple models:

```bash
WORLDKIT_MODELS="pusht:pusht_model.wk,cartpole:cartpole_model.wk" \
  worldkit serve --port 8000
```

Query a specific model:

```bash
curl -X POST http://localhost:8000/encode \
  -F "observation=@frame.png" \
  -G -d "model=pusht"
```

## Export format comparison

| Format | Platform | Speed | Size | Dependencies |
|--------|----------|-------|------|-------------|
| `.wk` | Any (Python) | Baseline | Smallest | worldkit |
| ONNX | Any | ~1.5x | ~1x | onnxruntime |
| TorchScript | Any (C++/Python) | ~1.2x | ~1x | libtorch |
| TensorRT | NVIDIA GPU | ~3-5x | ~0.5x | tensorrt |
| CoreML | Apple | ~2-3x | ~0.8x | coremltools |

## Next steps

- [REST API reference](../rest_api.md) — full endpoint documentation
- [CLI reference](../cli.md) — all export and serve commands
- [Benchmark your model](benchmark_your_model.md) — evaluate before deploying
