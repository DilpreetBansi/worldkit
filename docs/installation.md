# Installation

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later

## Basic install

```bash
pip install worldkit
```

This installs the core package with training, prediction, planning, and model I/O.

## Optional extras

WorldKit uses optional dependency groups to keep the base install lightweight.

```bash
# Training with WandB logging and Hydra configs
pip install worldkit[train]

# Gymnasium environment wrappers and recording
pip install worldkit[envs]

# FastAPI inference server
pip install worldkit[serve]

# ONNX and TorchScript export
pip install worldkit[export]

# TensorRT optimization (requires NVIDIA GPU)
pip install worldkit[tensorrt]

# CoreML export (macOS only)
pip install worldkit[coreml]

# Everything
pip install worldkit[all]

# Development (includes testing, linting, all extras)
pip install worldkit[dev]
```

### What each extra includes

| Extra | Packages | Purpose |
|-------|----------|---------|
| `train` | wandb, hydra-core, omegaconf, lightning, transformers, scikit-learn, matplotlib | Experiment tracking, config management, probe evaluation |
| `envs` | gymnasium, opencv-python | Environment wrappers and pixel recording |
| `serve` | fastapi, uvicorn, python-multipart | REST API inference server |
| `export` | onnx, onnxruntime | ONNX model export |
| `tensorrt` | tensorrt, onnx | NVIDIA TensorRT optimization |
| `coreml` | coremltools | Apple CoreML export |
| `all` | train + envs + serve + export | All of the above |
| `dev` | all + pytest, pytest-cov, ruff, pre-commit | Development and testing |

## Install from source

```bash
git clone https://github.com/worldkit-ai/worldkit.git
cd worldkit
pip install -e ".[dev]"
```

## Verify installation

```python
from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA

config = get_config("nano", action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device="cpu")

import numpy as np
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f"OK: {model.num_params:,} params, latent={z.shape}")
# OK: 3,500,000 params, latent=torch.Size([128])
```

## Device support

WorldKit automatically selects the best available device:

| Device | Support | Notes |
|--------|---------|-------|
| CPU | Full | Default fallback, works everywhere |
| CUDA | Full | Recommended for training |
| MPS | Full | Apple Silicon (M1/M2/M3/M4) |

To explicitly set the device:

```python
model = WorldModel.train(data="data.h5", config="base", device="cuda")
# or
model = WorldModel.load("model.wk", device="mps")
```

Use `device="auto"` (the default) to let WorldKit choose: CUDA > MPS > CPU.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

PyTorch must be installed separately if not pulled in by pip. Follow the [official PyTorch install guide](https://pytorch.org/get-started/locally/) for your platform.

### `ImportError: No module named 'gymnasium'`

Install the environments extra:

```bash
pip install worldkit[envs]
```

### `ImportError: No module named 'fastapi'`

Install the server extra:

```bash
pip install worldkit[serve]
```

### CUDA out of memory

Use a smaller config (`"nano"` or `"base"`) or reduce batch size:

```python
model = WorldModel.train(data="data.h5", config="nano", batch_size=32)
```

### MPS fallback warning on macOS

Some operations fall back to CPU on MPS. This is normal and handled automatically. Training still benefits from MPS acceleration for most operations.
