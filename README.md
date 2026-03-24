<p align="center">
  <h1 align="center">WorldKit</h1>
  <p align="center">
    <strong>The open-source world model runtime.</strong><br>
    Train physics-aware AI on a laptop. Deploy anywhere.
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
    <a href="https://github.com/DilpreetBansi/worldkit/issues"><img src="https://img.shields.io/github/issues/DilpreetBansi/worldkit" alt="Issues"></a>
  </p>
</p>

---

## What is WorldKit?

WorldKit is an **independent, open-source Python SDK** that provides a clean, developer-friendly interface for training, predicting, planning, and evaluating lightweight world models.

WorldKit is built on top of the **JEPA (Joint-Embedding Predictive Architecture)** introduced in the [LeWorldModel paper](#acknowledgments). We provide an original SDK layer — the underlying research architecture is credited fully below.

```python
from worldkit import WorldModel

model = WorldModel.train(data="my_data.h5", config="base", epochs=100)
plan  = model.plan(current_frame, goal_frame, max_steps=50)
```

**Key capabilities:**
- **5 lines to train** a world model from your own data
- **1 line to predict** future states from actions
- **1 line to plan** optimal action sequences to reach goals
- **Anomaly detection** via violation-of-expectation scoring
- **Multiple export targets** — ONNX, TorchScript for edge deployment

## Install

```bash
pip install worldkit
```

With optional extras:
```bash
pip install worldkit[train]    # Training deps (WandB, Hydra, transformers)
pip install worldkit[envs]     # Gymnasium environment wrappers
pip install worldkit[serve]    # FastAPI inference server
pip install worldkit[export]   # ONNX/TorchScript export
pip install worldkit[all]      # Everything
```

## Quickstart

```python
from worldkit import WorldModel
import numpy as np

# Train from HDF5 data
model = WorldModel.train(data="my_data.h5", config="base", epochs=100)

# Or load a pre-trained model
# model = WorldModel.from_hub("worldkit/pusht")

# Encode an observation to a compact latent vector
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)  # → (192,) latent vector

# Predict future states given actions
actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)

# Plan an action sequence to reach a goal
goal = np.random.rand(96, 96, 3).astype(np.float32)
plan = model.plan(obs, goal, max_steps=50)

# Score physical plausibility of a video sequence
frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(30)]
score = model.plausibility(frames)  # → 0.0 (impossible) to 1.0 (expected)

# Save and load
model.save("my_model.wk")
loaded = WorldModel.load("my_model.wk")

# Export for deployment
model.export(format="torchscript", output="./deploy/")
```

## Model Configurations

WorldKit ships with four model sizes. All share the same API.

| Config | Params | Latent Dim | Train Time (1 GPU) | Use Case |
|--------|--------|------------|---------------------|----------|
| `nano` | ~3.5M | 128 | ~1 hour | Edge devices, fast prototyping |
| `base` | ~13M | 192 | ~3 hours | Default — based on the paper's architecture |
| `large` | ~54M | 384 | ~8 hours | Complex 3D environments |
| `xl` | ~102M | 512 | ~20 hours | Multi-object, high-fidelity |

## CLI

```bash
# Train
worldkit train --data ./data.h5 --config base --epochs 100

# Serve as REST API
worldkit serve --model ./model.wk --port 8000

# Export for deployment
worldkit export --model ./model.wk --format onnx

# Model info
worldkit info --model ./model.wk

# Convert video data
worldkit convert --input ./videos/ --output ./data.h5 --fps 10

# Hub (Hugging Face)
worldkit hub list
worldkit hub download worldkit/pusht
```

## REST API

```bash
worldkit serve --model ./model.wk --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status |
| `/encode` | POST | Encode observation to latent vector |
| `/predict` | POST | Predict future states from actions |
| `/plan` | POST | Plan action sequence to reach goal |
| `/plausibility` | POST | Score physical plausibility of video |

## Architecture Overview

WorldKit implements a JEPA (Joint-Embedding Predictive Architecture) world model with three core components:

1. **Encoder** (Vision Transformer) — Compresses raw pixel observations into compact latent vectors via CLS token pooling
2. **Predictor** (Transformer with AdaLN-Zero) — Models environment dynamics in latent space: given state z and action a, predicts next state z'
3. **CEM Planner** (Cross-Entropy Method) — Searches for optimal action sequences by "imagining" outcomes entirely in latent space

The training loss has only **one tunable hyperparameter** (lambda):

```
L = L_prediction + λ · SIGReg(Z)
```

SIGReg (Sketch Isotropic Gaussian Regularizer) prevents representation collapse by enforcing a Gaussian structure on the latent space — replacing the 6 hyperparameters needed by prior approaches (VICReg, Barlow Twins, etc.) with just one.

## Project Structure

```
worldkit/
├── core/           # Model, encoder, predictor, planner, losses, config
├── data/           # HDF5 dataset loading, env recording, video conversion
├── cli/            # Click CLI (train, serve, export, hub, convert, record)
├── server/         # FastAPI inference server
├── envs/           # Gymnasium environment wrappers
├── eval/           # Evaluation tools (planning benchmarks, probing, visualization)
├── export/         # ONNX and TorchScript export
└── hub/            # Hugging Face Hub integration
```

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---------|-------------|
| [`01_quickstart.py`](examples/01_quickstart.py) | Train, predict, plan in 5 lines |
| [`02_train_from_gym.py`](examples/02_train_from_gym.py) | Record and train from Gymnasium |
| [`03_plan_to_goal.py`](examples/03_plan_to_goal.py) | Goal-conditioned CEM planning |
| [`04_anomaly_detection.py`](examples/04_anomaly_detection.py) | Plausibility-based anomaly detection |
| [`05_export_onnx.py`](examples/05_export_onnx.py) | Export to TorchScript/ONNX |
| [`06_serve_api.py`](examples/06_serve_api.py) | Deploy as REST API |
| [`07_latent_probing.py`](examples/07_latent_probing.py) | Explore the learned latent space |

## Acknowledgments

WorldKit is an **independent project** created by [Dilpreet Bansi](https://github.com/DilpreetBansi). It is **not affiliated with, endorsed by, or sponsored by** any of the researchers or institutions listed below.

WorldKit's architecture is based on research from the following paper:

> **LeWorldModel: Learning World Models with Joint-Embedding Predictive Architectures**
> Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero (2026)
> [Paper](https://le-wm.github.io/) | [Code](https://github.com/lucas-maes/le-wm)

This paper introduced the JEPA-based world model with SIGReg regularization that WorldKit wraps into a developer-friendly SDK. We are deeply grateful to the authors for making their research and code publicly available.

WorldKit also builds on top of these open-source projects:

- **[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)** — The underlying research library published on PyPI by the Galilai Group
- **[PyTorch](https://pytorch.org/)** — Deep learning framework (BSD License)
- **[Hugging Face Hub](https://huggingface.co/)** — Model hosting and distribution (Apache 2.0)
- **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)** — Encoder architecture by Dosovitskiy et al.
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST API framework (MIT License)

The SIGReg loss function used in WorldKit is based on the Sketch Isotropic Gaussian Regularizer from the LeWM paper. Our implementation is an original re-implementation for use within the WorldKit SDK.

## Disclaimer

This software is provided "as is" without warranty of any kind. WorldKit is an **independent open-source project**. It is not affiliated with, endorsed by, or connected to Meta, FAIR, NYU, AMI Labs, World Labs, Physical Intelligence, or any other company or research institution.

WorldKit does not include or redistribute any proprietary code, datasets, or model weights from third parties. Pre-trained models available on the WorldKit Hub are trained independently by the WorldKit community using publicly available environments and datasets.

The authors and contributors of WorldKit are not responsible for any use or misuse of this software. Users are responsible for ensuring their use complies with all applicable laws and regulations.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

WorldKit is released under the [MIT License](LICENSE).

```
MIT License — Copyright (c) 2026 WorldKit Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

See [LICENSE](LICENSE) for the full text.

---

<p align="center">
  Built by <a href="https://github.com/DilpreetBansi">Dilpreet Bansi</a>
</p>
