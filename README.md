<div align="center">

# WorldKit

**The open-source world model SDK.**<br>
Train, predict, plan, and deploy — on a laptop.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/DilpreetBansi/worldkit/actions/workflows/ci.yml/badge.svg)](https://github.com/DilpreetBansi/worldkit/actions)
[![Models on HF](https://img.shields.io/badge/Models-Hugging%20Face-orange)](https://huggingface.co/DilpreetBansi)

[Paper](https://le-wm.github.io/) | [Models](https://huggingface.co/DilpreetBansi) | [Demo](https://huggingface.co/spaces/DilpreetBansi/worldkit-demo) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)

<br>

<img src="demo.gif" alt="WorldKit Demo" width="720">

Try the interactive tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DilpreetBansi/worldkit/blob/main/notebooks/quickstart.ipynb)

</div>

---

## What is WorldKit?

WorldKit is a Python SDK for training and deploying lightweight **world models** — neural networks that learn how environments behave and can imagine future states without interacting with the real world.

```python
from worldkit import WorldModel

# Train a world model from your data
model = WorldModel.train(data="my_data.h5", config="base", epochs=100)

# Imagine the future: given a state and actions, predict what happens next
result = model.predict(current_frame, actions)

# Plan: find the actions that reach a goal state
plan = model.plan(current_frame, goal_frame, max_steps=50)
```

**Why world models matter:** Instead of trial-and-error in the real world (slow, expensive, dangerous), a world model lets an agent "think ahead" by simulating outcomes in a learned latent space. This is how robots can plan manipulation sequences, how game AI can anticipate physics, and how anomaly detectors can flag impossible events.

**Why WorldKit:** Existing world model implementations are research code — coupled to specific environments, hard to train, harder to deploy. WorldKit gives you a clean `train → predict → plan → deploy` pipeline with one hyperparameter.

### Key Features

- **Train in minutes** — 13M-param model trains in ~60 seconds on an M4 MacBook
- **One hyperparameter** — SIGReg regularization replaces 6+ collapse-prevention hyperparameters
- **Plan in latent space** — CEM planner "imagines" thousands of futures without rendering pixels
- **Hierarchical planning** — Long-horizon tasks via subgoal decomposition in latent space
- **Auto-config** — Automatically recommend the best model config for your data and hardware
- **Latent probing** — Linear probe evaluation to measure what the latent space has learned
- **Visualization** — PCA/t-SNE/UMAP latent space plots and rollout GIF generation
- **Model distillation** — Compress a trained model into a smaller student for edge deployment
- **Online learning** — Incrementally update models with streaming experience
- **Benchmarks** — Standardized benchmark suite for comparing world models
- **Federated training** — Privacy-preserving multi-client training with FedAvg aggregation
- **Multi-backend** — Pluggable architecture backends (LeWM default, extensible)
- **Deploy anywhere** — Export to ONNX, TorchScript, TensorRT, or CoreML
- **Hub integration** — Push and pull trained models from Hugging Face

### Supported Environments

WorldKit works across domains -- bring any environment that produces pixel observations:

- **Push-T** (manipulation) -- trained on real expert demonstrations
- **CartPole** (balance control) -- trained on Gymnasium pixel observations
- **Your environment** -- bring your own HDF5 data (see [examples/02_train_from_gym.py](examples/02_train_from_gym.py))

## Install

```bash
pip install worldkit
```

Optional extras:
```bash
pip install worldkit[train]    # WandB logging, Hydra configs
pip install worldkit[envs]     # Gymnasium environment wrappers
pip install worldkit[serve]    # FastAPI inference server
pip install worldkit[export]   # ONNX / TorchScript export
pip install worldkit[all]      # Everything
```

## Quickstart

### Train a model

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="my_data.h5",   # HDF5 with pixels + actions
    config="base",        # nano | base | large | xl
    epochs=100,
)
model.save("my_model.wk")
```

### Auto-select the best config

```python
config, explanation = WorldModel.auto_config(data="my_data.h5", max_training_time="2h")
print(explanation)  # "Recommended 'base' config — fits within 2h on this hardware"
model = WorldModel.train(data="my_data.h5", config=config, epochs=100)
```

### Load a pre-trained model

```python
model = WorldModel.from_hub("DilpreetBansi/pusht")
```

### Predict future states

```python
# Given current observation and a sequence of actions,
# roll out the dynamics model in latent space
result = model.predict(current_frame, actions=[action] * 10)
# result.latent_trajectory: (10, 192) predicted latent states
# result.confidence: prediction confidence score
```

### Plan to reach a goal

```python
# Find an action sequence that takes you from current_frame to goal_frame
plan = model.plan(current_frame, goal_frame, max_steps=50)
# plan.actions: optimized action sequence
# plan.cost: final planning cost (lower = closer to goal)

# For long-horizon tasks, use hierarchical planning
plan = model.hierarchical_plan(current_frame, goal_frame, max_subgoals=5)
# plan.actions: full action sequence connecting subgoals
# plan.subgoal_latents: intermediate subgoal representations
```

### Detect anomalies

```python
# Score whether a video sequence is physically plausible
score = model.plausibility(video_frames)
# 1.0 = expected behavior, 0.0 = physically impossible
```

### Probe the latent space

```python
# Measure what the encoder has learned
result = model.probe(data="my_data.h5", properties=["x", "y", "angle"], labels="labels.csv")
print(result.summary)  # R² per property

# Visualize latent space structure
model.visualize_latent_space(data="my_data.h5", method="pca", save_to="latents.png")

# Generate a rollout GIF
model.rollout_gif(observation, actions, save_to="rollout.gif")
```

### Distill and deploy

```python
# Compress a model for edge deployment
student = WorldModel.distill(teacher=model, student_config="nano", data="my_data.h5")
student.export(format="onnx", output="./deploy/")
```

### Online learning

```python
# Incrementally adapt a model to new experience
model.enable_online_learning(lr=1e-5, buffer_size=1000)
for obs, action, next_obs in stream:
    loss = model.update(obs, action, next_obs)
```

## Pre-trained Models

| Model | Config | Params | Latent Dim | Task | Download |
|-------|--------|--------|------------|------|----------|
| [`DilpreetBansi/pusht`](https://huggingface.co/DilpreetBansi/pusht) | base | 13M | 192 | Push-T manipulation | `WorldModel.from_hub("DilpreetBansi/pusht")` |
| [`DilpreetBansi/pusht-base`](https://huggingface.co/DilpreetBansi/pusht-base) | base | 13M | 192 | Push-T manipulation | `WorldModel.from_hub("DilpreetBansi/pusht-base")` |
| [`DilpreetBansi/pusht-nano`](https://huggingface.co/DilpreetBansi/pusht-nano) | nano | 3.5M | 128 | Push-T manipulation | `WorldModel.from_hub("DilpreetBansi/pusht-nano")` |
| [`DilpreetBansi/cartpole-base`](https://huggingface.co/DilpreetBansi/cartpole-base) | base | 13M | 192 | CartPole balance control | `WorldModel.from_hub("DilpreetBansi/cartpole-base")` |
| [`DilpreetBansi/cartpole-nano`](https://huggingface.co/DilpreetBansi/cartpole-nano) | nano | 3.5M | 128 | CartPole balance control | `WorldModel.from_hub("DilpreetBansi/cartpole-nano")` |

> Train your own and share it: `model.save("my_model.wk")` then upload to the Hub.

## Benchmarks

How does WorldKit compare to other world model frameworks?

| Framework | Architecture | Params | Training Time | Hardware Required | Hyperparams |
|-----------|-------------|--------|--------------|-------------------|-------------|
| **WorldKit** | JEPA + SIGReg | 3.5M--13M | ~60s | Laptop CPU/GPU | 1 (lambda) |
| Dreamer-v3 | RSSM + Actor-Critic | 20M--200M | 1--24 hours | GPU cluster | 30+ |
| TD-MPC2 | MPC + learned model | 5M--50M | 2--8 hours | Single GPU | 15+ |
| IRIS | Transformer + VQ-VAE | 15M--100M | 4--12 hours | Multi-GPU | 20+ |

> WorldKit prioritizes simplicity and accessibility. It trains on a laptop in under a minute with a single hyperparameter (lambda). For production RL, consider combining WorldKit with policy learning frameworks.

## Model Configurations

All configs share the same API. Pick the one that fits your compute budget.

| Config | Params | Latent Dim | Encoder | Predictor Depth | Train Time* |
|--------|--------|------------|---------|-----------------|------------|
| `nano` | ~3.5M | 128 | ViT-Tiny | 2 layers | ~30s |
| `base` | ~13M | 192 | ViT-Small | 3 layers | ~60s |
| `large` | ~54M | 384 | ViT-Base | 4 layers | ~8 min |
| `xl` | ~102M | 512 | ViT-Large | 6 layers | ~20 min |

*On Apple M4 Pro with MPS. GPU times will vary.

## Architecture

WorldKit implements a world model using the [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) (Joint-Embedding Predictive Architecture) pattern — an architecture class [proposed by Yann LeCun](https://openreview.net/forum?id=BZ5a1r-kVsf) where prediction happens in **latent space** rather than pixel space.

JEPA alone is an architecture, not a training method. Many architectures are JEPAs (including Siamese networks from 1993). The critical question is **how you prevent representation collapse** — how you stop the model from learning a trivial mapping where all inputs produce the same output.

WorldKit uses **SIGReg** (Sketch Isotropic Gaussian Regularizer), introduced in the [LeWorldModel paper](https://le-wm.github.io/), which solves collapse with a single hyperparameter:

```
L = L_prediction + λ · SIGReg(Z)

where:
  L_prediction = MSE between predicted and actual latent states
  SIGReg(Z)    = KL divergence approximation enforcing Gaussian structure on Z
  λ             = the ONE hyperparameter you tune (default: 1.0)
```

This replaces the 6+ hyperparameters required by prior methods (VICReg, Barlow Twins, BYOL).

### Components

```
Observation (96x96 RGB)
        │
        ▼
┌───────────────┐
│   ViT Encoder │ ── CLS token pooling ──▶ z ∈ R^192 (latent state)
└───────────────┘
        │
        ▼
┌───────────────────────┐
│ Predictor (AdaLN-Zero)│ ── conditioned on action embeddings
│   Transformer         │ ── causal attention
└───────────────────────┘
        │
        ▼
   z' ∈ R^192 (predicted next state)
        │
        ▼
┌───────────────┐
│  CEM Planner  │ ── samples action candidates
│               │ ── rolls out in latent space (no pixels)
│               │ ── refines toward goal
└───────────────┘
        │
        ▼
   Optimal action sequence
```

- **Encoder** — Vision Transformer (ViT) compresses 96x96 RGB images into compact latent vectors via CLS token pooling. ~200x more compact than patch-level representations.
- **Predictor** — Transformer with AdaLN-Zero conditioning. Given latent state z and action a, predicts next state z'. Autoregressive for multi-step rollouts.
- **Planner** — Cross-Entropy Method (CEM) that searches for optimal actions by "imagining" outcomes entirely in latent space — no rendering, no physics engine needed.

## CLI

```bash
# Train
worldkit train --data ./data.h5 --config base --epochs 100

# Serve as REST API
worldkit serve --model ./model.wk --port 8000

# Export for edge deployment
worldkit export --model ./model.wk --format onnx

# Inspect a model
worldkit info --model ./model.wk

# Convert video data to HDF5
worldkit convert --input ./videos/ --output ./data.h5 --fps 10

# Hub operations
worldkit hub download DilpreetBansi/pusht
```

## REST API

```bash
worldkit serve --model ./model.wk --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status and model info |
| `/encode` | POST | Encode observation to latent vector |
| `/predict` | POST | Predict future latent states from actions |
| `/plan` | POST | Plan optimal action sequence to reach a goal |
| `/plausibility` | POST | Score physical plausibility of a video |

## Examples

| Example | What it shows |
|---------|---------------|
| [`01_quickstart.py`](examples/01_quickstart.py) | Train, predict, plan in 5 lines |
| [`02_train_from_gym.py`](examples/02_train_from_gym.py) | Record a Gymnasium env and train |
| [`03_plan_to_goal.py`](examples/03_plan_to_goal.py) | Goal-conditioned CEM planning |
| [`04_anomaly_detection.py`](examples/04_anomaly_detection.py) | Detect physically impossible events |
| [`05_export_onnx.py`](examples/05_export_onnx.py) | Export to ONNX / TorchScript |
| [`06_serve_api.py`](examples/06_serve_api.py) | Deploy as a REST API |
| [`07_latent_probing.py`](examples/07_latent_probing.py) | Visualize what the latent space learns |

## Project Structure

```
worldkit/
├── core/           # WorldModel, ViT encoder, predictor, planners, SIGReg loss, distillation, online learning
├── data/           # HDF5 dataset, env recorder, video converter, multi-env datasets
├── cli/            # CLI commands (train, serve, export, hub, convert, bench, env)
├── server/         # FastAPI inference server with WebSocket support
├── envs/           # Gymnasium wrappers and environment registry
├── eval/           # Probing, visualization, comparison, rollout GIFs
├── bench/          # Standardized benchmark suite and leaderboard
├── export/         # ONNX, TorchScript, TensorRT, CoreML export
├── federated/      # Privacy-preserving federated training
└── hub/            # Hugging Face Hub integration
```

## Research & Acknowledgments

WorldKit is an **independent open-source project** created by [Dilpreet Bansi](https://github.com/DilpreetBansi). It is not affiliated with, endorsed by, or sponsored by any of the researchers or institutions listed below.

The concept of learning world models with neural networks was pioneered by:

> **Recurrent World Models Facilitate Policy Evolution**
> David Ha, Jürgen Schmidhuber (2018) — NIPS 2018
> [Paper](https://worldmodels.github.io/) | [Code](https://github.com/hardmaru/WorldModelsExperiments)

Ha & Schmidhuber demonstrated that agents can learn entirely inside their own "dreams" — training in a learned simulation of the environment and transferring policies back to reality. Their VAE + MDN-RNN architecture is the foundation that all modern world models build upon.

WorldKit implements the architecture and training methodology from:

> **LeWorldModel: Learning World Models with Joint-Embedding Predictive Architectures**
> Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero (2026)
> [Paper](https://le-wm.github.io/) | [Code](https://github.com/lucas-maes/le-wm)

LeWM builds on Ha & Schmidhuber's vision but replaces the generative approach (pixel reconstruction) with a JEPA-based approach (latent prediction), and uses SIGReg to solve the collapse problem with a single hyperparameter.

The JEPA architectural pattern was proposed in:

> **A Path Towards Autonomous Machine Intelligence**
> Yann LeCun (2022)
> [Paper](https://openreview.net/forum?id=BZ5a1r-kVsf)

WorldKit builds on these open-source projects:

- **[PyTorch](https://pytorch.org/)** — Deep learning framework (BSD License)
- **[Hugging Face Hub](https://huggingface.co/)** — Model hosting (Apache 2.0)
- **[Vision Transformer](https://arxiv.org/abs/2010.11929)** — Dosovitskiy et al., 2020
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST API framework (MIT License)

## Citation

If you use WorldKit in your research, please cite:

```bibtex
@software{worldkit2026,
  title={WorldKit: Open-Source SDK for JEPA World Models},
  author={Bansi, Dilpreet},
  year={2026},
  url={https://github.com/DilpreetBansi/worldkit},
  version={0.2.0}
}

@article{lewm2026,
  title   = {LeWorldModel: Learning World Models with Joint-Embedding Predictive Architectures},
  author  = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  year    = {2026},
  url     = {https://le-wm.github.io/}
}

@incollection{ha2018worldmodels,
  title     = {Recurrent World Models Facilitate Policy Evolution},
  author    = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages     = {2451--2463},
  year      = {2018},
  url       = {https://worldmodels.github.io}
}
```

## Roadmap

WorldKit v0.2 ships with the LeWM architecture and a full evaluation/deployment toolkit. The goal is to become a **unified SDK for all world model architectures** — same `train/predict/plan` API, multiple backends.

| Version | What's New | Status |
|---------|-----------|--------|
| **v0.1** | LeWM architecture (JEPA + SIGReg), CEM planning, Hub integration | Released |
| **v0.2** | Hierarchical planning, auto-config, probing, visualization, distillation, online learning, benchmarks, federated training, multi-backend architecture, TensorRT/CoreML export | **Available now** |
| v0.3 | [Ha & Schmidhuber (2018)](https://worldmodels.github.io/) (VAE + MDN-RNN) backend | Planned |
| v0.4 | [Dreamer V4](https://arxiv.org/abs/2301.04104) (VAE-based) backend | Planned |
| v0.5 | [TD-MPC2](https://arxiv.org/abs/2310.16828) (task-specific MPC) backend | Planned |
| v0.6 | [DIAMOND](https://arxiv.org/abs/2405.12399) (diffusion-based) backend | Planned |
| v0.7 | Custom architecture API | Planned |

The vision:

```python
# Today (v0.2) — LeWM is the default backend, with full eval + deployment
model = WorldModel.train(data="my_data.h5", config="base")

# Future (v0.3+) — choose your architecture, same API
model = WorldModel.train(data="my_data.h5", arch="lewm", config="base")
model = WorldModel.train(data="my_data.h5", arch="ha2018", config="base")
model = WorldModel.train(data="my_data.h5", arch="dreamer", config="medium")
model = WorldModel.train(data="my_data.h5", arch="td-mpc", config="large")
```

One API. Any world model. Train on a laptop, deploy anywhere.

Want to help build this? See [CONTRIBUTING.md](CONTRIBUTING.md).

## Known Limitations

- **Limited environments** — pre-trained models currently cover Push-T and CartPole. More environments and real-world robotics models coming soon.
- **No video decoder** — WorldKit predicts in latent space. It does not reconstruct pixel observations from latent states (by design — this is a feature of JEPA, not a limitation).
- **Single-task models** — each model is trained on one environment. Multi-task and transfer learning are planned.
- **CPU/MPS training only tested** — CUDA training works but is less extensively tested at this stage.

## Disclaimer

This software is provided "as is" without warranty of any kind. WorldKit is an independent open-source project. It is not affiliated with, endorsed by, or connected to Meta, FAIR, NYU, or any other company or research institution.

See [NOTICE](NOTICE) for complete third-party attribution.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT License](LICENSE) — Copyright (c) 2026 Dilpreet Bansi and WorldKit Contributors.

---

<div align="center">

Built by [Dilpreet Bansi](https://github.com/DilpreetBansi)

</div>
