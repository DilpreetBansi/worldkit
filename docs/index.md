# WorldKit

WorldKit is a Python SDK for training and deploying lightweight world models. It implements the [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) architecture with [SIGReg](https://le-wm.github.io/) regularization — 15M parameters, 1 hyperparameter, trains on a single GPU.

## What are world models?

A world model is a neural network that learns how an environment behaves. Given a current state and an action, it predicts the next state — entirely in a learned latent space, without rendering pixels or running a physics engine. This enables:

- **Planning** — find action sequences that reach a goal by "imagining" outcomes
- **Anomaly detection** — score whether observed behavior is physically plausible
- **Simulation** — generate training data for downstream policies

## Core workflow

```python
from worldkit import WorldModel

# Train from pixel observations + actions
model = WorldModel.train(data="my_data.h5", config="base", epochs=100)

# Predict future states
result = model.predict(current_frame, actions=[action] * 10)

# Plan to reach a goal
plan = model.plan(current_frame, goal_frame, max_steps=50)

# Score physical plausibility
score = model.plausibility(video_frames)

# Save and deploy
model.save("my_model.wk")
model.export(format="onnx", output="./deploy/")
```

## Documentation

| Section | Description |
|---------|-------------|
| [Installation](installation.md) | Install options, extras, troubleshooting |
| [Quickstart](quickstart.md) | Train your first model in 5 minutes |
| **API Reference** | |
| [WorldModel](api/worldmodel.md) | The main class — every method documented |
| [Config](api/config.md) | Model configurations and presets |
| [Prediction](api/prediction.md) | PredictionResult and prediction workflow |
| [Planning](api/planning.md) | CEM planner and hierarchical planning |
| [Plausibility](api/plausibility.md) | Anomaly detection via violation-of-expectation |
| [Probing](api/probing.md) | Linear probing to inspect latent representations |
| [Data](api/data.md) | HDF5 datasets, environment recording, video conversion |
| **Tutorials** | |
| [Train your first model](tutorials/train_first_model.md) | End-to-end training with Push-T |
| [Plan robot actions](tutorials/plan_robot_actions.md) | Goal-conditioned planning |
| [Anomaly detection](tutorials/anomaly_detection.md) | Detect physically impossible events |
| [Export and deploy](tutorials/export_deploy.md) | ONNX, TorchScript, REST API |
| [Custom environments](tutorials/custom_environment.md) | Bring your own environment |
| [Contribute a model](tutorials/contribute_model.md) | Train and upload to the Hub |
| [Benchmark your model](tutorials/benchmark_your_model.md) | Run WorldKit-Bench |
| **Guides** | |
| [Model configurations](guides/model_configs.md) | Choosing the right config |
| [Data preparation](guides/data_preparation.md) | Preparing training data |
| [.wk file format](guides/wk_format.md) | Model file format specification |
| [Architecture](guides/architecture.md) | Technical deep dive into JEPA + SIGReg |
| **Reference** | |
| [CLI](cli.md) | Command-line interface reference |
| [REST API](rest_api.md) | Inference server endpoints |
| [Model Zoo](model_zoo.md) | Pre-trained model catalog |
| [Benchmarks](benchmarks.md) | WorldKit-Bench documentation |
| [FAQ](faq.md) | Common questions |
| [Changelog](changelog.md) | Version history |
| [Contributing](contributing.md) | How to contribute |

## Links

- [GitHub](https://github.com/worldkit-ai/worldkit)
- [PyPI](https://pypi.org/project/worldkit/)
- [Hugging Face Models](https://huggingface.co/DilpreetBansi)
- [LeWM Paper](https://le-wm.github.io/)
