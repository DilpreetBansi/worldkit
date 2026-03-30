# Changelog

## v0.1.0 (2026-03-30)

Initial release. Implements the [LeWM](https://le-wm.github.io/) architecture (JEPA + SIGReg).

### Core

- `WorldModel` class with `train()`, `predict()`, `plan()`, `plausibility()`, `encode()`, `save()`, `load()`, `from_hub()`, `export()`
- Four model configs: `nano` (~3.5M), `base` (~13M), `large` (~54M), `xl` (~102M)
- SIGReg loss with single hyperparameter (`lambda_reg`)
- CEM planner for goal-conditioned action planning
- Hierarchical planner for long-horizon tasks
- Linear probing for latent space evaluation
- Model distillation (`WorldModel.distill()`)
- Online learning (`enable_online_learning()`, `update()`)
- Auto-configuration (`WorldModel.auto_config()`)

### Data

- `HDF5Dataset` for training data loading
- `MultiEnvironmentDataset` for multi-env training with action zero-padding
- `Recorder` for recording Gymnasium environments to HDF5
- `Converter` for video-to-HDF5 conversion

### Environments

- Environment registry with 10 pre-registered environments
- Categories: navigation, manipulation, control, games
- `worldkit env list/info/search/install` CLI commands

### Export

- ONNX export with optimization
- TorchScript export
- TensorRT export (FP16, INT8)
- CoreML export
- ROS2 package generation

### Server

- FastAPI inference server
- Endpoints: `/encode`, `/predict`, `/plan`, `/plausibility`
- Batch endpoints: `/batch/encode`, `/batch/predict`
- WebSocket streaming: `/ws/predict`
- Multi-model serving via environment variables
- Performance metrics via `/metrics`

### CLI

- `worldkit train` — train from HDF5 data
- `worldkit serve` — start REST API server
- `worldkit export` — export models
- `worldkit info` / `inspect` / `validate` — model inspection
- `worldkit convert` / `record` — data preparation
- `worldkit probe` — linear probing
- `worldkit compare` — model comparison
- `worldkit hub list/download` — Hub integration
- `worldkit bench run/quick/report` — benchmarking
- `worldkit env list/info/search/install` — environment management

### Evaluation

- `LinearProbe` for latent space probing
- `LatentVisualizer` with PCA, t-SNE, UMAP
- `RolloutGIFGenerator` for trajectory visualization
- `ModelComparator` with HTML report generation

### Benchmarks

- WorldKit-Bench with 11 tasks across 4 categories
- `BenchmarkSuite` (full, quick, category)
- `BenchmarkRunner` with JSON/HTML output
- Leaderboard entry formatting

### File Format

- `.wk` v2 format (ZIP + safetensors + JSON)
- Backward-compatible v1 loading (torch.save, deprecated)
- `WKFormat.save()`, `load()`, `validate()`, `inspect()`

### Pre-trained Models

- `DilpreetBansi/pusht` (base)
- `DilpreetBansi/pusht-nano` (nano)
- `DilpreetBansi/cartpole-base` (base)
- `DilpreetBansi/cartpole-nano` (nano)

### Backend System

- Pluggable `BaseWorldModelBackend` interface
- `LeWMBackend` (default JEPA + SIGReg)
- `BackendRegistry` for registering custom backends
