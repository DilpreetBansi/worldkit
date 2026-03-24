# API Reference

## WorldModel

### Construction

- `WorldModel.train(data, config, epochs, ...)` — Train from HDF5 data
- `WorldModel.from_hub(model_id)` — Load from WorldKit Hub
- `WorldModel.load(path)` — Load from .wk file

### Inference

- `model.encode(observation)` — Encode to latent vector
- `model.predict(observation, actions)` — Predict future states
- `model.plan(current, goal)` — Plan action sequence via CEM
- `model.plausibility(frames)` — Score physical plausibility

### Export

- `model.export(format, output)` — Export to ONNX/TorchScript
- `model.save(path)` — Save to .wk file

## Model Configs

| Config | Params | Latent Dim | Use Case |
|--------|--------|------------|----------|
| nano | 5M | 128 | Edge/prototyping |
| base | 15M | 192 | Default |
| large | 40M | 384 | Complex 3D |
| xl | 80M | 512 | Multi-object |
