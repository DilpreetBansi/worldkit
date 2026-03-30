# FAQ

## General

### What is a world model?

A world model is a neural network that learns how an environment behaves. Given the current state and an action, it predicts the next state. WorldKit does this entirely in a learned latent space — no pixel rendering needed.

### What is JEPA?

JEPA (Joint-Embedding Predictive Architecture) is an architecture pattern [proposed by Yann LeCun](https://openreview.net/forum?id=BZ5a1r-kVsf) where prediction happens in latent space rather than pixel space. WorldKit implements JEPA with a ViT encoder and a Transformer predictor.

### What is SIGReg?

SIGReg (Sketch Isotropic Gaussian Regularizer) is a loss function from the [LeWM paper](https://le-wm.github.io/) that prevents representation collapse with a single hyperparameter. It replaces the 6+ hyperparameters required by prior methods like VICReg and Barlow Twins.

### What does "1 hyperparameter" mean?

Traditional world model frameworks require tuning 15-30+ hyperparameters. WorldKit's SIGReg regularization has one: `lambda_reg` (default 1.0). This controls the balance between prediction accuracy and latent space structure.

### Can WorldKit reconstruct pixels?

No. WorldKit predicts in latent space by design. It does not decode latent vectors back into images. This is a feature of JEPA — predicting in latent space is faster, more compact, and avoids the blurriness of pixel-space prediction.

## Training

### How long does training take?

| Config | ~100 epochs (M4 Pro, MPS) |
|--------|--------------------------|
| nano | ~30 seconds |
| base | ~60 seconds |
| large | ~8 minutes |
| xl | ~20 minutes |

GPU times vary. CUDA is generally faster than MPS for large configs.

### How much data do I need?

- **Minimum**: 50 episodes, 50 steps each (~2,500 frames)
- **Recommended**: 200+ episodes, 200+ steps each (~40,000 frames)

More data generally produces better models. If results are poor, increasing data is usually more effective than changing the config.

### Can I train on video without actions?

Yes. Use the `Converter` to create HDF5 from video files. Without action labels, the model learns visual dynamics but cannot do action-conditioned prediction or planning. Plausibility scoring still works.

### Can I train on multiple environments?

Yes. Pass a list of HDF5 files to `WorldModel.train()`:

```python
model = WorldModel.train(
    data=["env1.h5", "env2.h5"],
    action_dim=4,  # max action dim across envs
)
```

Actions are zero-padded automatically.

### My model isn't learning. What should I try?

1. Check your data — verify HDF5 structure and pixel values
2. Increase epochs (try 200+)
3. Increase data (more episodes)
4. Try `lambda_reg=0.5` or `lambda_reg=2.0`
5. Use a larger config (`"base"` instead of `"nano"`)

## Inference

### How fast is planning?

With the base config and default CEM settings (200 candidates, 5 iterations):
- ~150ms per plan on CPU
- ~50ms per plan on GPU

Planning time scales with `n_candidates * n_iterations * max_steps`.

### What input formats are accepted?

The `encode()`, `predict()`, and `plan()` methods accept:
- Shape: `(H, W, C)` or `(C, H, W)`
- Type: `uint8` [0, 255] or `float32` [0, 1]
- Size: Any resolution (resized internally to `image_size`)

### Can I use WorldKit for real-time control?

Yes, using MPC (Model Predictive Control). Plan, execute a few actions, then re-plan:

```python
while not done:
    plan = model.plan(obs, goal, max_steps=30)
    for action in plan.actions[:5]:
        obs, reward, done, _, info = env.step(action)
```

## Deployment

### What export formats are supported?

| Format | Target |
|--------|--------|
| ONNX | Cross-platform (ONNX Runtime) |
| TorchScript | C++ inference (no Python) |
| TensorRT | NVIDIA GPUs (optimized) |
| CoreML | Apple devices (iOS, macOS) |
| ROS2 | Robot operating system |

### Can I serve multiple models?

Yes. Use the `WORLDKIT_MODELS` environment variable:

```bash
WORLDKIT_MODELS="model1:path1.wk,model2:path2.wk" worldkit serve
```

Query a specific model with `?model=model1`.

### Is the .wk format safe?

Yes. The v2 format uses safetensors (no pickle) and JSON for metadata. No arbitrary code execution on load.

## Troubleshooting

### "KeyError: No pixel data found in HDF5 file"

Your HDF5 file uses a key name that WorldKit doesn't recognize. Rename to one of: `pixels`, `observations`, `obs`, `images`.

### "ValueError: action_dim mismatch"

The model's `action_dim` doesn't match the actions in your data. Specify the correct `action_dim` when training:

```python
model = WorldModel.train(data="data.h5", config="base", action_dim=4)
```

### CUDA out of memory

- Use a smaller config (`"nano"` or `"base"`)
- Reduce `batch_size` (e.g., 32 instead of 64)
- Use `device="cpu"` for inference (slower but no VRAM limit)

### Import errors for optional features

Install the relevant extra:

```bash
pip install worldkit[envs]      # for gymnasium
pip install worldkit[serve]     # for fastapi
pip install worldkit[export]    # for onnx
pip install worldkit[train]     # for wandb, sklearn
```
