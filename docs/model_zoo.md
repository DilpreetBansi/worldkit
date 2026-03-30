# Model Zoo

Pre-trained WorldKit models hosted on [Hugging Face Hub](https://huggingface.co/DilpreetBansi).

## Available models

### Manipulation

| Model | Config | Params | Latent dim | Description |
|-------|--------|--------|------------|-------------|
| [`DilpreetBansi/pusht`](https://huggingface.co/DilpreetBansi/pusht) | base | 13M | 192 | Push T-block to target position |
| [`DilpreetBansi/pusht-base`](https://huggingface.co/DilpreetBansi/pusht-base) | base | 13M | 192 | Push-T (base variant) |
| [`DilpreetBansi/pusht-nano`](https://huggingface.co/DilpreetBansi/pusht-nano) | nano | 3.5M | 128 | Push-T (nano, edge-deployable) |

### Control

| Model | Config | Params | Latent dim | Description |
|-------|--------|--------|------------|-------------|
| [`DilpreetBansi/cartpole-base`](https://huggingface.co/DilpreetBansi/cartpole-base) | base | 13M | 192 | CartPole balance control |
| [`DilpreetBansi/cartpole-nano`](https://huggingface.co/DilpreetBansi/cartpole-nano) | nano | 3.5M | 128 | CartPole (nano) |

## Load a model

```python
from worldkit import WorldModel

# Load from the Hub
model = WorldModel.from_hub("DilpreetBansi/pusht")

# Check model info
print(f"Config: {model.config.name}")
print(f"Parameters: {model.num_params:,}")
print(f"Latent dim: {model.latent_dim}")
```

## Use a model

```python
import numpy as np

# Encode
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)

# Predict
actions = [np.array([0.1, 0.2])] * 10
result = model.predict(obs, actions)

# Plan
goal = np.random.rand(96, 96, 3).astype(np.float32)
plan = model.plan(obs, goal, max_steps=50)

# Plausibility
frames = [np.random.rand(96, 96, 3).astype(np.float32) for _ in range(20)]
score = model.plausibility(frames)
```

## Download via CLI

```bash
# List available models
worldkit hub list

# Download a model
worldkit hub download DilpreetBansi/pusht --output ./models/
```

## Browse models

```python
from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(author="DilpreetBansi", sort="downloads", direction=-1)
for m in models:
    print(f"{m.id}: {m.downloads} downloads")
```

## Contribute a model

Train your own model and share it on the Hub. See the [Contribute a model tutorial](tutorials/contribute_model.md) for a step-by-step guide.

```python
# Train
model = WorldModel.train(data="my_data.h5", config="base", epochs=200)
model.save("my_model.wk")

# Upload
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-username/my-model", exist_ok=True)
api.upload_file(
    path_or_fileobj="my_model.wk",
    path_in_repo="model.wk",
    repo_id="your-username/my-model",
)
```
