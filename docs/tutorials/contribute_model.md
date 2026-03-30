# Contribute a model

This tutorial shows how to train a world model, upload it to the Hugging Face Hub, and make it available via `WorldModel.from_hub()`.

## Prerequisites

```bash
pip install worldkit[train]
pip install huggingface-hub
```

Log in to Hugging Face:

```bash
huggingface-cli login
```

## Step 1: Train a model

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="my_env_data.h5",
    config="base",
    epochs=200,
    lambda_reg=1.0,
    device="auto",
)
```

## Step 2: Save with metadata

Include rich metadata so others understand the model:

```python
model.save(
    "my_env_model.wk",
    metadata={
        "dataset": "my_env",
        "epochs": 200,
        "final_train_loss": 0.023,
        "description": "Base model trained on 500 episodes of MyEnv-v1",
    },
    action_space={
        "dim": 3,
        "type": "continuous",
        "low": -1.0,
        "high": 1.0,
    },
    model_card={
        "name": "my_env_base",
        "description": "WorldKit base model trained on MyEnv-v1",
        "architecture": "JEPA + SIGReg",
        "parameters": model.num_params,
        "latent_dim": model.latent_dim,
        "tags": ["robotics", "manipulation"],
    },
)
```

## Step 3: Upload to Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()

# Create a repository
repo_id = "your-username/my-env-base"
api.create_repo(repo_id, exist_ok=True)

# Upload the model
api.upload_file(
    path_or_fileobj="my_env_model.wk",
    path_in_repo="model.wk",
    repo_id=repo_id,
)

print(f"Uploaded to https://huggingface.co/{repo_id}")
```

## Step 4: Verify download

```python
model = WorldModel.from_hub("your-username/my-env-base")
print(f"Loaded: {model.config.name}, {model.num_params:,} params")
```

## Step 5: Add a model card

Create a `README.md` for your Hugging Face repo:

```markdown
---
tags:
  - worldkit
  - world-model
  - robotics
license: mit
---

# MyEnv Base Model

A WorldKit world model trained on MyEnv-v1.

## Usage

```python
from worldkit import WorldModel

model = WorldModel.from_hub("your-username/my-env-base")
```

## Training details

- **Config**: base (13M params, latent_dim=192)
- **Data**: 500 episodes, 200 steps each
- **Epochs**: 200
- **Lambda**: 1.0
- **Final loss**: 0.023

## Environment

- **Name**: MyEnv-v1
- **Action space**: continuous, dim=3, range=[-1, 1]
- **Observation**: 96x96 RGB
```

Upload it:

```python
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)
```

## Upload multiple configs

Train and upload nano and base variants:

```python
for config_name in ["nano", "base"]:
    model = WorldModel.train(
        data="my_env_data.h5",
        config=config_name,
        epochs=200,
    )

    filename = f"my_env_{config_name}.wk"
    model.save(filename)

    repo_id = f"your-username/my-env-{config_name}"
    api.create_repo(repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo="model.wk",
        repo_id=repo_id,
    )
```

## Browse available models

```bash
worldkit hub list
```

```python
from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(author="worldkit", sort="downloads", direction=-1)
for m in models:
    print(f"{m.id}: {m.downloads} downloads")
```

## Next steps

- [Model Zoo](../model_zoo.md) — catalog of available pre-trained models
- [Benchmark your model](benchmark_your_model.md) — evaluate before sharing
- [Train your first model](train_first_model.md) — training walkthrough
