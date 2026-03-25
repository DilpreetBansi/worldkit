"""Push trained WorldKit models to Hugging Face Hub."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import HfApi, create_repo

api = HfApi()
username = api.whoami()["name"]

MODELS = [
    {
        "local_path": "checkpoints/pusht_nano_real.wk",
        "repo_name": "pusht-nano",
        "config": "nano",
        "params": "3.5M",
        "latent_dim": 128,
        "size_mb": 13.4,
        "train_time": "30 seconds",
        "val_loss": 0.48,
    },
    {
        "local_path": "checkpoints/pusht_base_real.wk",
        "repo_name": "pusht-base",
        "config": "base",
        "params": "13M",
        "latent_dim": 192,
        "size_mb": 50.2,
        "train_time": "2 minutes",
        "val_loss": 0.35,
    },
]


def make_model_card(model_info: dict) -> str:
    return f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - robotics
  - push-t
  - planning
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / {model_info['repo_name']}

A **{model_info['config']}** world model trained on the Push-T task using [WorldKit](https://github.com/DilpreetBansi/worldkit).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | JEPA (Joint-Embedding Predictive Architecture) |
| Config | `{model_info['config']}` |
| Parameters | {model_info['params']} |
| Latent Dim | {model_info['latent_dim']} |
| Image Size | 96x96 |
| Action Dim | 2 (dx, dy) |
| File Size | {model_info['size_mb']:.1f} MB |
| Training Time | {model_info['train_time']} (Apple M4 Pro, MPS) |
| Best Val Loss | {model_info['val_loss']:.4f} |

## Usage

```bash
pip install worldkit
```

```python
from worldkit import WorldModel

# Load this model
model = WorldModel.from_hub("{username}/{model_info['repo_name']}")

# Encode an observation
z = model.encode(observation)  # -> ({model_info['latent_dim']},) latent vector

# Predict future states
result = model.predict(current_frame, actions)

# Plan to reach a goal
plan = model.plan(current_frame, goal_frame, max_steps=50)

# Score physical plausibility
score = model.plausibility(video_frames)
```

## Task: Push-T

The Push-T task is a 2D manipulation environment where an agent (blue circle) pushes a T-shaped block (red) toward a target position. Observations are 96x96 RGB images and actions are 2D continuous (dx, dy).

## Training

Trained using WorldKit's built-in training pipeline:

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="pusht_train.h5",
    config="{model_info['config']}",
    epochs=50,
    batch_size=32,
    lr=3e-4,
    lambda_reg=0.5,
    action_dim=2,
)
```

## Architecture

Based on the LeWorldModel paper (Maes et al., 2026):
- **Encoder**: Vision Transformer (ViT) with CLS token pooling
- **Predictor**: Transformer with AdaLN-Zero conditioning on actions
- **Loss**: L_pred + lambda * SIGReg(Z)
- **Planner**: Cross-Entropy Method (CEM) in latent space

## Citation

If you use this model, please cite WorldKit and the LeWorldModel paper:

```bibtex
@software{{worldkit,
  title = {{WorldKit: The Open-Source World Model Runtime}},
  author = {{Bansi, Dilpreet}},
  year = {{2026}},
  url = {{https://github.com/DilpreetBansi/worldkit}}
}}
```

## License

MIT License. See [WorldKit LICENSE](https://github.com/DilpreetBansi/worldkit/blob/main/LICENSE).

---

Built with [WorldKit](https://github.com/DilpreetBansi/worldkit) by [Dilpreet Bansi](https://github.com/DilpreetBansi).
"""


def main():
    root = Path(__file__).parent.parent

    for model_info in MODELS:
        repo_id = f"{username}/{model_info['repo_name']}"
        local_path = root / model_info["local_path"]

        if not local_path.exists():
            print(f"SKIP: {local_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Pushing: {repo_id}")
        print(f"{'='*50}")

        # Create repo
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"  Repo created/exists: {repo_id}")
        except Exception as e:
            print(f"  Repo creation note: {e}")

        # Write model card
        card_path = root / "tmp_model_card.md"
        card_path.write_text(make_model_card(model_info))

        # Upload model file
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="model.wk",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded: model.wk ({model_info['size_mb']:.1f} MB)")

        # Upload model card
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded: README.md (model card)")

        card_path.unlink()

        print(f"  DONE: https://huggingface.co/{repo_id}")

    # Also create the "worldkit/pusht" alias pointing to base
    print(f"\n{'='*50}")
    print(f"Creating alias: {username}/pusht -> base model")
    print(f"{'='*50}")

    alias_repo = f"{username}/pusht"
    try:
        create_repo(alias_repo, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    base_path = root / "checkpoints" / "pusht_base_real.wk"
    api.upload_file(
        path_or_fileobj=str(base_path),
        path_in_repo="model.wk",
        repo_id=alias_repo,
        repo_type="model",
    )

    alias_card = f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - robotics
  - push-t
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / Push-T (Default)

The default Push-T world model for [WorldKit](https://github.com/DilpreetBansi/worldkit).
This is the `base` config (13M params, 192-D latent space).

```python
from worldkit import WorldModel

model = WorldModel.from_hub("{username}/pusht")
plan = model.plan(current_frame, goal_frame)
```

See [{username}/pusht-base](https://huggingface.co/{username}/pusht-base) for full details.

Built by [Dilpreet Bansi](https://github.com/DilpreetBansi).
"""

    alias_card_path = root / "tmp_alias_card.md"
    alias_card_path.write_text(alias_card)
    api.upload_file(
        path_or_fileobj=str(alias_card_path),
        path_in_repo="README.md",
        repo_id=alias_repo,
        repo_type="model",
    )
    alias_card_path.unlink()
    print(f"  DONE: https://huggingface.co/{alias_repo}")

    print(f"\n{'='*50}")
    print("ALL MODELS PUSHED SUCCESSFULLY!")
    print(f"  https://huggingface.co/{username}/pusht")
    print(f"  https://huggingface.co/{username}/pusht-nano")
    print(f"  https://huggingface.co/{username}/pusht-base")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
