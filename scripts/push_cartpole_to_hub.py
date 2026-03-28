"""Push trained CartPole WorldKit models to Hugging Face Hub."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN", "")
USERNAME = "DilpreetBansi"

api = HfApi(token=TOKEN)

MODELS = [
    {
        "local_path": "/Users/dilpreetbansi/WorldKit/models/cartpole_base.wk",
        "repo_name": "cartpole-base",
        "config": "base",
        "params": "13M",
        "latent_dim": 192,
        "val_loss": 0.2958,
    },
    {
        "local_path": "/Users/dilpreetbansi/WorldKit/models/cartpole_nano.wk",
        "repo_name": "cartpole-nano",
        "config": "nano",
        "params": "3.5M",
        "latent_dim": 128,
        "val_loss": 0.2417,
    },
]


def make_model_card(model_info: dict) -> str:
    return f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - cartpole
  - reinforcement-learning
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / {model_info['repo_name']}

A **{model_info['config']}** CartPole-v1 world model trained with [WorldKit](https://github.com/DilpreetBansi/worldkit).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | JEPA (Joint-Embedding Predictive Architecture) |
| Config | `{model_info['config']}` |
| Parameters | {model_info['params']} |
| Latent Dim | {model_info['latent_dim']} |
| Task | CartPole balance control |
| Training | 100 epochs on 200 episodes of pixel observations |
| Best Val Loss | {model_info['val_loss']:.4f} |

## Usage

```bash
pip install worldkit
```

```python
from worldkit import WorldModel

# Load this model
model = WorldModel.from_hub("{USERNAME}/{model_info['repo_name']}")

# Encode an observation
z = model.encode(observation)  # -> ({model_info['latent_dim']},) latent vector

# Predict future states
result = model.predict(current_frame, actions)

# Plan to reach a goal
plan = model.plan(current_frame, goal_frame, max_steps=50)

# Score physical plausibility
score = model.plausibility(video_frames)
```

## Task: CartPole-v1

The CartPole-v1 environment requires an agent to balance a pole on a cart by applying left/right forces. The world model learns to predict future visual observations from pixel inputs, enabling planning and control in latent space.

## Training

Trained using WorldKit's built-in training pipeline on 200 episodes of pixel observations for 100 epochs:

```python
from worldkit import WorldModel

model = WorldModel.train(
    data="cartpole_train.h5",
    config="{model_info['config']}",
    epochs=100,
    batch_size=32,
    lr=3e-4,
    lambda_reg=0.5,
)
```

## Architecture

Based on the LeWorldModel paper (Maes et al., 2026):
- **Encoder**: Vision Transformer (ViT) with CLS token pooling
- **Predictor**: Transformer with AdaLN-Zero conditioning on actions
- **Loss**: L_pred + lambda * SIGReg(Z)
- **Planner**: Cross-Entropy Method (CEM) in latent space

## Links

- **PyPI**: [pypi.org/project/worldkit](https://pypi.org/project/worldkit/)
- **GitHub**: [github.com/DilpreetBansi/worldkit](https://github.com/DilpreetBansi/worldkit)

## Citation

If you use this model, please cite WorldKit:

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

Built with [WorldKit](https://github.com/DilpreetBansi/worldkit) | [PyPI](https://pypi.org/project/worldkit/) | [GitHub](https://github.com/DilpreetBansi/worldkit)
"""


def main():
    for model_info in MODELS:
        repo_id = f"{USERNAME}/{model_info['repo_name']}"
        local_path = Path(model_info["local_path"])

        if not local_path.exists():
            print(f"SKIP: {local_path} not found")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)

        print(f"\n{'='*50}")
        print(f"Pushing: {repo_id}")
        print(f"{'='*50}")

        # Create repo
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True, token=TOKEN)
            print(f"  Repo created/exists: {repo_id}")
        except Exception as e:
            print(f"  Repo creation note: {e}")

        # Upload model file
        print(f"  Uploading model.wk ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="model.wk",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Uploaded: model.wk")

        # Write and upload model card
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(make_model_card(model_info))
            card_path = Path(f.name)

        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        card_path.unlink()
        print(f"  Uploaded: README.md (model card)")

        print(f"  DONE: https://huggingface.co/{repo_id}")

    print(f"\n{'='*50}")
    print("ALL CARTPOLE MODELS PUSHED SUCCESSFULLY!")
    print(f"  https://huggingface.co/{USERNAME}/cartpole-base")
    print(f"  https://huggingface.co/{USERNAME}/cartpole-nano")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
