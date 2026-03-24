"""Update model cards on Hugging Face with corrected technical language."""
from __future__ import annotations

from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()["name"]

NANO_CARD = f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - sigreg
  - robotics
  - push-t
  - planning
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / pusht-nano

A **nano** world model trained on the Push-T manipulation task using [WorldKit](https://github.com/DilpreetBansi/worldkit).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | JEPA pattern with SIGReg training |
| Config | `nano` |
| Parameters | 3.5M |
| Latent Dim | 128 |
| Image Size | 96x96 |
| Action Dim | 2 (dx, dy) |
| File Size | 13.4 MB |
| Training Time | ~30 seconds (Apple M4 Pro, MPS) |
| Best Val Loss | 0.5114 |

## Usage

```bash
pip install worldkit
```

```python
from worldkit import WorldModel

model = WorldModel.from_hub("{username}/pusht-nano")

# Predict future states
result = model.predict(current_frame, actions)

# Plan to reach a goal
plan = model.plan(current_frame, goal_frame, max_steps=50)

# Detect anomalies
score = model.plausibility(video_frames)
```

## Architecture

Uses the [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) (Joint-Embedding Predictive Architecture) pattern — prediction happens in latent space, not pixel space. Trained with [SIGReg](https://le-wm.github.io/) regularization to prevent representation collapse using a single hyperparameter.

- **Encoder**: ViT-Tiny with CLS token pooling
- **Predictor**: 2-layer Transformer with AdaLN-Zero conditioning
- **Loss**: L_pred + lambda * SIGReg(Z)
- **Planner**: Cross-Entropy Method (CEM) in latent space

Based on: LeWorldModel (Maes et al., 2026) | [Paper](https://le-wm.github.io/) | [Code](https://github.com/lucas-maes/le-wm)

## Citation

```bibtex
@software{{worldkit,
  title = {{WorldKit: The Open-Source World Model SDK}},
  author = {{Bansi, Dilpreet}},
  year = {{2026}},
  url = {{https://github.com/DilpreetBansi/worldkit}}
}}
```

## License

MIT License. See [WorldKit repo](https://github.com/DilpreetBansi/worldkit) for details.

---

Built with [WorldKit](https://github.com/DilpreetBansi/worldkit) by [Dilpreet Bansi](https://github.com/DilpreetBansi).
"""

BASE_CARD = f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - sigreg
  - robotics
  - push-t
  - planning
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / pusht-base

The **base** (default) world model trained on the Push-T manipulation task using [WorldKit](https://github.com/DilpreetBansi/worldkit). This is the recommended starting model.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | JEPA pattern with SIGReg training |
| Config | `base` |
| Parameters | 13M |
| Latent Dim | 192 |
| Image Size | 96x96 |
| Action Dim | 2 (dx, dy) |
| File Size | 50.2 MB |
| Training Time | ~60 seconds (Apple M4 Pro, MPS) |
| Best Val Loss | 0.3862 |

## Usage

```bash
pip install worldkit
```

```python
from worldkit import WorldModel

model = WorldModel.from_hub("{username}/pusht-base")

# Predict future states
result = model.predict(current_frame, actions)

# Plan to reach a goal
plan = model.plan(current_frame, goal_frame, max_steps=50)

# Detect anomalies
score = model.plausibility(video_frames)
```

## Architecture

Uses the [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) (Joint-Embedding Predictive Architecture) pattern — prediction happens in latent space, not pixel space. Trained with [SIGReg](https://le-wm.github.io/) regularization to prevent representation collapse using a single hyperparameter.

- **Encoder**: ViT-Small with CLS token pooling
- **Predictor**: 3-layer Transformer with AdaLN-Zero conditioning
- **Loss**: L_pred + lambda * SIGReg(Z)
- **Planner**: Cross-Entropy Method (CEM) in latent space

Based on: LeWorldModel (Maes et al., 2026) | [Paper](https://le-wm.github.io/) | [Code](https://github.com/lucas-maes/le-wm)

## Citation

```bibtex
@software{{worldkit,
  title = {{WorldKit: The Open-Source World Model SDK}},
  author = {{Bansi, Dilpreet}},
  year = {{2026}},
  url = {{https://github.com/DilpreetBansi/worldkit}}
}}
```

## License

MIT License. See [WorldKit repo](https://github.com/DilpreetBansi/worldkit) for details.

---

Built with [WorldKit](https://github.com/DilpreetBansi/worldkit) by [Dilpreet Bansi](https://github.com/DilpreetBansi).
"""

ALIAS_CARD = f"""---
license: mit
tags:
  - worldkit
  - world-model
  - jepa
  - sigreg
  - robotics
  - push-t
library_name: worldkit
pipeline_tag: reinforcement-learning
---

# WorldKit / Push-T (Default)

The default Push-T world model for [WorldKit](https://github.com/DilpreetBansi/worldkit).
Uses the `base` config (13M params, 192-D latent space, SIGReg training).

```python
from worldkit import WorldModel

model = WorldModel.from_hub("{username}/pusht")
plan = model.plan(current_frame, goal_frame)
```

See [{username}/pusht-base](https://huggingface.co/{username}/pusht-base) for full details.

Built by [Dilpreet Bansi](https://github.com/DilpreetBansi).
"""

import tempfile
from pathlib import Path

for repo_name, card_content in [
    ("pusht-nano", NANO_CARD),
    ("pusht-base", BASE_CARD),
    ("pusht", ALIAS_CARD),
]:
    repo_id = f"{username}/{repo_name}"
    tmp = Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text(card_content)
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    tmp.unlink()
    print(f"Updated: https://huggingface.co/{repo_id}")

print("\nAll model cards updated.")
