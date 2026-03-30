# Contributing

Contributions to WorldKit are welcome. This guide covers how to set up a development environment, run tests, and submit changes.

## Development setup

```bash
# Clone the repo
git clone https://github.com/worldkit-ai/worldkit.git
cd worldkit

# Install in development mode with all extras
pip install -e ".[dev]"

# Verify
python -c "from worldkit import WorldModel; print('OK')"
```

## Running tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Specific test
pytest tests/test_model.py::test_encode -v

# With coverage
pytest tests/ -v --cov=worldkit
```

Tests use the `"nano"` config for speed (3.5M params, runs on CPU in seconds). Test shapes and behavior, not exact values (neural net outputs are non-deterministic).

## Linting and formatting

```bash
# Check for issues
ruff check worldkit/ tests/

# Auto-fix
ruff check worldkit/ tests/ --fix

# Format
ruff format worldkit/ tests/
```

## Code style

- **Type hints** on all public methods
- **Docstrings** on all public methods (Google style)
- **Max line length**: 100 characters
- **Import order**: stdlib, third-party, local (handled by Ruff)
- Use `from __future__ import annotations` for union syntax (`X | Y`)

### Example

```python
from __future__ import annotations

import numpy as np
import torch

from worldkit.core.config import ModelConfig


def encode_observation(
    observation: np.ndarray,
    config: ModelConfig,
) -> torch.Tensor:
    """Encode a raw observation into a latent vector.

    Args:
        observation: Image as (H, W, C), uint8 or float32.
        config: Model configuration.

    Returns:
        Latent tensor of shape (latent_dim,).
    """
    ...
```

## Project structure

```
worldkit/
├── core/           # WorldModel, encoder, predictor, planner, losses, config
├── data/           # HDF5 loading, recording, conversion
├── envs/           # Environment wrappers and registry
├── eval/           # Probing, visualization, comparison
├── export/         # ONNX, TorchScript, TensorRT, CoreML
├── hub/            # Hugging Face Hub integration
├── server/         # FastAPI inference server
├── cli/            # Click CLI
└── bench/          # Benchmark suite
```

The `WorldModel` class in `core/model.py` is the public API. Everything else is internal.

## Submitting changes

### Branch naming

```
feat/F-XXX-short-description    # New feature
fix/short-description           # Bug fix
docs/short-description          # Documentation
test/short-description          # Tests
refactor/short-description      # Refactoring
```

### Commit messages

```
feat(core): add hierarchical planner
fix(data): handle missing action key in HDF5
docs(api): expand WorldModel reference
test(eval): add probing test with Ridge
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`.

### Before submitting

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check worldkit/ tests/

# Verify imports
python -c "from worldkit import WorldModel"
```

### Pull request guidelines

- One feature per PR
- Keep PRs small and reviewable
- Include tests for new features
- Update docstrings for changed APIs
- Reference the feature ID if applicable (e.g., "Implements F-019")

## Adding a new feature

1. Read the feature spec in `WORLDKIT_PLATFORM_FEATURES.md`
2. Check dependencies — some features build on others
3. Create a branch: `git checkout -b feat/F-XXX-description`
4. Implement with tests
5. Run `pytest tests/ -v && ruff check worldkit/`
6. Submit a PR

## Adding a test

Tests go in `tests/`. Use the `nano` config for speed:

```python
import numpy as np
import pytest

from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA


@pytest.fixture
def model():
    config = get_config("nano", action_dim=2)
    jepa = JEPA.from_config(config)
    return WorldModel(jepa, config, device="cpu")


def test_encode_shape(model):
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    z = model.encode(obs)
    assert z.shape == (model.latent_dim,)


def test_predict_shape(model):
    obs = np.random.rand(96, 96, 3).astype(np.float32)
    actions = [np.array([0.1, 0.2])] * 5
    result = model.predict(obs, actions)
    assert result.latent_trajectory.shape == (5, model.latent_dim)
    assert result.steps == 5
```

## Adding a dependency

Add to the correct optional group in `pyproject.toml`:

| Group | For |
|-------|-----|
| Core (no group) | Required by all users |
| `train` | Training features (logging, evaluation) |
| `envs` | Gymnasium environment support |
| `serve` | FastAPI server |
| `export` | ONNX/TorchScript export |
| `tensorrt` | TensorRT optimization |
| `coreml` | CoreML export |
| `dev` | Development and testing |

## Reporting issues

File issues at [github.com/worldkit-ai/worldkit/issues](https://github.com/worldkit-ai/worldkit/issues) with:
- WorldKit version (`worldkit --version`)
- Python version
- OS and device (CPU/GPU)
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).
