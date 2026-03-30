# CLAUDE.md — WorldKit Development Rules

> Drop this file at the root of your worldkit/ project. Claude Code reads it every session and follows these rules. After every mistake, add a new rule so it never happens again.

---

## Project Context

WorldKit is an open-source Python SDK for training and deploying lightweight world models. Built on the LeWorldModel (LeWM) architecture — 15M parameters, 1 hyperparameter, single GPU, MIT licensed.

- **Repo**: worldkit/
- **Package**: worldkit (on PyPI)
- **Python**: 3.10+
- **Framework**: PyTorch
- **Config**: Hydra-style dataclasses (not YAML files)
- **Tests**: pytest
- **Linting**: Ruff
- **CLI**: Click
- **Server**: FastAPI
- **Hub**: Hugging Face Hub

---

## Architecture — Know This Before Touching Anything

```
worldkit/
├── core/          # THE engine. Encoder, Predictor, JEPA, Planner, Losses, WorldModel class
├── data/          # HDF5 loading, Gym recording, video conversion
├── envs/          # Gymnasium and dm_control wrappers
├── eval/          # Benchmarks, probing, plausibility, visualization
├── export/        # ONNX, TensorRT, CoreML, TorchScript, WASM
├── hub/           # Hugging Face Hub upload/download/registry
├── server/        # FastAPI inference server
└── cli/           # Click CLI entry point
```

**The WorldModel class in core/model.py is the ONLY class developers interact with.** Everything else is internal. Never expose internal classes in public docs or examples unless there's a reason.

**The JEPA class in core/jepa.py combines encoder + predictor + action_encoder.** It has five key methods: encode, predict, rollout, criterion, get_cost. WorldModel wraps JEPA.

**SIGReg in core/losses.py is the critical loss function.** It prevents representation collapse with one hyperparameter (λ). If you break this, models stop learning. Test any changes to losses.py extensively.

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Reference the feature spec: read WORLDKIT_PLATFORM_FEATURES.md for feature requirements before implementing

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- Run tests in a subagent while building in the main session

### 3. Self-Improvement Loop
- After ANY correction from the user: update this CLAUDE.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review this file at session start

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Run `pytest tests/ -v` after every change to core/
- Run `ruff check worldkit/` after every change
- Run `python -c "from worldkit import WorldModel"` to verify imports
- Ask yourself: "Would a staff engineer approve this?"

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky, implement the elegant solution
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to tasks/todo.md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to tasks/todo.md
6. **Capture Lessons**: Update this CLAUDE.md after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Code Standards

### Python Style
- Use type hints everywhere: `def encode(self, observation: np.ndarray) -> torch.Tensor`
- Docstrings on every public method (Google style)
- Max line length: 100 (configured in pyproject.toml via Ruff)
- Use `from __future__ import annotations` for Python 3.10 union syntax (`X | Y`)
- Import order: stdlib → third-party → local (Ruff handles this)

### PyTorch Conventions
- Always use `@torch.no_grad()` for inference methods (predict, plan, plausibility, encode)
- Use `model.eval()` before inference, `model.train()` before training
- Tensor shapes always documented in comments: `# (B, T, D)`
- Use `torch.nn.functional` as `F`
- Clamp gradients with `torch.nn.utils.clip_grad_norm_` during training
- Use `.detach()` on target embeddings in the loss function — never backprop through targets

### WorldKit-Specific Rules
- The WorldModel class is the public API. Keep it clean. No PyTorch leaking out.
- Users pass numpy arrays in, get numpy arrays or dataclasses out. Tensors are internal.
- All pixel inputs should handle both (H, W, C) and (C, H, W) formats — detect and convert
- All pixel inputs should handle both [0, 255] and [0, 1] ranges — detect and normalize
- .wk files use torch.save() with a structured dict. Always include worldkit_version.
- Action spaces can be continuous (float vectors) or discrete (integers). Handle both.
- Config names are strings: "nano", "base", "large", "xl". Never use raw ModelConfig in public API.

### Testing Rules
- Every new feature gets a test in tests/
- Tests use the "nano" config for speed (5M params, trains in seconds on CPU)
- Use `tmp_path` fixture for file I/O tests (pytest built-in)
- Test with random data — don't depend on real datasets in unit tests
- Test shapes, not values (neural net outputs are non-deterministic)

### Error Handling
- Helpful error messages that tell the user what to do:
  - BAD: `KeyError: 'pixels'`
  - GOOD: `KeyError: No pixel data found in HDF5 file. Expected one of: 'pixels', 'observations', 'obs'. Found keys: ['images', 'actions']`
- Validate inputs at the boundary (WorldModel methods), not deep inside the engine
- Fail fast with clear messages, don't silently produce garbage

---

## Feature Implementation Guide

When building a new feature from WORLDKIT_PLATFORM_FEATURES.md:

1. **Read the feature spec** — find the feature by ID (F-001 through F-051) and read the full description
2. **Check dependencies** — some features depend on others (e.g., F-033 Playground needs F-029 WASM export)
3. **Plan the implementation** — enter plan mode, write steps
4. **Build it** — follow code standards above
5. **Test it** — add tests, run full suite
6. **Document it** — update docstrings, add to docs/ if public-facing
7. **Verify it** — does it actually work end-to-end? Not just "does it not crash?"

### Feature Priority Order
Build P0 features first (F-001 through F-017), then P1, then P2. Never skip ahead.

---

## File Reference

These files contain the complete specs. Read them before building.

| File | What It Contains |
|------|-----------------|
| `WORLDKIT_FULL_BLUEPRINT.md` | Product vision, market analysis, competitive positioning, launch content, investor pitch |
| `WORLDKIT_CLAUDE_CODE_BUILD_SPEC.md` | Every file, every class, every function. The complete implementation spec. |
| `WORLDKIT_PLATFORM_FEATURES.md` | 51 features across 8 layers. What to build and in what order. |
| `CLAUDE.md` | This file. Development rules. Updated after every mistake. |

---

## Key Dependencies — Exact Versions

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
h5py>=3.8
einops>=0.7
huggingface-hub>=0.20
safetensors>=0.4
click>=8.1          # CLI
fastapi>=0.110      # Server
gradio>=4.0         # Demo
ruff>=0.3           # Linting
pytest>=7.4         # Testing
```

When adding a new dependency: add it to the correct optional group in pyproject.toml ([train], [envs], [serve], [export], [dev]).

---

## Common Mistakes — Lessons Learned

> This section grows over time. After every correction, add a new rule here.

### Lesson 1: Tensor Shape Mismatches
SIGReg expects (T, B, D). The predictor outputs (B, T, D). Always check and transpose when passing between modules. Add shape comments on every tensor operation.

### Lesson 2: Forgetting .detach() on Targets
The prediction loss must use `target.detach()`. If you backprop through both predicted and target embeddings, the encoder has a shortcut to minimize loss by collapsing representations. SIGReg prevents this, but detaching targets is still required for stable training.

### Lesson 3: Pixel Normalization
Users will pass pixels as uint8 [0, 255], float32 [0, 1], or float32 [-1, 1]. The encoder must handle all three. Check `tensor.max() > 1.0` and normalize accordingly. Never assume the input format.

### Lesson 4: HDF5 Key Names
Different datasets use different key names: "pixels", "observations", "obs", "images" for frames and "actions", "action" for actions. Always check multiple key names and provide a helpful error listing the actual keys found.

### Lesson 5: CEM Planner Batch Dimension
The CEM planner adds a candidate dimension (S) to tensors, making them (B, S, T, D). When reshaping for the predictor (which expects (B, T, D)), you must reshape to (B*S, T, D) and reshape back after. Off-by-one on dimensions here causes silent bugs — the model runs but produces garbage plans.

### Lesson 6: Context Length vs Sequence Length
context_length (how many past frames the predictor sees) is NOT the same as sequence_length (total frames in a training sample). sequence_length = context_length + prediction_length. Mixing these up causes the training loop to pass wrong tensor slices.

### Lesson 7: Action Encoder Input Shape
Continuous actions are (B, T, action_dim) — 3D. Discrete actions are (B, T) — 2D. The ActionEncoder must handle both. Check `.dim()` before processing.

### Lesson 8: Model.eval() Before Inference
ALWAYS call `self._jepa.eval()` at the start of predict(), plan(), plausibility(), and encode(). Forgetting this means BatchNorm and Dropout layers behave differently, producing inconsistent results between training and inference.

---

## Git Conventions

- Branch naming: `feat/F-XXX-short-description` (e.g., `feat/F-019-worldkit-bench`)
- Commit messages: `feat(core): add CEM planner with configurable candidates`
- Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`
- One feature per PR. Keep PRs small and reviewable.
- Always run `ruff check worldkit/ && pytest tests/ -v` before committing.

---

## How to Run Common Tasks

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py::test_encode -v

# Lint
ruff check worldkit/ tests/

# Format
ruff format worldkit/ tests/

# Type check (optional)
mypy worldkit/ --ignore-missing-imports

# Build package
python -m build

# Train a model (CLI)
worldkit train --data ./data.h5 --config base --epochs 100

# Serve a model (CLI)
worldkit serve --model ./model.wk --port 8000

# Verify everything works
python -c "
from worldkit import WorldModel
from worldkit.core.config import get_config
from worldkit.core.jepa import JEPA
config = get_config('nano', action_dim=2)
jepa = JEPA.from_config(config)
model = WorldModel(jepa, config, device='cpu')
import numpy as np
obs = np.random.rand(96, 96, 3).astype(np.float32)
z = model.encode(obs)
print(f'OK: {model.num_params:,} params, latent={z.shape}')
"
```

---

*This file is alive. It grows smarter every session. Every bug that gets fixed becomes a permanent rule. Every pattern that works gets documented. The longer you use Claude Code with this file, the better it gets at building WorldKit.*
