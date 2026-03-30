# Architecture

This guide explains the technical architecture of WorldKit's world model — how observations are encoded, how predictions are made, and how the model is trained.

## Overview

WorldKit implements a [JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) (Joint-Embedding Predictive Architecture) trained with [SIGReg](https://le-wm.github.io/) regularization.

```
Observation (96x96 RGB)
        |
        v
┌───────────────┐
│   ViT Encoder │ ── CLS token pooling ──> z ∈ R^D (latent state)
└───────────────┘
        |
        v
┌───────────────────────┐
│ Predictor (AdaLN-Zero)│ ── conditioned on action embeddings
│   Transformer         │ ── causal attention
└───────────────────────┘
        |
        v
   z' ∈ R^D (predicted next state)
```

## Components

### ViT Encoder

The encoder is a Vision Transformer (ViT) that compresses 96x96 RGB images into compact latent vectors.

- **Input**: `(B, C, H, W)` pixel tensor, normalized to `[0, 1]`
- **Process**: Patch embedding (16x16 patches) → Transformer layers → CLS token pooling → Linear projection
- **Output**: `(B, latent_dim)` latent vector

The encoder uses pretrained ViT backbones from Hugging Face:

| Config | Backbone | Embedding dim | Latent dim |
|--------|----------|--------------|------------|
| nano | ViT-Tiny | 192 | 128 |
| base | ViT-Small | 384 | 192 |
| large | ViT-Base | 768 | 384 |
| xl | ViT-Large | 1024 | 512 |

CLS token pooling produces a single vector per image — ~200x more compact than keeping all patch tokens. This makes prediction and planning fast.

### Action Encoder

The action encoder maps raw actions into the same embedding space as latent states.

- **Continuous actions**: `(B, T, action_dim)` → Linear → `(B, T, latent_dim)`
- **Discrete actions**: `(B, T)` → Embedding table → `(B, T, latent_dim)`

The encoder auto-detects whether actions are continuous (3D tensor) or discrete (2D tensor).

### Autoregressive Predictor

The predictor is a Transformer with AdaLN-Zero (Adaptive Layer Normalization with Zero initialization) conditioning.

- **Input**: Latent state embeddings `(B, T, D)` and action embeddings `(B, T, D)`
- **Architecture**: Transformer with causal masking, `pred_depth` layers, `pred_heads` attention heads
- **Conditioning**: Actions modulate the predictor via AdaLN-Zero — each layer produces 6 modulation parameters (shift/scale/gate for attention and MLP)
- **Output**: Predicted next-state latent `(B, T, D)`

The causal mask ensures predictions at time `t` only depend on states and actions at times `≤ t`.

### CEM Planner

The Cross-Entropy Method (CEM) planner searches for optimal action sequences in latent space.

**Algorithm:**

```
1. Initialize: sample N random action sequences
2. For each iteration:
   a. Roll out each sequence through the predictor → predicted latent trajectory
   b. Score each sequence: cost = MSE(final_predicted_latent, goal_latent)
   c. Select the K lowest-cost sequences (elite set)
   d. Update sampling distribution: mean, std ← fit to elite set
   e. Sample N new sequences from the updated distribution
3. Return the best sequence found
```

Planning happens entirely in latent space — no rendering, no physics engine. This makes it fast: ~150ms for the base config with 200 candidates and 5 iterations.

### Hierarchical Planner

For long-horizon tasks, the hierarchical planner:

1. Encodes current and goal into latent vectors
2. Interpolates `K` intermediate subgoal latents between them
3. Runs CEM between each consecutive pair of subgoals
4. Concatenates all action sequences

## Training

### Loss function

WorldKit uses the LeWM loss: a prediction loss plus SIGReg regularization.

```
L = L_pred + λ · SIGReg(Z)
```

**Prediction loss** (`L_pred`): MSE between predicted and actual latent states. The target embeddings are `detached()` — gradients do not flow through them. This prevents the encoder from collapsing to a trivial mapping.

**SIGReg loss**: Sketch Isotropic Gaussian Regularizer. Forces the latent distribution to be isotropic Gaussian, preventing representation collapse. Uses random projections and a trigonometric basis to approximate the KL divergence between the empirical latent distribution and a unit Gaussian.

**λ**: The single hyperparameter. Default 1.0. Controls the balance between prediction accuracy and latent structure.

### Why SIGReg?

Without regularization, the encoder can learn a "shortcut" — map all inputs to the same output. The prediction loss is trivially zero, but the representation is useless.

Prior methods solve this with:
- **VICReg**: 3 loss terms (variance, invariance, covariance) with 6+ hyperparameters
- **Barlow Twins**: Cross-correlation matrix with multiple trade-off weights
- **BYOL**: Momentum encoder, EMA updates, asymmetric architecture

SIGReg replaces all of this with a single regularization term and one hyperparameter (λ).

### Training loop

```
For each epoch:
    For each batch of (pixels, actions):
        1. Encode pixels → embeddings (B, T, D)
        2. Encode actions → action embeddings (B, T, D)
        3. Split into context (t < context_length) and target (t ≥ context_length)
        4. Predict next states from context embeddings + action embeddings
        5. Compute L_pred = MSE(predicted, target.detach())
        6. Compute L_sigreg = SIGReg(all_embeddings)
        7. L = L_pred + λ · L_sigreg
        8. Backpropagate, clip gradients, optimizer step
```

### Key invariants

- **`target.detach()`**: Always detach target embeddings to prevent collapse shortcuts.
- **`model.eval()` before inference**: BatchNorm and Dropout behave differently in train/eval mode.
- **Tensor shapes**: SIGReg expects `(T, B, D)`, predictor outputs `(B, T, D)`. Transpose when crossing module boundaries.
- **`context_length` vs `sequence_length`**: `sequence_length = context_length + prediction_length`. The predictor sees `context_length` past frames and predicts the remaining frames.

## Backend system

WorldKit uses a pluggable backend architecture. The `WorldModel` class delegates architecture-specific logic to a `BaseWorldModelBackend` subclass.

```python
from worldkit.core.backends import BackendRegistry, BaseWorldModelBackend

# Current backend
class LeWMBackend(BaseWorldModelBackend):
    """JEPA + SIGReg (default)"""
    def build(self, config): ...
    def encode(self, model, pixels): ...
    def predict(self, model, state, actions): ...
    def rollout(self, model, pixels, actions, action_sequence, context_length): ...
    def get_cost(self, model, pixels, actions, goal, candidates, context_length): ...
    def training_step(self, model, batch, config): ...
```

This enables future backends (Ha & Schmidhuber 2018, Dreamer, TD-MPC2, DIAMOND) to plug in with the same `WorldModel` API.

## Inference flow

```python
# Encoding
obs → normalize → resize to (96, 96) → to tensor → ViT encoder → projection → z

# Prediction
z, actions → action encoder → predictor (autoregressive) → [z₁, z₂, ..., zₜ]

# Planning
current, goal → encode both → CEM loop:
    sample candidates → rollout → score → select elite → refit → repeat
    → best action sequence

# Plausibility
frames → encode each → predict consecutive → MSE errors → aggregate → score
```

## Related

- [Config reference](../api/config.md) — model configuration details
- [Training tutorial](../tutorials/train_first_model.md) — train a model
- [Planning API](../api/planning.md) — CEM planner details
- [LeWM paper](https://le-wm.github.io/) — original research paper
