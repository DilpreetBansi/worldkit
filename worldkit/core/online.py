"""Online learning for WorldKit world models.

Provides incremental weight updates from streaming experience using a
circular replay buffer and periodic mini-batch gradient steps.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class OnlineLearner:
    """Incremental online learner that attaches to a WorldModel.

    Maintains a circular replay buffer of (obs, action, next_obs) tuples.
    Every ``update_every`` calls to :meth:`step`, samples a mini-batch from
    the buffer and performs one gradient step on the underlying model.

    Optionally maintains an exponential moving average (EMA) of model
    weights for more stable inference.

    Args:
        model: The WorldModel instance to update.
        lr: Learning rate for online updates.
        buffer_size: Maximum replay buffer capacity.
        batch_size: Mini-batch size sampled from the buffer.
        update_every: Perform a gradient step every N calls to step().
        ema_decay: EMA decay rate for weight averaging (0 = disabled).
    """

    def __init__(
        self,
        model: object,
        lr: float = 1e-5,
        buffer_size: int = 1000,
        batch_size: int = 16,
        update_every: int = 4,
        ema_decay: float = 0.0,
    ):
        self._model = model
        self._lr = lr
        self._buffer: deque = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self._update_every = update_every
        self._ema_decay = ema_decay
        self._step_count = 0

        self._optimizer = torch.optim.AdamW(
            model._model.parameters(), lr=lr, weight_decay=0.01
        )

        # Store EMA weights if enabled
        if ema_decay > 0:
            self._ema_state: dict[str, torch.Tensor] = {
                k: v.clone() for k, v in model._model.state_dict().items()
            }
        else:
            self._ema_state = {}

    @property
    def buffer_size(self) -> int:
        """Current number of transitions in the buffer."""
        return len(self._buffer)

    def step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> float | None:
        """Add a transition and optionally perform a gradient step.

        Args:
            observation: Current observation (H, W, C) or (C, H, W).
            action: Action taken, shape (action_dim,).
            next_observation: Resulting observation.

        Returns:
            Loss value if a gradient step was performed, else None.
        """
        self._buffer.append((observation, action, next_observation))
        self._step_count += 1

        if (
            self._step_count % self._update_every == 0
            and len(self._buffer) >= self._batch_size
        ):
            return self._gradient_step()
        return None

    def _gradient_step(self) -> float:
        """Sample a mini-batch from the buffer and do one gradient update."""
        model = self._model
        device = model._device

        # Sample random indices from buffer
        indices = np.random.choice(len(self._buffer), self._batch_size, replace=False)
        obs_list, act_list, next_obs_list = [], [], []
        for i in indices:
            obs, act, next_obs = self._buffer[i]
            obs_list.append(obs)
            act_list.append(act)
            next_obs_list.append(next_obs)

        # Prepare observations: stack and normalize
        obs_batch = self._prepare_batch(obs_list, device)  # (B, C, H, W)
        next_obs_batch = self._prepare_batch(next_obs_list, device)  # (B, C, H, W)

        # Prepare actions: (B, 1, action_dim)
        actions = torch.tensor(
            np.array(act_list), dtype=torch.float32, device=device
        )
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        actions = actions.unsqueeze(1)  # (B, 1, action_dim)

        # Forward pass
        model._model.train()

        # Encode current and next observations
        ctx_emb = model._backend.encode(
            model._model, obs_batch.unsqueeze(1)
        )  # (B, 1, D)
        target_emb = model._backend.encode(
            model._model, next_obs_batch.unsqueeze(1)
        ).detach()  # (B, 1, D)

        # Predict next state
        act_emb = model._model.encode_actions(actions)  # (B, 1, D)
        pred_emb = model._model.predict(ctx_emb, act_emb)  # (B, T_pred, D)

        # Loss: MSE between predicted first step and target
        loss = torch.nn.functional.mse_loss(pred_emb[:, :1], target_emb)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model._model.parameters(), max_norm=1.0)
        self._optimizer.step()

        # EMA update
        if self._ema_decay > 0:
            with torch.no_grad():
                for name, param in model._model.named_parameters():
                    if name in self._ema_state:
                        self._ema_state[name].mul_(self._ema_decay).add_(
                            param.data, alpha=1.0 - self._ema_decay
                        )

        return loss.item()

    def apply_ema(self) -> None:
        """Replace model weights with the EMA weights.

        Only has effect if ema_decay > 0 was set at construction time.
        """
        if not self._ema_state:
            return
        state = self._model._model.state_dict()
        for k in self._ema_state:
            if k in state:
                state[k] = self._ema_state[k]
        self._model._model.load_state_dict(state)

    def _prepare_batch(
        self, obs_list: list[np.ndarray], device: str
    ) -> torch.Tensor:
        """Stack observations and normalize to (B, C, H, W) in [0, 1]."""
        batch = torch.tensor(
            np.array(obs_list), dtype=torch.float32, device=device
        )
        if batch.max() > 1.0:
            batch = batch / 255.0
        # (B, H, W, C) -> (B, C, H, W)
        if batch.dim() == 4 and batch.shape[-1] in (1, 3):
            batch = batch.permute(0, 3, 1, 2)
        return batch
