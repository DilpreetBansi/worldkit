"""Federated training client for WorldKit.

Trains a WorldModel locally on HDF5 data and exchanges weight deltas
with a :class:`FederatedServer` over WebSocket.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from worldkit.core.model import WorldModel

from . import protocol


class FederatedClient:
    """Client for federated training of a WorldModel.

    Args:
        model: A WorldModel instance to train locally.
        server_url: WebSocket URL of the federated server
            (e.g. ``ws://localhost:8080/ws/federated``).
        client_id: Optional human-readable client identifier.
            Auto-generated if not provided.
    """

    def __init__(
        self,
        model: WorldModel,
        server_url: str,
        client_id: str | None = None,
    ) -> None:
        self._model = model
        self._server_url = server_url
        self._client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
        self._initial_weights: dict[str, torch.Tensor] | None = None
        self._ws: object | None = None  # websockets connection

    @property
    def client_id(self) -> str:
        """The unique identifier for this client."""
        return self._client_id

    # ── Weight management ────────────────────────────────

    def _snapshot_weights(self) -> dict[str, torch.Tensor]:
        """Deep copy current model weights to CPU."""
        return {
            k: v.clone().cpu()
            for k, v in self._model._model.state_dict().items()
        }

    def compute_delta(self) -> dict[str, torch.Tensor]:
        """Compute weight delta from the initial weights snapshot.

        Returns:
            Dictionary of parameter deltas (current - initial).
        """
        if self._initial_weights is None:
            raise RuntimeError("No weight snapshot — call _snapshot_weights() first")
        current = self._model._model.state_dict()
        return protocol.compute_delta(self._initial_weights, current)

    def apply_global_delta(self, aggregated_delta: dict[str, torch.Tensor]) -> None:
        """Apply the server's aggregated delta to the initial weights.

        This sets the model weights to ``initial + aggregated_delta``,
        which is the new global model for the next round.

        Args:
            aggregated_delta: Aggregated weight delta from the server.
        """
        if self._initial_weights is None:
            raise RuntimeError("No weight snapshot to apply delta to")
        new_state = protocol.apply_delta(self._initial_weights, aggregated_delta)
        # Move to model device
        device = self._model.device
        new_state = {k: v.to(device) for k, v in new_state.items()}
        self._model._model.load_state_dict(new_state)

    # ── Local training ───────────────────────────────────

    def train_local(
        self,
        data: str | Path,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> tuple[float, int]:
        """Train the model in-place on local HDF5 data.

        Follows the same training loop as
        :meth:`WorldModel.train` but operates on the existing model
        without creating a new one.

        Args:
            data: Path to an HDF5 file with pixel and action data.
            epochs: Number of local training epochs.
            batch_size: Training batch size.
            lr: Learning rate.

        Returns:
            Tuple of (average_loss, num_samples).
        """
        import h5py

        data_path = Path(data)
        config = self._model.config
        device = self._model.device

        # Load HDF5 data — same key resolution as WorldModel.train()
        with h5py.File(data_path, "r") as f:
            for key in ("pixels", "observations", "obs"):
                if key in f:
                    pixels = torch.tensor(
                        np.array(f[key]), dtype=torch.float32,
                    )
                    break
            else:
                keys = list(f.keys())
                raise KeyError(
                    f"Expected 'pixels', 'observations', or 'obs' "
                    f"in HDF5. Found: {keys}"
                )

            for key in ("actions", "action"):
                if key in f:
                    actions = torch.tensor(
                        np.array(f[key]), dtype=torch.float32,
                    )
                    break
            else:
                keys = list(f.keys())
                raise KeyError(
                    f"Expected 'actions' or 'action' in HDF5. Found: {keys}"
                )

        # Normalize pixels to [0, 1]
        if pixels.max() > 1.0:
            pixels = pixels / 255.0

        # Ensure (N, T, C, H, W) shape
        if pixels.dim() == 4:  # (N, H, W, C) — single frame per episode
            pixels = pixels.unsqueeze(1)
        if pixels.shape[-1] in (1, 3):  # (N, T, H, W, C) → (N, T, C, H, W)
            pixels = pixels.permute(0, 1, 4, 2, 3)

        dataset = TensorDataset(pixels, actions)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )

        # Training loop (modeled on WorldModel.train() lines 279–304)
        optimizer = torch.optim.AdamW(
            self._model._model.parameters(),
            lr=lr,
            weight_decay=config.weight_decay,
        )

        self._model._model.train()
        total_loss_sum = 0.0
        total_steps = 0

        for _epoch in range(epochs):
            for batch_pixels, batch_actions in loader:
                batch_pixels = batch_pixels.to(device)  # (B, T, C, H, W)
                batch_actions = batch_actions.to(device)  # (B, T, action_dim)

                optimizer.zero_grad()

                total_loss, _loss_dict = self._model._backend.training_step(
                    self._model._model,
                    (batch_pixels, batch_actions),
                    config,
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    optimizer.zero_grad()
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model._model.parameters(), max_norm=1.0,
                )
                optimizer.step()

                total_loss_sum += total_loss.item()
                total_steps += 1

        avg_loss = total_loss_sum / max(total_steps, 1)
        num_samples = len(dataset)
        return avg_loss, num_samples

    # ── WebSocket communication ──────────────────────────

    async def connect(self) -> None:
        """Connect to the federated server and register."""
        import websockets

        self._ws = await websockets.connect(
            self._server_url, max_size=2**30,
        )
        num_params = sum(
            p.numel() for p in self._model._model.state_dict().values()
        )
        msg = protocol.to_json(protocol.RegisterMessage(
            client_id=self._client_id,
            num_params=num_params,
        ))
        await self._ws.send(json.dumps(msg))
        print(
            f"WorldKit | Federated client {self._client_id!r} "
            f"connected to {self._server_url}"
        )

    async def send_update(
        self,
        delta: dict[str, torch.Tensor],
        num_samples: int,
        train_loss: float,
        round_number: int,
    ) -> None:
        """Send weight delta to the server.

        Args:
            delta: Weight delta from local training.
            num_samples: Number of training samples used.
            train_loss: Average training loss.
            round_number: Current round number.
        """
        delta_b64 = protocol.encode_delta(delta)
        msg = protocol.to_json(protocol.UpdateMessage(
            client_id=self._client_id,
            round_number=round_number,
            delta_b64=delta_b64,
            num_samples=num_samples,
            train_loss=train_loss,
        ))
        await self._ws.send(json.dumps(msg))

    async def receive_global(self) -> dict[str, torch.Tensor] | None:
        """Receive aggregated global weights from the server.

        Returns:
            Aggregated weight delta, or ``None`` if the server signaled
            completion.

        Raises:
            RuntimeError: If the server sends an error message.
        """
        raw = await self._ws.recv()
        data = json.loads(raw)
        msg = protocol.from_json(data)

        if isinstance(msg, protocol.GlobalWeightsMessage):
            return protocol.decode_delta(msg.aggregated_delta_b64)
        elif isinstance(msg, protocol.DoneMessage):
            return None
        elif isinstance(msg, protocol.ErrorMessage):
            raise RuntimeError(f"Server error: {msg.detail}")
        else:
            raise RuntimeError(
                f"Unexpected message type: {data.get('type')}"
            )

    # ── Main loop ────────────────────────────────────────

    async def run(
        self,
        data: str | Path,
        rounds: int = 10,
        epochs_per_round: int = 1,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> None:
        """Run the federated training loop.

        Connects to the server, then for each round: trains locally,
        sends the weight delta, and receives the aggregated update.

        Args:
            data: Path to local HDF5 training data.
            rounds: Maximum number of rounds to participate in.
            epochs_per_round: Local training epochs per round.
            batch_size: Local training batch size.
            lr: Local learning rate.
        """
        await self.connect()

        try:
            for round_num in range(rounds):
                # Wait for round_start from server
                raw = await self._ws.recv()
                msg_data = json.loads(raw)
                msg = protocol.from_json(msg_data)

                if isinstance(msg, protocol.DoneMessage):
                    print(
                        f"WorldKit | Federated client {self._client_id!r}: "
                        f"server signaled done ({msg.reason})"
                    )
                    break

                if not isinstance(msg, protocol.RoundStartMessage):
                    raise RuntimeError(
                        f"Expected round_start, got {msg_data.get('type')}"
                    )

                # Snapshot weights before local training
                self._initial_weights = self._snapshot_weights()

                # Train locally
                avg_loss, num_samples = self.train_local(
                    data,
                    epochs=epochs_per_round,
                    batch_size=batch_size,
                    lr=lr,
                )
                print(
                    f"WorldKit | Federated client {self._client_id!r}: "
                    f"round {round_num + 1} local training done "
                    f"(loss={avg_loss:.4f}, samples={num_samples})"
                )

                # Send delta
                delta = self.compute_delta()
                await self.send_update(delta, num_samples, avg_loss, round_num)

                # Receive aggregated global update
                global_delta = await self.receive_global()
                if global_delta is None:
                    print(
                        f"WorldKit | Federated client {self._client_id!r}: "
                        f"server signaled done after round {round_num + 1}"
                    )
                    break

                # Apply global update
                self.apply_global_delta(global_delta)

        finally:
            if self._ws:
                await self._ws.close()
                self._ws = None
