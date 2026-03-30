"""Federated training server for WorldKit.

Orchestrates synchronous federated averaging rounds over WebSocket.
Multiple clients connect, train locally, send weight deltas, and the
server aggregates them via FedAvg (weighted by dataset size).
"""

from __future__ import annotations

import asyncio
import json
import logging

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from worldkit.core.model import WorldModel

from . import protocol

logger = logging.getLogger(__name__)


class FederatedServer:
    """Server for federated training of WorldKit world models.

    Args:
        model: The global WorldModel whose weights are aggregated.
        min_clients: Minimum number of clients before training starts.
        strategy: Aggregation strategy. Currently only ``"fedavg"`` is
            supported.
    """

    def __init__(
        self,
        model: WorldModel,
        min_clients: int = 2,
        strategy: str = "fedavg",
    ) -> None:
        if strategy != "fedavg":
            raise ValueError(
                f"Unknown aggregation strategy: {strategy!r}. "
                f"Supported: 'fedavg'"
            )
        self._model = model
        self._min_clients = min_clients
        self._strategy = strategy

        # Global model state (kept on CPU)
        self._global_state: dict[str, torch.Tensor] = {
            k: v.clone().cpu()
            for k, v in model._model.state_dict().items()
        }

        # Connected client websockets keyed by client_id
        self._clients: dict[str, WebSocket] = {}

        # Updates received in the current round
        self._round_updates: dict[str, protocol.UpdateMessage] = {}

        # Synchronization primitives (created lazily in the running event loop)
        self._registration_event: asyncio.Event | None = None
        self._updates_event: asyncio.Event | None = None

        self._current_round = 0
        self._total_rounds = 0

        # Build FastAPI app
        self._app = FastAPI(title="WorldKit Federated Server")
        self._app.add_api_websocket_route(
            "/ws/federated", self._ws_handler,
        )

    @property
    def app(self) -> FastAPI:
        """The FastAPI application (useful for testing)."""
        return self._app

    # ── WebSocket handler ────────────────────────────────

    async def _ws_handler(self, ws: WebSocket) -> None:
        """Handle a single client WebSocket connection."""
        self._ensure_events()
        await ws.accept()
        client_id: str | None = None

        try:
            # First message must be a register
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg = protocol.from_json(data)

            if not isinstance(msg, protocol.RegisterMessage):
                await ws.send_text(json.dumps(protocol.to_json(
                    protocol.ErrorMessage(detail="First message must be 'register'"),
                )))
                await ws.close()
                return

            client_id = msg.client_id
            expected_params = sum(
                p.numel() for p in self._global_state.values()
            )
            if msg.num_params != expected_params:
                await ws.send_text(json.dumps(protocol.to_json(
                    protocol.ErrorMessage(
                        detail=(
                            f"Parameter count mismatch: client has "
                            f"{msg.num_params}, server expects {expected_params}"
                        ),
                    ),
                )))
                await ws.close()
                return

            self._clients[client_id] = ws
            print(
                f"WorldKit | Federated: client {client_id!r} registered "
                f"({len(self._clients)}/{self._min_clients} clients)"
            )

            # Signal if we've reached min_clients
            if len(self._clients) >= self._min_clients:
                self._registration_event.set()

            # Wait for round messages from the orchestrator
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                msg = protocol.from_json(data)

                if isinstance(msg, protocol.UpdateMessage):
                    self._round_updates[msg.client_id] = msg
                    print(
                        f"WorldKit | Federated: received update from "
                        f"{msg.client_id!r} (round {msg.round_number}, "
                        f"loss={msg.train_loss:.4f}, "
                        f"samples={msg.num_samples})"
                    )
                    # Check if all connected clients have sent updates
                    if len(self._round_updates) >= len(self._clients):
                        self._updates_event.set()

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Client %s error: %s", client_id, exc)
        finally:
            if client_id and client_id in self._clients:
                del self._clients[client_id]
                print(
                    f"WorldKit | Federated: client {client_id!r} disconnected "
                    f"({len(self._clients)} remaining)"
                )
                # Re-check update barrier in case a disconnected client
                # was the one we were waiting for
                if (
                    self._round_updates
                    and len(self._round_updates) >= len(self._clients)
                ):
                    self._updates_event.set()

    # ── Aggregation ──────────────────────────────────────

    def aggregate(self) -> dict[str, torch.Tensor]:
        """FedAvg: weighted average of client deltas by num_samples.

        Returns:
            Aggregated weight delta.
        """
        total_samples = sum(
            u.num_samples for u in self._round_updates.values()
        )
        if total_samples == 0:
            raise RuntimeError("No samples received from any client")

        # Decode all deltas first (avoid decoding per-key)
        decoded: dict[str, dict[str, torch.Tensor]] = {}
        weights: dict[str, float] = {}
        for cid, update in self._round_updates.items():
            decoded[cid] = protocol.decode_delta(update.delta_b64)
            weights[cid] = update.num_samples / total_samples

        # Weighted average across clients
        aggregated: dict[str, torch.Tensor] = {}
        for key in self._global_state:
            weighted_sum = torch.zeros_like(self._global_state[key])
            for cid in decoded:
                weighted_sum += weights[cid] * decoded[cid][key].to(
                    weighted_sum.device
                )
            aggregated[key] = weighted_sum

        return aggregated

    # ── Broadcasting ─────────────────────────────────────

    async def _broadcast(self, msg_dict: dict) -> None:
        """Send a JSON message to all connected clients."""
        payload = json.dumps(msg_dict)
        disconnected: list[str] = []
        for cid, ws in list(self._clients.items()):
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(cid)
        for cid in disconnected:
            self._clients.pop(cid, None)

    # ── Round orchestration ──────────────────────────────

    def _ensure_events(self) -> None:
        """Create asyncio events if not yet initialized (must be called in the running loop)."""
        if self._registration_event is None:
            self._registration_event = asyncio.Event()
        if self._updates_event is None:
            self._updates_event = asyncio.Event()

    async def _run_rounds(self, rounds: int) -> None:
        """Run the federated training loop for the given number of rounds."""
        self._ensure_events()
        self._total_rounds = rounds

        # Wait for enough clients
        print(
            f"WorldKit | Federated: waiting for {self._min_clients} "
            f"clients to register..."
        )
        await self._registration_event.wait()
        print(
            f"WorldKit | Federated: {len(self._clients)} clients "
            f"connected. Starting {rounds} rounds."
        )

        for round_num in range(rounds):
            self._current_round = round_num
            self._round_updates.clear()
            self._updates_event.clear()

            # Broadcast round_start
            round_msg = protocol.to_json(protocol.RoundStartMessage(
                round_number=round_num,
            ))
            await self._broadcast(round_msg)

            # Wait for all connected clients to send updates
            await self._updates_event.wait()

            if not self._round_updates:
                print(
                    f"WorldKit | Federated: no updates received in "
                    f"round {round_num}, aborting."
                )
                break

            # Aggregate
            aggregated_delta = self.aggregate()
            self._global_state = protocol.apply_delta(
                self._global_state, aggregated_delta,
            )

            avg_loss = sum(
                u.train_loss for u in self._round_updates.values()
            ) / len(self._round_updates)
            print(
                f"WorldKit | Federated: round {round_num + 1}/{rounds} "
                f"complete | clients={len(self._round_updates)} | "
                f"avg_loss={avg_loss:.4f}"
            )

            # Broadcast aggregated delta to clients
            global_msg = protocol.to_json(protocol.GlobalWeightsMessage(
                round_number=round_num,
                aggregated_delta_b64=protocol.encode_delta(aggregated_delta),
                num_clients=len(self._round_updates),
            ))
            await self._broadcast(global_msg)

        # Broadcast done
        done_msg = protocol.to_json(protocol.DoneMessage(
            reason="all rounds complete",
            final_round=self._current_round,
        ))
        await self._broadcast(done_msg)

        # Update the model with final global state
        self._model._model.load_state_dict(self._global_state)
        print("WorldKit | Federated: training complete.")

    # ── Public API ───────────────────────────────────────

    async def run(
        self,
        rounds: int = 10,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """Start the server and run federated training.

        Args:
            rounds: Number of federated rounds.
            host: Bind address.
            port: Bind port.
        """
        import uvicorn

        config = uvicorn.Config(
            self._app, host=host, port=port, log_level="warning",
            ws_max_size=2**30,
        )
        server = uvicorn.Server(config)

        # Run uvicorn and the round orchestrator concurrently
        async def _orchestrate():
            await self._run_rounds(rounds)
            server.should_exit = True

        await asyncio.gather(server.serve(), _orchestrate())
