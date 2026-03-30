"""Tests for federated training (F-042)."""

from __future__ import annotations

import asyncio

import h5py
import numpy as np
import torch

from worldkit.core.config import get_config
from worldkit.core.model import WorldModel
from worldkit.federated import protocol

# ── Helpers ──────────────────────────────────────────────


def _make_h5(path, n_episodes=5, ep_len=20, h=32, w=32, action_dim=2):
    """Create a minimal HDF5 dataset for testing."""
    pixels = np.random.randint(0, 255, (n_episodes, ep_len, h, w, 3), dtype=np.uint8)
    actions = np.random.randn(n_episodes, ep_len, action_dim).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("actions", data=actions)


def _build_model(action_dim=2, device="cpu"):
    """Build a nano WorldModel for testing."""
    from worldkit.core.backends import backend_registry

    config = get_config("nano", action_dim=action_dim, image_size=32)
    backend_cls = backend_registry.get(config.backend)
    backend = backend_cls()
    module = backend.build(config)
    return WorldModel(module, config, device, backend=backend)


# ── Protocol serialization tests ─────────────────────────


class TestProtocolSerialization:
    def test_encode_decode_roundtrip(self):
        """Weight deltas survive encode → decode roundtrip."""
        state_dict = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(16, 64),
        }
        b64 = protocol.encode_delta(state_dict)
        assert isinstance(b64, str)

        restored = protocol.decode_delta(b64)
        assert set(restored.keys()) == set(state_dict.keys())
        for key in state_dict:
            torch.testing.assert_close(
                restored[key], state_dict[key], atol=1e-6, rtol=1e-6,
            )

    def test_compute_delta(self):
        """compute_delta returns current - initial."""
        initial = {"w": torch.tensor([1.0, 2.0, 3.0])}
        current = {"w": torch.tensor([1.5, 2.5, 3.5])}
        delta = protocol.compute_delta(initial, current)
        expected = torch.tensor([0.5, 0.5, 0.5])
        torch.testing.assert_close(delta["w"], expected)

    def test_apply_delta(self):
        """apply_delta returns state + delta."""
        state = {"w": torch.tensor([1.0, 2.0])}
        delta = {"w": torch.tensor([0.1, 0.2])}
        result = protocol.apply_delta(state, delta)
        expected = torch.tensor([1.1, 2.2])
        torch.testing.assert_close(result["w"], expected)

    def test_compute_then_apply_is_identity(self):
        """apply_delta(initial, compute_delta(initial, current)) == current."""
        initial = {"a": torch.randn(10, 10), "b": torch.randn(5)}
        current = {"a": torch.randn(10, 10), "b": torch.randn(5)}
        delta = protocol.compute_delta(initial, current)
        restored = protocol.apply_delta(initial, delta)
        for key in current:
            torch.testing.assert_close(
                restored[key], current[key], atol=1e-6, rtol=1e-6,
            )

    def test_message_to_json_from_json(self):
        """Message dataclasses survive to_json → from_json roundtrip."""
        messages = [
            protocol.RegisterMessage(client_id="c1", num_params=1000),
            protocol.RoundStartMessage(round_number=5),
            protocol.UpdateMessage(
                client_id="c1", round_number=3,
                delta_b64="abc123", num_samples=500, train_loss=0.42,
            ),
            protocol.GlobalWeightsMessage(
                round_number=2, aggregated_delta_b64="xyz", num_clients=3,
            ),
            protocol.DoneMessage(reason="complete", final_round=9),
            protocol.ErrorMessage(detail="something went wrong"),
        ]
        for msg in messages:
            data = protocol.to_json(msg)
            assert "type" in data
            restored = protocol.from_json(data)
            assert type(restored) is type(msg)
            # Verify all fields match
            import dataclasses

            for field in dataclasses.fields(msg):
                assert getattr(restored, field.name) == getattr(msg, field.name)

    def test_encode_non_contiguous_tensor(self):
        """Non-contiguous tensors are handled correctly."""
        t = torch.randn(10, 10).T  # Transpose makes it non-contiguous
        assert not t.is_contiguous()
        state = {"w": t}
        b64 = protocol.encode_delta(state)
        restored = protocol.decode_delta(b64)
        torch.testing.assert_close(restored["w"], t.contiguous())


# ── FedAvg aggregation tests ────────────────────────────


class TestFedAvgAggregation:
    def test_weighted_average_with_equal_samples(self):
        """Equal sample counts produce a simple average of deltas."""
        from worldkit.federated.server import FederatedServer

        model = _build_model()
        server = FederatedServer(model, min_clients=2)

        # Create two deltas with equal sample counts
        delta_a = {"w": torch.tensor([1.0, 0.0])}
        delta_b = {"w": torch.tensor([0.0, 1.0])}

        server._round_updates = {
            "client_a": protocol.UpdateMessage(
                client_id="client_a", round_number=0,
                delta_b64=protocol.encode_delta(delta_a),
                num_samples=100, train_loss=0.5,
            ),
            "client_b": protocol.UpdateMessage(
                client_id="client_b", round_number=0,
                delta_b64=protocol.encode_delta(delta_b),
                num_samples=100, train_loss=0.4,
            ),
        }
        # Override global_state to match our simple test
        server._global_state = {"w": torch.zeros(2)}

        aggregated = server.aggregate()
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(aggregated["w"], expected)

    def test_weighted_average_with_unequal_samples(self):
        """Unequal sample counts weight deltas proportionally."""
        from worldkit.federated.server import FederatedServer

        model = _build_model()
        server = FederatedServer(model, min_clients=2)

        # Client A: 300 samples, delta=[3.0]
        # Client B: 100 samples, delta=[1.0]
        # Expected: (300*3.0 + 100*1.0) / 400 = 2.5
        delta_a = {"w": torch.tensor([3.0])}
        delta_b = {"w": torch.tensor([1.0])}

        server._round_updates = {
            "a": protocol.UpdateMessage(
                client_id="a", round_number=0,
                delta_b64=protocol.encode_delta(delta_a),
                num_samples=300, train_loss=0.3,
            ),
            "b": protocol.UpdateMessage(
                client_id="b", round_number=0,
                delta_b64=protocol.encode_delta(delta_b),
                num_samples=100, train_loss=0.5,
            ),
        }
        server._global_state = {"w": torch.zeros(1)}

        aggregated = server.aggregate()
        expected = torch.tensor([2.5])
        torch.testing.assert_close(aggregated["w"], expected)

    def test_three_clients_multiple_params(self):
        """Aggregation works with 3 clients and multiple parameter tensors."""
        from worldkit.federated.server import FederatedServer

        model = _build_model()
        server = FederatedServer(model, min_clients=2)

        deltas = [
            {"a": torch.ones(3), "b": torch.ones(2, 2)},
            {"a": torch.ones(3) * 2, "b": torch.ones(2, 2) * 2},
            {"a": torch.ones(3) * 3, "b": torch.ones(2, 2) * 3},
        ]
        samples = [100, 200, 300]

        server._round_updates = {}
        for i, (d, s) in enumerate(zip(deltas, samples)):
            server._round_updates[f"c{i}"] = protocol.UpdateMessage(
                client_id=f"c{i}", round_number=0,
                delta_b64=protocol.encode_delta(d),
                num_samples=s, train_loss=0.1,
            )
        server._global_state = {
            "a": torch.zeros(3),
            "b": torch.zeros(2, 2),
        }

        aggregated = server.aggregate()

        # Weighted avg: (100*1 + 200*2 + 300*3) / 600 = 1400/600 ≈ 2.333
        expected_val = (100 * 1 + 200 * 2 + 300 * 3) / 600
        torch.testing.assert_close(
            aggregated["a"],
            torch.ones(3) * expected_val,
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            aggregated["b"],
            torch.ones(2, 2) * expected_val,
            atol=1e-5, rtol=1e-5,
        )


# ── Client-server integration test ──────────────────────


class TestClientServerIntegration:
    def test_two_clients_one_round(self, tmp_path):
        """Two clients complete 1 round of federated training on localhost."""
        from worldkit.federated.client import FederatedClient
        from worldkit.federated.server import FederatedServer

        # Create two separate HDF5 datasets
        data_a = tmp_path / "data_a.h5"
        data_b = tmp_path / "data_b.h5"
        _make_h5(data_a, n_episodes=3, ep_len=20, h=32, w=32, action_dim=2)
        _make_h5(data_b, n_episodes=3, ep_len=20, h=32, w=32, action_dim=2)

        # Build three copies of the same model (server + 2 clients)
        server_model = _build_model()
        client_model_a = _build_model()
        client_model_b = _build_model()

        # Sync initial weights across all models
        init_state = server_model._model.state_dict()
        client_model_a._model.load_state_dict(
            {k: v.clone() for k, v in init_state.items()}
        )
        client_model_b._model.load_state_dict(
            {k: v.clone() for k, v in init_state.items()}
        )

        # Snapshot initial weights for comparison
        initial_weights = {
            k: v.clone().cpu() for k, v in init_state.items()
        }

        port = 18942  # Use a high port to avoid conflicts

        server = FederatedServer(server_model, min_clients=2)
        client_a = FederatedClient(
            client_model_a,
            server_url=f"ws://127.0.0.1:{port}/ws/federated",
            client_id="alice",
        )
        client_b = FederatedClient(
            client_model_b,
            server_url=f"ws://127.0.0.1:{port}/ws/federated",
            client_id="bob",
        )

        async def run_all():
            import uvicorn

            config = uvicorn.Config(
                server.app, host="127.0.0.1", port=port,
                log_level="warning", ws_max_size=2**30,
            )
            uvi = uvicorn.Server(config)

            # Start uvicorn server in background
            server_task = asyncio.ensure_future(uvi.serve())

            # Wait for server to start
            await asyncio.sleep(0.5)

            # Run round orchestration and both clients concurrently
            async def _orchestrate():
                await server._run_rounds(1)
                uvi.should_exit = True

            await asyncio.gather(
                _orchestrate(),
                client_a.run(
                    data=str(data_a), rounds=1, epochs_per_round=1,
                    batch_size=4, lr=1e-3,
                ),
                client_b.run(
                    data=str(data_b), rounds=1, epochs_per_round=1,
                    batch_size=4, lr=1e-3,
                ),
            )
            await server_task

        asyncio.run(run_all())

        # Verify: server model weights changed from initial
        final_server_state = server_model._model.state_dict()
        weights_changed = False
        for key in initial_weights:
            if not torch.equal(final_server_state[key].cpu(), initial_weights[key]):
                weights_changed = True
                break
        assert weights_changed, "Server model weights should have changed"

        # Verify: both clients end with the same weights
        state_a = client_model_a._model.state_dict()
        state_b = client_model_b._model.state_dict()
        for key in state_a:
            torch.testing.assert_close(
                state_a[key].cpu(), state_b[key].cpu(),
                atol=1e-6, rtol=1e-6,
                msg=f"Client weights differ on key {key!r}",
            )
