"""Federated training for WorldKit world models.

Enables privacy-preserving collaborative training where clients train
locally on their own data and only exchange weight deltas with a
central server.

Example::

    from worldkit import WorldModel
    from worldkit.federated import FederatedClient, FederatedServer

    model = WorldModel.load("base_model.wk")

    # On the server machine:
    server = FederatedServer(model, min_clients=3)
    await server.run(rounds=50, port=8080)

    # On each client machine:
    client = FederatedClient(model, server_url="ws://server:8080/ws/federated")
    await client.run(data="local_data.h5", rounds=50)
"""

from .client import FederatedClient
from .protocol import (
    DoneMessage,
    ErrorMessage,
    GlobalWeightsMessage,
    MessageType,
    RegisterMessage,
    RoundStartMessage,
    UpdateMessage,
    apply_delta,
    compute_delta,
    decode_delta,
    encode_delta,
    from_json,
    to_json,
)
from .server import FederatedServer

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "MessageType",
    "RegisterMessage",
    "RoundStartMessage",
    "UpdateMessage",
    "GlobalWeightsMessage",
    "DoneMessage",
    "ErrorMessage",
    "encode_delta",
    "decode_delta",
    "compute_delta",
    "apply_delta",
    "to_json",
    "from_json",
]
