"""World model backends — pluggable architecture implementations.

Each backend encapsulates a full world model architecture (encoder, predictor,
training loop) behind the ``BaseWorldModelBackend`` interface.  WorldModel
resolves the backend from ``config.backend`` via the global registry.

Usage (registering a custom backend)::

    from worldkit.core.backends import BaseWorldModelBackend, backend_registry

    class MyBackend(BaseWorldModelBackend):
        ...

    backend_registry.register("my_backend", MyBackend)
"""

from .base import BaseWorldModelBackend as BaseWorldModelBackend
from .dreamer_stub import DreamerV3Backend as DreamerV3Backend
from .lewm import LeWMBackend as LeWMBackend
from .registry import BackendRegistry as BackendRegistry
from .registry import backend_registry as backend_registry

# Pre-register built-in backends.
backend_registry.register("lewm", LeWMBackend)
backend_registry.register("dreamerv3", DreamerV3Backend)
