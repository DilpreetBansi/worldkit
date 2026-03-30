"""Backend registry for discovering and retrieving world model backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseWorldModelBackend


class BackendRegistry:
    """Registry for world model backend classes.

    Backends are registered by name and can be retrieved later to construct
    instances. The registry is the single source of truth for which
    architectures are available.
    """

    def __init__(self) -> None:
        self._backends: dict[str, type[BaseWorldModelBackend]] = {}

    def register(self, name: str, backend_class: type[BaseWorldModelBackend]) -> None:
        """Register a backend class under a given name.

        Args:
            name: Short identifier (e.g. ``"lewm"``, ``"dreamerv3"``).
            backend_class: The backend class to register.
        """
        self._backends[name] = backend_class

    def get(self, name: str) -> type[BaseWorldModelBackend]:
        """Retrieve a backend class by name.

        Args:
            name: The backend identifier.

        Returns:
            The backend class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._backends:
            available = list(self._backends.keys())
            raise KeyError(
                f"Unknown backend: '{name}'. Available backends: {available}"
            )
        return self._backends[name]

    def list(self) -> list[str]:
        """Return the names of all registered backends."""
        return list(self._backends.keys())


# Module-level singleton used by WorldModel and external code.
backend_registry = BackendRegistry()
