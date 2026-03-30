"""Environment registry for WorldKit.

Provides a central registry of environments that can be discovered,
installed, and recorded with WorldKit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Configuration for a registered environment.

    Args:
        env_id: Unique identifier (e.g., "worldkit/pusht").
        display_name: Human-readable name.
        category: One of navigation, manipulation, control, games, simulation.
        gym_id: Gymnasium environment ID, if applicable.
        action_dim: Dimensionality of the action space.
        action_type: "continuous" or "discrete".
        action_low: Lower bound for continuous actions.
        action_high: Upper bound for continuous actions.
        observation_shape: Shape of pixel observations (H, W, C).
        description: Short description of the environment.
        install_cmd: pip command to install dependencies.
        dataset_url: URL to download pre-recorded data.
        pretrained_models: List of Hub model IDs with pretrained weights.
    """

    env_id: str
    display_name: str
    category: str
    gym_id: str | None = None
    action_dim: int = 2
    action_type: str = "continuous"
    action_low: float = -1.0
    action_high: float = 1.0
    observation_shape: tuple[int, ...] = (96, 96, 3)
    description: str = ""
    install_cmd: str | None = None
    dataset_url: str | None = None
    pretrained_models: list[str] = field(default_factory=list)


class EnvironmentRegistry:
    """Registry of environments available for use with WorldKit.

    Provides methods to register, retrieve, list, filter, and search
    environments by ID, category, or free-text query.
    """

    def __init__(self) -> None:
        self._registry: dict[str, EnvConfig] = {}

    def register(self, env_id: str, **kwargs) -> EnvConfig:
        """Register a new environment.

        Args:
            env_id: Unique identifier (e.g., "worldkit/cartpole").
            **kwargs: Fields for EnvConfig.

        Returns:
            The registered EnvConfig.
        """
        config = EnvConfig(env_id=env_id, **kwargs)
        self._registry[env_id] = config
        return config

    def get(self, env_id: str) -> EnvConfig:
        """Get an environment config by ID.

        Args:
            env_id: The environment identifier.

        Returns:
            The EnvConfig for the given ID.

        Raises:
            ValueError: If the environment is not found.
        """
        if env_id not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Unknown environment: '{env_id}'. "
                f"Available environments: {available}"
            )
        return self._registry[env_id]

    def list_all(self) -> list[EnvConfig]:
        """List all registered environments.

        Returns:
            List of all EnvConfig entries, sorted by env_id.
        """
        return sorted(self._registry.values(), key=lambda e: e.env_id)

    def list_by_category(self, category: str) -> list[EnvConfig]:
        """List environments filtered by category.

        Args:
            category: Category to filter by (e.g., "control", "games").

        Returns:
            List of matching EnvConfig entries, sorted by env_id.
        """
        cat = category.lower()
        return sorted(
            [e for e in self._registry.values() if e.category.lower() == cat],
            key=lambda e: e.env_id,
        )

    def search(self, query: str) -> list[EnvConfig]:
        """Search environments by free-text query.

        Performs case-insensitive substring matching against env_id,
        display_name, and description.

        Args:
            query: Search string.

        Returns:
            List of matching EnvConfig entries, sorted by env_id.
        """
        q = query.lower()
        results = []
        for env in self._registry.values():
            if (
                q in env.env_id.lower()
                or q in env.display_name.lower()
                or q in env.description.lower()
            ):
                results.append(env)
        return sorted(results, key=lambda e: e.env_id)


# ---------------------------------------------------------------------------
# Default registry instance with pre-registered environments
# ---------------------------------------------------------------------------

registry = EnvironmentRegistry()

# --- Navigation ---
registry.register(
    "worldkit/two-room",
    display_name="Two-Room Navigation",
    category="navigation",
    gym_id=None,
    action_dim=2,
    action_type="continuous",
    action_low=-1.0,
    action_high=1.0,
    observation_shape=(96, 96, 3),
    description="Agent navigates between two connected rooms. From LeWM.",
    install_cmd="pip install stable-worldmodel",
)

registry.register(
    "worldkit/grid-world",
    display_name="Grid World",
    category="navigation",
    gym_id="MiniGrid-Empty-8x8-v0",
    action_dim=1,
    action_type="discrete",
    action_low=0.0,
    action_high=6.0,
    observation_shape=(96, 96, 3),
    description="Empty 8x8 grid navigation environment.",
    install_cmd="pip install minigrid",
)

# --- Manipulation ---
registry.register(
    "worldkit/pusht",
    display_name="Push-T",
    category="manipulation",
    gym_id=None,
    action_dim=2,
    action_type="continuous",
    action_low=-1.0,
    action_high=1.0,
    observation_shape=(96, 96, 3),
    description="Push a T-shaped block to a target pose. From LeWM.",
    install_cmd="pip install stable-worldmodel",
)

# --- Control ---
registry.register(
    "worldkit/cartpole",
    display_name="CartPole",
    category="control",
    gym_id="CartPole-v1",
    action_dim=1,
    action_type="discrete",
    action_low=0.0,
    action_high=1.0,
    observation_shape=(400, 600, 3),
    description="Balance a pole on a cart by moving left or right.",
)

registry.register(
    "worldkit/pendulum",
    display_name="Pendulum",
    category="control",
    gym_id="Pendulum-v1",
    action_dim=1,
    action_type="continuous",
    action_low=-2.0,
    action_high=2.0,
    observation_shape=(500, 500, 3),
    description="Swing up and balance an inverted pendulum.",
)

registry.register(
    "worldkit/acrobot",
    display_name="Acrobot",
    category="control",
    gym_id="Acrobot-v1",
    action_dim=1,
    action_type="discrete",
    action_low=0.0,
    action_high=2.0,
    observation_shape=(500, 500, 3),
    description="Swing the lower link of a two-link robot above a threshold.",
)

registry.register(
    "worldkit/reacher",
    display_name="Reacher",
    category="control",
    gym_id=None,
    action_dim=2,
    action_type="continuous",
    action_low=-1.0,
    action_high=1.0,
    observation_shape=(96, 96, 3),
    description="Two-joint arm reaching a random target. From LeWM.",
    install_cmd="pip install stable-worldmodel",
)

registry.register(
    "worldkit/mountain-car",
    display_name="Mountain Car (Continuous)",
    category="control",
    gym_id="MountainCarContinuous-v0",
    action_dim=1,
    action_type="continuous",
    action_low=-1.0,
    action_high=1.0,
    observation_shape=(400, 600, 3),
    description="Drive a car up a steep hill using momentum.",
)

# --- Games ---
registry.register(
    "worldkit/pong",
    display_name="Pong",
    category="games",
    gym_id="ALE/Pong-v5",
    action_dim=1,
    action_type="discrete",
    action_low=0.0,
    action_high=5.0,
    observation_shape=(210, 160, 3),
    description="Classic Atari Pong.",
    install_cmd="pip install ale-py",
)

registry.register(
    "worldkit/breakout",
    display_name="Breakout",
    category="games",
    gym_id="ALE/Breakout-v5",
    action_dim=1,
    action_type="discrete",
    action_low=0.0,
    action_high=3.0,
    observation_shape=(210, 160, 3),
    description="Classic Atari Breakout.",
    install_cmd="pip install ale-py",
)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def register(env_id: str, **kwargs) -> EnvConfig:
    """Register a new environment in the default registry."""
    return registry.register(env_id, **kwargs)


def get(env_id: str) -> EnvConfig:
    """Get an environment config by ID from the default registry."""
    return registry.get(env_id)


def list_all() -> list[EnvConfig]:
    """List all registered environments."""
    return registry.list_all()


def list_by_category(category: str) -> list[EnvConfig]:
    """List environments filtered by category."""
    return registry.list_by_category(category)


def search(query: str) -> list[EnvConfig]:
    """Search environments by free-text query."""
    return registry.search(query)
