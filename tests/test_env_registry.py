"""Tests for the WorldKit environment registry."""

import pytest
from click.testing import CliRunner

from worldkit.cli.main import cli
from worldkit.envs.registry import (
    EnvConfig,
    EnvironmentRegistry,
    registry,
)

# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------


def test_list_all_envs():
    """Default registry should have at least 10 pre-registered envs."""
    envs = registry.list_all()
    assert len(envs) >= 10


def test_get_env_config():
    """Should retrieve a known env by ID and have correct fields."""
    env = registry.get("worldkit/cartpole")
    assert isinstance(env, EnvConfig)
    assert env.env_id == "worldkit/cartpole"
    assert env.display_name == "CartPole"
    assert env.category == "control"
    assert env.gym_id == "CartPole-v1"
    assert env.action_dim == 1
    assert env.action_type == "discrete"


def test_get_unknown_env():
    """Should raise ValueError with helpful message for unknown env."""
    with pytest.raises(ValueError, match="Unknown environment"):
        registry.get("worldkit/does-not-exist")


def test_search():
    """Search should match on env_id, display_name, and description."""
    results = registry.search("pong")
    assert len(results) >= 1
    assert any(e.env_id == "worldkit/pong" for e in results)


def test_search_case_insensitive():
    """Search should be case-insensitive."""
    results = registry.search("CARTPOLE")
    assert len(results) >= 1
    assert any(e.env_id == "worldkit/cartpole" for e in results)


def test_search_no_results():
    """Search with no matches should return empty list."""
    results = registry.search("zzz_nonexistent_xyz")
    assert results == []


def test_filter_by_category():
    """Should return only envs matching the given category."""
    control_envs = registry.list_by_category("control")
    assert len(control_envs) >= 4
    for env in control_envs:
        assert env.category == "control"


def test_filter_by_category_case_insensitive():
    """Category filtering should be case-insensitive."""
    games = registry.list_by_category("Games")
    assert len(games) >= 2
    for env in games:
        assert env.category == "games"


def test_filter_by_category_empty():
    """Unknown category should return empty list."""
    envs = registry.list_by_category("nonexistent")
    assert envs == []


def test_register_custom_env():
    """Should be able to register a custom environment."""
    custom_registry = EnvironmentRegistry()
    custom_registry.register(
        "test/my-env",
        display_name="My Custom Env",
        category="simulation",
        action_dim=3,
        action_type="continuous",
        description="A custom test environment.",
    )
    env = custom_registry.get("test/my-env")
    assert env.env_id == "test/my-env"
    assert env.display_name == "My Custom Env"
    assert env.action_dim == 3


def test_list_all_sorted():
    """list_all should return envs sorted by env_id."""
    envs = registry.list_all()
    env_ids = [e.env_id for e in envs]
    assert env_ids == sorted(env_ids)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


def test_cli_env_list(runner):
    """'worldkit env list' should show registered environments."""
    result = runner.invoke(cli, ["env", "list"])
    assert result.exit_code == 0
    assert "worldkit/cartpole" in result.output
    assert "worldkit/pong" in result.output


def test_cli_env_list_category(runner):
    """'worldkit env list --category games' should filter."""
    result = runner.invoke(cli, ["env", "list", "--category", "games"])
    assert result.exit_code == 0
    assert "worldkit/pong" in result.output
    assert "worldkit/cartpole" not in result.output


def test_cli_env_info(runner):
    """'worldkit env info worldkit/pusht' should show details."""
    result = runner.invoke(cli, ["env", "info", "worldkit/pusht"])
    assert result.exit_code == 0
    assert "Push-T" in result.output
    assert "manipulation" in result.output
    assert "continuous" in result.output


def test_cli_env_info_unknown(runner):
    """'worldkit env info' with unknown ID should error."""
    result = runner.invoke(cli, ["env", "info", "worldkit/nope"])
    assert result.exit_code != 0


def test_cli_env_search(runner):
    """'worldkit env search cart' should find cartpole."""
    result = runner.invoke(cli, ["env", "search", "cart"])
    assert result.exit_code == 0
    assert "worldkit/cartpole" in result.output


def test_cli_env_search_no_results(runner):
    """'worldkit env search zzz' should show no results message."""
    result = runner.invoke(cli, ["env", "search", "zzz_none"])
    assert result.exit_code == 0
    assert "No environments matching" in result.output
