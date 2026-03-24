# Contributing to WorldKit

Thank you for your interest in contributing to WorldKit!

## Development Setup

```bash
git clone https://github.com/worldkit-ai/worldkit.git
cd worldkit
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check worldkit/ tests/
ruff format worldkit/ tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Reporting Issues

Please use [GitHub Issues](https://github.com/worldkit-ai/worldkit/issues) to report bugs or request features.

## Code of Conduct

Be respectful and constructive. We welcome contributions from everyone.
