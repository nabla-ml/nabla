# Contributing to Nabla

This section provides guidance on how to contribute to the Nabla project.

```{toctree}
:maxdepth: 2

deploy_docs
```

## Getting Started with Development

Nabla welcomes contributions from the community. To get started:

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine.
3. **Create a branch** for your changes.
4. **Make your changes** and commit them.
5. **Push your branch** to your fork.
6. **Create a pull request** to the main repository.

## Development Setup

To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/nabla.git
cd nabla

# Install development dependencies
pip install -r requirements-dev.txt

# Install nabla in development mode
pip install -e .
```

## Running Tests

To run the test suite:

```bash
pytest tests/
```

## Documentation

See the [Documentation Deployment Guide](deploy_docs.md) for details on building and deploying the documentation.

## Code Style

Nabla follows best practices for code style, including:

- PEP 8 for Python code style
- Comprehensive docstrings in Google/NumPy format
- Type annotations
- Clear, descriptive variable and function names

## Reporting Issues

If you find a bug or have a suggestion for improvement, please [open an issue](https://github.com/nabla-ml/nabla/issues) on GitHub.
