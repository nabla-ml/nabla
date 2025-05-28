# Nabla ∇

> **A JAX-like deep learning framework with MAX and Mojo backends for fast, hardware-agnostic kernels.**

[![Development Status](https://img.shields.io/badge/status-pre--alpha-red)](https://github.com/nabla-ml/nabla)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## 🚀 About

Nabla aims to provide a familiar, JAX-inspired API for numerical computation and automatic differentiation, with a focus on leveraging the [MAX Engine](https://www.modular.com/max) and Mojo programming language for high-performance, portable execution across various hardware platforms.

**⚠️ Current Status: Pre-Alpha - Under Heavy Development**

## 🛠️ Setup & Installation

### Prerequisites

Before you begin, ensure you have:
- **Python 3.10+** installed
- **Git** for cloning the repository
- **MAX SDK** (for full functionality)

### 1. Clone the Repository

```bash
git clone https://github.com/nabla-ml/nabla.git
cd nabla
```

### 2. Set Up Python Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Development Installation (Recommended)

For active development and testing:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install nabla in editable mode
pip install -e .
```

This approach:
- Installs all required dependencies including MAX SDK
- Allows you to modify the code and see changes immediately
- Keeps your development environment clean

#### Option B: Direct Installation with pyproject.toml

```bash
# Upgrade pip first
pip install --upgrade pip

# Install Nabla with all dependencies (including MAX SDK)
pip install -e ".[dev]"
```

This will automatically install:
- **Core dependencies**: `numpy>=1.22`, `modular>=25.0.0` (MAX SDK)
- **Development tools**: `pytest`, `ruff`, `black`, `mypy`, `build`, `twine`, `pre-commit`

### 4. Verify Installation

Test that everything is working correctly:

```bash
python -c "import nabla as nb; print('Nabla installed successfully!')"
```

## 🏃‍♂️ Quick Start

```python
import nabla as nb

# Create arrays
a = nb.arange(shape=(4, 4), dtype=nb.DType.float32)
b = nb.randn(shape=(4, 4), dtype=nb.DType.float32)

# Perform operations
c = nb.add(a, b)
d = nb.matmul(c, nb.transpose(b))

# Realize computation
result = nb.realize_(d)
print(result)
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/verification/  # Verification tests

# Run with verbose output
pytest -v
```

### Code Quality

The project uses several tools for code quality:

```bash
# Format code
black nabla/
ruff format nabla/

# Lint code
ruff check nabla/

# Type checking
mypy nabla/
```

### Pre-commit Hooks

Set up pre-commit hooks for automated code quality checks:

```bash
pre-commit install
```

## 📦 Building for Distribution

To build the package for distribution:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build source and wheel distributions
python -m build

# Check the built package
twine check dist/*
```

## 🗂️ Project Structure

```
nabla/
├── nabla/                  # Main package
│   ├── core/              # Core array and execution engine
│   ├── ops/               # Mathematical operations
│   ├── mojo_kernels/      # Mojo kernel implementations
│   └── utils/             # Utilities and helpers
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── verification/      # End-to-end verification
├── pyproject.toml         # Project configuration
└── readme.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the code quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the Apache License v2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: [https://github.com/nabla-ml/nabla](https://github.com/nabla-ml/nabla)
- **MAX Engine**: [https://www.modular.com/max](https://www.modular.com/max)
- **Mojo Language**: [https://docs.modular.com/mojo/](https://docs.modular.com/mojo/)

---

**Built with ❤️ by the Nabla team**

