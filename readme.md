# Nabla ∇

**Nabla: A JAX-like deep learning framework with MAX and Mojo backends for fast, hardware-agnostic kernels.**

**Current Status: Pre-Alpha - Under Heavy Development**

## About
Nabla aims to provide a familiar, JAX-inspired API for numerical computation and automatic differentiation, with a focus on leveraging the MAX engine and Mojo programming language for high-performance, portable execution across various hardware.

## Installation
This package is not yet on PyPI. For development within this repository:
1. Ensure you are on the correct branch.
2. Navigate to the `nabla` directory (i.e., `cd path/to/repo/nabla`).
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

   pip install modular 

   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

