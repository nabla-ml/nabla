# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Shared fixtures and helpers for nabla.nn test suite.

Convention:
  - Imperative / stateful tests (Module, .backward(), stateful AdamW) → PyTorch as reference
  - Functional tests (functional.*, optim functional, LoRA/QLoRA) → JAX as reference
"""

from __future__ import annotations

import numpy as np
import pytest

import nabla as nb

# ---------------------------------------------------------------------------
# Deterministic seed helpers
# ---------------------------------------------------------------------------
SEED = 42


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Return a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def np_randn(*shape: int, seed: int = SEED, dtype=np.float32) -> np.ndarray:
    """Deterministic random normal ndarray."""
    return make_rng(seed).normal(size=shape).astype(dtype)


def nb_from_np(arr: np.ndarray, *, requires_grad: bool = False) -> nb.Tensor:
    """Convert a NumPy array to a Nabla tensor."""
    t = nb.Tensor.from_dlpack(arr)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# Optional framework skip markers
# ---------------------------------------------------------------------------


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _has_jax() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not _has_jax(), reason="JAX not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def rng():
    """A default reproducible RNG."""
    return make_rng(SEED)


@pytest.fixture()
def torch_mod():
    """Import and return the torch module, skip if unavailable."""
    return pytest.importorskip("torch")


@pytest.fixture()
def jax_mod():
    """Import and return jax + jax.numpy, skip if unavailable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    return jax, jnp


@pytest.fixture()
def simple_linear():
    """A small Linear(4, 3) model with deterministic weights."""
    rng = make_rng(100)
    model = nb.nn.Linear(4, 3)
    model.weight = nb_from_np(
        rng.normal(size=(4, 3)).astype(np.float32), requires_grad=True
    )
    model.bias = nb_from_np(
        rng.normal(size=(1, 3)).astype(np.float32), requires_grad=True
    )
    return model


@pytest.fixture()
def simple_data_4x3():
    """Return (x_np, y_np) of shapes (8, 4) and (8, 3)."""
    rng = make_rng(200)
    x = rng.normal(size=(8, 4)).astype(np.float32)
    y = rng.normal(size=(8, 3)).astype(np.float32)
    return x, y


@pytest.fixture()
def simple_sequential():
    """A Sequential(Linear(4,6), ReLU(), Linear(6,3)) model."""
    return nb.nn.Sequential(
        nb.nn.Linear(4, 6),
        nb.nn.ReLU(),
        nb.nn.Linear(6, 3),
    )
