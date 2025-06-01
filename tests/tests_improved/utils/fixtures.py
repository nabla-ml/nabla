# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Test Fixtures
# ===----------------------------------------------------------------------=== #

"""Reusable pytest fixtures."""

import numpy as np
import pytest

# Check for optional dependencies
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Pytest Fixtures ---


@pytest.fixture(scope="session")
def jax_available():
    """Session-scoped fixture indicating JAX availability."""
    return JAX_AVAILABLE


@pytest.fixture(scope="session")
def torch_available():
    """Session-scoped fixture indicating PyTorch availability."""
    return TORCH_AVAILABLE


@pytest.fixture
def random_seed():
    """Fixture to set reproducible random seed."""
    seed = 42
    np.random.seed(seed)

    if JAX_AVAILABLE:
        jax_key = jax.random.PRNGKey(seed)
        return {"numpy": seed, "jax": jax_key}

    return {"numpy": seed}


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    """Parametrized fixture for floating point data types."""
    return request.param


@pytest.fixture(
    params=[np.float32, np.float64, np.int32], ids=["float32", "float64", "int32"]
)
def dtype_with_int(request):
    """Parametrized fixture including integer types."""
    return request.param


@pytest.fixture
def tolerance_config():
    """Fixture providing tolerance configuration for different dtypes."""
    return {
        np.float32: {"value": (1e-5, 1e-6), "gradient": (1e-4, 1e-5)},
        np.float64: {"value": (1e-7, 1e-8), "gradient": (1e-6, 1e-7)},
        np.int32: {"value": (0, 0), "gradient": (0, 0)},
    }


@pytest.fixture
def tolerances(dtype, tolerance_config):
    """Fixture providing appropriate tolerances for the current dtype."""
    return tolerance_config.get(dtype, tolerance_config[np.float32])


# Skip markers for optional dependencies
requires_jax = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
