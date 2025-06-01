# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Improved Unit Test Configuration
# ===----------------------------------------------------------------------=== #

"""Pytest configuration and fixtures for unit tests."""

import numpy as np
import pytest

# Import JAX availability check
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# --- Test Configuration ---
DEFAULT_DTYPES = [np.float32, np.float64]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark (slow)")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Automatically apply markers and skip logic."""
    for item in items:
        # Skip benchmark tests unless explicitly requested
        if "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(
                pytest.mark.skip(
                    reason="Benchmark tests skipped (use --benchmark to run)"
                )
            )

        # Skip JAX-dependent tests if JAX not available
        if "jax" in item.keywords and not JAX_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="JAX not available"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="Run benchmark tests"
    )


# --- Pytest Fixtures ---
@pytest.fixture(scope="session")
def jax_available():
    """Session-scoped fixture for JAX availability."""
    return JAX_AVAILABLE


@pytest.fixture
def random_seed():
    """Fixture to ensure reproducible random data."""
    np.random.seed(42)
    if JAX_AVAILABLE:
        import jax

        return jax.random.PRNGKey(42)
    return None


@pytest.fixture
def tolerance_config():
    """Fixture providing tolerance configuration."""
    return {
        "float32": {"value": (1e-5, 1e-6), "gradient": (1e-4, 1e-5)},
        "float64": {"value": (1e-7, 1e-8), "gradient": (1e-6, 1e-7)},
    }


# Simple data generation utilities (no external dependencies)
def generate_random_data(shape, dtype=np.float32, seed=42):
    """Generate reproducible random test data."""
    np.random.seed(seed)
    return np.random.uniform(-5.0, 5.0, size=shape).astype(dtype)


def assert_arrays_close(actual, expected, rtol=1e-7, atol=1e-8, msg=""):
    """Assert that two arrays are numerically close."""
    if hasattr(actual, "to_numpy"):
        actual_np = actual.to_numpy()
    else:
        actual_np = np.asarray(actual)

    expected_np = np.asarray(expected)

    assert actual_np.shape == expected_np.shape, (
        f"Shape mismatch: {actual_np.shape} vs {expected_np.shape}. {msg}"
    )

    assert np.allclose(actual_np, expected_np, rtol=rtol, atol=atol), (
        f"Arrays not close enough (rtol={rtol}, atol={atol}). {msg}\n"
        f"Actual:\n{actual_np}\nExpected:\n{expected_np}\n"
        f"Difference:\n{actual_np - expected_np}"
    )
