# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Global Test Configuration
# ===----------------------------------------------------------------------=== #

"""Global pytest configuration for all test modules."""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark (slow)"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "reference: mark test as reference implementation comparison"
    )
    config.addinivalue_line(
        "markers", "property: mark test as mathematical property test"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="Run benchmark tests"
    )
    parser.addoption(
        "--reference",
        action="store_true",
        default=False,
        help="Run reference implementation comparison tests",
    )


def pytest_collection_modifyitems(config, items):
    """Automatically apply skip logic based on markers and options."""
    skip_benchmark = pytest.mark.skip(
        reason="Benchmark tests skipped (use --benchmark to run)"
    )
    skip_reference = pytest.mark.skip(
        reason="Reference tests skipped (use --reference to run)"
    )

    for item in items:
        # Skip benchmark tests unless explicitly requested
        if "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(skip_benchmark)

        # Skip reference tests unless explicitly requested
        if "reference" in item.keywords and not config.getoption("--reference"):
            item.add_marker(skip_reference)
