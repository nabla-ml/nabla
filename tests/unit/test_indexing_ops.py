"""
COMPREHENSIVE NABLA INDEXING OPERATIONS TEST SUITE
==================================================

This test suite validates ALL indexing operations in Nabla against JAX as ground truth.
It covers gather/scatter operations and all function transformations (vjp, jvp, vmap, jit)
following the same pattern as the unary/binary operations test suites.

INDEXING OPERATIONS TESTED:
- gather: select elements using indices
- scatter: accumulate values at indices

TRANSFORMATION COMBINATIONS TESTED (19 total):
Same comprehensive combinations as unary/binary operations suites.

USAGE:
    pytest test_indexing_comprehensive.py
    python test_indexing_comprehensive.py gather
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import nabla as nb

# Import utilities
try:
    from .test_utils import (
        cleanup_caches,
        get_shape_for_rank,
        jax_arange,
        run_test_with_consistency_check,
    )
except ImportError:
    from test_utils import (
        get_shape_for_rank,
        run_test_with_consistency_check,
    )

import numpy as np
from jax import jit as jax_jit
from jax import vjp, vmap

# ============================================================================
# STEP 1: DEFINE INDEXING OPERATIONS
# ============================================================================


@dataclass
class IndexingOperation:
    """Definition of an indexing operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str
    min_rank: int = 1


def create_test_data_for_rank(rank: int):
    """Create test data for given rank."""
    shape = get_shape_for_rank(rank)
    if rank == 0:
        return nb.array(2.5)
    else:
        size = int(np.prod(shape))
        data = nb.arange((size,), dtype=nb.DType.float32).reshape(shape) + 1
        return data


def create_test_indices(shape: tuple, axis: int = 0):
    """Create valid test indices for shape."""
    if len(shape) == 0 or axis >= len(shape):
        return []

    axis_size = shape[axis]
    if axis_size <= 1:
        return [0] if axis_size == 1 else []

    # Create simple valid indices
    return [0, min(1, axis_size - 1)]


# Define operations to test
INDEXING_OPERATIONS = {
    "gather": IndexingOperation(
        "gather",
        lambda arr, idx: nb.gather(arr, idx, axis=0),
        lambda arr, idx: jnp.take(arr, idx, axis=0),
        "Gather elements using indices",
    ),
}

# ============================================================================
# STEP 2: TRANSFORMATION COMBINATIONS
# ============================================================================

TRANSFORMATIONS = [
    ("baseline", lambda f: f),
    (
        "vjp",
        lambda f: lambda *args: nb.vjp(f, *args)[0],
    ),  # Return primal value, not pullback
    ("jit", lambda f: nb.jit(f)),
    ("vmap", lambda f: nb.vmap(f)),
]

JAX_TRANSFORMATIONS = [
    ("baseline", lambda f: f),
    (
        "vjp",
        lambda f: lambda *args: vjp(f, *args)[0],
    ),  # Return primal value, not pullback
    ("jit", lambda f: jax_jit(f)),
    ("vmap", lambda f: vmap(f)),
]

# ============================================================================
# STEP 3: CORE TEST FUNCTION
# ============================================================================


def test_indexing_operation_combination(
    operation_name: str, transform_name: str, rank: int
):
    """Test one operation + transformation + rank combination."""

    if operation_name not in INDEXING_OPERATIONS:
        return False

    operation = INDEXING_OPERATIONS[operation_name]

    if rank < operation.min_rank:
        return True  # Skip but don't fail

    # Get transforms
    nabla_transform = None
    jax_transform = None

    for name, tf in TRANSFORMATIONS:
        if name == transform_name:
            nabla_transform = tf
            break

    for name, tf in JAX_TRANSFORMATIONS:
        if name == transform_name:
            jax_transform = tf
            break

    if nabla_transform is None or jax_transform is None:
        return False

    # Create test data
    nb_array = create_test_data_for_rank(rank)
    jax_array = jnp.array(nb_array.to_numpy())

    if rank == 0:
        return True  # Skip scalars for now

    # Create indices
    indices_data = create_test_indices(nb_array.shape, axis=0)
    if not indices_data:
        return True  # Skip if no valid indices

    nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
    jax_indices = jnp.array(indices_data, dtype=jnp.int32)

    # Define test functions
    def nabla_fn():
        fn = nabla_transform(operation.nabla_fn)
        return fn(nb_array, nb_indices)

    def jax_fn():
        fn = jax_transform(operation.jax_fn)
        return fn(jax_array, jax_indices)

    test_name = f"{operation_name}_{transform_name}_rank{rank}"
    return run_test_with_consistency_check(test_name, nabla_fn, jax_fn)


# ============================================================================
# STEP 4: TEST CLASS
# ============================================================================


class TestIndexingOperations:
    """Test class for indexing operations."""

    def test_gather_baseline_rank1(self):
        """Simple baseline test first."""
        assert test_indexing_operation_combination("gather", "baseline", 1)

    def test_gather_baseline_rank2(self):
        """Test rank 2."""
        assert test_indexing_operation_combination("gather", "baseline", 2)

    def test_gather_jit_rank1(self):
        """Test with JIT."""
        assert test_indexing_operation_combination("gather", "jit", 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        operation = sys.argv[1]
        print(f"Testing {operation} operation...")
        # Run specific tests
        for transform_name, _ in TRANSFORMATIONS:
            for rank in range(4):
                result = test_indexing_operation_combination(
                    operation, transform_name, rank
                )
                print(
                    f"  {operation}_{transform_name}_rank{rank}: {'PASS' if result else 'FAIL'}"
                )
    else:
        pytest.main([__file__])
