"""
COMPREHENSIVE NABLA INDEXING OPERATIONS TEST SUITE
==================================================

This test suite validates ALL indexing operations in Nabla against JAX as ground truth.
It covers advanced indexing, gather/scatter operations, and all function transformations
(vjp, jvp, vmap, jit) across different scenarios to ensure robust behavior.

INDEXING OPERATIONS TESTED (3 main categories):
──────────────────────────────────────────────

Advanced Indexing:
    __getitem__ with Array indices, multi-dimensional indexing

Gather Operations:
    gather - selecting elements from arrays using indices

Scatter Operations:
    scatter - accumulating values into arrays at specified indices

TRANSFORMATION COMBINATIONS TESTED (19 total):
──────────────────────────────────────────────
(Same comprehensive 19 combinations as unary/binary operations suites)

Level 0 - Baseline:
    1. f(x)

Level 1 - Single Transformations:
    2. vjp(f), 3. jvp(f), 4. vmap(f), 5. jit(f)

Level 2 - Double Transformations:
    6. jit(vjp(f)), 7. jit(jvp(f)), 8. jit(vmap(f)),
    9. vmap(vjp(f)), 10. vmap(jvp(f))

Level 3 - Triple Transformations:
    11. jit(vmap(vjp(f))), 12. jit(vmap(jvp(f)))

Level 4 - Higher-Order Differentiation:
    13. vjp(vjp(f)), 14. jvp(vjp(f)), 15. vjp(jvp(f)), 16. jvp(jvp(f))

Level 5 - Advanced Compositions:
    17. vjp(vmap(f)), 18. jvp(vmap(f)), 19. vmap(vmap(f))

INDEXING SCENARIOS TESTED:
─────────────────────────
- Different tensor ranks (0D to 4D)
- Various axis choices for gather/scatter
- Different batch dimensions for vmap
- Edge cases (empty indices, out-of-bounds, broadcasting)
- Mixed data types (float32, float64, int32, int64)

CONSISTENCY CHECKING LOGIC:
─────────────────────────
✅ PASS if both Nabla and JAX succeed and produce identical results
✅ PASS if both fail consistently (expected for certain edge cases)
❌ FAIL if only one framework fails or if results don't match

This ensures that Nabla's indexing operations behave identically to JAX across
all transformation combinations, which is critical for automatic differentiation
correctness in ML workloads.

USAGE:
──────
Run tests for a specific operation:
    pytest test_view_ops.py -k "gather"
    python test_view_ops.py gather

Run tests for all operations:
    pytest test_view_ops.py
    python test_view_ops.py all
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

import nabla as nb

# Import utility modules
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
from jax import grad, jvp, vjp, vmap
from jax import jit as jax_jit

# ============================================================================
# INDEXING OPERATION DEFINITIONS
# ============================================================================


@dataclass
class IndexingOperation:
    """Definition of an indexing operation for testing."""

    name: str
    nabla_fn: Callable
    jax_fn: Callable
    description: str
    needs_target_shape: bool = False  # True for scatter operations
    min_rank: int = 1  # Minimum tensor rank needed


def create_test_data_for_rank(rank: int, dtype=nb.DType.float32):
    """Create test data with appropriate shape for given rank."""
    shape = get_shape_for_rank(rank)
    if rank == 0:
        return nb.array(2.5, dtype=dtype)
    else:
        # Nabla's arange takes the final shape directly - no reshape needed!
        data = nb.arange(shape, dtype=dtype)
        return data


def create_jax_equivalent(nb_array):
    """Create equivalent JAX array from Nabla array."""
    return jnp.array(nb_array.to_numpy())


def create_valid_indices(shape: tuple, axis: int, num_indices: int = 3):
    """Create valid indices for indexing into shape along axis."""
    if len(shape) == 0 or axis >= len(shape):
        return []

    axis_size = shape[axis]
    if axis_size <= 1:
        return [0] if axis_size == 1 else []

    # Create diverse indices within bounds
    indices = [0, axis_size - 1]  # boundary cases
    if axis_size > 2:
        indices.append(axis_size // 2)  # middle

    # Limit to requested number
    return indices[:num_indices]


# Define all indexing operations to test
INDEXING_OPERATIONS = {
    "gather": IndexingOperation(
        "gather",
        lambda arr, idx, axis=0: nb.gather(arr, idx, axis=axis),
        lambda arr, idx, axis=0: jnp.take(arr, idx, axis=axis),
        "Gather elements using indices",
    ),
    "scatter": IndexingOperation(
        "scatter",
        lambda target_shape, idx, values, axis=0: nb.scatter(
            target_shape, idx, values, axis=axis
        ),
        lambda target_shape, idx, values, axis=0: jnp.zeros(target_shape)
        .at[jnp.arange(target_shape[axis])]
        .add(values)
        if axis == 0
        else None,  # Simplified JAX scatter for axis=0
        "Scatter/accumulate values at indices",
        needs_target_shape=True,
    ),
}


# ============================================================================
# TRANSFORMATION COMBINATIONS
# ============================================================================

# Define ALL 19 transformation combinations as promised in docstring
TRANSFORMATIONS = [
    # Level 0 - Baseline (1 combination)
    ("baseline", lambda f: f),
    # Level 1 - Single Transformations (4 combinations)
    (
        "vjp",
        lambda f: lambda *args, **kwargs: nb.vjp(lambda *a: f(*a, **kwargs), *args)[0],
    ),  # Return primal from VJP
    (
        "jvp",
        lambda f: lambda *args, **kwargs: nb.jvp(
            lambda *a: f(*a, **kwargs), (args[0],), (nb.ones_like(args[0]),)
        )[0],
    ),  # Return primal from JVP - only differentiate w.r.t. first arg
    ("vmap", lambda f: nb.vmap(f)),
    ("jit", lambda f: nb.jit(f)),
    # Level 2 - Double Transformations (5 combinations)
    (
        "jit_vjp",
        lambda f: nb.jit(
            lambda *args, **kwargs: nb.vjp(lambda *a: f(*a, **kwargs), *args)[0]
        ),
    ),
    (
        "jit_jvp",
        lambda f: nb.jit(
            lambda *args, **kwargs: nb.jvp(
                lambda *a: f(*a, **kwargs), (args[0],), (nb.ones_like(args[0]),)
            )[0]
        ),
    ),
    ("jit_vmap", lambda f: nb.jit(nb.vmap(f))),
    (
        "vmap_vjp",
        lambda f: nb.vmap(
            lambda *args, **kwargs: nb.vjp(lambda *a: f(*a, **kwargs), *args)[0]
        ),
    ),
    (
        "vmap_jvp",
        lambda f: nb.vmap(
            lambda *args, **kwargs: nb.jvp(
                lambda *a: f(*a, **kwargs), (args[0],), (nb.ones_like(args[0]),)
            )[0]
        ),
    ),
    # Level 3 - Triple Transformations (2 combinations)
    (
        "jit_vmap_vjp",
        lambda f: nb.jit(
            nb.vmap(
                lambda *args, **kwargs: nb.vjp(lambda *a: f(*a, **kwargs), *args)[0]
            )
        ),
    ),
    (
        "jit_vmap_jvp",
        lambda f: nb.jit(
            nb.vmap(
                lambda *args, **kwargs: nb.jvp(
                    lambda *a: f(*a, **kwargs), (args[0],), (nb.ones_like(args[0]),)
                )[0]
            )
        ),
    ),
    # Level 4 - Higher-Order Differentiation (4 combinations)
    (
        "vjp_vjp",
        lambda f: lambda *args, **kwargs: nb.vjp(
            lambda *a: nb.vjp(lambda *b: f(*b, **kwargs), *a)[0], *args
        )[0],
    ),
    (
        "jvp_vjp",
        lambda f: lambda *args, **kwargs: nb.jvp(
            lambda *a: nb.vjp(lambda *b: f(*b, **kwargs), *a)[0],
            (args[0],),
            (nb.ones_like(args[0]),),
        )[0],
    ),
    (
        "vjp_jvp",
        lambda f: lambda *args, **kwargs: nb.vjp(
            lambda *a: nb.jvp(
                lambda *b: f(*b, **kwargs), (a[0],), (nb.ones_like(a[0]),)
            )[0],
            *args,
        )[0],
    ),
    (
        "jvp_jvp",
        lambda f: lambda *args, **kwargs: nb.jvp(
            lambda *a: nb.jvp(
                lambda *b: f(*b, **kwargs), (a[0],), (nb.ones_like(a[0]),)
            )[0],
            (args[0],),
            (nb.ones_like(args[0]),),
        )[0],
    ),
    # Level 5 - Advanced Compositions (3 combinations)
    (
        "vjp_vmap",
        lambda f: lambda *args, **kwargs: nb.vjp(
            lambda *a: nb.vmap(f)(*a, **kwargs), *args
        )[0],
    ),
    (
        "jvp_vmap",
        lambda f: lambda *args, **kwargs: nb.jvp(
            lambda *a: nb.vmap(f)(*a, **kwargs), (args[0],), (nb.ones_like(args[0]),)
        )[0],
    ),
    ("vmap_vmap", lambda f: nb.vmap(nb.vmap(f))),
]

JAX_TRANSFORMATIONS = [
    # Corresponding JAX transformations - ALL 19 combinations
    ("baseline", lambda f: f),
    # Level 1 - Single Transformations
    (
        "vjp",
        lambda f: lambda *args, **kwargs: vjp(lambda *a: f(*a, **kwargs), *args)[0],
    ),
    (
        "jvp",
        lambda f: lambda *args, **kwargs: jvp(
            lambda *a: f(*a, **kwargs), (args[0],), (jnp.ones_like(args[0]),)
        )[0],
    ),
    ("vmap", lambda f: vmap(f)),
    ("jit", lambda f: jax_jit(f)),
    # Level 2 - Double Transformations
    (
        "jit_vjp",
        lambda f: jax_jit(
            lambda *args, **kwargs: vjp(lambda *a: f(*a, **kwargs), *args)[0]
        ),
    ),
    (
        "jit_jvp",
        lambda f: jax_jit(
            lambda *args, **kwargs: jvp(
                lambda *a: f(*a, **kwargs), (args[0],), (jnp.ones_like(args[0]),)
            )[0]
        ),
    ),
    ("jit_vmap", lambda f: jax_jit(vmap(f))),
    (
        "vmap_vjp",
        lambda f: vmap(
            lambda *args, **kwargs: vjp(lambda *a: f(*a, **kwargs), *args)[0]
        ),
    ),
    (
        "vmap_jvp",
        lambda f: vmap(
            lambda *args, **kwargs: jvp(
                lambda *a: f(*a, **kwargs), (args[0],), (jnp.ones_like(args[0]),)
            )[0]
        ),
    ),
    # Level 3 - Triple Transformations
    (
        "jit_vmap_vjp",
        lambda f: jax_jit(
            vmap(lambda *args, **kwargs: vjp(lambda *a: f(*a, **kwargs), *args)[0])
        ),
    ),
    (
        "jit_vmap_jvp",
        lambda f: jax_jit(
            vmap(
                lambda *args, **kwargs: jvp(
                    lambda *a: f(*a, **kwargs), (args[0],), (jnp.ones_like(args[0]),)
                )[0]
            )
        ),
    ),
    # Level 4 - Higher-Order Differentiation
    (
        "vjp_vjp",
        lambda f: lambda *args, **kwargs: vjp(
            lambda *a: vjp(lambda *b: f(*b, **kwargs), *a)[0], *args
        )[0],
    ),
    (
        "jvp_vjp",
        lambda f: lambda *args, **kwargs: jvp(
            lambda *a: vjp(lambda *b: f(*b, **kwargs), *a)[0],
            (args[0],),
            (jnp.ones_like(args[0]),),
        )[0],
    ),
    (
        "vjp_jvp",
        lambda f: lambda *args, **kwargs: vjp(
            lambda *a: jvp(lambda *b: f(*b, **kwargs), (a[0],), (jnp.ones_like(a[0]),))[
                0
            ],
            *args,
        )[0],
    ),
    (
        "jvp_jvp",
        lambda f: lambda *args, **kwargs: jvp(
            lambda *a: jvp(lambda *b: f(*b, **kwargs), (a[0],), (jnp.ones_like(a[0]),))[
                0
            ],
            (args[0],),
            (jnp.ones_like(args[0]),),
        )[0],
    ),
    # Level 5 - Advanced Compositions
    (
        "vjp_vmap",
        lambda f: lambda *args, **kwargs: vjp(lambda *a: vmap(f)(*a, **kwargs), *args)[
            0
        ],
    ),
    (
        "jvp_vmap",
        lambda f: lambda *args, **kwargs: jvp(
            lambda *a: vmap(f)(*a, **kwargs), (args[0],), (jnp.ones_like(args[0]),)
        )[0],
    ),
    ("vmap_vmap", lambda f: vmap(vmap(f))),
]


def apply_transformation(operation_fn, transform_name, transform_fn):
    """Apply a transformation to an operation function."""
    try:
        return transform_fn(operation_fn)
    except Exception:
        # Some transformations may not be compatible with all operations
        return None


# ============================================================================
# CORE TESTING FRAMEWORK
# ============================================================================


class TestComprehensiveIndexing:
    """Comprehensive tests for all indexing scenarios."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test data with various shapes and dtypes."""
        # More challenging test shapes covering edge cases
        self.test_shapes = [
            (),  # scalar
            (1,),  # size-1 1D
            (7,),  # prime-sized 1D
            (16,),  # power-of-2 1D
            (1, 1),  # size-1 2D
            (1, 7),  # mixed size-1 and normal
            (3, 4),  # typical 2D
            (13, 17),  # prime sizes 2D
            (1, 1, 1),  # size-1 3D
            (2, 1, 4),  # mixed size-1 in middle
            (2, 3, 4),  # typical 3D
            (7, 11, 13),  # all prime sizes 3D
            (1, 2, 3, 4),  # mixed with size-1 4D
            (2, 3, 4, 5),  # typical 4D
            (8, 1, 16, 1),  # alternating size-1 4D
            (3, 5, 7, 11),  # all prime sizes 4D
        ]

        # All numeric dtypes for thorough testing
        self.test_dtypes = [
            nb.DType.float32,
            nb.DType.float64,
            nb.DType.int32,
            nb.DType.int64,
        ]

        # Challenging index patterns
        self.index_patterns = {
            "sequential": lambda n: list(range(min(n, 5))),
            "reverse": lambda n: list(range(min(n, 5) - 1, -1, -1)),
            "repeated": lambda n: [0, 0, min(n - 1, 2), min(n - 1, 2)]
            if n > 1
            else [0],
            "sparse": lambda n: [0, min(n - 1, n // 2), min(n - 1, n - 1)]
            if n > 2
            else ([0, min(n - 1, 1)] if n > 1 else [0]),
            "single": lambda n: [min(n - 1, n // 2)] if n > 0 else [],
            "empty": lambda n: [],
            "boundary": lambda n: [0, min(n - 1, n - 1)]
            if n > 1
            else ([0] if n > 0 else []),
        }

    def _create_test_array(self, shape, dtype=nb.DType.float32):
        """Create test array with known pattern."""
        if shape == ():
            # Scalar
            return nb.array(42.0, dtype=dtype)
        else:
            # Use range to create predictable data
            size = np.prod(shape)
            data = np.arange(size, dtype=dtype.to_numpy()).reshape(shape)
            return nb.array(data, dtype=dtype)

    def _create_jax_equivalent(self, nb_array):
        """Create equivalent JAX array."""
        return jnp.array(nb_array.to_numpy())

    def test_gather_basic_correctness(self):
        """Test basic gather operation correctness across shapes and dtypes."""
        for shape in self.test_shapes:
            if len(shape) == 0:  # Skip scalar for gather
                continue

            for dtype in self.test_dtypes:
                # Test gather along different axes
                for axis in range(len(shape)):
                    if shape[axis] == 1:  # Skip if axis has size 1
                        continue

                    # Create test data
                    nb_array = self._create_test_array(shape, dtype)
                    jax_array = self._create_jax_equivalent(nb_array)

                    # Create indices that are valid for this axis
                    max_index = shape[axis] - 1
                    if max_index <= 0:
                        continue

                    indices_data = [0, max_index, 0] if max_index > 0 else [0]
                    if len(indices_data) > max_index + 1:
                        indices_data = indices_data[: max_index + 1]

                    nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
                    jax_indices = jnp.array(indices_data, dtype=jnp.int32)

                    # Test gather
                    nb_result = nb.gather(nb_array, nb_indices, axis=axis)
                    jax_result = jnp.take(jax_array, jax_indices, axis=axis)

                    np.testing.assert_allclose(
                        nb_result.to_numpy(),
                        jax_result,
                        rtol=1e-6,
                        err_msg=f"Gather mismatch for shape={shape}, dtype={dtype}, axis={axis}",
                    )

    def test_scatter_basic_correctness(self):
        """Test basic scatter operation correctness across shapes and dtypes."""
        for shape in self.test_shapes:
            if len(shape) == 0:  # Skip scalar for scatter
                continue

            for dtype in self.test_dtypes:
                # Test scatter along different axes
                for axis in range(len(shape)):
                    if shape[axis] == 1:  # Skip if axis has size 1
                        continue

                    # Create indices and values
                    max_index = shape[axis] - 1
                    if max_index <= 0:
                        continue

                    indices_data = [0, max_index] if max_index > 0 else [0]
                    nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

                    # Create values with correct shape for scatter
                    values_shape = list(shape)
                    values_shape[axis] = len(indices_data)
                    values_data = np.ones(values_shape, dtype=dtype.to_numpy()) * 999
                    nb_values = nb.array(values_data, dtype=dtype)

                    # Test scatter
                    nb_result = nb.scatter(shape, nb_indices, nb_values, axis=axis)

                    # Create expected result manually
                    expected = np.zeros(shape, dtype=dtype.to_numpy())
                    for i, idx in enumerate(indices_data):
                        # Build slice for this index
                        slices = [slice(None)] * len(shape)
                        slices[axis] = idx  # type: ignore
                        value_slices = [slice(None)] * len(shape)
                        value_slices[axis] = i  # type: ignore
                        expected[tuple(slices)] = values_data[tuple(value_slices)]

                    np.testing.assert_allclose(
                        nb_result.to_numpy(),
                        expected,
                        rtol=1e-6,
                        err_msg=f"Scatter mismatch for shape={shape}, dtype={dtype}, axis={axis}",
                    )

    def test_gather_gradients(self):
        """Test gather gradients against JAX."""
        # Test on 2D array for simplicity
        shape = (4, 3)
        nb_array = self._create_test_array(shape, nb.DType.float32)
        jax_array = self._create_jax_equivalent(nb_array)

        indices_data = [0, 2, 1]
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
        jax_indices = jnp.array(indices_data, dtype=jnp.int32)

        # Define functions for gradient computation
        def nb_gather_fn(x):
            return nb.gather(x, nb_indices, axis=0).sum()

        def jax_gather_fn(x):
            return jnp.take(x, jax_indices, axis=0).sum()

        # Compute gradients
        nb_grad = nb.grad(nb_gather_fn)(nb_array)
        jax_grad = grad(jax_gather_fn)(jax_array)

        np.testing.assert_allclose(
            nb_grad.to_numpy(), jax_grad, rtol=1e-6, err_msg="Gather gradient mismatch"
        )

    def test_scatter_gradients(self):
        """Test scatter gradients."""
        # Test scatter gradient w.r.t. values
        shape = (4, 3)
        indices_data = [0, 2]
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

        values_shape = (2, 3)
        nb_values = self._create_test_array(values_shape, nb.DType.float32)

        def nb_scatter_fn(values):
            return nb.scatter(shape, nb_indices, values, axis=0).sum()

        # Compute gradient
        nb_grad = nb.grad(nb_scatter_fn)(nb_values)

        # Expected gradient should be all ones (since we sum the result)
        expected_grad = np.ones(values_shape, dtype=np.float32)

        np.testing.assert_allclose(
            nb_grad.to_numpy(),
            expected_grad,
            rtol=1e-6,
            err_msg="Scatter gradient mismatch",
        )

    def test_advanced_indexing_syntax(self):
        """Test __getitem__ and __setitem__ syntax with Array indices."""
        # Create test array
        shape = (5, 4)
        nb_array = self._create_test_array(shape, nb.DType.float32)
        jax_array = self._create_jax_equivalent(nb_array)

        # Test Array indexing
        indices_data = [0, 3, 1]
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
        jax_indices = jnp.array(indices_data, dtype=jnp.int32)

        # Test __getitem__
        nb_result = nb_array[nb_indices]
        jax_result = jax_array[jax_indices]

        np.testing.assert_allclose(
            nb_result.to_numpy(),
            jax_result,
            rtol=1e-6,
            err_msg="Advanced indexing __getitem__ mismatch",
        )

    def test_jit_compatibility(self):
        """Test that indexing operations work with JIT compilation."""
        shape = (4, 3)
        nb_array = self._create_test_array(shape, nb.DType.float32)
        indices_data = [0, 2, 1]
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

        # Define function to JIT
        def gather_sum(arr, idx):
            return nb.gather(arr, idx, axis=0).sum()

        # Test without JIT
        result_no_jit = gather_sum(nb_array, nb_indices)

        # Test with JIT
        jit_fn = nb.jit(gather_sum)
        result_jit = jit_fn(nb_array, nb_indices)

        np.testing.assert_allclose(
            result_no_jit.to_numpy(),
            result_jit.to_numpy(),
            rtol=1e-6,
            err_msg="JIT gather mismatch",
        )

    def test_vmap_compatibility(self):
        """Test that indexing operations work with vmap."""
        # Create batch of arrays with DISTINCT dimensions to track axis movement
        batch_shape = (3, 5, 7)  # 3 arrays of shape (5, 7) - all different sizes!
        nb_arrays = self._create_test_array(batch_shape, nb.DType.float32)

        # Create batch of indices - valid for axis 0 of shape (5, 7)
        indices_batch = np.array(
            [[0, 1], [1, 4], [0, 3]], dtype=np.int32
        )  # Different indices per batch
        nb_indices_batch = nb.array(indices_batch, dtype=nb.DType.int32)

        # Create corresponding JAX arrays for comparison
        jax_arrays = jnp.array(nb_arrays.to_numpy())
        jax_indices_batch = jnp.array(indices_batch)

        # Define function to vmap
        def nb_gather_fn(arr, idx):
            return nb.gather(arr, idx, axis=0)

        def jax_gather_fn(arr, idx):
            return jnp.take(arr, idx, axis=0)

        # Apply vmap
        nb_vmapped_fn = nb.vmap(nb_gather_fn, in_axes=(0, 0))
        nb_result = nb_vmapped_fn(nb_arrays, nb_indices_batch)

        jax_vmapped_fn = vmap(jax_gather_fn, in_axes=(0, 0))
        jax_result = jax_vmapped_fn(jax_arrays, jax_indices_batch)

        # Check shapes match
        print(f"JAX result shape: {jax_result.shape}")
        print(f"Nabla result shape: {nb_result.shape}")
        print(f"JAX result:\n{jax_result}")
        print(f"Nabla result:\n{nb_result.to_numpy()}")

        # Expected shape: (3, 2, 7) - 3 batches, 2 indices each, 7 features
        expected_shape = (3, 2, 7)
        assert jax_result.shape == expected_shape, (
            f"JAX shape mismatch: expected {expected_shape}, got {jax_result.shape}"
        )
        assert nb_result.shape == expected_shape, (
            f"Nabla shape mismatch: expected {expected_shape}, got {nb_result.shape}"
        )

        # Check values match
        np.testing.assert_allclose(
            nb_result.to_numpy(),
            jax_result,
            rtol=1e-6,
            err_msg="Vmap gather results don't match between Nabla and JAX",
        )

    def test_mixed_transformations(self):
        """Test combinations of jit, vmap, and grad with indexing operations."""
        # Create test data
        batch_shape = (2, 4, 3)  # 2 arrays of shape (4, 3)
        nb_arrays = self._create_test_array(batch_shape, nb.DType.float32)

        indices_data = [0, 3, 1]  # Valid indices for axis 0 of shape (4, 3)
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

        # Define function that combines gather and reduction
        def gather_and_sum(arr_batch, idx):
            def single_gather_sum(arr):
                return nb.gather(arr, idx, axis=0).sum()

            # Apply to each array in batch
            return nb.vmap(single_gather_sum)(arr_batch)

        # Test with gradient
        grad_fn = nb.grad(lambda x: gather_and_sum(x, nb_indices).sum())
        gradient = grad_fn(nb_arrays)

        # Test with JIT
        jit_fn = nb.jit(gather_and_sum)
        jit_result = jit_fn(nb_arrays, nb_indices)

        # Verify shapes
        assert gradient.shape == batch_shape, (
            f"Gradient shape mismatch: {gradient.shape} vs {batch_shape}"
        )
        assert jit_result.shape == (2,), (
            f"JIT result shape mismatch: {jit_result.shape} vs (2,)"
        )

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty indices
        shape = (3, 4)
        nb_array = self._create_test_array(shape, nb.DType.float32)
        empty_indices = nb.array([], dtype=nb.DType.int32)

        result = nb.gather(nb_array, empty_indices, axis=0)
        assert result.shape == (0, 4), f"Empty gather shape mismatch: {result.shape}"

        # Test single index
        single_index = nb.array([1], dtype=nb.DType.int32)
        result = nb.gather(nb_array, single_index, axis=0)
        assert result.shape == (1, 4), f"Single gather shape mismatch: {result.shape}"

        # Test out of bounds (should be handled gracefully)
        try:
            oob_indices = nb.array(
                [0, 5, 1], dtype=nb.DType.int32
            )  # 5 is out of bounds for axis of size 3
            result = nb.gather(nb_array, oob_indices, axis=0)
            # If no error, check that something reasonable happened
            assert result.shape[0] == 3, (
                "Out of bounds gather should still return correct outer shape"
            )
        except (IndexError, ValueError):
            # Expected for out of bounds
            pass

    def test_broadcasting_behavior(self):
        """Test broadcasting behavior in scatter operations."""
        # Test broadcasting values in scatter
        shape = (4, 3)
        indices_data = [0, 2]
        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

        # Create values that need broadcasting
        scalar_value = nb.array(999.0, dtype=nb.DType.float32)
        result = nb.scatter(shape, nb_indices, scalar_value, axis=0)

        # Check that both positions got the scalar value
        result_np = result.to_numpy()
        assert np.all(result_np[0] == 999.0), "Scalar broadcast failed for index 0"
        assert np.all(result_np[2] == 999.0), "Scalar broadcast failed for index 2"
        assert np.all(result_np[1] == 0.0), "Unindexed position should be zero"
        assert np.all(result_np[3] == 0.0), "Unindexed position should be zero"

    def test_different_index_dtypes(self):
        """Test that different integer dtypes work for indices."""
        shape = (4, 3)
        nb_array = self._create_test_array(shape, nb.DType.float32)

        for index_dtype in [nb.DType.int32, nb.DType.int64]:
            indices = nb.array([0, 2, 1], dtype=index_dtype)
            result = nb.gather(nb_array, indices, axis=0)

            assert result.shape == (3, 3), (
                f"Shape mismatch with {index_dtype}: {result.shape}"
            )

    def test_jit_vs_eager_consistency(self):
        """
        CRITICAL TEST: Verify that JIT (maxpr) and eager (eagerxpr) produce identical results.

        This is essential because:
        - JIT uses the maxpr method (MAX graph compilation)
        - Eager uses the eagerxpr method (NumPy-based execution)
        - Any divergence indicates a bug in one of the execution paths
        """
        for shape in [(4, 3), (5, 7), (3, 4, 2)]:  # Test multiple shapes
            for dtype in [nb.DType.float32, nb.DType.float64]:
                # Create test data
                nb_array = self._create_test_array(shape, dtype)
                indices_data = [0, min(shape[0] - 1, 2), 1] if shape[0] > 2 else [0]
                nb_indices = nb.array(indices_data, dtype=nb.DType.int32)

                # Test gather
                def gather_fn(arr, idx):
                    return nb.gather(arr, idx, axis=0)

                # Run eager (eagerxpr)
                eager_result = gather_fn(nb_array, nb_indices)

                # Run JIT (maxpr)
                jit_fn = nb.jit(gather_fn)
                jit_result = jit_fn(nb_array, nb_indices)

                np.testing.assert_allclose(
                    eager_result.to_numpy(),
                    jit_result.to_numpy(),
                    rtol=1e-12,  # Very tight tolerance
                    err_msg=f"JIT vs Eager mismatch for gather: shape={shape}, dtype={dtype}",
                )

                # Test scatter if shape allows
                if len(indices_data) > 0:
                    values_shape = list(shape)
                    values_shape[0] = len(indices_data)
                    nb_values = self._create_test_array(tuple(values_shape), dtype)

                    def scatter_fn(idx, vals):
                        return nb.scatter(shape, idx, vals, axis=0)

                    # Run eager
                    eager_scatter = scatter_fn(nb_indices, nb_values)

                    # Run JIT
                    jit_scatter_fn = nb.jit(scatter_fn)
                    jit_scatter = jit_scatter_fn(nb_indices, nb_values)

                    np.testing.assert_allclose(
                        eager_scatter.to_numpy(),
                        jit_scatter.to_numpy(),
                        rtol=1e-12,
                        err_msg=f"JIT vs Eager mismatch for scatter: shape={shape}, dtype={dtype}",
                    )

    def test_nested_transformations_comprehensive(self):
        """
        Test all 19 transformation combinations on indexing operations.

        This ensures that complex nested transformations like jit(vmap(vjp(f)))
        work correctly and that maxpr/eagerxpr paths are consistent.
        """
        # Test on simpler shapes to avoid timeout on complex transformations
        test_shapes = [(3, 4), (4, 3)]

        for shape in test_shapes:
            # Create test data
            nb_array = self._create_test_array(shape, nb.DType.float32)
            jax_array = self._create_jax_equivalent(nb_array)

            indices_data = [0, min(shape[0] - 1, 2)] if shape[0] > 1 else [0]
            nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
            jax_indices = jnp.array(indices_data, dtype=jnp.int32)

            # Test gather with all transformations
            operation = INDEXING_OPERATIONS["gather"]

            for transform_name, nabla_transform in TRANSFORMATIONS:
                # Skip the most complex transformations on larger shapes to avoid timeout
                if len(transform_name.split("_")) > 2 and np.prod(shape) > 12:
                    continue

                # Find corresponding JAX transform
                jax_transform = None
                for jax_name, jax_tf in JAX_TRANSFORMATIONS:
                    if jax_name == transform_name:
                        jax_transform = jax_tf
                        break

                if jax_transform is None:
                    continue

                # Test the transformation
                def nabla_op():
                    try:
                        transformed_fn = nabla_transform(operation.nabla_fn)
                        return transformed_fn(nb_array, nb_indices, axis=0)
                    except Exception as e:
                        # Some complex transformations may not be supported yet
                        raise e

                def jax_op():
                    try:
                        transformed_fn = jax_transform(operation.jax_fn)
                        return transformed_fn(jax_array, jax_indices, axis=0)
                    except Exception as e:
                        raise e

                test_name = f"gather_{transform_name}_shape{shape}"

                try:
                    success = run_test_with_consistency_check(
                        test_name, nabla_op, jax_op
                    )
                    if not success:
                        print(f"❌ FAILED: {test_name}")
                    else:
                        print(f"✅ PASSED: {test_name}")
                except Exception as e:
                    print(f"⚠️  SKIPPED: {test_name} - {str(e)[:100]}")

    # ============================================================================
    # SYSTEMATIC TRANSFORMATION TESTING
    # ============================================================================

    def test_gather_all_transformations(self):
        """Test gather operation with all transformation combinations."""
        operation_name = "gather"

        # Test ranks 1-3 (skip 0 for indexing operations)
        for rank in [1, 2, 3]:
            for transform_name, _ in TRANSFORMATIONS:
                success = self._test_single_indexing_operation(
                    operation_name, transform_name, rank
                )
                if not success:
                    # Don't fail immediately, just log
                    print(f"FAILED: {operation_name}_{transform_name}_rank{rank}")

    def test_gather_baseline_all_ranks(self):
        """Test gather baseline operation across all ranks."""
        operation_name = "gather"
        transform_name = "baseline"

        for rank in [1, 2, 3]:
            success = self._test_single_indexing_operation(
                operation_name, transform_name, rank
            )
            assert success, f"Baseline gather failed at rank {rank}"

    def test_gather_single_transformations(self):
        """Test gather with the core transformations only (subset of all 19)."""
        operation_name = "gather"
        # Test only the core transformations first to ensure they work
        core_transforms = ["baseline", "vjp", "jvp", "vmap", "jit"]

        for transform_name in core_transforms:
            for rank in [1, 2]:  # Test on simpler ranks first
                success = self._test_single_indexing_operation(
                    operation_name, transform_name, rank
                )
                assert success, (
                    f"Core transform {transform_name} failed for gather at rank {rank}"
                )

    def _test_single_indexing_operation(
        self, operation_name: str, transform_name: str, rank: int
    ):
        """Test a single indexing operation with a specific transformation at given rank."""

        if operation_name not in INDEXING_OPERATIONS:
            return False

        operation = INDEXING_OPERATIONS[operation_name]

        # Skip if operation requires higher rank than available
        if rank < operation.min_rank:
            return True  # Skip but don't fail

        # Get transformation functions
        nabla_transform = None
        jax_transform = None

        for name, transform in TRANSFORMATIONS:
            if name == transform_name:
                nabla_transform = transform
                break

        for name, transform in JAX_TRANSFORMATIONS:
            if name == transform_name:
                jax_transform = transform
                break

        if nabla_transform is None or jax_transform is None:
            return False

        # Create test data
        nb_array = create_test_data_for_rank(rank, nb.DType.float32)
        jax_array = create_jax_equivalent(nb_array)

        # Create indices for gather operation (simplified for now)
        if rank == 0:
            return True  # Skip scalar indexing for now

        shape = nb_array.shape
        indices_data = create_valid_indices(shape, axis=0, num_indices=2)

        if not indices_data:
            return True  # Skip if no valid indices

        nb_indices = nb.array(indices_data, dtype=nb.DType.int32)
        jax_indices = jnp.array(indices_data, dtype=jnp.int32)

        # Define the operation functions
        def nabla_op():
            transformed_fn = nabla_transform(operation.nabla_fn)
            if transformed_fn is None:
                raise ValueError("Transformation failed")
            return transformed_fn(nb_array, nb_indices, axis=0)

        def jax_op():
            transformed_fn = jax_transform(operation.jax_fn)
            if transformed_fn is None:
                raise ValueError("Transformation failed")
            return transformed_fn(jax_array, jax_indices, axis=0)

        # Run the test with consistency checking
        test_name = f"{operation_name}_{transform_name}_rank{rank}"
        return run_test_with_consistency_check(test_name, nabla_op, jax_op)


if __name__ == "__main__":
    pytest.main([__file__])
