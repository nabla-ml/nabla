"""
Test utilities for binary operations test suite.

This module contains reusable utility functions that are not specific to binary operations
and can be used across different test files.
"""

import gc
import threading
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

import nabla as nb
from nabla.core.execution_context import global_execution_context


def cleanup_caches():
    """Clean up both JAX and Nabla caches to prevent memory accumulation."""
    try:
        # Clear JAX's compilation cache
        jax.clear_caches()

        # Clear Nabla's execution context cache
        global_execution_context.clear()

        # Force garbage collection
        gc.collect()

    except Exception as e:
        # Don't let cleanup errors break tests, just print a warning
        print(f"Warning: Cache cleanup failed: {e}")


def cleanup_jax_caches():
    """Clean up JAX compilation caches to prevent memory exhaustion (legacy function)"""
    try:
        # Clear JAX compilation cache
        jax.clear_caches()
        # Force garbage collection
        gc.collect()
    except Exception as e:
        print(f"Warning: Failed to clear caches: {e}")


def with_timeout(timeout_seconds=30):
    """Decorator to add timeout protection to test functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = []

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Thread is still running, likely stuck
                raise TimeoutError(
                    f"Test function timed out after {timeout_seconds} seconds"
                )

            if exception:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


def jax_arange(shape, dtype=jnp.float32):
    """Create JAX array matching nabla.arange"""
    return jax.numpy.arange(np.prod(shape), dtype=dtype).reshape(shape)


def get_shape_for_rank(rank: int) -> tuple:
    """Get appropriate shape for given tensor rank"""
    if rank == 0:
        return ()  # scalar
    elif rank == 1:
        return (4,)  # vector
    elif rank == 2:
        return (2, 3)  # matrix
    elif rank == 3:
        return (2, 2, 3)  # 3D tensor
    else:
        raise ValueError(f"Unsupported rank: {rank}")


def get_test_data_for_ranks(
    rank_x: int, rank_y: int
) -> tuple[nb.Array, nb.Array, jnp.ndarray, jnp.ndarray]:
    """Get test data for specific tensor ranks"""
    shape_x = get_shape_for_rank(rank_x)
    shape_y = get_shape_for_rank(rank_y)

    # Handle scalar case specially
    if rank_x == 0:
        x_nb = 1 + nb.array(2.5)
        x_jax = 1 + jnp.array(2.5)
    else:
        x_nb = 1 + nb.ndarange(shape_x)
        x_jax = 1 + jax_arange(shape_x)

    if rank_y == 0:
        y_nb = 1 + nb.array(1.5)
        y_jax = 1 + jnp.array(1.5)
    else:
        y_nb = 1 + nb.ndarange(shape_y)
        y_jax = 1 + jax_arange(shape_y)

    return x_nb, y_nb, x_jax, y_jax


def get_rank_combinations() -> list[tuple[int, int]]:
    """Get all rank combinations to test"""
    return [
        (0, 0),  # scalar + scalar
        (1, 1),  # vector + vector
        (2, 2),  # matrix + matrix (original)
        (3, 3),  # 3D + 3D
        (0, 1),  # scalar + vector
        (0, 2),  # scalar + matrix
        (0, 3),  # scalar + 3D
        (1, 2),  # vector + matrix
        (1, 3),  # vector + 3D
        (2, 3),  # matrix + 3D
    ]


def compare_nested_structures(nabla_item, jax_item, test_name, item_path=""):
    """Recursively compare nested structures of arrays and tuples."""
    # If both are tuples, compare recursively
    if isinstance(nabla_item, tuple) and isinstance(jax_item, tuple):
        if len(nabla_item) != len(jax_item):
            print(
                f"Tuple length mismatch at {item_path}: Nabla {len(nabla_item)} vs JAX {len(jax_item)}"
            )
            return False

        for i, (nb_sub, jax_sub) in enumerate(zip(nabla_item, jax_item, strict=False)):
            if not compare_nested_structures(
                nb_sub, jax_sub, test_name, f"{item_path}[{i}]"
            ):
                return False
        return True

    # If both are arrays, compare them
    elif not isinstance(nabla_item, tuple) and not isinstance(jax_item, tuple):
        if hasattr(nabla_item, "to_numpy"):
            nb_numpy = nabla_item.to_numpy()
        else:
            nb_numpy = np.array(nabla_item)

        # Handle JAX float0 (zero tangent space) - convert to regular zeros
        if hasattr(jax_item, "dtype") and str(jax_item.dtype).startswith("[('float0"):
            jax_item = jnp.zeros_like(jax_item, dtype=jnp.float32)

        if not np.allclose(nb_numpy, np.array(jax_item), rtol=1e-5, atol=1e-6):
            print(
                f"Array mismatch at {item_path}: shapes {nb_numpy.shape} vs {np.array(jax_item).shape}"
            )
            return False
        return True

    # Structure mismatch (one is tuple, other is not)
    else:
        print(
            f"Structure mismatch at {item_path}: Nabla {type(nabla_item)} vs JAX {type(jax_item)}"
        )
        return False


def run_test_with_consistency_check(
    test_name: str, nabla_fn: Callable, jax_fn: Callable
) -> bool:
    """
    Run Nabla and JAX functions separately and check for consistency.

    Returns True if:
    - Both succeed and give same result (numeric or boolean arrays)
    - Both fail consistently

    Returns False if:
    - Only one fails
    - Results don't match when both succeed
    """
    # Import here to avoid circular imports
    try:
        from .test_errors import ErrorType, format_error_message
    except ImportError:
        from test_errors import ErrorType, format_error_message

    nabla_result = None
    jax_result = None
    nabla_error = None
    jax_error = None

    # Try Nabla
    try:
        nabla_result = nabla_fn()
    except Exception as e:
        nabla_error = str(e)

    # Try JAX
    try:
        jax_result = jax_fn()
    except Exception as e:
        jax_error = str(e)

    # Case 1: Both succeeded - check if results match
    if nabla_result is not None and jax_result is not None:
        try:
            # Use recursive comparison for nested structures
            if compare_nested_structures(nabla_result, jax_result, test_name):
                print(format_error_message(test_name, ErrorType.SUCCESS))
                return True
            else:
                print(
                    format_error_message(
                        test_name,
                        ErrorType.RESULTS_MISMATCH,
                        "Structure comparison failed",
                    )
                )
                return False

        except Exception as e:
            print(format_error_message(test_name, ErrorType.COMPARISON_FAILED, str(e)))
            return False

    # Case 2: Both failed - this is consistent behavior, so it's a pass
    elif nabla_error is not None and jax_error is not None:
        # Summarize the errors briefly
        error_summary = f"Nabla: {nabla_error[:50]}..., JAX: {jax_error[:50]}..."
        print(
            format_error_message(
                test_name, ErrorType.BOTH_FAILED_CONSISTENTLY, error_summary
            )
        )
        return True

    # Case 3: Only one failed - this is a discrepancy
    else:
        if nabla_error is not None:
            print(
                format_error_message(
                    test_name, ErrorType.NABLA_ONLY_FAILED, nabla_error
                )
            )
        else:
            print(format_error_message(test_name, ErrorType.JAX_ONLY_FAILED, jax_error))
        return False
