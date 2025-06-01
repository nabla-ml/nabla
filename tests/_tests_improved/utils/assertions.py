# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Test Assertion Utilities
# ===----------------------------------------------------------------------=== #

"""Custom assertion utilities for better test error reporting."""

from typing import Any

import numpy as np


def assert_arrays_close(
    actual: Any,
    expected: Any,
    rtol: float = 1e-7,
    atol: float = 1e-8,
    msg: str = "",
    check_dtype: bool = True,
) -> None:
    """Assert that two arrays are numerically close with detailed error reporting.

    Args:
        actual: The actual result (Nabla Array, numpy array, or scalar)
        expected: The expected result (numpy array or scalar)
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Additional message for assertion error
        check_dtype: Whether to check dtype compatibility
    """
    # Convert actual to numpy
    if hasattr(actual, "to_numpy"):
        actual_np = actual.to_numpy()
    else:
        actual_np = np.asarray(actual)

    # Convert expected to numpy
    expected_np = np.asarray(expected)

    # Check shapes match
    if actual_np.shape != expected_np.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual_np.shape} vs expected {expected_np.shape}. {msg}"
        )

    # Check dtype compatibility if requested
    if check_dtype:
        if not np.can_cast(actual_np.dtype, expected_np.dtype, casting="safe"):
            # Allow some flexibility for float32/float64
            if not (
                np.issubdtype(actual_np.dtype, np.floating)
                and np.issubdtype(expected_np.dtype, np.floating)
            ):
                raise AssertionError(
                    f"Dtype incompatibility: actual {actual_np.dtype} vs expected {expected_np.dtype}. {msg}"
                )

    # Check numerical closeness
    if not np.allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=True):
        # Calculate statistics for better error reporting
        diff = actual_np - expected_np
        max_abs_diff = np.max(np.abs(diff))
        max_rel_diff = np.max(np.abs(diff) / (np.abs(expected_np) + atol))

        error_msg = [
            f"Arrays not close enough (rtol={rtol}, atol={atol}). {msg}",
            f"Max absolute difference: {max_abs_diff}",
            f"Max relative difference: {max_rel_diff}",
            f"Actual shape: {actual_np.shape}, dtype: {actual_np.dtype}",
            f"Expected shape: {expected_np.shape}, dtype: {expected_np.dtype}",
        ]

        # Show small arrays in full, large arrays partially
        if actual_np.size <= 20:
            error_msg.extend(
                [
                    f"Actual:\n{actual_np}",
                    f"Expected:\n{expected_np}",
                    f"Difference:\n{diff}",
                ]
            )
        else:
            error_msg.extend(
                [
                    f"Actual (first 5 elements): {actual_np.flat[:5]}",
                    f"Expected (first 5 elements): {expected_np.flat[:5]}",
                    f"Difference (first 5 elements): {diff.flat[:5]}",
                ]
            )

        raise AssertionError("\n".join(error_msg))


def assert_shapes_equal(actual: Any, expected_shape: tuple, msg: str = "") -> None:
    """Assert that an array has the expected shape."""
    if hasattr(actual, "shape"):
        actual_shape = actual.shape
    else:
        actual_shape = np.asarray(actual).shape

    if actual_shape != expected_shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual_shape} vs expected {expected_shape}. {msg}"
        )


def assert_dtype_equal(actual: Any, expected_dtype: np.dtype, msg: str = "") -> None:
    """Assert that an array has the expected dtype."""
    if hasattr(actual, "dtype"):
        actual_dtype = actual.dtype
    elif hasattr(actual, "to_numpy"):
        actual_dtype = actual.to_numpy().dtype
    else:
        actual_dtype = np.asarray(actual).dtype

    if actual_dtype != expected_dtype:
        raise AssertionError(
            f"Dtype mismatch: actual {actual_dtype} vs expected {expected_dtype}. {msg}"
        )


def assert_gradients_close(
    actual_grads: list,
    expected_grads: list,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    msg: str = "",
) -> None:
    """Assert that gradient lists are close with detailed error reporting."""
    if len(actual_grads) != len(expected_grads):
        raise AssertionError(
            f"Gradient list length mismatch: actual {len(actual_grads)} vs expected {len(expected_grads)}. {msg}"
        )

    for i, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=False)
    ):
        try:
            assert_arrays_close(
                actual_grad, expected_grad, rtol, atol, f"Gradient {i}: {msg}"
            )
        except AssertionError as e:
            raise AssertionError(f"Gradient mismatch at index {i}: {str(e)}")


def get_tolerance_for_dtype(dtype: np.dtype, is_gradient: bool = False) -> tuple:
    """Get appropriate numerical tolerances for a given dtype.

    Args:
        dtype: NumPy dtype
        is_gradient: Whether these tolerances are for gradient computations

    Returns:
        Tuple of (rtol, atol)
    """
    # Gradients typically need looser tolerances
    if np.dtype(dtype).itemsize >= 8:  # float64 or complex128
        if is_gradient:
            return 1e-6, 1e-7
        else:
            return 1e-7, 1e-8
    else:  # float32 or complex64
        if is_gradient:
            return 1e-4, 1e-5
        else:
            return 1e-5, 1e-6


def assert_operation_preserves_shape(
    operation, input_arrays: list, expected_output_shape: tuple, msg: str = ""
) -> None:
    """Assert that an operation produces output with expected shape."""
    if len(input_arrays) == 1:
        result = operation(input_arrays[0])
    elif len(input_arrays) == 2:
        result = operation(input_arrays[0], input_arrays[1])
    else:
        result = operation(*input_arrays)

    assert_shapes_equal(result, expected_output_shape, msg)
