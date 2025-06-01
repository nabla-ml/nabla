# ===----------------------------------------------------------------------=== #
# Nabla 2025 - Test Data Generators
# ===----------------------------------------------------------------------=== #

"""Test data generation utilities."""

from typing import Optional

import numpy as np

import nabla as nb


def generate_test_array(
    shape: tuple[int, ...],
    dtype: str = "float32",
    low: float = -5.0,
    high: float = 5.0,
    seed: Optional[int] = None,
    ensure_positive: bool = False,
    avoid_zeros: bool = False,
    requires_grad: bool = False,
    for_binary_op_rhs: bool = False,
) -> nb.Array:
    """Generate a test array with specified properties.

    Args:
        shape: Shape of the array to generate
        dtype: Data type for the array (string like "float32", "float64")
        low: Lower bound for random values
        high: Upper bound for random values
        seed: Random seed for reproducibility
        ensure_positive: If True, ensure all values are positive
        avoid_zeros: If True, avoid values close to zero (useful for division)
        requires_grad: If True, enable gradient tracking for the array
        for_binary_op_rhs: If True, adjust values for right-hand side of binary ops

    Returns:
        Nabla Array with random test data
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert string dtype to numpy dtype
    if isinstance(dtype, str):
        np_dtype = getattr(np, dtype)
    else:
        np_dtype = dtype

    # Handle special case for binary operation right-hand side
    if for_binary_op_rhs:
        if ensure_positive:
            # For operations like division, avoid very small positive values
            low = max(0.5, low)
            high = max(low + 0.5, high)
        else:
            # Avoid values close to zero for division safety
            avoid_zeros = True

    if ensure_positive:
        low = max(0.1, low)
        high = max(low + 0.1, high)

    data = np.random.uniform(low, high, size=shape).astype(np_dtype)

    if avoid_zeros:
        # Replace values close to zero
        near_zero_mask = np.abs(data) < 1e-3
        data[near_zero_mask] = np.sign(data[near_zero_mask]) * 0.1
        # Handle case where sign is also zero
        data[data == 0] = 0.1

    # Create nabla Array from numpy data
    array = nb.Array.from_numpy(data)

    # Enable gradient tracking if requested
    if requires_grad:
        array.requires_grad_(True)

    return array


def generate_test_data_numpy(
    shape: tuple[int, ...], dtype: str = "float32", **kwargs
) -> np.ndarray:
    """Generate test data as NumPy array (for reference comparisons)."""
    # Remove gradient-specific parameters before creating array
    numpy_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["requires_grad", "for_binary_op_rhs"]
    }
    array = generate_test_array(shape, dtype, **numpy_kwargs)
    return array.to_numpy()


def generate_compatible_shapes(
    operation: str, max_dims: int = 3, max_size: int = 10, seed: Optional[int] = None
) -> tuple[tuple[int, ...], ...]:
    """Generate compatible shapes for specific operations.

    Args:
        operation: Type of operation ('matmul', 'broadcast', 'elementwise')
        max_dims: Maximum number of dimensions
        max_size: Maximum size per dimension
        seed: Random seed

    Returns:
        Tuple of compatible shapes
    """
    if seed is not None:
        np.random.seed(seed)

    if operation == "matmul":
        # Generate compatible matrix shapes
        k = np.random.randint(1, max_size + 1)
        m = np.random.randint(1, max_size + 1)
        n = np.random.randint(1, max_size + 1)

        if max_dims <= 2:
            return (m, k), (k, n)
        else:
            # Add batch dimensions
            batch_size = np.random.randint(1, 4)
            return (batch_size, m, k), (batch_size, k, n)

    elif operation == "broadcast":
        # Generate broadcastable shapes
        base_shape = tuple(np.random.randint(1, max_size + 1, size=max_dims))
        # Create a shape that can broadcast with base_shape
        broadcast_shape = list(base_shape)
        for i in range(len(broadcast_shape)):
            if np.random.random() < 0.3:  # 30% chance to make dimension 1
                broadcast_shape[i] = 1
        return base_shape, tuple(broadcast_shape)

    elif operation == "elementwise":
        # Generate identical shapes
        shape = tuple(np.random.randint(1, max_size + 1, size=max_dims))
        return shape, shape

    else:
        raise ValueError(f"Unknown operation type: {operation}")


# Common test shapes for different operations
SMALL_SHAPES = [
    (1,),  # scalar
    (3,),  # small vector
    (2, 3),  # small matrix
    (2, 3, 4),  # small tensor
]

MEDIUM_SHAPES = [
    (10,),
    (10, 10),
    (5, 10, 15),
    (2, 5, 10, 15),
]

MATMUL_COMPATIBLE_SHAPES = [
    ((2, 3), (3, 4)),
    ((1, 3), (3, 1)),
    ((4, 5), (5, 2)),
    ((2, 3, 4), (2, 4, 5)),  # Batched
]

BROADCAST_COMPATIBLE_SHAPES = [
    ((3,), (3,)),  # identical
    ((1,), (3,)),  # scalar broadcast
    ((3, 1), (3, 4)),  # column broadcast
    ((1, 4), (3, 4)),  # row broadcast
    ((3, 1, 4), (3, 2, 4)),  # middle dimension broadcast
]
