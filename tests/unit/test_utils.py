# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Shared utilities and constants for unit tests."""

import numpy as np
import pytest

try:
    import jax

    # Enable 64-bit precision in JAX to match numpy float64 behavior
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# --- Test Constants ---
DTYPES_TO_TEST = [np.float32, np.float64]

# Tolerance configurations
RTOL_F32_VAL, ATOL_F32_VAL = 1e-5, 1e-6
RTOL_F64_VAL, ATOL_F64_VAL = 1e-7, 1e-8
RTOL_F32_GRAD, ATOL_F32_GRAD = 1e-4, 1e-5
RTOL_F64_GRAD, ATOL_F64_GRAD = 1e-6, 1e-7

# Shape configurations
UNARY_SHAPES = [
    ((1,), "scalar"),
    ((3,), "vector"),
    ((2, 3), "matrix"),
]

BINARY_SHAPES_BROADCAST = [
    ((1,), (1,), "scalar_scalar"),
    ((3,), (3,), "vector_vector"),
    ((2, 3), (2, 3), "matrix_matrix"),
    ((1,), (2, 3), "scalar_matrix"),
    ((3,), (2, 3), "vector_matrix_row_bcast"),
    ((2, 1), (2, 3), "vector_matrix_col_bcast"),
    ((4,), (2, 3, 4), "vector_tensor_bcast"),
]

MATMUL_SHAPES = [
    ((2, 3), (3, 4), "matmul_2x3_3x4"),
    ((2, 3, 4), (2, 4, 5), "matmul_batch_2x3x4_2x4x5"),
    # ((3,4), (4,), "matmul_matrix_vector"), TODO!
    # ((4,), (4,3), "matmul_vector_matrix"), TODO!
    # ((3,), (3,), "matmul_vector_vector"), TODO! IndexError in validation
]

# Simplified shapes for transform tests
SIMPLE_UNARY_SHAPES = [((2, 3),)]
SIMPLE_BINARY_SHAPES = [((3,), (3,)), ((2, 3), (2, 3))]
SIMPLE_MATMUL_SHAPES = [((2, 3), (3, 4))]


# --- Utility Functions ---
def get_tolerances(dtype, is_gradient=False):
    """Get appropriate tolerances for the given dtype."""
    is_complex = np.issubdtype(dtype, np.complexfloating)
    is_float = np.issubdtype(dtype, np.floating)

    if is_complex or is_float:
        if np.dtype(dtype).itemsize >= 8:  # float64 or complex128
            return (
                (RTOL_F64_GRAD, ATOL_F64_GRAD)
                if is_gradient
                else (RTOL_F64_VAL, ATOL_F64_VAL)
            )
        else:  # float32 or complex64
            return (
                (RTOL_F32_GRAD, ATOL_F32_GRAD)
                if is_gradient
                else (RTOL_F32_VAL, ATOL_F32_VAL)
            )
    return (0, 0)  # For exact comparison (e.g., int, bool)


def generate_test_data(
    shape, dtype, low=-5.0, high=5.0, ensure_positive=False, for_binary_op_rhs=False
):
    """Generate random test data with optional constraints."""
    if ensure_positive:
        low = max(0.1, low)
        high = max(low + 0.1, high)

    data = np.random.uniform(low, high, size=shape).astype(dtype)

    if for_binary_op_rhs:  # Avoid zeros for division/power operations
        data[np.abs(data) < 1e-3] = np.sign(data[np.abs(data) < 1e-3] + 1e-3) * 0.1
        if np.issubdtype(dtype, np.integer):
            data[data == 0] = 1

    return data


def allclose_recursive(nabla_result, jax_result, rtol, atol):
    """Recursively compare nested structures with proper error reporting."""
    if isinstance(nabla_result, list | tuple) and isinstance(jax_result, list | tuple):
        assert len(nabla_result) == len(jax_result), (
            f"Result structures have different lengths: Nabla {len(nabla_result)}, JAX {len(jax_result)}"
        )
        for i, (nb_item, jax_item) in enumerate(
            zip(nabla_result, jax_result, strict=False)
        ):
            assert allclose_recursive(nb_item, jax_item, rtol, atol), (
                f"Mismatch in item {i} of recursive structure."
            )
        return True

    # Convert to numpy arrays
    if hasattr(nabla_result, "to_numpy"):
        nabla_np = nabla_result.to_numpy()
    elif hasattr(nabla_result, "__array__"):
        nabla_np = np.array(nabla_result)
    else:
        nabla_np = nabla_result

    if hasattr(jax_result, "to_numpy"):
        jax_np = jax_result.to_numpy()
    elif hasattr(jax_result, "__array__") and hasattr(jax_result, "shape"):
        jax_np = np.asarray(jax_result)
    else:
        jax_np = np.array(jax_result)

    assert nabla_np.shape == jax_np.shape, (
        f"Shape mismatch: Nabla {nabla_np.shape}, JAX {jax_np.shape}.\nNabla: {nabla_np}\nJAX: {jax_np}"
    )

    is_close = np.allclose(nabla_np, jax_np, rtol=rtol, atol=atol, equal_nan=True)
    if not is_close:
        print(f"Numerical mismatch detected with rtol={rtol}, atol={atol}:")
        print(f"Nabla ({nabla_np.dtype}):\n{nabla_np}")
        print(f"JAX ({jax_np.dtype}):\n{jax_np}")
        print(f"Difference:\n{nabla_np - jax_np}")
    return is_close


# Skip marker for when JAX is not available
requires_jax = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
