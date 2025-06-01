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

import numpy as np
import pytest

import nabla as nb


def test_vmap_batched_matmul():
    """Test batched matrix multiplication using nested vmap."""

    def dot(args: list[nb.Array]) -> list[nb.Array]:
        return [
            nb.reduce_sum(
                args[0] * args[1],
                axes=[0],
            )
        ]

    def mv_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(dot, [0, None])(args)

    def mm_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(mv_prod, [None, 1], [1])(args)

    def batched_matmul(args: list[nb.Array]) -> list[nb.Array]:
        return [nb.vmap(mm_prod, [0, None])([args[0], args[1]])[0]]

    # Test data
    batch_a = nb.arange((2, 3, 4), nb.DType.float32)  # Batch of 2 matrices (3x4)
    mat_b = nb.arange((4, 5), nb.DType.float32)  # Single matrix (4x5)

    # Test that expression can be compiled
    try:
        expr = nb.xpr(batched_matmul, [batch_a, mat_b])
        assert expr is not None, "Failed to compile batched matmul expression"
    except Exception as e:
        pytest.fail(f"Failed to compile batched matmul expression: {e}")

    # Execute the batched matmul
    result = batched_matmul([batch_a, mat_b])

    # Verify shape: (2, 3, 4) @ (4, 5) -> (2, 3, 5)
    expected_shape = (2, 3, 5)
    assert result[0].shape == expected_shape, (
        f"Expected result shape {expected_shape}, got {result[0].shape}"
    )

    # Verify values by computing expected result with numpy
    batch_a_np = batch_a.to_numpy()
    mat_b_np = mat_b.to_numpy()

    # Manually compute expected result
    expected = np.zeros((2, 3, 5), dtype=np.float32)
    for i in range(2):  # For each batch
        expected[i] = batch_a_np[i] @ mat_b_np

    assert np.allclose(result[0].to_numpy(), expected, rtol=1e-5), (
        "Batched matmul result doesn't match expected numpy computation"
    )


def test_simple_dot_product():
    """Test the basic dot product function used in batched matmul."""

    def dot(args: list[nb.Array]) -> list[nb.Array]:
        return [
            nb.reduce_sum(
                args[0] * args[1],
                axes=[0],
            )
        ]

    # Test with simple vectors
    a = nb.array([1.0, 2.0, 3.0], nb.DType.float32)
    b = nb.array([4.0, 5.0, 6.0], nb.DType.float32)

    result = dot([a, b])

    # Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expected = 32.0
    assert np.isclose(result[0].to_numpy().item(), expected, rtol=1e-6), (
        f"Dot product result {result[0].to_numpy().item()} doesn't match expected {expected}"
    )


def test_matrix_vector_product():
    """Test matrix-vector multiplication using vmap."""

    def dot(args: list[nb.Array]) -> list[nb.Array]:
        return [
            nb.reduce_sum(
                args[0] * args[1],
                axes=[0],
            )
        ]

    def mv_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(dot, [0, None])(args)

    # Test data: 2x3 matrix times 3-element vector
    matrix = nb.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], nb.DType.float32)
    vector = nb.array([1.0, 1.0, 1.0], nb.DType.float32)

    result = mv_prod([matrix, vector])

    # Expected: [1+2+3, 4+5+6] = [6, 15]
    expected = np.array([6.0, 15.0], dtype=np.float32)
    assert np.allclose(result[0].to_numpy(), expected, rtol=1e-6), (
        "Matrix-vector product result doesn't match expected"
    )


@pytest.mark.parametrize("batch_size,inner_dim", [(1, 2), (2, 3), (3, 4)])
def test_batched_matmul_parametrized(batch_size, inner_dim):
    """Test batched matmul with different dimensions."""

    def dot(args: list[nb.Array]) -> list[nb.Array]:
        return [nb.reduce_sum(args[0] * args[1], axes=[0])]

    def mv_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(dot, [0, None])(args)

    def mm_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(mv_prod, [None, 1], [1])(args)

    def batched_matmul(args: list[nb.Array]) -> list[nb.Array]:
        return [nb.vmap(mm_prod, [0, None])([args[0], args[1]])[0]]

    # Create test matrices
    batch_a = nb.ones((batch_size, 2, inner_dim), nb.DType.float32)
    mat_b = nb.ones((inner_dim, 3), nb.DType.float32)

    result = batched_matmul([batch_a, mat_b])

    # Expected shape: (batch_size, 2, 3)
    expected_shape = (batch_size, 2, 3)
    assert result[0].shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result[0].shape}"
    )

    # For matrices of ones, result should be all inner_dim
    expected_value = float(inner_dim)
    assert np.allclose(result[0].to_numpy(), expected_value, rtol=1e-6), (
        f"Expected all values to be {expected_value}, but got varying values"
    )


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_simple_dot_product()
    test_matrix_vector_product()
    test_vmap_batched_matmul()

    for batch_size, inner_dim in [(1, 2), (2, 3)]:
        test_batched_matmul_parametrized(batch_size, inner_dim)

    print("All vmap matmul tests passed!")
