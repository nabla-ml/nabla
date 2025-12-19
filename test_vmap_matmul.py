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

"""Test vmap transform for the eager module - matmul-focused tests.

This file tests different ways to define matrix multiplication using vmap,
following the patterns from examples/vmap_examples.py. This is a key validation
of the vmap implementation because it:

1. Tests nested vmap patterns (dot -> mv_prod -> mm_prod -> batched_matmul)
2. Tests broadcasting (in_axes=None) with batched inputs
3. Tests different out_axes configurations
4. Compares results against numpy for correctness

The tests build up from simple primitives to complex batched operations:
- dot: element-wise multiply + reduce_sum -> scalar
- mv_prod: vmap(dot) over rows -> matrix-vector product
- mm_prod: vmap(mv_prod) over columns -> full matrix multiply
- batched_matmul: vmap(mm_prod) over batches -> batched matrix multiply
"""

import numpy as np

from eager.tensor import Tensor
from eager.vmap_trafo import vmap
from eager import reduce_sum, reshape


# =============================================================================
# Helper Functions: Building matmul from primitives
# =============================================================================

def dot(a: Tensor, b: Tensor) -> Tensor:
    """Compute dot product via element-wise multiply + reduce.
    
    a: shape (k,)
    b: shape (k,)
    result: scalar
    """
    return reduce_sum(a * b, axis=0)


def mv_prod(matrix: Tensor, vector: Tensor) -> Tensor:
    """Matrix-vector product using vmap over dot.
    
    matrix: shape (m, k)
    vector: shape (k,)
    result: shape (m,)
    
    vmap(dot, in_axes=(0, None)) maps dot over rows of matrix,
    broadcasting the vector.
    """
    return vmap(dot, in_axes=(0, None))(matrix, vector)


def mm_prod(a: Tensor, b: Tensor) -> Tensor:
    """Matrix-matrix product using vmap over mv_prod.
    
    a: shape (m, k)
    b: shape (k, n)
    result: shape (m, n)
    
    vmap(mv_prod, in_axes=(None, 1), out_axes=1) maps mv_prod over
    columns of b (axis 1), placing results in columns (out_axes=1).
    """
    return vmap(mv_prod, in_axes=(None, 1), out_axes=1)(a, b)


def batched_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Batched matrix multiplication using vmap over mm_prod.
    
    a: shape (batch, m, k)
    b: shape (k, n)
    result: shape (batch, m, n)
    
    vmap(mm_prod, in_axes=(0, None)) maps mm_prod over batch dimension of a,
    broadcasting b to all batches.
    """
    return vmap(mm_prod, in_axes=(0, None))(a, b)


# =============================================================================
# Basic Dot Product Tests
# =============================================================================

def test_dot_product():
    """Test basic dot product: element-wise multiply + sum."""
    print("=" * 50)
    print("Test: dot product")
    print("=" * 50)
    
    a = Tensor.arange(1, 4)  # [1, 2, 3]
    b = Tensor.arange(4, 7)  # [4, 5, 6]
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = dot(a, b)
    print(f"result.shape: {tuple(result.shape)}")
    
    # Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    expected = np.dot(a_np, b_np)
    
    result_np = result.to_numpy()
    print(f"result: {result_np}, expected: {expected}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        f"Dot product result {result_np} doesn't match expected {expected}"
    )
    
    print("✓ dot product works!\n")


def test_dot_product_different_values():
    """Test dot product with different values."""
    print("=" * 50)
    print("Test: dot product with different values")
    print("=" * 50)
    
    # Use ones - simpler to verify
    a = Tensor.ones((5,))
    b = Tensor.ones((5,))
    print(f"a.shape: {tuple(a.shape)}, b.shape: {tuple(b.shape)}")
    
    result = dot(a, b)
    
    # Expected: 1+1+1+1+1 = 5
    expected = 5.0
    result_np = result.to_numpy()
    print(f"result: {result_np}, expected: {expected}")
    
    assert np.isclose(result_np, expected, rtol=1e-5), (
        f"Dot product result {result_np} doesn't match expected {expected}"
    )
    
    print("✓ dot product with different values works!\n")


# =============================================================================
# Matrix-Vector Product Tests (vmap over dot)
# =============================================================================

def test_matrix_vector_product():
    """Test matrix-vector product: vmap(dot) over rows."""
    print("=" * 50)
    print("Test: matrix-vector product (vmap over dot)")
    print("=" * 50)
    
    # 2x3 matrix
    matrix = reshape(Tensor.arange(1, 7), (2, 3))  # [[1,2,3], [4,5,6]]
    # 3-element vector
    vector = Tensor.ones((3,))  # [1, 1, 1]
    
    print(f"matrix.shape: {tuple(matrix.shape)}")
    print(f"vector.shape: {tuple(vector.shape)}")
    
    result = mv_prod(matrix, vector)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (2,), f"Expected shape (2,), got {tuple(result.shape)}"
    
    # Expected: [1+2+3, 4+5+6] = [6, 15]
    matrix_np = matrix.to_numpy()
    vector_np = vector.to_numpy()
    expected = matrix_np @ vector_np
    
    result_np = result.to_numpy()
    print(f"result: {result_np}, expected: {expected}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        f"Matrix-vector product {result_np} doesn't match expected {expected}"
    )
    
    print("✓ matrix-vector product works!\n")


def test_matrix_vector_product_larger():
    """Test matrix-vector product with larger matrices."""
    print("=" * 50)
    print("Test: matrix-vector product (larger)")
    print("=" * 50)
    
    # 4x5 matrix with sequential values
    m, k = 4, 5
    matrix = reshape(Tensor.arange(0, m * k), (m, k))
    vector = Tensor.arange(1, k + 1)  # [1, 2, 3, 4, 5]
    
    print(f"matrix.shape: {tuple(matrix.shape)}")
    print(f"vector.shape: {tuple(vector.shape)}")
    
    result = mv_prod(matrix, vector)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (m,), f"Expected shape ({m},), got {tuple(result.shape)}"
    
    # Compare with numpy
    matrix_np = matrix.to_numpy()
    vector_np = vector.to_numpy()
    expected = matrix_np @ vector_np
    
    result_np = result.to_numpy()
    print(f"result[:3]: {result_np[:3]}, expected[:3]: {expected[:3]}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        f"Matrix-vector product doesn't match expected"
    )
    
    print("✓ matrix-vector product (larger) works!\n")


# =============================================================================
# Matrix-Matrix Product Tests (nested vmap)
# =============================================================================

def test_matrix_matrix_product():
    """Test matrix-matrix product: vmap(mv_prod) over columns."""
    print("=" * 50)
    print("Test: matrix-matrix product (nested vmap)")
    print("=" * 50)
    
    # 2x3 @ 3x4 -> 2x4
    m, k, n = 2, 3, 4
    
    a = reshape(Tensor.arange(0, m * k), (m, k))  # 2x3
    b = reshape(Tensor.arange(0, k * n), (k, n))  # 3x4
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = mm_prod(a, b)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (m, n), f"Expected shape ({m}, {n}), got {tuple(result.shape)}"
    
    # Compare with numpy
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    expected = a_np @ b_np
    
    result_np = result.to_numpy()
    print(f"result:\n{result_np}")
    print(f"expected:\n{expected}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        f"Matrix-matrix product doesn't match expected"
    )
    
    print("✓ matrix-matrix product works!\n")


def test_matrix_matrix_product_ones():
    """Test matrix-matrix product with ones - easy to verify."""
    print("=" * 50)
    print("Test: matrix-matrix product (ones)")
    print("=" * 50)
    
    # 3x4 @ 4x5 -> 3x5
    m, k, n = 3, 4, 5
    
    a = Tensor.ones((m, k))  # All 1s
    b = Tensor.ones((k, n))  # All 1s
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = mm_prod(a, b)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (m, n), f"Expected shape ({m}, {n}), got {tuple(result.shape)}"
    
    # All 1s @ All 1s = each element should be k (sum of k 1s)
    expected = np.full((m, n), k, dtype=np.float32)
    
    result_np = result.to_numpy()
    print(f"result:\n{result_np}")
    print(f"expected:\n{expected}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        f"Matrix-matrix product doesn't match expected"
    )
    
    print("✓ matrix-matrix product (ones) works!\n")


def test_matrix_matrix_matches_native():
    """Test that vmap-based mm_prod matches native matmul."""
    print("=" * 50)
    print("Test: mm_prod matches native matmul")
    print("=" * 50)
    
    m, k, n = 3, 4, 5
    
    a = reshape(Tensor.arange(0, m * k), (m, k))
    b = reshape(Tensor.arange(0, k * n), (k, n))
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    # vmap-based mm_prod
    result_vmap = mm_prod(a, b)
    
    # Native matmul
    result_native = a @ b
    
    print(f"result_vmap.shape: {tuple(result_vmap.shape)}")
    print(f"result_native.shape: {tuple(result_native.shape)}")
    
    result_vmap_np = result_vmap.to_numpy()
    result_native_np = result_native.to_numpy()
    
    print(f"result_vmap:\n{result_vmap_np}")
    print(f"result_native:\n{result_native_np}")
    
    assert np.allclose(result_vmap_np, result_native_np, rtol=1e-5), (
        f"vmap-based mm_prod doesn't match native matmul"
    )
    
    print("✓ mm_prod matches native matmul!\n")


# =============================================================================
# Batched Matrix Multiplication Tests
# =============================================================================

def test_batched_matmul():
    """Test batched matmul: vmap(mm_prod) over batches."""
    print("=" * 50)
    print("Test: batched matmul (triple nested vmap)")
    print("=" * 50)
    
    # (batch, m, k) @ (k, n) -> (batch, m, n)
    batch, m, k, n = 2, 3, 4, 5
    
    a = reshape(Tensor.arange(0, batch * m * k), (batch, m, k))  # 2x3x4
    b = reshape(Tensor.arange(0, k * n), (k, n))  # 4x5
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = batched_matmul(a, b)
    
    expected_shape = (batch, m, n)
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == expected_shape, (
        f"Expected shape {expected_shape}, got {tuple(result.shape)}"
    )
    
    # Compare with numpy
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    expected = np.zeros((batch, m, n), dtype=np.float32)
    for i in range(batch):
        expected[i] = a_np[i] @ b_np
    
    result_np = result.to_numpy()
    print(f"result[0]:\n{result_np[0]}")
    print(f"expected[0]:\n{expected[0]}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        "Batched matmul doesn't match expected numpy computation"
    )
    
    print("✓ batched matmul works!\n")


def test_batched_matmul_ones():
    """Test batched matmul with ones - easy to verify."""
    print("=" * 50)
    print("Test: batched matmul (ones)")
    print("=" * 50)
    
    batch, m, k, n = 3, 2, 4, 5
    
    a = Tensor.ones((batch, m, k))  # All 1s
    b = Tensor.ones((k, n))  # All 1s
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = batched_matmul(a, b)
    
    expected_shape = (batch, m, n)
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == expected_shape, (
        f"Expected shape {expected_shape}, got {tuple(result.shape)}"
    )
    
    # All 1s @ All 1s = each element should be k
    expected = np.full((batch, m, n), k, dtype=np.float32)
    
    result_np = result.to_numpy()
    print(f"result:\n{result_np}")
    print(f"expected[0]:\n{expected[0]}")
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        "Batched matmul (ones) doesn't match expected"
    )
    
    print("✓ batched matmul (ones) works!\n")


def test_batched_matmul_matches_numpy():
    """Test that vmap-based batched_matmul matches numpy computation."""
    print("=" * 50)
    print("Test: batched_matmul matches numpy")
    print("=" * 50)
    
    batch, m, k, n = 2, 3, 4, 5
    
    a = reshape(Tensor.arange(0, batch * m * k), (batch, m, k))
    b = reshape(Tensor.arange(0, k * n), (k, n))
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    # vmap-based batched_matmul
    result_vmap = batched_matmul(a, b)
    
    # Numpy computation (manual loop since numpy @ doesn't broadcast like this)
    a_np = a.to_numpy()
    b_np = b.to_numpy()
    expected = np.zeros((batch, m, n), dtype=np.float32)
    for i in range(batch):
        expected[i] = a_np[i] @ b_np
    
    result_vmap_np = result_vmap.to_numpy()
    
    print(f"result_vmap[0]:\n{result_vmap_np[0]}")
    print(f"expected[0]:\n{expected[0]}")
    
    assert np.allclose(result_vmap_np, expected, rtol=1e-5), (
        "vmap-based batched_matmul doesn't match numpy"
    )
    
    print("✓ batched_matmul matches numpy!\n")


# =============================================================================
# Alternative Implementations: Compare different vmap strategies
# =============================================================================

def test_direct_native_matmul_vs_vmap():
    """Compare native Tensor @ operator with vmap-based mm_prod."""
    print("=" * 50)
    print("Test: native @ vs vmap mm_prod")
    print("=" * 50)
    
    m, k, n = 4, 5, 6
    
    a = reshape(Tensor.arange(0, m * k), (m, k))
    b = reshape(Tensor.arange(0, k * n), (k, n))
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    # Method 1: Native @ operator
    result_native = a @ b
    
    # Method 2: vmap-based mm_prod
    result_vmap = mm_prod(a, b)
    
    result_native_np = result_native.to_numpy()
    result_vmap_np = result_vmap.to_numpy()
    
    print(f"Native result shape: {tuple(result_native.shape)}")
    print(f"vmap result shape: {tuple(result_vmap.shape)}")
    
    assert result_native_np.shape == result_vmap_np.shape, (
        f"Shape mismatch: native {result_native_np.shape} vs vmap {result_vmap_np.shape}"
    )
    
    assert np.allclose(result_native_np, result_vmap_np, rtol=1e-5), (
        "Native @ and vmap mm_prod produce different results"
    )
    
    print("✓ native @ matches vmap mm_prod!\n")


def test_alternative_batched_via_direct_vmap():
    """Test alternative: directly vmap over native matmul."""
    print("=" * 50)
    print("Test: vmap over native matmul")
    print("=" * 50)
    
    batch, m, k, n = 2, 3, 4, 5
    
    a = reshape(Tensor.arange(0, batch * m * k), (batch, m, k))
    b = reshape(Tensor.arange(0, k * n), (k, n))
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    # Method 1: vmap over native @ operator
    def matmul_fn(x, y):
        return x @ y
    
    result_vmap_native = vmap(matmul_fn, in_axes=(0, None))(a, b)
    
    # Method 2: Our primitive-based batched_matmul
    result_primitive = batched_matmul(a, b)
    
    result_vmap_native_np = result_vmap_native.to_numpy()
    result_primitive_np = result_primitive.to_numpy()
    
    print(f"vmap(native @) shape: {tuple(result_vmap_native.shape)}")
    print(f"primitive batched_matmul shape: {tuple(result_primitive.shape)}")
    
    assert np.allclose(result_vmap_native_np, result_primitive_np, rtol=1e-5), (
        "vmap(native @) doesn't match primitive-based batched_matmul"
    )
    
    print("✓ vmap(native @) matches primitive batched_matmul!\n")


# =============================================================================
# Edge Cases
# =============================================================================

def test_single_element_dot():
    """Test dot product with single-element vectors."""
    print("=" * 50)
    print("Test: single element dot product")
    print("=" * 50)
    
    a = Tensor.full((1,), 3.0)
    b = Tensor.full((1,), 4.0)
    
    print(f"a.shape: {tuple(a.shape)}, b.shape: {tuple(b.shape)}")
    
    result = dot(a, b)
    
    print(f"result.shape: {tuple(result.shape)}")
    
    expected = 12.0  # 3 * 4
    result_np = result.to_numpy()
    print(f"result: {result_np}, expected: {expected}")
    
    # Handle both scalar and 0-d array cases
    result_val = float(result_np) if result_np.ndim == 0 else float(result_np.flat[0])
    assert np.isclose(result_val, expected, rtol=1e-5), (
        f"Single element dot product {result_np} doesn't match expected {expected}"
    )
    
    print("✓ single element dot product works!\n")


def test_single_row_mv_prod():
    """Test matrix-vector product with single-row matrix."""
    print("=" * 50)
    print("Test: single row matrix-vector product")
    print("=" * 50)
    
    # 1x4 matrix @ 4-element vector -> 1-element result
    matrix = reshape(Tensor.arange(1, 5), (1, 4))  # [[1,2,3,4]]
    vector = Tensor.ones((4,))  # [1,1,1,1]
    
    print(f"matrix.shape: {tuple(matrix.shape)}")
    print(f"vector.shape: {tuple(vector.shape)}")
    
    result = mv_prod(matrix, vector)
    
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == (1,), f"Expected shape (1,), got {tuple(result.shape)}"
    
    expected = 10.0  # 1+2+3+4
    result_np = result.to_numpy()
    print(f"result: {result_np}, expected: [{expected}]")
    
    assert np.isclose(result_np[0], expected, rtol=1e-5), (
        f"Single row mv_prod {result_np} doesn't match expected [{expected}]"
    )
    
    print("✓ single row matrix-vector product works!\n")


def test_single_batch_batched_matmul():
    """Test batched matmul with batch size 1."""
    print("=" * 50)
    print("Test: single batch batched matmul")
    print("=" * 50)
    
    # (1, 2, 3) @ (3, 4) -> (1, 2, 4)
    batch, m, k, n = 1, 2, 3, 4
    
    a = Tensor.ones((batch, m, k))
    b = Tensor.ones((k, n))
    
    print(f"a.shape: {tuple(a.shape)}")
    print(f"b.shape: {tuple(b.shape)}")
    
    result = batched_matmul(a, b)
    
    expected_shape = (batch, m, n)
    print(f"result.shape: {tuple(result.shape)}")
    assert tuple(result.shape) == expected_shape, (
        f"Expected shape {expected_shape}, got {tuple(result.shape)}"
    )
    
    # All 1s @ All 1s = each element should be k
    expected = np.full((batch, m, n), k, dtype=np.float32)
    result_np = result.to_numpy()
    
    assert np.allclose(result_np, expected, rtol=1e-5), (
        "Single batch batched matmul doesn't match expected"
    )
    
    print("✓ single batch batched matmul works!\n")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print(" VMAP MATMUL TESTS")
    print(" (Testing different ways to define matmul via vmap)")
    print("=" * 60 + "\n")
    
    # Basic dot product
    test_dot_product()
    test_dot_product_different_values()
    
    # Matrix-vector product (single vmap)
    test_matrix_vector_product()
    test_matrix_vector_product_larger()
    
    # Matrix-matrix product (nested vmap)
    test_matrix_matrix_product()
    test_matrix_matrix_product_ones()
    test_matrix_matrix_matches_native()
    
    # Batched matmul (triple nested vmap)
    test_batched_matmul()
    test_batched_matmul_ones()
    test_batched_matmul_matches_numpy()
    
    # Alternative implementations
    test_direct_native_matmul_vs_vmap()
    test_alternative_batched_via_direct_vmap()
    
    # Edge cases
    test_single_element_dot()
    test_single_row_mv_prod()
    test_single_batch_batched_matmul()
    
    print("=" * 60)
    print(" ALL VMAP MATMUL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
