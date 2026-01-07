"""Test dynamic batch sizes with vmap + compile.

This file documents and tests the RECOMMENDED PATTERN for dynamic batching:
    @compile(dynamic_dims={arg_idx: {dim_idx: "name"}})
    @vmap(in_axes=...)
    def fn(...): ...

This pattern already works without any special vmap modifications!
The compile decorator's dynamic_dims handles the symbolic dimensions,
and vmap just does axis transformation.

NOTE: JAX's jit + vmap also recompiles for each batch size by default.
Their solution requires jax.export with symbolic_shape, which is for
export-time only. Our compile(dynamic_dims=...) works at runtime!
"""

from nabla.core.tensor import Tensor
from nabla.transforms.vmap import vmap
from nabla.transforms.compile import compile
from nabla.core.compute_graph import GRAPH


def reset_graph():
    """Reset global graph state to avoid pollution between tests."""
    GRAPH._reset(GRAPH.context, 0)


# =============================================================================
# RECOMMENDED PATTERN: compile(dynamic_dims=...) + vmap
# =============================================================================

def test_compile_dynamic_dims_with_vmap():
    """Recommended pattern: compile(dynamic_dims=...) + vmap.
    
    This enables a single compiled model to handle any batch size.
    """
    print("=" * 60)
    print("Test: compile(dynamic_dims=...) + vmap (recommended pattern)")
    print("=" * 60)
    
    reset_graph()
    
    # Explicit dynamic_dims in compile
    @compile(dynamic_dims={0: {0: "batch"}})
    @vmap(in_axes=0)
    def batched_square(x):
        return x * x
    
    # First call with batch size 5
    x1 = Tensor.arange(1, 6)  # [1, 2, 3, 4, 5]
    print(f"x1.shape: {tuple(x1.shape)}")
    
    result1 = batched_square(x1)
    print(f"result1.shape: {tuple(result1.shape)}")
    
    stats1 = batched_square.stats
    print(f"After first call: misses={stats1.misses}, hits={stats1.hits}")
    
    # Second call with DIFFERENT batch size (8)
    x2 = Tensor.arange(1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"x2.shape: {tuple(x2.shape)}")
    
    result2 = batched_square(x2)
    print(f"result2.shape: {tuple(result2.shape)}")
    
    stats2 = batched_square.stats
    print(f"After second call: misses={stats2.misses}, hits={stats2.hits}")
    
    # Should be 1 miss, 1 hit (same compiled model for different batch sizes)
    assert stats2.misses == 1, f"Expected 1 miss, got {stats2.misses}"
    assert stats2.hits == 1, f"Expected 1 hit, got {stats2.hits}"
    
    print("✓ compile(dynamic_dims=...) + vmap works correctly!")
    print()


def test_compile_dynamic_dims_matmul():
    """Recommended pattern with batched matrix multiplication."""
    print("=" * 60)
    print("Test: compile(dynamic_dims=...) + vmap matmul")
    print("=" * 60)
    
    reset_graph()
    
    # Only x has dynamic batch (arg 0, dim 0)
    @compile(dynamic_dims={0: {0: "batch"}})
    @vmap(in_axes=(0, None))
    def batched_matmul(x, W):
        return x @ W
    
    # Weight matrix (static, broadcast)
    W = Tensor.ones((4, 8))
    
    # First call with batch=3
    x1 = Tensor.ones((3, 4))
    print(f"x1.shape: {tuple(x1.shape)}, W.shape: {tuple(W.shape)}")
    
    result1 = batched_matmul(x1, W)
    print(f"result1.shape: {tuple(result1.shape)}")
    
    # Second call with batch=7 (different!)
    x2 = Tensor.ones((7, 4))
    print(f"x2.shape: {tuple(x2.shape)}")
    
    result2 = batched_matmul(x2, W)
    print(f"result2.shape: {tuple(result2.shape)}")
    
    stats = batched_matmul.stats
    print(f"Stats: misses={stats.misses}, hits={stats.hits}")
    
    assert stats.hits == 1, f"Expected 1 hit, got {stats.hits}"
    
    print("✓ compile(dynamic_dims=...) + vmap matmul works!")
    print()


def test_compile_dynamic_dims_multi_output():
    """Recommended pattern with multiple outputs."""
    print("=" * 60)
    print("Test: compile(dynamic_dims=...) + vmap multi-output")
    print("=" * 60)
    
    reset_graph()
    
    @compile(dynamic_dims={0: {0: "batch"}})
    @vmap(in_axes=0)
    def split_process(x):
        return x * 2, x + 1
    
    # First call with batch=4
    x1 = Tensor.ones((4, 3))
    result1_a, result1_b = split_process(x1)
    print(f"x1.shape: {tuple(x1.shape)}")
    print(f"results: {tuple(result1_a.shape)}, {tuple(result1_b.shape)}")
    
    # Second call with batch=10
    x2 = Tensor.ones((10, 3))
    result2_a, result2_b = split_process(x2)
    print(f"x2.shape: {tuple(x2.shape)}")
    print(f"results: {tuple(result2_a.shape)}, {tuple(result2_b.shape)}")
    
    stats = split_process.stats
    print(f"Stats: misses={stats.misses}, hits={stats.hits}")
    
    assert stats.hits == 1, f"Expected 1 hit, got {stats.hits}"
    
    print("✓ compile(dynamic_dims=...) + vmap multi-output works!")
    print()


# =============================================================================
# COMPARISON: Without dynamic_dims (Recompiles)
# =============================================================================

def test_without_dynamic_dims_recompiles():
    """Show that without dynamic_dims, different batch sizes cause recompilation."""
    print("=" * 60)
    print("Test: without dynamic_dims (recompilation)")
    print("=" * 60)
    
    reset_graph()
    
    @compile  # No dynamic_dims
    @vmap(in_axes=0)
    def batched_square_static(x):
        return x * x
    
    # First call with batch=5
    x1 = Tensor.arange(1, 6)
    result1 = batched_square_static(x1)
    print(f"x1.shape: {tuple(x1.shape)} -> result: {tuple(result1.shape)}")
    
    # Second call with DIFFERENT batch=8
    x2 = Tensor.arange(1, 9)
    result2 = batched_square_static(x2)
    print(f"x2.shape: {tuple(x2.shape)} -> result: {tuple(result2.shape)}")
    
    stats = batched_square_static.stats
    print(f"Stats: misses={stats.misses}, hits={stats.hits}")
    
    # Should be 2 misses (recompiled for different batch size)
    assert stats.misses == 2, f"Expected 2 misses (recompilation), got {stats.misses}"
    assert stats.hits == 0, f"Expected 0 hits, got {stats.hits}"
    
    print("✓ Without dynamic_dims: different batch sizes cause recompilation")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(" DYNAMIC BATCH SIZE TESTS (vmap + compile)")
    print("=" * 70 + "\n")
    
    print("--- RECOMMENDED PATTERN: compile(dynamic_dims=...) + vmap ---\n")
    test_compile_dynamic_dims_with_vmap()
    test_compile_dynamic_dims_matmul()
    test_compile_dynamic_dims_multi_output()
    
    print("\n--- COMPARISON (Without dynamic_dims) ---\n")
    test_without_dynamic_dims_recompiles()
    
    print("=" * 70)
    print(" ALL DYNAMIC BATCH TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
