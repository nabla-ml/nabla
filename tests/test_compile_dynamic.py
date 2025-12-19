"""Comprehensive tests for compile transform.

Test categories (increasing complexity):
1. Basic: Single input/output, static shapes
2. Multiple inputs: 2-3 tensor inputs
3. Dynamic dims: Symbolic batch dimensions
4. Multiple outputs: Functions returning tuples
5. Mixed outputs: Tensors + static values
6. Cache behavior: Hits, misses, eviction
7. Edge cases: No tensors, side effects, kwargs
"""

from max import driver
from nabla import Tensor, compile
import numpy as np


def make_tensor(shape, value=1.0):
    """Helper to create realized tensors."""
    return Tensor(storage=driver.Tensor.from_numpy(
        np.full(shape, value, dtype=np.float32)
    ))


def run_tests():
    print("=" * 70)
    print("COMPILE TRANSFORM TEST SUITE")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1
    
    # =========================================================================
    # 1. BASIC: Single input/output, static shapes
    # =========================================================================
    print("\n1. BASIC STATIC COMPILATION")
    print("-" * 40)
    
    @compile
    def square(x):
        return x * x
    
    x = make_tensor((3, 3))
    y = square(x)
    check("Single input compiles", square.stats.misses == 1)
    check("Output shape correct", list(y.shape) == [3, 3])
    
    # Same shape → cache hit
    x2 = make_tensor((3, 3), value=2.0)
    y2 = square(x2)
    check("Same shape → cache hit", square.stats.hits == 1)
    check("Values computed correctly", np.allclose(y2.to_numpy(), 4.0))
    
    # Different shape → cache miss
    x3 = make_tensor((4, 4))
    y3 = square(x3)
    check("Different shape → cache miss", square.stats.misses == 2)
    
    # =========================================================================
    # 2. MULTIPLE INPUTS
    # =========================================================================
    print("\n2. MULTIPLE INPUTS")
    print("-" * 40)
    
    @compile
    def matmul_add(x, W, b):
        return x @ W + b
    
    x = make_tensor((2, 4))
    W = make_tensor((4, 3))
    b = make_tensor((3,))
    
    y = matmul_add(x, W, b)
    check("3 inputs compile", matmul_add.stats.misses == 1)
    check("Matmul+add shape correct", list(y.shape) == [2, 3])
    
    # Different x, same W, b → cache hit
    x2 = make_tensor((2, 4), value=2.0)
    y2 = matmul_add(x2, W, b)
    check("Different values, same shapes → hit", matmul_add.stats.hits == 1)
    
    # Different W shape → cache miss
    W2 = make_tensor((4, 5))
    b2 = make_tensor((5,))
    y3 = matmul_add(x, W2, b2)
    check("Different W shape → miss", matmul_add.stats.misses == 2)
    
    # =========================================================================
    # 3. DYNAMIC DIMS: Symbolic batch
    # =========================================================================
    print("\n3. DYNAMIC DIMENSIONS")
    print("-" * 40)
    
    @compile(dynamic_dims={0: {0: "batch"}})
    def dynamic_matmul(x, W):
        return x @ W
    
    W = make_tensor((4, 2))
    
    # First call with batch=3
    x1 = make_tensor((3, 4))
    y1 = dynamic_matmul(x1, W)
    check("Dynamic compile works", dynamic_matmul.stats.misses == 1)
    check("Output batch=3", list(y1.shape) == [3, 2])
    
    # Different batch=5 → cache HIT (dynamic!)
    x2 = make_tensor((5, 4))
    y2 = dynamic_matmul(x2, W)
    check("Different batch → cache HIT", dynamic_matmul.stats.hits == 1)
    check("Output batch=5", list(y2.shape) == [5, 2])
    
    # Different batch=10 → still cache hit
    x3 = make_tensor((10, 4))
    y3 = dynamic_matmul(x3, W)
    check("batch=10 → still cache HIT", dynamic_matmul.stats.hits == 2)
    
    # Check cache key has symbolic signature
    key = list(dynamic_matmul._cache.keys())[0]
    check("Cache key has $batch", "$batch" in str(key.tensor_sigs))
    
    # =========================================================================
    # 4. MULTIPLE DYNAMIC DIMS
    # =========================================================================
    print("\n4. MULTIPLE DYNAMIC DIMS")
    print("-" * 40)
    
    @compile(dynamic_dims={0: {0: "batch"}, 1: {0: "batch"}})
    def add_dynamic(a, b):
        return a + b
    
    a1 = make_tensor((3, 4))
    b1 = make_tensor((3, 4))
    y1 = add_dynamic(a1, b1)
    check("Two dynamic args compile", add_dynamic.stats.misses == 1)
    
    # Different batch sizes → cache hit
    a2 = make_tensor((7, 4))
    b2 = make_tensor((7, 4))
    y2 = add_dynamic(a2, b2)
    check("Same signature, different batch → hit", add_dynamic.stats.hits == 1)
    
    # =========================================================================
    # 5. MULTIPLE OUTPUTS (tuple)
    # =========================================================================
    print("\n5. MULTIPLE OUTPUTS")
    print("-" * 40)
    
    @compile
    def split_compute(x):
        doubled = x * 2
        squared = x * x
        return doubled, squared
    
    x = make_tensor((2, 3))
    d, s = split_compute(x)
    check("Tuple output compiles", split_compute.stats.misses == 1)
    check("First output correct", np.allclose(d.to_numpy(), 2.0))
    check("Second output correct", np.allclose(s.to_numpy(), 1.0))
    
    # Cache hit preserves tuple structure
    x2 = make_tensor((2, 3), value=3.0)
    d2, s2 = split_compute(x2)
    check("Tuple cache hit works", split_compute.stats.hits == 1)
    check("First output on hit", np.allclose(d2.to_numpy(), 6.0))
    check("Second output on hit", np.allclose(s2.to_numpy(), 9.0))
    
    # =========================================================================
    # 6. MIXED OUTPUTS (tensors + static)
    # =========================================================================
    print("\n6. MIXED OUTPUTS")
    print("-" * 40)
    
    @compile
    def compute_with_info(x, scale):
        result = x * scale
        return result, "metadata", 42
    
    x = make_tensor((2, 2))
    result, meta, num = compute_with_info(x, 2.0)
    check("Mixed output compiles", compute_with_info.stats.misses == 1)
    check("Tensor output correct", np.allclose(result.to_numpy(), 2.0))
    check("String preserved", meta == "metadata")
    check("Int preserved", num == 42)
    
    # =========================================================================
    # 7. STATIC ARGS (non-tensor)
    # =========================================================================
    print("\n7. STATIC ARGS")
    print("-" * 40)
    
    @compile
    def scale_by(x, factor):
        return x * factor
    
    x = make_tensor((2, 2))
    y1 = scale_by(x, 2.0)
    check("Static arg compiles", scale_by.stats.misses == 1)
    
    # Same static arg → cache hit
    y2 = scale_by(x, 2.0)
    check("Same static arg → hit", scale_by.stats.hits == 1)
    
    # Different static arg → cache miss (recompile!)
    y3 = scale_by(x, 3.0)
    check("Different static arg → miss", scale_by.stats.misses == 2)
    
    # =========================================================================
    # 8. KWARGS
    # =========================================================================  
    print("\n8. KEYWORD ARGUMENTS")
    print("-" * 40)
    
    @compile
    def with_kwargs(x, *, multiplier=1.0):
        return x * multiplier
    
    x = make_tensor((2, 2))
    y1 = with_kwargs(x, multiplier=2.0)
    check("Kwargs compile", with_kwargs.stats.misses == 1)
    
    y2 = with_kwargs(x, multiplier=2.0)
    check("Same kwargs → hit", with_kwargs.stats.hits == 1)
    
    y3 = with_kwargs(x, multiplier=5.0)
    check("Different kwargs → miss", with_kwargs.stats.misses == 2)
    
    # =========================================================================
    # 9. DYNAMIC + STATIC ARGS TOGETHER
    # =========================================================================
    print("\n9. DYNAMIC + STATIC TOGETHER")
    print("-" * 40)
    
    @compile(dynamic_dims={0: {0: "batch"}})
    def batched_scale(x, factor):
        return x * factor
    
    x1 = make_tensor((3, 4))
    y1 = batched_scale(x1, 2.0)
    check("Dynamic+static compiles", batched_scale.stats.misses == 1)
    
    # Different batch, same factor → hit
    x2 = make_tensor((7, 4))
    y2 = batched_scale(x2, 2.0)
    check("Different batch, same factor → hit", batched_scale.stats.hits == 1)
    
    # Same batch, different factor → miss
    x3 = make_tensor((7, 4))
    y3 = batched_scale(x3, 5.0)
    check("Same batch, different factor → miss", batched_scale.stats.misses == 2)
    
    # =========================================================================
    # 10. CACHE SIZE LIMIT
    # =========================================================================
    print("\n10. CACHE SIZE LIMIT")
    print("-" * 40)
    
    @compile(max_cache_size=3)
    def limited_cache(x):
        return x * 2
    
    # Fill cache with 3 different shapes
    for size in [2, 3, 4]:
        limited_cache(make_tensor((size, size)))
    
    check("Cache has 3 entries", len(limited_cache._cache) == 3)
    
    # Add 4th → should evict oldest
    limited_cache(make_tensor((5, 5)))
    check("4th entry evicts oldest", len(limited_cache._cache) == 3)
    check("Stats show 4 misses", limited_cache.stats.misses == 4)
    
    # =========================================================================
    # 11. FULLGRAPH MODE (would error on side effects - just test compilation)
    # =========================================================================
    print("\n11. FULLGRAPH MODE")
    print("-" * 40)
    
    @compile(fullgraph=True)
    def strict_fn(x):
        return x * x
    
    x = make_tensor((2, 2))
    y = strict_fn(x)
    check("Fullgraph compiles cleanly", strict_fn.stats.misses == 1)
    
    # =========================================================================  
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
