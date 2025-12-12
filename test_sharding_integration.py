#!/usr/bin/env python3
"""Integration tests for Shardy sharding integration into eager module.

These tests validate that:
1. Metadata caching works for all tensor operations
2. Sharding annotations can be attached to tensors
3. The sharding compiler can walk the graph and access metadata
4. Shardy propagation runs correctly

Run with: python test_sharding_integration.py
"""

from eager import (
    Tensor, add, matmul,
    DeviceMesh, DimSpec, ShardingSpec,
    compile_with_sharding,
    get_topological_order,
)
from eager.multi_output_ops import split


def test_metadata_caching_single_output():
    """Test that single-output ops cache metadata correctly."""
    print("\n=== Test: Metadata Caching (Single Output) ===")
    
    x = Tensor.zeros((4, 8), traced=True)
    y = Tensor.ones((4, 8), traced=True)
    z = add(x, y)
    
    # Verify cached metadata
    assert z._impl.cached_shape is not None, "cached_shape should be set"
    assert z._impl.cached_dtype is not None, "cached_dtype should be set"
    assert z._impl.cached_device is not None, "cached_device should be set"
    
    print(f"  z.cached_shape: {z._impl.cached_shape}")
    print(f"  z.cached_dtype: {z._impl.cached_dtype}")
    print("  ✓ Single output caching works!")


def test_metadata_caching_multi_output():
    """Test that multi-output ops cache metadata correctly."""
    print("\n=== Test: Metadata Caching (Multi Output) ===")
    
    x = Tensor.zeros((4, 8), traced=True)
    a, b = split(x, num_splits=2, axis=0)
    
    # Both outputs should have cached metadata
    assert a._impl.cached_shape is not None, "a.cached_shape should be set"
    assert b._impl.cached_shape is not None, "b.cached_shape should be set"
    
    print(f"  a.cached_shape: {a._impl.cached_shape}")
    print(f"  b.cached_shape: {b._impl.cached_shape}")
    print("  ✓ Multi-output caching works!")


def test_sharding_annotation():
    """Test that sharding annotations can be attached to tensors."""
    print("\n=== Test: Sharding Annotation ===")
    
    # Create a 2x4 mesh (8 devices total)
    mesh = DeviceMesh("cluster", (2, 4), ("dp", "tp"))
    print(f"  Mesh: {mesh}")
    
    # Create a sharding spec: shard dim 0 on "dp", dim 1 on "tp"
    spec = ShardingSpec(
        mesh,
        [DimSpec(["dp"]), DimSpec(["tp"])],  # 2D tensor
    )
    print(f"  Spec: {spec}")
    
    # Attach to tensor
    x = Tensor.zeros((8, 16), traced=True)
    x._impl.sharding = spec
    
    assert x._impl.sharding is not None, "sharding should be set"
    assert x._impl.sharding.mesh.name == "cluster", "mesh name should match"
    
    print("  ✓ Sharding annotation works!")


def test_graph_walking_with_sharding():
    """Test that we can walk the graph and access sharding info."""
    print("\n=== Test: Graph Walking with Sharding ===")
    
    # Create a simple computation graph
    x = Tensor.zeros((4, 8), traced=True)
    y = Tensor.ones((4, 8), traced=True)
    z = add(x, y)
    w = add(z, x)  # Reuse x to test DAG structure
    
    # Create mesh and attach sharding to input
    mesh = DeviceMesh("test", (2,), ("x",))
    x._impl.sharding = ShardingSpec(mesh, [DimSpec(["x"]), DimSpec([])])
    
    # Walk the graph
    all_impls = get_topological_order(w._impl)
    
    print(f"  Graph has {len(all_impls)} nodes")
    for i, impl in enumerate(all_impls):
        has_sharding = "yes" if impl.sharding else "no"
        shape_str = impl.cached_shape if impl.cached_shape else impl.logical_shape
        print(f"    {i}: op={impl.op_name or 'leaf'}, shape={shape_str}, sharding={has_sharding}")
    
    # Verify we can access all metadata
    for impl in all_impls:
        if impl.op is not None:  # Skip leaf tensors
            assert impl.cached_shape is not None, f"Missing cached_shape for {impl.op_name}"
    
    print("  ✓ Graph walking with sharding works!")


def test_compile_with_sharding_validation():
    """Test that compile_with_sharding validates inputs correctly."""
    print("\n=== Test: Compile with Sharding Validation ===")
    
    mesh = DeviceMesh("test", (2,), ("x",))
    
    # Test 1: Untraced tensor should fail
    untraced = Tensor.zeros((4,), traced=False)
    try:
        compile_with_sharding([untraced], mesh)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "traced" in str(e).lower()
        print("  ✓ Correctly rejects untraced tensors")
    
    # Test 2: Traced tensor with metadata should pass validation
    # (but fail at NotImplementedError since we haven't implemented the compiler)
    traced = Tensor.zeros((4,), traced=True)
    z = add(traced, traced)  # This caches metadata
    try:
        compile_with_sharding([z], mesh)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "not yet implemented" in str(e).lower()
        print("  ✓ Correctly reaches NotImplementedError (validation passed)")


def test_shardy_propagation_standalone():
    """Test that Shardy propagation works independently."""
    print("\n=== Test: Shardy Propagation (Standalone) ===")
    
    from eager.sharding import (
        DeviceMesh, DimSpec, ShardingSpec,
    )
    from eager.sharding_propagation import (
        OpShardingRule, propagate_sharding,
    )
    
    # Create a simple 1D mesh
    mesh = DeviceMesh("1d", (4,), ("x",))
    
    # Create input and output specs for a simple identity op
    in_spec = ShardingSpec(mesh, [DimSpec(["x"], is_open=False)])  # Sharded on x
    out_spec = ShardingSpec(mesh, [DimSpec([], is_open=True)])     # Open (unknown)
    
    # Create rule: identity (m) -> (m)
    rule = OpShardingRule([{0: ["m"]}], [{0: ["m"]}], {"m": 4})
    
    # Propagate
    changed = propagate_sharding(rule, [in_spec], [out_spec])
    
    print(f"  Input spec:  {in_spec}")
    print(f"  Output spec: {out_spec}")
    print(f"  Changed: {changed}")
    
    # Output should now have "x" sharding propagated from input
    assert out_spec.dim_specs[0].axes == ["x"], "Sharding should propagate"
    
    print("  ✓ Shardy propagation works!")


def main():
    print("=" * 60)
    print(" SHARDING INTEGRATION TESTS")
    print("=" * 60)
    
    test_metadata_caching_single_output()
    test_metadata_caching_multi_output()
    test_sharding_annotation()
    test_graph_walking_with_sharding()
    test_compile_with_sharding_validation()
    test_shardy_propagation_standalone()
    
    print("\n" + "=" * 60)
    print(" ALL SHARDING INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
