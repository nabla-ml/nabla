#!/usr/bin/env python3
"""Integration tests for Shardy sharding integration into eager module.

These tests validate that:
1. Metadata caching works for all tensor operations
2. Sharding annotations can be attached to tensors
3. The sharding compiler can walk the graph and access metadata
4. Shardy propagation runs correctly

Run with: python test_sharding_integration.py
"""

from nabla import (
    Tensor, add, matmul,
    DeviceMesh, DimSpec, ShardingSpec,
    get_topological_order,
)
from nabla.ops.multi_output import split


def test_metadata_caching_single_output():
    """Test that single-output ops cache metadata correctly."""
    print("\n=== Test: Metadata Caching (Single Output) ===")
    
    x = Tensor.zeros((4, 8), traced=True)
    y = Tensor.ones((4, 8), traced=True)
    z = add(x, y)
    
    # Verify cached metadata
    assert z.global_shape is not None, "cached_shape should be set"
    assert z._impl.cached_dtype is not None, "cached_dtype should be set"
    assert z._impl.cached_device is not None, "cached_device should be set"
    
    print(f"  z.cached_shape: {z.global_shape}")
    print(f"  z.cached_dtype: {z._impl.cached_dtype}")
    print("  ✓ Single output caching works!")


def test_metadata_caching_multi_output():
    """Test that multi-output ops cache metadata correctly."""
    print("\n=== Test: Metadata Caching (Multi Output) ===")
    
    x = Tensor.zeros((4, 8), traced=True)
    a, b = split(x, num_splits=2, axis=0)
    
    # Both outputs should have cached metadata
    assert a.global_shape is not None, "a.cached_shape should be set"
    assert b.global_shape is not None, "b.cached_shape should be set"
    
    print(f"  a.cached_shape: {a.global_shape}")
    print(f"  b.cached_shape: {b.global_shape}")
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





def test_shardy_propagation_standalone():
    """Test that Shardy propagation works independently."""
    print("\n=== Test: Shardy Propagation (Standalone) ===")
    
    from nabla.sharding import (
        DeviceMesh, DimSpec, ShardingSpec,
    )
    from nabla.sharding.propagation import (
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


def test_tensor_shard_method():
    """Test the Tensor.shard() method - functional API returns new tensor."""
    print("\n=== Test: Tensor.shard() Method ===")
    
    mesh = DeviceMesh("test", (2,), ("x",))
    
    # Create tensor and shard it - shard() returns NEW tensor
    # No .trace() needed - sharding triggers SPMD path automatically
    A = Tensor.ones((4, 8))
    A_sharded = A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
    
    # shard() is functional - returns new tensor, NOT self
    # Original tensor unchanged, new tensor has sharding
    
    # Should have sharding attached to returned tensor
    assert A_sharded._impl.sharding is not None, "sharding should be set"
    assert isinstance(A_sharded._impl.sharding, ShardingSpec), "sharding should be ShardingSpec"
    assert A_sharded._impl.sharding.mesh.name == "test", "mesh name should match"
    assert A_sharded._impl.sharding.dim_specs[0].axes == ["x"], "first dim should be sharded on x"
    assert A_sharded._impl.sharding.dim_specs[1].axes == [], "second dim should be unsharded"
    
    # Verify it has multiple shard values
    assert A_sharded._impl.num_shards == 2, "should have 2 shards"
    
    print("  ✓ Tensor.shard() works!")


def test_eager_sharding_propagation_matmul():
    """Test that sharding propagates automatically through matmul operations.
    
    In the eager sharding architecture, propagation happens automatically
    in Operation.__call__ when sharded inputs are detected.
    """
    print("\n=== Test: Eager Sharding Propagation (Matmul) ===")
    
    from nabla.core.tensor_impl import get_topological_order
    
    mesh = DeviceMesh("test", (2,), ("x",))
    
    # A sharded on first dim (4, 8) -> [x, -]
    # shard() is functional - use the returned tensor!
    # No .trace() needed - sharding triggers SPMD path automatically
    A = Tensor.ones((4, 8))
    A_sharded = A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
    
    # B unsharded
    B = Tensor.ones((8, 4))
    
    # C = A_sharded @ B - sharding propagates automatically!
    C = A_sharded @ B
    
    # Verify sharding was propagated eagerly
    assert C._impl.sharding is not None, "C should have sharding after op"
    assert isinstance(C._impl.sharding, ShardingSpec), "C.sharding should be ShardingSpec"
    
    # C's first dim should be sharded on "x" (propagated from A's m dimension)
    # Matmul: (m, k) @ (k, n) -> (m, n), so m maps to x
    c_first_dim_axes = C._impl.sharding.dim_specs[0].axes
    print(f"  A_sharded sharding: {A_sharded._impl.sharding}")
    print(f"  C sharding: {C._impl.sharding}")
    print(f"  C first dim axes: {c_first_dim_axes}")
    
    assert "x" in c_first_dim_axes, "C's first dim should have 'x' sharding from A"
    
    print("  ✓ Eager sharding propagation works!")


def test_collectives_interface():
    """Test that collective ops have correct interface."""
    print("\n=== Test: Collectives Interface ===")
    
    from nabla.ops.communication import shard, all_gather, all_reduce, ShardOp, AllGatherOp, AllReduceOp, ReduceScatterOp
    
    # Check names
    assert AllGatherOp().name == "all_gather"
    assert ReduceScatterOp().name == "reduce_scatter"
    assert AllReduceOp().name == "all_reduce"
    
    print("  ✓ Collective ops have correct names!")


def test_end_to_end_sharded_execution():
    """Test actual sharded compilation and execution path."""
    print("\n=== Test: End-to-End Sharded Execution ===")
    import asyncio
    
    mesh = DeviceMesh("test", (2,), ("x",))
    
    # A sharded on first dim - use functional shard() API
    # No .trace() needed - sharding triggers SPMD path automatically
    A = Tensor.ones((4, 8))
    A_sharded = A.shard(mesh, [DimSpec(["x"]), DimSpec([])])
    
    # B unsharded
    B = Tensor.ones((8, 4))
    
    # C = A_sharded @ B
    C = A_sharded @ B
    
    print(f"  Before await: C._impl.sharding = {C._impl.sharding}")
    print(f"  Before await: C._impl._values count = {len(C._impl._values)}")
    
    # This should trigger sharded execution
    try:
        asyncio.run(C.realize)
        print(f"  After await: C._impl._storages count = {len(C._impl._storages) if C._impl._storages else 0}")
        print("  ✓ End-to-end sharded execution works!")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        raise


def main():
    print("=" * 60)
    print(" SHARDING INTEGRATION TESTS")
    print("=" * 60)
    
    test_metadata_caching_single_output()
    test_metadata_caching_multi_output()
    test_sharding_annotation()
    test_graph_walking_with_sharding()

    test_shardy_propagation_standalone()
    
    # Eager sharding tests
    test_tensor_shard_method()
    test_eager_sharding_propagation_matmul()
    test_collectives_interface()
    
    # End-to-end test
    test_end_to_end_sharded_execution()
    
    print("\n" + "=" * 60)
    print(" ALL SHARDING INTEGRATION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
