
import pytest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.ops.communication import shard_batch_dims
from nabla.ops._physical import incr_batch_dims, decr_batch_dims

@pytest.fixture
def mesh():
    return DeviceMesh("test_mesh", (4,), ("dp",))

def test_shard_prepends_batch_specs(mesh):
    """Test that Tensor.shard() prepends replicated DimSpecs for batch dimensions."""
    # 1. Create tensor and manually force batch_dims for testing
    # Physical shape: (4, 32)
    x = Tensor.normal((4, 32))
    x._impl.batch_dims = 1
    # Logical shape is now (32,)
    
    # 2. Shard logical dimension 0 (physical 1) on "dp"
    # User intent: shard the 32-dim on "dp"
    sharded_x = x.shard(mesh, [DimSpec(["dp"])])
    
    # 3. Verify physical spec
    spec = sharded_x._impl.sharding
    assert spec is not None
    assert len(spec.dim_specs) == 2  # Physical rank match
    
    # Batch dim (physical 0) should be replicated (prepended)
    assert spec.dim_specs[0].is_replicated()
    
    # Logical dim 0 (physical 1) should be sharded on "dp"
    assert spec.dim_specs[1].axes == ["dp"]

def test_shard_inherits_existing_batch_sharding():
    """Test that Tensor.shard() preserves existing batch dimension sharding."""
    mesh2 = DeviceMesh("mesh2", (2, 2), ("x", "y"))
    
    x2 = Tensor.normal((2, 2, 32)) # (Batch, Logical1, Logical2)
    x2._impl.batch_dims = 1
    
    # Pre-existing spec: Batch on "x", Replicated, Replicated
    x2._impl.sharding = ShardingSpec(mesh2, [DimSpec(["x"]), DimSpec([]), DimSpec([])])
    
    # Shard logical dim 0 on "y" (physical 1)
    # User provides spec for (Logical1, Logical2) -> [y, R]
    # Input batch specs (len 1) should be prepended
    sharded_x2 = x2.shard(mesh2, [DimSpec(["y"]), DimSpec([])])
    
    spec = sharded_x2._impl.sharding
    assert len(spec.dim_specs) == 3
    assert spec.dim_specs[0].axes == ["x"] # Batch preserved
    assert spec.dim_specs[1].axes == ["y"] # Logical 0 applied
    assert spec.dim_specs[2].is_replicated()

def test_shard_batch_dims_function(mesh):
    """Test explicit shard_batch_dims API."""
    x = Tensor.normal((4, 32))
    x._impl.batch_dims = 1 # Simulate vmap
    
    # 1. Shard batch dim on "dp"
    sharded = shard_batch_dims(x, mesh, "dp")
    
    spec = sharded._impl.sharding
    assert spec.dim_specs[0].axes == ["dp"]
    assert spec.dim_specs[1].is_replicated() 

def test_incr_batch_dims_preserves_spec(mesh):
    """Test that incr_batch_dims does NOT modify spec (rank is constant)."""
    # x: (4, 32). Spec [dp, R].
    x = Tensor.normal((4, 32))
    x._impl.sharding = ShardingSpec(mesh, [DimSpec(["dp"]), DimSpec([])])
    
    assert x.batch_dims == 0
    
    # Simulate vmap entry
    y = incr_batch_dims(x)
    
    assert y.batch_dims == 1
    assert y._impl.sharding is not None
    assert len(y._impl.sharding.dim_specs) == 2
    # Spec should be identical
    assert y._impl.sharding.dim_specs[0].axes == ["dp"]
