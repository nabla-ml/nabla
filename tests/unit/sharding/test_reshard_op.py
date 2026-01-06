
import pytest
import numpy as np
from nabla.core.tensor import Tensor
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec
from nabla.ops.communication import reshard

@pytest.fixture
def mesh():
    return DeviceMesh("test_mesh", (4,), ("dp",))

def test_reshard_logical_specs_prepending(mesh):
    """Test that ReshardOp correctly prepends batch specs when given logical specs."""
    x = Tensor.normal((4, 32))
    x._impl.batch_dims = 1
    # Logical rank = 1 (shape (32,)), Physical rank = 2
    
    # Pass logical specs (len 1)
    # Target: Shard logical axis 0 (physical 1) on "dp"
    # Expected: [Replicated, "dp"]
    
    y = reshard(x, mesh, [DimSpec(["dp"])])
    
    assert y._impl.sharding is not None
    assert len(y._impl.sharding.dim_specs) == 2
    assert y._impl.sharding.dim_specs[0].is_replicated()
    assert y._impl.sharding.dim_specs[1].axes == ["dp"]

def test_reshard_physical_specs_direct(mesh):
    """Test that ReshardOp accepts full physical specs without modification."""
    x = Tensor.normal((4, 32))
    x._impl.batch_dims = 1
    
    # Pass physical specs (len 2)
    # Target: Shard batch axis (physical 0) on "dp"
    
    specs = [DimSpec(["dp"]), DimSpec([])]
    y = reshard(x, mesh, specs)
    
    assert y._impl.sharding is not None
    assert len(y._impl.sharding.dim_specs) == 2
    assert y._impl.sharding.dim_specs[0].axes == ["dp"]
    assert y._impl.sharding.dim_specs[1].is_replicated()

def test_reshard_noop(mesh):
    """Test that ReshardOp returns same tensor object if no change needed."""
    x = Tensor.normal((32,))
    spec = ShardingSpec(mesh, [DimSpec(["dp"])])
    x._impl.sharding = spec
    
    # Request same sharding
    y = reshard(x, mesh, [DimSpec(["dp"])])
    
    # Should be identital object (optimization)
    assert y is x

def test_reshard_replicated_axes(mesh):
    """Test that ReshardOp handles replicated_axes argument."""
    x = Tensor.normal((32,))
    
    # Shard on "dp" but explicitly mark as replicated? 
    # Usually replicated_axes are for partial replication in tensor parallelism.
    # Here we just verify it sets the metadata.
    
    # Logic: replicated_axes=replicated_axes passed to ShardingSpec
    # Axis "dp" is explicitly replicated, so it must NOT appear in dim_specs
    y = reshard(x, mesh, [DimSpec([])], replicated_axes={"dp"})
    
    assert y._impl.sharding is not None
    assert "dp" in y._impl.sharding.replicated_axes

def test_tensor_shard_delegation(mesh):
    """Test that Tensor.shard correctly calls ReshardOp logic (integration check)."""
    x = Tensor.normal((4, 32))
    x._impl.batch_dims = 1
    
    # Call Tensor.shard with logical specs
    y = x.shard(mesh, [DimSpec(["dp"])])
    
    # Verify prepending happened
    assert len(y._impl.sharding.dim_specs) == 2
    assert y._impl.sharding.dim_specs[0].is_replicated()
    assert y._impl.sharding.dim_specs[1].axes == ["dp"]

