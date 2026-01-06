
import pytest
from nabla.core.tensor import Tensor
from nabla.transforms.vmap import vmap
from nabla.sharding.spec import DeviceMesh, DimSpec, ShardingSpec

@pytest.fixture
def mesh():
    return DeviceMesh("test_mesh", (4,), ("dp",))

def test_vmap_spmd_axis_name(mesh):
    """Test using spmd_axis_name to shard the batch dimension."""
    # Input has batch dim of size 4
    x = Tensor.normal((4, 32))
    # Must assign sharding to use spmd_axis_name (even if replicated input)
    x._impl.sharding = ShardingSpec(mesh, [DimSpec([]), DimSpec([])])
    
    @vmap(spmd_axis_name="dp")
    def f(x):
        # x inside should be sharded on "dp" along batch dim (physical axis 0)
        assert x._impl.sharding is not None
        assert x._impl.sharding.dim_specs[0].axes == ["dp"]
        assert x._impl.sharding.dim_specs[1].is_replicated()
        return x * 2

    y = f(x)
    
    # Output should inherit sharding from x
    assert tuple(y.shape) == (4, 32)
    assert y._impl.sharding is not None
    assert y._impl.sharding.dim_specs[0].axes == ["dp"]

def test_vmap_over_sharded_input(mesh):
    """Test vmap processing an already sharded input."""
    x = Tensor.normal((4, 32))
    # Shard axis 0 on "dp"
    x = x.shard(mesh, [DimSpec(["dp"]), DimSpec([])])
    
    @vmap
    def f(x_val):
        # x_val inside vmap should preserve physical sharding
        assert x_val.batch_dims == 1
        spec = x_val._impl.sharding
        assert spec is not None
        assert spec.dim_specs[0].axes == ["dp"]
        return x_val + 1
        
    y = f(x)
    assert tuple(y.shape) == (4, 32)
    assert y._impl.sharding.dim_specs[0].axes == ["dp"]

def test_sharding_inside_vmap(mesh):
    """Test applying sharding constraints inside vmap."""
    x = Tensor.normal((4, 32))
    x._impl.sharding = ShardingSpec(mesh, [DimSpec([]), DimSpec([])])
    
    @vmap
    def f(x_val):
        # Shard logical axis 0 (physical 1) on "dp"
        # Input: [R, R]
        # Target: [R, "dp"]
        sharded = x_val.shard(mesh, [DimSpec(["dp"])])
        
        # Verify physical spec contains batch [R] prepended
        spec = sharded._impl.sharding
        assert len(spec.dim_specs) == 2
        assert spec.dim_specs[0].is_replicated()
        assert spec.dim_specs[1].axes == ["dp"]
        
        return sharded
    
    y = f(x)
    # Output should be [R, dp]
    assert y._impl.sharding.dim_specs[0].is_replicated()
    assert y._impl.sharding.dim_specs[1].axes == ["dp"]
