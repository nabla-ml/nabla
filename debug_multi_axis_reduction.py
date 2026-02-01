"""Debug the multi-axis sharded reduction issue."""
import nabla as nb
from nabla.core.sharding import DeviceMesh, DimSpec
import jax
import jax.numpy as jnp
import numpy as np
from tests.unit_v2.common import to_jax


def debug_multi_axis_reduction():
    """Debug what happens with multi-axis sharded reductions."""
    
    # Create mesh and data just like the failing test
    mesh = DeviceMesh("debug_mesh", (2, 2), ("x", "y"))
    
    # Create test data 
    np.random.seed(55)
    jax_x = jnp.array(np.random.randn(8, 8).astype(np.float32))
    x = nb.Tensor.from_dlpack(jax_x)
    
    print(f"Original tensor shape: {jax_x.shape}")
    print(f"Original tensor:\n{jax_x}")
    
    # Shard on both dimensions
    x_sharded = x.shard(mesh, [DimSpec(["x"]), DimSpec(["y"])])
    
    print(f"\nSharded tensor spec: {x_sharded.sharding}")
    print(f"Sharded tensor shape: {x_sharded.shape}")
    print(f"Sharded tensor num_shards: {x_sharded.num_shards}")
    
    # Print each shard
    for i in range(x_sharded.num_shards):
        local_shape = x_sharded.physical_local_shape(i)
        print(f"Shard {i} local shape: {local_shape}")
    
    # Test reduction axis=0
    print("\n=== Testing axis=0 reduction ===")
    result = nb.mean(x_sharded, axis=0, keepdims=False)
    expected = jnp.mean(jax_x, axis=0)
    
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: {expected.shape}")
    
    result_array = result.to_jax()
    print(f"Result values:\n{result_array}")
    print(f"Expected values:\n{expected}")
    
    # Show the difference
    print(f"Differences:\n{result_array - expected}")
    
    # Test if first 4 elements are correct
    print(f"First 4 elements match: {jnp.allclose(result_array[:4], expected[:4])}")
    print(f"Last 4 elements match: {jnp.allclose(result_array[4:], expected[4:])}")
    print(f"Last 4 == First 4: {jnp.allclose(result_array[4:], result_array[:4])}")

if __name__ == "__main__":
    debug_multi_axis_reduction()