
import jax
import jax.numpy as jnp
import nabla as nb
from nabla.core.sharding.spec import DeviceMesh, P
from nabla.core.tensor import Tensor

# Setup
mesh = DeviceMesh("mesh_debug", (4,), ("d",))
np_x = jax.random.normal(jax.random.PRNGKey(0), (4, 16))
from tests.unit_v2.common import tensor_from_jax

# Setup
mesh = DeviceMesh("mesh_debug", (4,), ("d",))
np_x = jax.random.normal(jax.random.PRNGKey(0), (4, 16))
x_nb = tensor_from_jax(np_x)

# Shard it
x_sharded = x_nb.shard(mesh, P("d", None))
print(f"X Sharded Spec Before: {x_sharded.sharding}")

# Simulate vmap transformation
# 1. moveaxis (batch dim 0 -> 0)
# 2. incr_batch_dims
# moveaxis usually logically moves axes. If batch dim logic is not separate.
# But vmap uses internal logical ops.
# Let's verify if 'sharding' is preserved on shallow copy or modification

# Check incr_batch_dims logic (we can't import it easily if it's internal op, but we can check implementation)
# nabla/core/tensor/impl.py

# Instead, we define a vmap function that checks the input tensor props
def check_props(x):
    print(f"Inside VMAP: Batch Dims: {x.batch_dims}")
    print(f"Inside VMAP: Sharding: {x.sharding}")
    print(f"Inside VMAP: Values Count: {len(x.values)}")
    if x.sharding:
         print(f"Inside VMAP: Dim Specs: {x.sharding.dim_specs}")
    return x # Identity

print("Running VMAP Identity...")
res = nb.vmap(check_props)(x_sharded)
