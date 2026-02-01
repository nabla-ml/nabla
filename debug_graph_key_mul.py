from functools import partial
import jax.numpy as jnp
import nabla as nb
from nabla.core.graph import GRAPH
from tests.unit_v2.common import OpConfig, standard_get_args, cleanup_caches
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec

config = OpConfig("Rank2", ranks=(2,2))
(args_nb, kwargs_nb), (args_jax, kwargs_jax) = standard_get_args(config)

# base add
cleanup_caches()
nb.add(*args_nb)
key_add_base = tuple(GRAPH._graph_key)

# base mul
cleanup_caches()
nb.mul(*args_nb)
key_mul_base = tuple(GRAPH._graph_key)

mesh = DeviceMesh("test_mesh", (2,2), ("x","y"))
spec = ShardingSpec(mesh, (("x",), ()))

def sharded_mul():
    a, b = args_nb
    a = a.with_sharding(spec.mesh, spec.dim_specs)
    return nb.mul(a, b)

cleanup_caches()
sharded_mul()
key_mul_shard = tuple(GRAPH._graph_key)

print('add base key == mul base key:', key_add_base == key_mul_base)
print('mul shard key == add base key:', key_mul_shard == key_add_base)
print('mul shard key == mul base key:', key_mul_shard == key_mul_base)

print('add base key:', key_add_base)
print('mul base key:', key_mul_base)
print('mul shard key:', key_mul_shard)
