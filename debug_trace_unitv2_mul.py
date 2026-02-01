from functools import partial
import jax.numpy as jnp
import nabla as nb
from nabla.core.graph import trace
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec
from tests.unit_v2.common import OpConfig, standard_get_args

config = OpConfig("Rank2", ranks=(2,2))
(args_nb, kwargs_nb), (args_jax, kwargs_jax) = standard_get_args(config)
mesh = DeviceMesh("test_mesh", (2,2), ("x","y"))

# spec index 2 in get_sharding_configs -> ShardingSpec(mesh, (("x",),()))
spec = ShardingSpec(mesh, (("x",), ()))

nb_fn = partial(nb.mul, **kwargs_nb)

def sharded_exec(a, b):
    a = a.with_sharding(spec.mesh, spec.dim_specs)
    return nb_fn(a, b)

tr = trace(sharded_exec, *args_nb)
print(tr)
