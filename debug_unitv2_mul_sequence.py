import nabla as nb
import jax.numpy as jnp
from functools import partial
from nabla.core.sharding.spec import DeviceMesh, ShardingSpec
from tests.unit_v2.common import standard_get_args, OpConfig

# create args like test
config = OpConfig("Rank2", ranks=(2,2))
(args_nb, kwargs_nb), (args_jax, kwargs_jax) = standard_get_args(config)

mesh = DeviceMesh("test_mesh", (2,2), ("x","y"))
spec = ShardingSpec(mesh, (("x",), ()))

# simulate add test before mul
nb.add(*args_nb)
nb.mul(*args_nb)

# now mul sharded
args = list(args_nb)
args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs)
res = nb.mul(*args)

print('res[0,0]=', res.numpy()[0,0])
print('expected[0,0]=', (args_jax[0]*args_jax[1])[0,0])
