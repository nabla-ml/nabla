from functools import partial
import nabla as nb
import jax.numpy as jnp
from tests.unit_v2.common import OpConfig, standard_get_args, get_sharding_configs, run_test_with_consistency_check
from nabla.core.sharding.spec import DeviceMesh

config = OpConfig("Rank2", ranks=(2,2))
(args_nb, kwargs_nb), (args_jax, kwargs_jax) = standard_get_args(config)

mesh = DeviceMesh("test_mesh", (2,2), ("x","y"))
specs = get_sharding_configs(mesh, len(args_nb[0].shape))

nb_fn = partial(nb.mul, **kwargs_nb)
jax_fn = partial(jnp.multiply, **kwargs_jax)

# simulate earlier tests
pre_ops = [
    (nb.add, jnp.add),
    (nb.mul, jnp.multiply),
    (nb.sub, jnp.subtract),
    (nb.div, jnp.divide),
]
for nb_op, jax_op in pre_ops:
    run_test_with_consistency_check("pre", lambda: nb_op(*args_nb), lambda: jax_op(*args_jax))

# now mul_Shard_2
spec = specs[2]

def sharded_exec():
    args = list(args_nb)
    if spec:
        args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs)
    return nb_fn(*args)

run_test_with_consistency_check("mul_Shard_2", sharded_exec, lambda: jax_fn(*args_jax))
print("done")
