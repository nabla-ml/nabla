from functools import partial
import nabla as nb
import jax.numpy as jnp
from tests.unit_v2.common import (
    OpConfig,
    Operation,
    standard_get_args,
    get_sharding_configs,
    cleanup_caches,
)
from nabla.core.sharding.spec import DeviceMesh

# Define ops like test
OPS = {}
OPS["add"] = Operation("add","BINARY", nb.add, jnp.add, [OpConfig("Rank2", ranks=(2,2))], standard_get_args)
OPS["mul"] = Operation("mul","BINARY", nb.mul, jnp.multiply, [OpConfig("Rank2", ranks=(2,2))], standard_get_args)

# Level0 base for add, mul
for op_name in ["add","mul"]:
    op=OPS[op_name]
    config=op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    nb_fn=partial(op.nabla_fn, **kwargs_nb)
    cleanup_caches()
    nb_fn(*args_nb)

# Level0 vmap skipped (not needed)

# Level1 sharding
for op_name in ["add","mul"]:
    op=OPS[op_name]
    config=op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    mesh=DeviceMesh("test_mesh", (2,2), ("x","y"))
    specs=get_sharding_configs(mesh, len(args_nb[0].shape))
    nb_fn=partial(op.nabla_fn, **kwargs_nb)
    for i, spec in enumerate(specs):
        def sharded_exec():
            args=list(args_nb)
            if spec:
                args[0]=args[0].with_sharding(spec.mesh, spec.dim_specs)
            return nb_fn(*args)
        cleanup_caches()
        res=sharded_exec()
        # trigger eval
        val=res.numpy()
        if op_name=="mul" and i==2:
            print("mul shard2 val[0,0]", val[0,0])
            print("expected", (args_jax[0]*args_jax[1])[0,0])
