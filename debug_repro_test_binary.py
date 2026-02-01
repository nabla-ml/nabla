from functools import partial
import jax.numpy as jnp
import nabla as nb
from tests.unit_v2.common import OpConfig, standard_get_args, get_sharding_configs, cleanup_caches
from nabla.core.sharding.spec import DeviceMesh
from nabla.core.graph import trace

# setup like test_binary
ops = [
    ("add", nb.add, jnp.add),
    ("mul", nb.mul, jnp.multiply),
    ("sub", nb.sub, jnp.subtract),
    ("div", nb.div, jnp.divide),
]

config = OpConfig("Rank2", ranks=(2,2))
(args_nb, kwargs_nb), (args_jax, kwargs_jax) = standard_get_args(config)

# run level0 base
for name, nb_fn, jax_fn in ops:
    cleanup_caches()
    nb_fn(*args_nb)

# run level0 vmap
for name, nb_fn, jax_fn in ops:
    cleanup_caches()
    batched_args_nb = tuple(nb.stack([a, a]) for a in args_nb)
    nb.vmap(partial(nb_fn, **kwargs_nb))(*batched_args_nb)

# run sharding
mesh = DeviceMesh("test_mesh", (2,2), ("x","y"))
specs = get_sharding_configs(mesh, len(args_nb[0].shape))

# focus on mul
nb_fn = partial(nb.mul, **kwargs_nb)
for i, spec in enumerate(specs):
    cleanup_caches()
    args = list(args_nb)
    if spec:
        args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs)
    if i == 2:
        tr = trace(lambda a,b: nb_fn(a,b), *args)
        print("TRACE for mul spec 2:")
        print(tr)
    res = nb_fn(*args)
    val = res.numpy()
    exp = (args_jax[0]*args_jax[1])
    print(f"mul spec {i} val[0,0]=", float(val[0,0]), "exp[0,0]=", float(exp[0,0]))
