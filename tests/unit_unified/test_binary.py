
# ===----------------------------------------------------------------------=== #
# Unified Test: Binary Operations
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_test_with_consistency_check,
    get_sharding_configs, DeviceMesh, jax_matmul_wrapper
)

OPS = {}

OPS["add"] = Operation(
    "add", "BINARY", nb.add, jnp.add,
    [OpConfig("Rank2", ranks=(2, 2))], standard_get_args
)
OPS["mul"] = Operation(
    "mul", "BINARY", nb.mul, jnp.multiply,
    [OpConfig("Rank2", ranks=(2, 2))], standard_get_args
)
OPS["sub"] = Operation(
    "sub", "BINARY", nb.sub, jnp.subtract,
    [OpConfig("Rank2", ranks=(2, 2))], standard_get_args
)
OPS["div"] = Operation(
    "div", "BINARY", nb.div, jnp.divide,
    [OpConfig("Rank2", ranks=(2, 2), use_stable_floats=True)], standard_get_args
)
OPS["matmul"] = Operation(
    "matmul", "LINALG", nb.matmul, jax_matmul_wrapper,
    [
        OpConfig("Matrix_@_Matrix", primal_shapes=((3, 4), (4, 5))),
    ],
    standard_get_args
)

@pytest.mark.parametrize("op_name", OPS.keys())
def test_binary_level0_base(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    run_test_with_consistency_check(f"{op_name}_Base", lambda: nb_fn(*args_nb), lambda: jax_fn(*args_jax))

@pytest.mark.parametrize("op_name", OPS.keys())
def test_binary_level0_vmap(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    if not config.supports_vmap: pytest.skip("vmap not supported")
    
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    # To simulate vmap behavior matching Nabla's implicit batching or behavior
    # We need to manually batchify the arguments because nb.vmap accepts unbatched args
    # and vectorizes OVER them if in_axes is default (all args).
    # Wait, nb.vmap(f)(x) implies x has a batch dim.
    # The existing test passes args_nb which are SINGLE items.
    # nb.vmap(fn)(*args_nb) would mean args_nb must have the batch dimension?
    # NO, vmap is a transform that adds a batch dimension.
    # If args_nb are unbatched, we must stack them to create a batch.
    
    # Create a batch of size 2 for all inputs
    batched_args_nb = tuple(nb.stack([a, a]) for a in args_nb)
    batched_args_jax = tuple(jnp.stack([a, a]) for a in args_jax)
    
    run_test_with_consistency_check(f"{op_name}_Vmap", 
        lambda: nb.vmap(nb_fn)(*batched_args_nb), 
        lambda: jax.vmap(jax_fn)(*batched_args_jax)
    )

@pytest.mark.parametrize("op_name", OPS.keys())
def test_binary_level1_sharding(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    mesh = DeviceMesh("test_mesh", (2, 2), ("x", "y"))
    
    # Only shard first arg for simplicity in basic test
    specs = get_sharding_configs(mesh, len(args_nb[0].shape))
    
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    for i, spec in enumerate(specs):
        def sharded_exec():
            args = list(args_nb)
            if spec:
                args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs) 
            return nb_fn(*args)
            
        run_test_with_consistency_check(f"{op_name}_Shard_{i}", sharded_exec, lambda: jax_fn(*args_jax))
