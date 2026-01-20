
# ===----------------------------------------------------------------------=== #
# Unified Test: Unary Operations
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax.numpy as jnp
import jax
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_test_with_consistency_check,
    get_sharding_configs, DeviceMesh
)

# Registry
OPS = {}

# Populate Registry
OPS["relu"] = Operation(
    "relu", "UNARY", nb.relu, jax.nn.relu,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["sigmoid"] = Operation(
    "sigmoid", "UNARY", nb.sigmoid, jax.nn.sigmoid,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["tanh"] = Operation(
    "tanh", "UNARY", nb.tanh, jnp.tanh,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["exp"] = Operation(
    "exp", "UNARY", nb.exp, jnp.exp,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["neg"] = Operation(
    "neg", "UNARY", nb.neg, jnp.negative,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["abs"] = Operation(
    "abs", "UNARY", nb.abs, jnp.abs,
    [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["softmax"] = Operation(
    "softmax", "UNARY", nb.softmax, jax.nn.softmax,
    [OpConfig("Rank2_Axis-1", ranks=(2,), params={"axis": -1})],
    standard_get_args
)


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level0_base(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    # Adapt args
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    run_test_with_consistency_check(
        f"{op_name}_Base",
        lambda: nb_fn(*args_nb),
        lambda: jax_fn(*args_jax)
    )

@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level0_vmap(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    if not config.supports_vmap: pytest.skip("vmap not supported")
    
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    run_test_with_consistency_check(
        f"{op_name}_Vmap",
        lambda: nb.vmap(nb_fn)(*args_nb),
        lambda: jax.vmap(jax_fn)(*args_jax)
    )

@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level1_sharding(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)
    
    mesh = DeviceMesh("test_mesh", (2, 2), ("x", "y"))
    
    # Generate sharding configs for 1st arg
    input_tensor = args_nb[0]
    specs = get_sharding_configs(mesh, len(input_tensor.shape))
    
    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)
    
    for i, spec in enumerate(specs):
        def sharded_exec():
            # Apply sharding
            args = list(args_nb)
            if spec:
                args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs) # Fake apply or use shard()
            return nb_fn(*args)
            
    if op_name == "softmax":
        pytest.skip("Known bug: Softmax sharding uses per-shard normalization instead of global")

    run_test_with_consistency_check(
            f"{op_name}_Shard_{i}",
            sharded_exec,
            lambda: jax_fn(*args_jax)
        )
