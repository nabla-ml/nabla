
# ===----------------------------------------------------------------------=== #
# Unified Test: Reduction & Comparison
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_test_with_consistency_check,
    get_sharding_configs, DeviceMesh
)

# REDUCTION
OPS_RED = {}
OPS_RED["reduce_sum"] = Operation(
    "reduce_sum", "REDUCTION", nb.reduce_sum, jnp.sum,
    [OpConfig("Axis1", ranks=(2,), params={"axis": 1})], standard_get_args
)
OPS_RED["mean"] = Operation(
    "mean", "REDUCTION", nb.mean, jnp.mean,
    [OpConfig("Axis1", ranks=(2,), params={"axis": 1})], standard_get_args
)

# COMPARISON
OPS_CMP = {}
for name, nb_fn, jax_fn in [
    ("equal", nb.equal, jnp.equal),
    ("not_equal", nb.not_equal, jnp.not_equal),
    ("greater", nb.greater, jnp.greater),
    ("less", nb.less, jnp.less),
    ("greater_equal", nb.greater_equal, jnp.greater_equal),
    ("less_equal", nb.less_equal, jnp.less_equal),
]:
    OPS_CMP[name] = Operation(
        name, "COMPARISON", nb_fn, jax_fn,
        [OpConfig("Rank2", ranks=(2, 2))], standard_get_args
    )


@pytest.mark.parametrize("op_name", OPS_RED.keys())
def test_reduction_base(op_name):
    op = OPS_RED[op_name]
    config = op.configs[0]
    (args_nb, _), (args_jax, kwargs_jax) = op.get_args(config)
    # Mapping params: nabla args vs jax kwargs?
    # Helper standard_get_args returns params for both.
    
    # Nabla reductions usually take axis as kwarg too?
    # nb.reduce_sum(x, axis=...)
    nb_fn = partial(op.nabla_fn, **config.params)
    jax_fn = partial(op.jax_fn, **config.params)
    
    run_test_with_consistency_check(f"{op_name}_Base", lambda: nb_fn(*args_nb), lambda: jax_fn(*args_jax))


@pytest.mark.parametrize("op_name", OPS_CMP.keys())
def test_comparison_base(op_name):
    op = OPS_CMP[op_name]
    config = op.configs[0]
    (args_nb, _), (args_jax, _) = op.get_args(config)
    
    run_test_with_consistency_check(
        f"{op_name}_Base", 
        lambda: op.nabla_fn(*args_nb), 
        lambda: op.jax_fn(*args_jax)
    )
