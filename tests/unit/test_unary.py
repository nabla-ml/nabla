# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from functools import partial

import jax
import jax.numpy as jnp
import pytest

import nabla as nb

from .common import (
    DeviceMesh,
    OpConfig,
    Operation,
    get_sharding_configs,
    run_test_with_consistency_check,
    standard_get_args,
)

OPS = {}

OPS["relu"] = Operation(
    "relu",
    "UNARY",
    nb.relu,
    jax.nn.relu,
    [OpConfig("Rank2", ranks=(2,))],
    standard_get_args,
)
OPS["sigmoid"] = Operation(
    "sigmoid",
    "UNARY",
    nb.sigmoid,
    jax.nn.sigmoid,
    [OpConfig("Rank2", ranks=(2,))],
    standard_get_args,
)
OPS["tanh"] = Operation(
    "tanh",
    "UNARY",
    nb.tanh,
    jnp.tanh,
    [OpConfig("Rank2", ranks=(2,))],
    standard_get_args,
)
OPS["exp"] = Operation(
    "exp", "UNARY", nb.exp, jnp.exp, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["neg"] = Operation(
    "neg",
    "UNARY",
    nb.neg,
    jnp.negative,
    [OpConfig("Rank2", ranks=(2,))],
    standard_get_args,
)
OPS["abs"] = Operation(
    "abs", "UNARY", nb.abs, jnp.abs, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["softmax"] = Operation(
    "softmax",
    "UNARY",
    nb.softmax,
    jax.nn.softmax,
    [OpConfig("Rank2_Axis-1", ranks=(2,), params={"axis": -1})],
    standard_get_args,
)
OPS["logsoftmax"] = Operation(
    "logsoftmax",
    "UNARY",
    nb.logsoftmax,
    jax.nn.log_softmax,
    [OpConfig("Rank2_Axis-1", ranks=(2,), params={"axis": -1})],
    standard_get_args,
)
OPS["cos"] = Operation(
    "cos", "UNARY", nb.cos, jnp.cos, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["sin"] = Operation(
    "sin", "UNARY", nb.sin, jnp.sin, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["acos"] = Operation(
    "acos", "UNARY", nb.acos, jnp.acos, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["atanh"] = Operation(
    "atanh", "UNARY", nb.atanh, jnp.atanh, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["erf"] = Operation(
    "erf", "UNARY", nb.erf, jax.lax.erf, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["floor"] = Operation(
    "floor", "UNARY", nb.floor, jnp.floor, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["log1p"] = Operation(
    "log1p", "UNARY", nb.log1p, jnp.log1p, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["rsqrt"] = Operation(
    "rsqrt", "UNARY", nb.rsqrt, jax.lax.rsqrt, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["silu"] = Operation(
    "silu", "UNARY", nb.silu, jax.nn.silu, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["trunc"] = Operation(
    "trunc", "UNARY", nb.trunc, jnp.trunc, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["gelu"] = Operation(
    "gelu", "UNARY", nb.gelu, jax.nn.gelu, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["round"] = Operation(
    "round", "UNARY", nb.round, jnp.round, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["is_inf"] = Operation(
    "is_inf", "UNARY", nb.is_inf, jnp.isinf, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["is_nan"] = Operation(
    "is_nan", "UNARY", nb.is_nan, jnp.isnan, [OpConfig("Rank2", ranks=(2,))], standard_get_args
)
OPS["cast"] = Operation(
    "cast",
    "UNARY",
    nb.cast,
    lambda x, dtype: x.astype(dtype.name),
    [OpConfig("Cast_f32", ranks=(2,), params={"dtype": nb.DType.float32})],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level0_base(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)

    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)

    run_test_with_consistency_check(
        f"{op_name}_Base", lambda: nb_fn(*args_nb), lambda: jax_fn(*args_jax)
    )


@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level0_vmap(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    if not config.supports_vmap:
        pytest.skip("vmap not supported")

    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)

    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)

    run_test_with_consistency_check(
        f"{op_name}_Vmap",
        lambda: nb.vmap(nb_fn)(*args_nb),
        lambda: jax.vmap(jax_fn)(*args_jax),
    )


@pytest.mark.parametrize("op_name", OPS.keys())
def test_unary_level1_sharding(op_name):
    op = OPS[op_name]
    config = op.configs[0]
    (args_nb, kwargs_nb), (args_jax, kwargs_jax) = op.get_args(config)

    mesh = DeviceMesh("test_mesh", (2, 2), ("x", "y"))

    input_tensor = args_nb[0]
    specs = get_sharding_configs(mesh, len(input_tensor.shape))

    nb_fn = partial(op.nabla_fn, **kwargs_nb)
    jax_fn = partial(op.jax_fn, **kwargs_jax)

    for i, spec in enumerate(specs):

        def sharded_exec():

            args = list(args_nb)
            if spec:
                args[0] = args[0].with_sharding(spec.mesh, spec.dim_specs)
            return nb_fn(*args)

    run_test_with_consistency_check(
        f"{op_name}_Shard_{i}", sharded_exec, lambda: jax_fn(*args_jax)
    )
