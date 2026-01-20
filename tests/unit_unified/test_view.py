# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax
import jax.numpy as jnp
import pytest

import nabla as nb
from nabla.ops.view import slice_tensor

from .common import (
    OpConfig,
    Operation,
    jax_squeeze_wrapper,
    jax_transpose_wrapper,
    jax_unsqueeze_wrapper,
    run_unified_test,
    standard_get_args,
)

OPS = {}


def get_reshape_args(config):
    (p_nb, _), (p_jax, _) = standard_get_args(config)
    return (p_nb, config.params), (p_jax, config.params)


OPS["swap_axes"] = Operation(
    "swap_axes",
    "VIEW",
    nb.swap_axes,
    jax_transpose_wrapper,
    [
        OpConfig("Swap_Last", ranks=(2,), params={"axis1": -2, "axis2": -1}),
        OpConfig("Swap_0_1", ranks=(3,), params={"axis1": 0, "axis2": 1}),
    ],
    standard_get_args,
)

OPS["transpose"] = Operation(
    "transpose",
    "VIEW",
    nb.swap_axes,
    jax_transpose_wrapper,
    [
        OpConfig("Transpose_Basic", ranks=(2,), params={"axis1": 0, "axis2": 1}),
    ],
    standard_get_args,
)

OPS["reshape"] = Operation(
    "reshape",
    "VIEW",
    nb.reshape,
    jnp.reshape,
    [
        OpConfig("Flatten", ranks=(3,), params={"shape": (-1,)}),
        OpConfig(
            "Reshape_Split", ranks=(1,), params={"shape": (2, 4)}, primal_shapes=((8,),)
        ),
    ],
    get_reshape_args,
)

OPS["squeeze"] = Operation(
    "squeeze",
    "VIEW",
    nb.squeeze,
    jax_squeeze_wrapper,
    [
        OpConfig("Squeeze_1", primal_shapes=((2, 1, 3),), params={"axis": 1}),
        OpConfig("Squeeze_Last", primal_shapes=((2, 3, 1),), params={"axis": -1}),
    ],
    standard_get_args,
)

OPS["unsqueeze"] = Operation(
    "unsqueeze",
    "VIEW",
    nb.unsqueeze,
    jax_unsqueeze_wrapper,
    [
        OpConfig("Unsqueeze_0", ranks=(2,), params={"axis": 0}),
        OpConfig("Unsqueeze_Last", ranks=(2,), params={"axis": -1}),
    ],
    standard_get_args,
)

OPS["broadcast_to"] = Operation(
    "broadcast_to",
    "VIEW",
    nb.broadcast_to,
    jnp.broadcast_to,
    [
        OpConfig("Broadcast_Simple", primal_shapes=((3,),), params={"shape": (2, 3)}),
        OpConfig(
            "Broadcast_RankUp", primal_shapes=((3,),), params={"shape": (2, 2, 3)}
        ),
    ],
    standard_get_args,
)

OPS["concatenate"] = Operation(
    "concatenate",
    "VIEW",
    nb.concatenate,
    jnp.concatenate,
    [
        OpConfig("Concat_0", ranks=(2, 2), params={"axis": 0}, is_list_input=True),
        OpConfig("Concat_1", ranks=(2, 2), params={"axis": 1}, is_list_input=True),
    ],
    standard_get_args,
)

OPS["stack"] = Operation(
    "stack",
    "VIEW",
    nb.stack,
    jnp.stack,
    [
        OpConfig("Stack_0", ranks=(2, 2), params={"axis": 0}, is_list_input=True),
    ],
    standard_get_args,
)


def jax_moveaxis(a, source, destination):
    return jnp.moveaxis(a, source, destination)


OPS["moveaxis"] = Operation(
    "moveaxis",
    "VIEW",
    nb.moveaxis,
    jax_moveaxis,
    [
        OpConfig("Move_0_to_Last", ranks=(3,), params={"source": 0, "destination": -1}),
    ],
    standard_get_args,
)


def jax_slice_tensor_wrapper(x, start, size):
    return jax.lax.dynamic_slice(x, start, size)


OPS["slice_tensor"] = Operation(
    "slice_tensor",
    "VIEW",
    slice_tensor,
    jax_slice_tensor_wrapper,
    [
        OpConfig(
            "Slice_Basic",
            ranks=(2,),
            params={"start": (0, 0), "size": (1, 1)},
            supports_vmap=False,
            supports_sharding=False,
        ),
        OpConfig(
            "Slice_Mid",
            ranks=(2,),
            primal_shapes=((4, 4),),
            params={"start": (1, 1), "size": (2, 2)},
            supports_vmap=False,
            supports_sharding=False,
        ),
    ],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1])
def test_view_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        return
    config = op.configs[config_idx]
    if op.name == "moveaxis" and config.supports_vmap:
        config = OpConfig(
            description=config.description,
            ranks=config.ranks,
            params=config.params,
            supports_vmap=False,
            supports_sharding=config.supports_sharding,
        )

    run_unified_test(op, config)
