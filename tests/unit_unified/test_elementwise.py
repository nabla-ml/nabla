# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

import jax.numpy as jnp
import pytest

import nabla as nb

from .common import OpConfig, Operation, run_unified_test, standard_get_args

OPS = {}

OPS["relu"] = Operation(
    "relu",
    "UNARY",
    nb.relu,
    lambda x: jnp.maximum(x, 0),
    [
        OpConfig("Relu_Basic", ranks=(2,), domain_positive=False),
        OpConfig("Relu_Pos", ranks=(2,), domain_positive=True),
    ],
    standard_get_args,
)

OPS["mul"] = Operation(
    "mul",
    "BINARY",
    nb.mul,
    jnp.multiply,
    [
        OpConfig("Mul_SameShape", ranks=(2, 2)),
        OpConfig("Mul_Broadcast_Scalar", ranks=(2, 0)),
    ],
    standard_get_args,
)


@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1, 2])
def test_elementwise_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        return
    config = op.configs[config_idx]

    print(f"\n[DEBUG] Running {op_name} - {config.description}")
    (args_nb, _), _ = op.get_args(config)
    for i, arg in enumerate(args_nb):
        print(f"Arg {i}: shape={arg.shape}")

    run_unified_test(op, config)
