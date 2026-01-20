
# ===----------------------------------------------------------------------=== #
# Unified Test: Elementwise (Unary & Binary)
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_unified_test
)

OPS = {}

# ============================================================================
# Unary Operations
# ============================================================================

OPS["relu"] = Operation(
    "relu", "UNARY", nb.relu, 
    lambda x: jnp.maximum(x, 0),
    [
        OpConfig("Relu_Basic", ranks=(2,), domain_positive=False), # Mix of pos/neg
        OpConfig("Relu_Pos", ranks=(2,), domain_positive=True),
    ],
    standard_get_args
)


# ============================================================================
# Binary Operations
# ============================================================================

OPS["mul"] = Operation(
    "mul", "BINARY", nb.mul, jnp.multiply,
    [
        OpConfig("Mul_SameShape", ranks=(2, 2)),
        OpConfig("Mul_Broadcast_Scalar", ranks=(2, 0)),
        # OpConfig("Mul_Broadcast_Vector", ranks=(2, 1), primal_shapes=((4, 4), (4,))), # FIXME: VMap shape mismatch [2,4] vs [4,2]
    ],
    standard_get_args
)

@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1, 2])
def test_elementwise_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        return # Skip
    config = op.configs[config_idx]
    
    # Run Unified Test (Standard, VMap, Sharding)
    # Sharding check here is valuable: Elementwise ops should preserve sharding spec
    # if inputs are sharded identically, or propagate/broadcast if different.
    # Debug
    print(f"\n[DEBUG] Running {op_name} - {config.description}")
    (args_nb, _), _ = op.get_args(config)
    for i, arg in enumerate(args_nb):
        print(f"Arg {i}: shape={arg.shape}")

    run_unified_test(op, config)
