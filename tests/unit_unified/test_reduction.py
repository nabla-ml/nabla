
# ===----------------------------------------------------------------------=== #
# Unified Test: Reduction
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
# Operations
# ============================================================================

OPS["sum"] = Operation(
    "sum", "REDUCTION", nb.reduce_sum, jnp.sum,
    [
        OpConfig("Sum_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Sum_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
        # OpConfig("Sum_All", ranks=(2,), params={"axis": None}), # Axis None not supported by reduce_sum yet?
    ],
    standard_get_args
)

OPS["mean"] = Operation(
    "mean", "REDUCTION", nb.mean, jnp.mean,
    [
        OpConfig("Mean_Axis0", ranks=(2,), params={"axis": 0}),
        OpConfig("Mean_Axis1_KeepDims", ranks=(2,), params={"axis": 1, "keepdims": True}),
    ],
    standard_get_args
)

# Note: Max, Min, Prod not yet available in nabla/ops/reduction.py

@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1, 2])
def test_reduction_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        pass 
        return

    config = op.configs[config_idx]
    run_unified_test(op, config)
