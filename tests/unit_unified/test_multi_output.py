
# ===----------------------------------------------------------------------=== #
# Unified Test: Multi-Output Operations
# ===----------------------------------------------------------------------=== #

import pytest
import nabla as nb
import jax.numpy as jnp
from functools import partial

from .common import (
    Operation, OpConfig, standard_get_args, run_unified_test
)

# ============================================================================
# JAX Wrappers
# ============================================================================

def jax_split_wrapper(x, num_splits, axis=0):
    # JAX split uses sections (count), which matches our API
    return jnp.split(x, num_splits, axis=axis)

def jax_chunk_wrapper(x, chunks, axis=0):
    # JAX chunk is also split with sections
    return jnp.split(x, chunks, axis=axis)

def jax_unbind_wrapper(x, axis=0):
    # JAX unbind equivalent: tuple of slices along axis
    return tuple(jnp.squeeze(s, axis=axis) for s in jnp.split(x, x.shape[axis], axis=axis))

# ============================================================================
# Operations
# ============================================================================

OPS = {}

OPS["split"] = Operation(
    "split", "MULTI_OUTPUT", nb.split, jax_split_wrapper,
    [
        OpConfig("Split_2", ranks=(2,), params={"num_splits": 2, "axis": 0}, primal_shapes=((4, 4),)),
        OpConfig("Split_Axis1", ranks=(2,), params={"num_splits": 2, "axis": 1}, primal_shapes=((4, 4),)),
    ],
    standard_get_args
)

OPS["chunk"] = Operation(
    "chunk", "MULTI_OUTPUT", nb.chunk, jax_chunk_wrapper,
    [
        OpConfig("Chunk_2", ranks=(2,), params={"chunks": 2, "axis": 0}, primal_shapes=((4, 4),)),
    ],
    standard_get_args
)

OPS["unbind"] = Operation(
    "unbind", "MULTI_OUTPUT", nb.unbind, jax_unbind_wrapper,
    [
        OpConfig("Unbind_Axis0", ranks=(2,), params={"axis": 0}, primal_shapes=((4, 4),)),
    ],
    standard_get_args
)

@pytest.mark.parametrize("op_name", OPS.keys())
@pytest.mark.parametrize("config_idx", [0, 1])
def test_multi_output_ops(op_name, config_idx):
    op = OPS[op_name]
    if config_idx >= len(op.configs):
        return
    config = op.configs[config_idx]
    run_unified_test(op, config)
