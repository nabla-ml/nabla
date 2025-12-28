"""Re-export communication ops for sharding module.

This module provides collective operations for sharded tensor execution:
- AllGatherOp: Gather shards to produce replicated tensors
- AllReduceOp: Reduce across shards (sum, mean, etc.)
- ReduceScatterOp: Reduce then scatter across shards
- shard: Split a replicated tensor into shards
"""

from ..ops.communication import (
    # Op classes
    ShardOp,
    AllGatherOp,
    AllReduceOp,
    ReduceScatterOp,
    # Singletons
    shard_op,
    all_gather_op,
    all_reduce_op,
    reduce_scatter_op,
    # Public functions
    shard,
    all_gather,
    all_reduce,
)

__all__ = [
    # Op classes
    "ShardOp",
    "AllGatherOp",
    "AllReduceOp",
    "ReduceScatterOp",
    # Singletons
    "shard_op",
    "all_gather_op",
    "all_reduce_op",
    "reduce_scatter_op",
    # Public functions
    "shard",
    "all_gather",
    "all_reduce",
]
