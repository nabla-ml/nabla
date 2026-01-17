# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .base import CollectiveOperation

from .shard import ShardOp, shard, shard_op
from .all_gather import AllGatherOp, all_gather, all_gather_op, GatherAllAxesOp, gather_all_axes, gather_all_axes_op
from .all_reduce import AllReduceOp, all_reduce, all_reduce_op, PMeanOp, pmean, pmean_op
from .reduce_scatter import ReduceScatterOp, reduce_scatter, reduce_scatter_op
from .p_permute import PPermuteOp, ppermute, ppermute_op
from .all_to_all import AllToAllOp, all_to_all, all_to_all_op
from .axis_index import AxisIndexOp, axis_index, axis_index_op
from .reshard import ReshardOp, reshard, reshard_op

__all__ = [
    # Classes
    "CollectiveOperation",
    "ShardOp",
    "AllGatherOp", "GatherAllAxesOp",
    "AllReduceOp", "PMeanOp",
    "ReduceScatterOp",
    "PPermuteOp",
    "AllToAllOp",
    "AxisIndexOp",
    "ReshardOp",
    
    # Functions
    "shard",
    "all_gather", "gather_all_axes",
    "all_reduce", "pmean",
    "reduce_scatter",
    "ppermute",
    "all_to_all",
    "axis_index",
    "reshard",
    
    # Singletons (internal use mostly, but exposed for completeness)
    "shard_op",
    "all_gather_op", "gather_all_axes_op",
    "all_reduce_op", "pmean_op",
    "reduce_scatter_op",
    "ppermute_op",
    "all_to_all_op",
    "axis_index_op",
    "reshard_op",
]
