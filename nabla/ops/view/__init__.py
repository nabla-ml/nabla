# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .axes import (
    UnsqueezeOp,
    SqueezeOp,
    SwapAxesOp,
    unsqueeze,
    squeeze,
    swap_axes,
    MoveAxisOp,
    UnsqueezePhysicalOp,
    SqueezePhysicalOp,
    moveaxis,
    unsqueeze_physical,
    squeeze_physical,
)

from .shape import (
    broadcast_to,
    reshape,
    stack,
    concatenate,
    slice_tensor,
    BroadcastToOp,
    ReshapeOp,
    SliceTensorOp,
    ConcatenateOp,
    BroadcastToPhysicalOp,
    broadcast_to_physical,
)

from .batch import (
    IncrBatchDimsOp,
    DecrBatchDimsOp,
    MoveAxisToBatchDimsOp,
    MoveAxisFromBatchDimsOp,
    BroadcastBatchDimsOp,
    incr_batch_dims,
    decr_batch_dims,
    move_axis_to_batch_dims,
    move_axis_from_batch_dims,
    broadcast_batch_dims,
)

from .indexing import (
    GatherOp,
    ScatterOp,
    gather,
    scatter,
)

__all__ = [
    # Axis operations
    "UnsqueezeOp", "unsqueeze",
    "SqueezeOp", "squeeze",
    "SwapAxesOp", "swap_axes",
    
    # Shape operations
    "BroadcastToOp", "broadcast_to",
    "ReshapeOp", "reshape",
    "SliceTensorOp", "slice_tensor",
    "ConcatenateOp", "concatenate", "stack",
    
    # Indexing operations
    "GatherOp", "gather",
    "ScatterOp", "scatter",
]
