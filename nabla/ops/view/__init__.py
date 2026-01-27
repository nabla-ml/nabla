# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .axes import (
    MoveAxisOp,
    SqueezeOp,
    SqueezePhysicalOp,
    SwapAxesOp,
    UnsqueezeOp,
    UnsqueezePhysicalOp,
    moveaxis,
    squeeze,
    squeeze_physical,
    swap_axes,
    unsqueeze,
    unsqueeze_physical,
)
from .batch import (
    BroadcastBatchDimsOp,
    DecrBatchDimsOp,
    IncrBatchDimsOp,
    MoveAxisFromBatchDimsOp,
    MoveAxisToBatchDimsOp,
    broadcast_batch_dims,
    decr_batch_dims,
    incr_batch_dims,
    move_axis_from_batch_dims,
    move_axis_to_batch_dims,
)
from .indexing import (
    GatherOp,
    ScatterOp,
    gather,
    scatter,
)
from .shape import (
    BroadcastToOp,
    BroadcastToPhysicalOp,
    ConcatenateOp,
    ReshapeOp,
    SliceTensorOp,
    SliceUpdateOp,
    broadcast_to,
    broadcast_to_physical,
    concatenate,
    reshape,
    slice_tensor,
    slice_update,
    stack,
)

__all__ = [
    "UnsqueezeOp",
    "unsqueeze",
    "SqueezeOp",
    "squeeze",
    "SwapAxesOp",
    "swap_axes",
    "BroadcastToOp",
    "broadcast_to",
    "ReshapeOp",
    "reshape",
    "SliceTensorOp",
    "slice_tensor",
    "SliceUpdateOp",
    "slice_update",
    "ConcatenateOp",
    "concatenate",
    "stack",
    "GatherOp",
    "gather",
    "ScatterOp",
    "scatter",
]
