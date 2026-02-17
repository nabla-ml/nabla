# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from .axes import (
    SqueezePhysicalOp,  # noqa: F401
    flip,
    moveaxis,
    permute,
    squeeze,
    squeeze_physical,
    swap_axes,
    transpose,
    unsqueeze,
    unsqueeze_physical,
)
from .batch import (
    broadcast_batch_dims,
    decr_batch_dims,
    incr_batch_dims,
    move_axis_from_batch_dims,
    move_axis_to_batch_dims,
    moveaxis_physical,
)
from .complex import (
    as_interleaved_complex,
    view_as_real_interleaved,
)
from .indexing import (
    gather,
    scatter,
)
from .shape import (
    broadcast_to,
    broadcast_to_physical,
    concatenate,
    flatten,
    pad,
    rebind,
    reshape,
    slice_tensor,
    slice_update,
    slice_update_inplace,
    stack,
)

__all__ = [
    "unsqueeze",
    "squeeze",
    "swap_axes",
    "transpose",
    "broadcast_to",
    "reshape",
    "slice_tensor",
    "slice_update",
    "slice_update_inplace",
    "concatenate",
    "stack",
    "gather",
    "scatter",
    "flatten",
    "rebind",
    "as_interleaved_complex",
    "view_as_real_interleaved",
    "flip",
    "permute",
    "pad",
    "squeeze_physical",
    "unsqueeze_physical",
    "broadcast_to_physical",
    "moveaxis",
    "moveaxis_physical",
    "broadcast_batch_dims",
    "decr_batch_dims",
    "incr_batch_dims",
    "move_axis_from_batch_dims",
    "move_axis_to_batch_dims",
]
