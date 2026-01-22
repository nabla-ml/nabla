# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""VJP and JVP rules for all operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core import Tensor


def reduce_sum_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for reduce_sum: broadcast cotangent back to input shape."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    from ..ops.view.shape import broadcast_to
    return broadcast_to(cotangent, tuple(x.shape))


def reduce_sum_jvp(primals: Any, tangents: Any, output: Any) -> Any:
    """JVP for reduce_sum: sum tangents along same axis."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    return t  # Placeholder - need axis info


def mean_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for mean: broadcast and scale by 1/n."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    from ..ops.view.shape import broadcast_to
    from ..ops.binary import div
    
    # Need axis info to determine size - for now broadcast only
    broadcasted = broadcast_to(cotangent, tuple(x.shape))
    # Should divide by reduction size but we don't have axis here
    return broadcasted


def mean_jvp(primals: Any, tangents: Any, output: Any) -> Any:
    """JVP for mean: take mean of tangents."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    return t  # Placeholder


def broadcast_to_vjp(primals: Any, cotangent: Any, output: Any, target_shape: tuple) -> Any:
    """VJP for broadcast_to: sum over broadcasted dimensions."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    
    from ..ops.reduction import reduce_sum
    
    input_shape = tuple(x.shape)
    
    # Find broadcasted axes (new dims or dims that were size 1)
    broadcasted_axes = []
    input_rank = len(input_shape)
    output_rank = len(target_shape)
    
    # New axes added at the beginning
    for i in range(output_rank - input_rank):
        broadcasted_axes.append(i)
    
    # Existing axes that were size 1 and got broadcast
    for i, (in_dim, out_dim) in enumerate(zip(input_shape, target_shape[output_rank - input_rank:])):
        if in_dim == 1 and out_dim > 1:
            broadcasted_axes.append(i + (output_rank - input_rank))
    
    # Sum over broadcasted axes
    result = cotangent
    for axis in sorted(broadcasted_axes, reverse=True):
        result = reduce_sum(result, axis=axis, keepdims=(axis >= output_rank - input_rank))
    
    return result


def broadcast_to_jvp(primals: Any, tangents: Any, output: Any, target_shape: tuple) -> Any:
    """JVP for broadcast_to: broadcast tangents to target shape."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    from ..ops.view.shape import broadcast_to
    return broadcast_to(t, target_shape)


def reshape_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for reshape: reshape cotangent back to input shape."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    from ..ops.view.shape import reshape
    return reshape(cotangent, tuple(x.shape))


def reshape_jvp(primals: Any, tangents: Any, output: Any, target_shape: tuple) -> Any:
    """JVP for reshape: reshape tangents to target shape."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    from ..ops.view.shape import reshape
    return reshape(t, target_shape)


# Unary operations

def relu_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for relu: cotangent * (x > 0)."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    from ..ops.comparison import greater
    from ..ops.binary import mul
    
    # Derivative is 1 where x > 0, else 0
    mask = greater(x, 0.0)
    return mul(cotangent, mask)


def relu_jvp(primals: Any, tangents: Any, output: Any) -> Any:
    """JVP for relu: tangent * (x > 0)."""
    if isinstance(primals, tuple):
        x = primals[0]
    else:
        x = primals
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    
    from ..ops.comparison import greater
    from ..ops.binary import mul
    
    mask = greater(x, 0.0)
    return mul(t, mask)


def exp_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for exp: cotangent * exp(x) = cotangent * output."""
    from ..ops.binary import mul
    return mul(cotangent, output)


def exp_jvp(primals: Any, tangents: Any, output: Any) -> Any:
    """JVP for exp: tangent * exp(x) = tangent * output."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    from ..ops.binary import mul
    return mul(output, t)


def neg_vjp(primals: Any, cotangent: Any, output: Any) -> Any:
    """VJP for neg: -cotangent."""
    from ..ops.unary import neg  
    return neg(cotangent)


def neg_jvp(primals: Any, tangents: Any, output: Any) -> Any:
    """JVP for neg: -tangent."""
    if isinstance(tangents, tuple):
        t = tangents[0]
    else:
        t = tangents
    from ..ops.unary import neg
    return neg(t)


__all__ = [
    "reduce_sum_vjp",
    "reduce_sum_jvp",
    "mean_vjp",
    "mean_jvp",
    "broadcast_to_vjp",
    "broadcast_to_jvp",
    "reshape_vjp",
    "reshape_jvp",
    "relu_vjp",
    "relu_jvp",
    "exp_vjp",
    "exp_jvp",
    "neg_vjp",
    "neg_jvp",
]
