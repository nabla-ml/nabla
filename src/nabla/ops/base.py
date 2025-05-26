"""Base operation utilities and registration functions."""

from typing import List, Callable
from ..core.array import (
    Array,
    MaxprCallable,
    VJPRule,
    JVPRule,
)  # Ensure Array is imported from core
from ..utils.broadcasting import get_broadcasted_shape

# Import broadcast_to from .view to avoid circular dependency if it's defined there
# If broadcast_to is a core utility, it might need to be in ..utils or ..core
from ..ops.view import (
    broadcast_to,
)  # This might be a source of circular import if view ops also use base.
from max.dtype import DType  # Import DType

# Global execution mode flag
EAGERMODE: bool = False


def _validate_binary_args(args: List[Array], op_name: str) -> None:
    """Validate arguments for binary operations."""
    if len(args) != 2:
        raise ValueError(f"{op_name} operation requires 2 arguments, got {len(args)}")
    if not all(isinstance(arg, Array) for arg in args):
        raise TypeError(
            f"All arguments must be instances of Array, got {[type(arg) for arg in args]}"
        )
    if args[0].dtype != args[1].dtype:
        raise ValueError(
            f"Dtypes {args[0].dtype} and {args[1].dtype} are not compatible for {op_name}."
        )
    if args[0].device != args[1].device:
        raise ValueError(
            f"Devices {args[0].device} and {args[1].device} are not compatible for {op_name}."
        )


def _validate_unary_arg(arg: Array, op_name: str) -> None:
    """Validate argument for unary operations."""
    if not isinstance(arg, Array):
        raise TypeError(f"Argument must be an instance of Array, got {type(arg)}")


def _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule) -> None:
    """Validate that all operation functions are callable."""
    if not callable(maxpr):
        raise TypeError(f"maxpr must be callable, got {type(maxpr)}")
    if not callable(eagerxpr):
        raise TypeError(f"eagerxpr must be callable, got {type(eagerxpr)}")
    if not callable(vjp_rule):
        raise TypeError(f"vjp_rule must be callable, got {type(vjp_rule)}")
    if not callable(jvp_rule):
        raise TypeError(f"jvp_rule must be callable, got {type(jvp_rule)}")


def register_binary_op(
    args: List[Array],
    op_name: str,
    maxpr: MaxprCallable,
    eagerxpr: Callable[[List[Array], Array], None],
    vjp_rule: VJPRule,
    jvp_rule: JVPRule,
    # op_params: dict = None,
) -> Array:
    """Register a binary operation with validation and broadcasting."""
    _validate_binary_args(args, op_name)
    _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule)

    target_shape = get_broadcasted_shape(args[0].shape, args[1].shape)
    # broadcast_to itself is an operation, ensure it's correctly imported and used.
    # It might be better to have broadcast_to defined in a way that doesn't create circular deps
    # For example, if broadcast_to is in .view, and .view imports from .base, this is tricky.
    arg0_broadcasted = broadcast_to(args[0], target_shape)
    arg1_broadcasted = broadcast_to(args[1], target_shape)

    res = Array(
        shape=target_shape,
        dtype=args[0].dtype,
        device=args[0].device,
        materialize=False,
        name=op_name,
    )
    res.set_maxpr(maxpr)
    res.add_argument(arg0_broadcasted)
    res.add_argument(arg1_broadcasted)
    res.vjp_rule = vjp_rule
    res.jvp_rule = jvp_rule
    # if op_params:
    #     res.op_params = op_params

    if EAGERMODE:
        eagerxpr([arg0_broadcasted, arg1_broadcasted], res)

    return res


def register_unary_op(
    arg: Array,
    op_name: str,
    maxpr: MaxprCallable,
    eagerxpr: Callable[[List[Array], Array], None],
    vjp_rule: VJPRule,
    jvp_rule: JVPRule,
    # op_params: dict = None,
    output_shape_fn: Callable[
        [tuple], tuple
    ] = None,  # For ops that change shape, like Cast potentially if it could change bitwidth affecting shape in some contexts (unlikely for basic cast)
    output_dtype: "DType" = None,  # For ops like Cast
) -> Array:
    """Register a unary operation with validation."""
    _validate_unary_arg(arg, op_name)
    _validate_callables(maxpr, eagerxpr, vjp_rule, jvp_rule)

    final_shape = output_shape_fn(arg.shape) if output_shape_fn else arg.shape
    final_dtype = output_dtype if output_dtype else arg.dtype

    res = Array(
        shape=final_shape,
        dtype=final_dtype,
        device=arg.device,
        materialize=False,
        name=op_name,
    )
    res.set_maxpr(maxpr)
    res.add_argument(arg)
    res.vjp_rule = vjp_rule
    res.jvp_rule = jvp_rule
    # if op_params:
    #     res.op_params = op_params

    # Special handling for CastOp's dtype, which is set on the result Array directly
    # This is now handled by passing output_dtype to Array constructor

    if EAGERMODE:
        eagerxpr([arg], res)

    return res
