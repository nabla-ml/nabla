# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Callable, Any

from .utils import backward_on_trace
from ..graph.tracing import trace
from ..common import pytree
from ..tensor.api import Tensor


def grad(
    fun: Callable, argnums: int | tuple[int, ...] = 0, create_graph: bool = False
) -> Callable:
    """Creates a function that evaluates the gradient of `fun`.

    Args:
        fun: Function to be differentiated. Must return a scalar.
        argnums: Index or indices of arguments to differentiate with respect to.
        create_graph: Whether to trace the backward computations (enabling higher-order derivs)

    Returns:
        A function with the same signature as `fun` that returns the gradient
        with respect to its inputs.
    """

    def wrapper(*args, **kwargs):
        # Trace the function execution
        t = trace(fun, *args, **kwargs)

        output = t.outputs

        from ...ops.creation import ones_like
        from ..tensor.api import Tensor

        # If output is a single Tensor, create ones_like
        if isinstance(output, Tensor):
            cotangent = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map = backward_on_trace(t, cotangent, create_graph=create_graph)
        input_leaves = pytree.tree_leaves(args)
        grad_leaves = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            elif isinstance(inp, Tensor):
                grad_leaves.append(None)  # Or zeros?
            else:
                grad_leaves.append(None)

        # Unflatten
        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)

        # Handle argnums
        if isinstance(argnums, int):
            # If default 0 and args is dict/tuple?
            # JAX argnums refers to positional args.
            # If args[0] is a dict (params), then grad is w.r.t params.
            if len(args) > argnums:
                return grads_struct[argnums]
            else:
                # If using kwargs exclusively?
                pass
            return grads_struct  # Fallback
        elif isinstance(argnums, (tuple, list)):
            return tuple(grads_struct[i] for i in argnums)
        else:
            return grads_struct

    return wrapper


def value_and_grad(
    fun: Callable, argnums: int | tuple[int, ...] = 0, create_graph: bool = False
) -> Callable:
    def wrapper(*args, **kwargs):
        t = trace(fun, *args, **kwargs)
        output = t.outputs

        from ...ops.creation import ones_like
        from ..tensor.api import Tensor

        if isinstance(output, Tensor):
            cotangent = ones_like(output)
        else:
            raise TypeError(f"grad: output must be a Tensor, got {type(output)}")

        grads_map = backward_on_trace(t, cotangent, create_graph=create_graph)

        input_leaves = pytree.tree_leaves(args)
        grad_leaves = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            else:
                grad_leaves.append(None)

        grads_struct = pytree.tree_unflatten(pytree.tree_structure(args), grad_leaves)

        if isinstance(argnums, int):
            if len(args) > argnums:
                g = grads_struct[argnums]
            else:
                g = grads_struct
        elif isinstance(argnums, (tuple, list)):
            g = tuple(grads_struct[i] for i in argnums)
        else:
            g = grads_struct

        return output, g

    return wrapper
