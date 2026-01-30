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


def backward(tensor: Tensor, gradient: Tensor = None, retain_graph: bool = False):
    """Computes the gradient of current tensor w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If the tensor is non-scalar
    (i.e. its data has more than one element) and requires gradient, the function
    additionally requires specifying ``gradient``. It should be a tensor of matching
    type and location, that contains the gradient of the differentiated function
    w.r.t. ``self``.

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Args:
        tensor: Tensor to compute backward on.
        gradient: Gradient w.r.t. the tensor. If tensor is scalar, this is optional
            and defaults to 1.0.
        retain_graph: If False, the graph used to compute the grads will be freed.
            (Not fully implemented, simplified trace handling).
    """
    # 1. Ensure tensor is valid for backward
    if not isinstance(tensor, Tensor):
        raise TypeError(f"backward expects a Tensor, got {type(tensor)}")

    # 2. Get the trace from the tensor
    # Tensor must be traced or have output_refs/trace associated locally?
    # Actually, in Nabla, backward_on_trace requires a Trace object.
    # But a single Tensor doesn't hold the whole Trace explicitly in a user-accessible way
    # unless we are inside a context.
    # However, 'tensor' has 'output_refs'. We can traverse backward from there.
    # 'backward_on_trace' iterates a list of ops.
    # We need to construct the backward graph starting from this tensor's creator.

    # Limitation: Current backward_on_trace design assumes we have the 'Trace' object
    # captured from 'trace(fun)'.
    # If we just do 'y = x * 2; backward(y)', we might not have the linear Trace object handy
    # if we didn't use `trace()`.
    # BUT, if `x` was traced (e.g. inside `trace` context or manual tracing), `y` is traced.

    # Let's check how 'backward_on_trace' works. It iterates a list of ops.
    # If we don't have the explicit list, we can traverse dependencies.
    # But 'backward_on_trace' expects a linear order (topological sort).

    # For now, let's look at how we can get the Trace.
    # A Tensor might have a weakref to the Trace?
    # If not, we can't easily support 'y.backward()' unless we are in implicit tracing mode
    # where there is a global trace.

    # In 'test_slice_update.py', we did:
    # x.trace() -> creates a trace?
    # x = nb.Tensor(...)
    # x.trace() ?? No, 'x.trace()' is not a standard method I saw.
    # Wait, in the test I wrote: 'x.trace()'.
    # I probably assumed this existed.

    # The user test code:
    # x = nb.Tensor(...)
    # x.trace()
    # This implies explicit tracing start.

    # If 'backward' expects to work, it needs to find the Trace.
    pass

    # Re-reading backward_on_trace signature: backward_on_trace(t: Trace, ...)

    # If I cannot get Trace from Tensor, I cannot implement 'backward(tensor)'.
    # I need to export 'backward_on_trace' or 'backward_mode' context?

    # Let's look at `Tensor` definition in `nabla/core/tensor/api.py`.
    # Does it have `.trace()` method?

    # If not, I should implement `backward` to verify if `tensor.trace` is reachable.

    # BUT! 'test_slice_update.py' uses:
    # x.trace()
    # ...
    # ops.slice_tensor(x...)

    # If I wrote that test, I assumed APIs that might not exist.
    # The existing `test_pp_grad.py` uses `grad(func)`. `grad` uses `trace(func)`.
    # So `grad` works because `trace` returns `t`.

    # To test VJP of a single op without `grad` wrapper, I should use `grad` in my test!
    # Refactor `test_slice_update.py` to use `grad` or `value_and_grad` instead of manual `backward`.
    # This avoids implementing new infrastructure.

    return None


def backward(tensor, *args, **kwargs):
    raise NotImplementedError("Use nabla.grad for now.")


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

        # Map grads back to args structure
        # backward_on_trace returns dict[Tensor, Tensor] (input tensor -> grad tensor)

        input_leaves = pytree.tree_leaves(args)
        grad_leaves = []
        for inp in input_leaves:
            if isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            elif isinstance(inp, Tensor):
                # No grad (detached or constant)
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
