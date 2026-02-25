# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""VJP (Vector-Jacobian Product) — reverse-mode autodiff primitive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

from .utils import collect_grads, realize_tensors, split_aux


@overload
def vjp(
    fn: Callable[..., Any], *primals: Any, has_aux: Literal[False] = False
) -> tuple[Any, Callable[..., tuple[Any, ...]]]: ...


@overload
def vjp(
    fn: Callable[..., Any], *primals: Any, has_aux: Literal[True]
) -> tuple[Any, Callable[..., tuple[Any, ...]], Any]: ...


def vjp(
    fn: Callable[..., Any],
    *primals: Any,
    has_aux: bool = False,
    create_graph: bool = True,
) -> (
    tuple[Any, Callable[..., tuple[Any, ...]]]
    | tuple[Any, Callable[..., tuple[Any, ...]], Any]
):
    """Compute the Vector-Jacobian Product (VJP) of *fn* at *primals*.

    Evaluates *fn* and returns a pullback function that multiplies a
    cotangent vector by the Jacobian. This is the fundamental building
    block for reverse-mode automatic differentiation.

    Args:
        fn: Differentiable function to differentiate.
        *primals: Input values at which to evaluate *fn* and the VJP.
        has_aux: If ``True``, *fn* must return ``(output, aux)``.
            The auxiliary data *aux* is returned as a third element and
            excluded from differentiation.
        create_graph: If ``True`` (default), the pullback is
            differentiable, enabling higher-order AD.

    Returns:
        - ``(output, pullback)`` when *has_aux* is ``False``.
        - ``(output, pullback, aux)`` when *has_aux* is ``True``.

        The returned *pullback* is a function that takes a cotangent
        vector (with the same structure as *output*) and returns
        a tuple of input cotangents.
    """
    from ..core.autograd.backward import backward_on_trace
    from ..core.common import pytree
    from ..core.graph.tracing import trace as capture_trace
    from ..core.tensor.api import Tensor

    _aux_box: list[Any] = []
    if has_aux:

        def traced_fn(*a: Any) -> Any:
            out, aux = split_aux(fn(*a), has_aux=True, name="vjp")
            _aux_box.append(aux)
            return out

    else:
        traced_fn = fn

    t = capture_trace(traced_fn, *primals)
    output, aux = t.outputs, (_aux_box[0] if has_aux else None)

    def vjp_fn(cotangent: Any) -> tuple[Any, ...]:
        grads_map = backward_on_trace(t, cotangent, create_graph=create_graph)
        input_leaves = pytree.tree_leaves(primals)
        grad_leaves = collect_grads(grads_map, input_leaves)
        if not create_graph:
            realize_tensors(grad_leaves)
        grad_struct: tuple[Any, ...] = pytree.tree_unflatten(
            pytree.tree_structure(primals), grad_leaves
        )
        if not isinstance(grad_struct, tuple):
            grad_struct = (grad_struct,)
        return grad_struct

    if has_aux:
        return output, vjp_fn, aux
    return output, vjp_fn


__all__ = ["vjp"]
