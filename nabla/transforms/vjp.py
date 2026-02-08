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
    fn: Callable[..., Any], *primals: Any, has_aux: bool = False
) -> tuple[Any, Callable[..., tuple[Any, ...]]] | tuple[Any, Callable[..., tuple[Any, ...]], Any]:
    """Compute VJP of *fn* at *primals*. Returns ``(output, vjp_fn[, aux])``."""
    from ..core.graph.tracing import trace as capture_trace
    from ..core.autograd.utils import backward_on_trace
    from ..core.common import pytree

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
        grads_map = backward_on_trace(t, cotangent)
        input_leaves = pytree.tree_leaves(primals)
        grad_leaves = collect_grads(grads_map, input_leaves)
        realize_tensors(grad_leaves)
        grad_struct = pytree.tree_unflatten(pytree.tree_structure(primals), grad_leaves)
        if not isinstance(grad_struct, tuple):
            grad_struct = (grad_struct,)
        return grad_struct

    if has_aux:
        return output, vjp_fn, aux
    return output, vjp_fn


__all__ = ["vjp"]
