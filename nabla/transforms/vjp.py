# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""VJP (Vector-Jacobian Product) transform — reverse-mode autodiff primitive.

Usage matches JAX's vjp API:
    primals_out, vjp_fn = vjp(f, *primals)
    cotangents_in = vjp_fn(cotangents_out)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload


@overload
def vjp(
    fn: Callable[..., Any], *primals: Any, has_aux: Literal[False] = False
) -> tuple[Any, Callable]: ...


@overload
def vjp(
    fn: Callable[..., Any], *primals: Any, has_aux: Literal[True]
) -> tuple[Any, Callable, Any]: ...


def vjp(
    fn: Callable[..., Any], *primals: Any, has_aux: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    """Compute the VJP (reverse-mode) of *fn* evaluated at *primals*.

    Args:
        fn: Function ``f(*primals) -> output`` (or ``(output, aux)`` when *has_aux*).
        *primals: Positional arguments — arbitrary pytrees of Tensors.
        has_aux: If True, *fn* returns ``(output, aux)`` and aux is passed through.

    Returns:
        ``(output, vjp_fn)`` — or ``(output, vjp_fn, aux)`` when *has_aux* is True.

        ``vjp_fn(cotangent)`` returns a tuple of cotangent pytrees, one per primal arg.
    """
    from ..core.graph.tracing import trace as capture_trace
    from ..core.autograd.utils import backward_on_trace
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    # For has_aux, we wrap fn so the Trace only records the differentiable output.
    # Aux is captured in a closure side-channel.
    _aux_container: list[Any] = []

    if has_aux:
        def traced_fn(*args):
            raw = fn(*args)
            if not isinstance(raw, tuple) or len(raw) != 2:
                raise ValueError(
                    "vjp with has_aux=True expects fn to return (output, aux), "
                    f"got {type(raw)}"
                )
            out, aux = raw
            _aux_container.append(aux)
            return out
    else:
        traced_fn = fn

    # 1. Trace forward execution, capturing the computation graph.
    t = capture_trace(traced_fn, *primals)

    output = t.outputs
    aux = _aux_container[0] if has_aux else None

    # 3. Build the pullback (vjp_fn) closure.
    def vjp_fn(cotangent: Any) -> tuple[Any, ...]:
        """Pullback: maps output cotangent → input cotangents."""
        grads_map = backward_on_trace(t, cotangent)

        # Non-differentiable dtypes: gradients are meaningless for these.
        from max.dtype import DType
        _NON_DIFF_DTYPES = {DType.bool, DType.int8, DType.int16, DType.int32, DType.int64,
                            DType.uint8, DType.uint16, DType.uint32, DType.uint64}

        # Collect gradients for each primal arg, preserving pytree structure.
        input_leaves = pytree.tree_leaves(primals)
        grad_leaves = []
        for inp in input_leaves:
            # Non-differentiable dtypes (bool, int) → None, no meaningful gradient.
            if isinstance(inp, Tensor) and inp.dtype in _NON_DIFF_DTYPES:
                grad_leaves.append(None)
            elif isinstance(inp, Tensor) and inp in grads_map:
                grad_leaves.append(grads_map[inp])
            elif isinstance(inp, Tensor):
                from ..ops.creation import zeros_like
                grad_leaves.append(zeros_like(inp))
            else:
                grad_leaves.append(None)

        # Realize lazily-computed gradients.
        unrealized = [g for g in grad_leaves if isinstance(g, Tensor) and not g.real]
        if unrealized:
            from ..core.tensor.api import realize_all
            realize_all(*unrealized)

        # Reconstruct per-arg pytree structure.
        grad_struct = pytree.tree_unflatten(pytree.tree_structure(primals), grad_leaves)

        # Always return a tuple matching len(primals).
        if not isinstance(grad_struct, tuple):
            grad_struct = (grad_struct,)
        return grad_struct

    if has_aux:
        return output, vjp_fn, aux
    return output, vjp_fn


__all__ = ["vjp"]
