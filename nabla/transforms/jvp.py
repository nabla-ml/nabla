# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""JVP (Jacobian-Vector Product) — forward-mode autodiff primitive.

Architecture: Trace-then-forward (mirrors VJP's trace-then-backward).
  1. Trace the function to capture the computation graph.
  2. Walk the graph forward, applying jvp_rule at each node.
  3. Extract output tangents from the tangent map.

This replaces the old inline tangent-slot mechanism (apply_jvp in __call__).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

from .utils import split_aux


@overload
def jvp(
    fn: Callable[..., Any],
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    has_aux: Literal[False] = False,
    create_graph: bool = True,
) -> tuple[Any, Any]: ...


@overload
def jvp(
    fn: Callable[..., Any],
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    has_aux: Literal[True],
    create_graph: bool = True,
) -> tuple[Any, Any, Any]: ...


def jvp(
    fn: Callable[..., Any],
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    has_aux: bool = False,
    create_graph: bool = True,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Compute JVP of *fn* at *primals* with *tangents*.

    Returns ``(output, tangent_out)`` or ``(output, tangent_out, aux)``
    when *has_aux* is True.

    Uses a trace-then-forward architecture mirroring VJP's trace-then-backward.
    """
    from ..core.autograd.forward import forward_on_trace
    from ..core.common import pytree
    from ..core.graph.tracing import trace as capture_trace
    from ..core.tensor.api import Tensor
    from ..ops.creation import zeros_like

    if not isinstance(primals, tuple):
        raise TypeError(f"primals must be a tuple, got {type(primals)}")
    if not isinstance(tangents, tuple):
        raise TypeError(f"tangents must be a tuple, got {type(tangents)}")
    if len(primals) != len(tangents):
        raise ValueError(
            f"primals and tangents must have the same length, "
            f"got {len(primals)} and {len(tangents)}"
        )

    # --- Trace the function ---
    _aux_box: list[Any] = []
    if has_aux:

        def traced_fn(*a: Any) -> Any:
            out, aux = split_aux(fn(*a), has_aux=True, name="jvp")
            _aux_box.append(aux)
            return out

    else:
        traced_fn = fn

    t = capture_trace(traced_fn, *primals)
    output = t.outputs
    aux = _aux_box[0] if has_aux else None

    # --- Forward-mode AD on the traced graph ---
    tangent_map = forward_on_trace(t, tangents, create_graph=create_graph)

    # --- Extract output tangents ---
    def _get_output_tangent(x: Any) -> Any:
        if isinstance(x, Tensor) and x._impl in tangent_map:
            return Tensor(impl=tangent_map[x._impl])
        if isinstance(x, Tensor):
            return zeros_like(x)
        return x

    output_tangents = pytree.tree_map(_get_output_tangent, output)

    if has_aux:
        return output, output_tangents, aux
    return output, output_tangents


def _clear_jvp_cache() -> None:
    """Clear all global JVP state. No-op in trace-then-forward architecture."""
    pass


__all__ = ["jvp", "_clear_jvp_cache"]
