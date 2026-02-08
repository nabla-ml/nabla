# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""JVP (Jacobian-Vector Product) — forward-mode autodiff primitive."""

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
) -> tuple[Any, Any]: ...


@overload
def jvp(
    fn: Callable[..., Any],
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    has_aux: Literal[True],
) -> tuple[Any, Any, Any]: ...


def jvp(
    fn: Callable[..., Any],
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    has_aux: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    """Compute JVP of *fn* at *primals* with *tangents*. Returns ``(out, tangent_out[, aux])``."""
    if not isinstance(primals, tuple):
        raise TypeError(f"primals must be a tuple, got {type(primals)}")
    if not isinstance(tangents, tuple):
        raise TypeError(f"tangents must be a tuple, got {type(tangents)}")
    if len(primals) != len(tangents):
        raise ValueError(
            f"primals and tangents must have the same length, "
            f"got {len(primals)} and {len(tangents)}"
        )

    _attach_tangents(primals, tangents)
    try:
        raw_output = fn(*primals)
    finally:
        _detach_tangents(primals)

    output, aux = split_aux(raw_output, has_aux, name="jvp")
    output_tangents = _extract_tangents(output)
    _detach_tangents_from_tree(output)

    if has_aux:
        return output, output_tangents, aux
    return output, output_tangents


def _attach_tangents(primals: tuple, tangents: tuple) -> None:
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    for primal, tangent in zip(primals, tangents):
        primal_leaves = (
            pytree.tree_leaves(primal) if not isinstance(primal, Tensor) else [primal]
        )
        tangent_leaves = (
            pytree.tree_leaves(tangent)
            if not isinstance(tangent, Tensor)
            else [tangent]
        )

        for p, t in zip(primal_leaves, tangent_leaves):
            if isinstance(p, Tensor) and isinstance(t, Tensor):
                p._impl.tangent = t._impl


def _detach_tangents(primals: tuple) -> None:
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    for primal in primals:
        leaves = (
            pytree.tree_leaves(primal) if not isinstance(primal, Tensor) else [primal]
        )
        for p in leaves:
            if isinstance(p, Tensor):
                p._impl.tangent = None


def _extract_tangents(output: Any) -> Any:
    from ..core.common import pytree
    from ..core.tensor.api import Tensor
    from ..ops.creation import zeros_like

    def _get_tangent(x: Any) -> Any:
        if isinstance(x, Tensor):
            if x._impl.tangent is not None:
                return Tensor(impl=x._impl.tangent)
            return zeros_like(x)
        return x

    return pytree.tree_map(_get_tangent, output)


def _detach_tangents_from_tree(tree: Any) -> None:
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    def _clear(x: Any) -> Any:
        if isinstance(x, Tensor):
            x._impl.tangent = None
        return x

    pytree.tree_map(_clear, tree)


__all__ = ["jvp"]
