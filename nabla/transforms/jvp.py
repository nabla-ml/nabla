# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""JVP (Jacobian-Vector Product) — forward-mode autodiff primitive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

from .utils import split_aux


_ATTACHED_TANGENT_PARENTS: dict[tuple[int, int], Any] = {}


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

    saved_tangents = _save_and_attach_tangents(primals, tangents)
    try:
        raw_output = fn(*primals)
    finally:
        _restore_tangents(saved_tangents)

    output, aux = split_aux(raw_output, has_aux, name="jvp")
    output_tangents = _extract_tangents(output)
    _detach_tangents_from_tree(output)

    if has_aux:
        return output, output_tangents, aux
    return output, output_tangents


def _save_and_attach_tangents(
    primals: tuple[Any, ...], tangents: tuple[Any, ...]
) -> list[tuple[Any, Any, Any | None]]:
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    saved: list[tuple[Any, Any, Any | None]] = []

    for primal, tangent in zip(primals, tangents, strict=False):
        primal_leaves = (
            pytree.tree_leaves(primal) if not isinstance(primal, Tensor) else [primal]
        )
        tangent_leaves = (
            pytree.tree_leaves(tangent)
            if not isinstance(tangent, Tensor)
            else [tangent]
        )

        for p, t in zip(primal_leaves, tangent_leaves, strict=False):
            if isinstance(p, Tensor) and isinstance(t, Tensor):
                old_tangent = p._impl.tangent
                attached_impl = t._impl
                _ATTACHED_TANGENT_PARENTS[(id(p._impl), id(attached_impl))] = old_tangent
                saved.append((p, old_tangent, attached_impl))
                p._impl.tangent = t._impl

    return saved


def _restore_tangents(saved_tangents: list[tuple[Any, Any, Any | None]]) -> None:
    for tensor, old_tangent, attached_impl in reversed(saved_tangents):
        if attached_impl is not None:
            _ATTACHED_TANGENT_PARENTS.pop((id(tensor._impl), id(attached_impl)), None)
        tensor._impl.tangent = old_tangent


def get_attached_tangent_parent(primal_impl: Any, attached_tangent_impl: Any) -> Any:
    """Return the parent tangent that was active before *attached_tangent_impl*.

    This enables nested forward-mode levels to clear only the current level,
    while preserving an outer tangent level if present.
    """
    if primal_impl is None or attached_tangent_impl is None:
        return None
    return _ATTACHED_TANGENT_PARENTS.get((id(primal_impl), id(attached_tangent_impl)))


def set_attached_tangent_parent(
    primal_impl: Any, attached_tangent_impl: Any, parent_tangent_impl: Any
) -> None:
    """Record a parent tangent for an attached tangent.

    This is used for intermediates during nested JVP to ensure all tangent
    levels are preserved across operation boundaries.
    """
    if primal_impl is not None and attached_tangent_impl is not None:
        _ATTACHED_TANGENT_PARENTS[
            (id(primal_impl), id(attached_tangent_impl))
        ] = parent_tangent_impl


def _clear_jvp_cache() -> None:
    """Clear all global JVP state. Internal use only."""
    _ATTACHED_TANGENT_PARENTS.clear()


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


__all__ = ["jvp", "_clear_jvp_cache"]
