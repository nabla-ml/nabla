# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""JVP (Jacobian-Vector Product) transform — forward-mode autodiff primitive.

Usage matches JAX's jvp API:
    primals_out, tangents_out = jvp(f, primals, tangents)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload


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
    """Compute the JVP (forward-mode) of *fn* evaluated at *primals*.

    Tangent vectors are propagated through operations automatically via
    each Operation's ``jvp_rule``, which is invoked inside ``Operation.__call__``
    whenever any input carries a tangent (see ``apply_jvp`` in ops/utils.py).

    Args:
        fn: Function ``f(*primals) -> output`` (or ``(output, aux)`` when *has_aux*).
        primals: Tuple of primal inputs — arbitrary pytrees of Tensors.
        tangents: Tuple of tangent vectors, matching the structure of *primals*.
        has_aux: If True, *fn* returns ``(output, aux)`` and aux is passed through.

    Returns:
        ``(primals_out, tangents_out)`` — or ``(primals_out, tangents_out, aux)`` if *has_aux*.
    """
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    if not isinstance(primals, tuple):
        raise TypeError(f"primals must be a tuple, got {type(primals)}")
    if not isinstance(tangents, tuple):
        raise TypeError(f"tangents must be a tuple, got {type(tangents)}")
    if len(primals) != len(tangents):
        raise ValueError(
            f"primals and tangents must have the same length, "
            f"got {len(primals)} and {len(tangents)}"
        )

    # 1. Attach tangent TensorImpls to primal Tensors.
    _attach_tangents(primals, tangents)

    try:
        # 2. Run the function — tangents propagate through each op automatically.
        raw_output = fn(*primals)
    finally:
        # 3. Detach tangents from primals (clean up), regardless of success/failure.
        _detach_tangents(primals)

    # 4. Separate output / aux.
    if has_aux:
        if not isinstance(raw_output, tuple) or len(raw_output) != 2:
            raise ValueError(
                "jvp with has_aux=True expects fn to return (output, aux), "
                f"got {type(raw_output)}"
            )
        output, aux = raw_output
    else:
        output = raw_output
        aux = None

    # 5. Extract tangents from output tensors.
    output_tangents = _extract_tangents(output)

    # 6. Clean tangents from output tensors.
    _detach_tangents_from_tree(output)

    if has_aux:
        return output, output_tangents, aux
    return output, output_tangents


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attach_tangents(primals: tuple, tangents: tuple) -> None:
    """Attach tangent TensorImpls to corresponding primal Tensors."""
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    for primal, tangent in zip(primals, tangents):
        primal_leaves = pytree.tree_leaves(primal) if not isinstance(primal, Tensor) else [primal]
        tangent_leaves = pytree.tree_leaves(tangent) if not isinstance(tangent, Tensor) else [tangent]

        for p, t in zip(primal_leaves, tangent_leaves):
            if isinstance(p, Tensor) and isinstance(t, Tensor):
                p._impl.tangent = t._impl


def _detach_tangents(primals: tuple) -> None:
    """Remove tangent references from primal Tensors."""
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    for primal in primals:
        leaves = pytree.tree_leaves(primal) if not isinstance(primal, Tensor) else [primal]
        for p in leaves:
            if isinstance(p, Tensor):
                p._impl.tangent = None


def _extract_tangents(output: Any) -> Any:
    """Extract tangent values from output, mirroring the output's pytree structure."""
    from ..core.common import pytree
    from ..core.tensor.api import Tensor
    from ..ops.creation import zeros_like

    def _get_tangent(x: Any) -> Any:
        if isinstance(x, Tensor):
            if x._impl.tangent is not None:
                return Tensor(impl=x._impl.tangent)
            else:
                return zeros_like(x)
        return x

    return pytree.tree_map(_get_tangent, output)


def _detach_tangents_from_tree(tree: Any) -> None:
    """Remove tangent fields from all Tensors in a pytree."""
    from ..core.common import pytree
    from ..core.tensor.api import Tensor

    def _clear(x: Any) -> Any:
        if isinstance(x, Tensor):
            x._impl.tangent = None
        return x

    pytree.tree_map(_clear, tree)


__all__ = ["jvp"]
