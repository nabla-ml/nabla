# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian via reverse-mode autodiff (jacrev).

Uses ``vmap(pullback)`` — the same pattern as JAX and the original nabla
implementation.  The outer ``vmap`` adds a batch dimension over the standard
basis cotangent directions so that all VJPs are computed in a single batched
call.  This naturally composes for higher-order derivatives: each nesting
level adds another batch dimension, and all intermediate ops handle batch
dims generically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.tensor.api import Tensor

from .utils import create_jacobian_helpers, lift_basis_to_batch_prefix, std_basis


def jacrev(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Compute Jacobian of *fn* via reverse-mode (``vmap`` over VJP cotangents)."""

    def jacrev_fn(*args: Any) -> Any:
        from ..core.common.pytree import tree_flatten, tree_unflatten
        from ..core.tensor.api import Tensor
        from ..ops.view.shape import stack
        from .vjp import vjp

        diff_args, partial_func = create_jacobian_helpers(fn, argnums, args)

        # ── VJP: forward pass + capture pullback ──
        if has_aux:
            output, pullback, aux = vjp(
                partial_func, *diff_args, has_aux=True
            )
        else:
            output, pullback = vjp(partial_func, *diff_args)
            aux = None

        # ── Build standard basis for OUTPUT arguments ──
        flat_out, out_td = tree_flatten(
            output, is_leaf=lambda x: isinstance(x, Tensor)
        )
        sizes, cotangent_basis = std_basis(flat_out)
        cotangent_basis = lift_basis_to_batch_prefix(cotangent_basis, flat_out)

        total_out = sum(sizes)
        pullback_rows: list[Any] = []
        for i in range(total_out):
            cot_flat = [basis_leaf[i] for basis_leaf in cotangent_basis]
            cot_tree = tree_unflatten(out_td, cot_flat)
            pullback_rows.append(pullback(cot_tree))

        first_row_flat, row_td = tree_flatten(
            pullback_rows[0], is_leaf=lambda x: isinstance(x, Tensor)
        )
        stacked_rows: list[list[Tensor]] = [[] for _ in range(len(first_row_flat))]
        for row in pullback_rows:
            row_flat, _ = tree_flatten(row, is_leaf=lambda x: isinstance(x, Tensor))
            for j, t in enumerate(row_flat):
                stacked_rows[j].append(t)

        grads = tree_unflatten(
            row_td,
            [
                rows[0] if len(rows) == 1 else stack(rows, axis=0)
                for rows in stacked_rows
            ],
        )

        # ── Split and reshape into Jacobian ──
        flat_diff_args, _ = tree_flatten(
            diff_args, is_leaf=lambda x: isinstance(x, Tensor)
        )

        jacobian = _reshape_jacrev(
            grads, flat_out, flat_diff_args, sizes, diff_args
        )

        if has_aux:
            return jacobian, aux
        return jacobian

    return jacrev_fn


def _reshape_jacrev(
    batched_grads: Any,
    flat_out: list[Tensor],
    flat_diff_args: list[Tensor],
    sizes: list[int],
    diff_args: tuple[Any, ...],
) -> Any:
    """Reshape vmap(pullback) results into Jacobian ``(*out, *in)``.

    ``batched_grads`` is a tuple of tensors each with leading dim ``total_out``.
    We split by output sizes, then reshape each block to ``(*out_shape, *in_shape)``.
    """
    from ..ops.view.shape import reshape

    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1
    total_out = sum(sizes)

    # batched_grads is the result from vmap(pullback)(basis).
    # For single-arg it's a Tensor; for multi-arg it's a tuple.
    if single_arg:
        all_grads = [
            batched_grads[0] if isinstance(batched_grads, tuple) else batched_grads
        ]
    else:
        all_grads = list(batched_grads)

    if len(flat_out) == 1:
        out_shape = tuple(int(d) for d in flat_out[0].shape)

        # Handle scalar outputs that got an extra dim from vmap
        if flat_out[0].batch_dims > 0 and out_shape == (1,):
            out_shape = ()

        if len(all_grads) == 1:
            grad = all_grads[0]
            expected_in_shape = tuple(int(d) for d in flat_diff_args[0].shape)
            if out_shape == ():
                # Scalar-output case: vmap over a single cotangent direction can
                # already return gradient shape ``(*in_shape)`` (no leading
                # output-basis dim to strip). Only reshape if a singleton leading
                # basis dimension is present.
                grad_shape = tuple(int(d) for d in grad.shape)
                if total_out == 1:
                    if grad_shape == expected_in_shape:
                        return grad
                    if len(grad_shape) == len(expected_in_shape) + 1 and grad_shape[0] == 1:
                        return reshape(grad, expected_in_shape)
                    return reshape(grad, expected_in_shape)
                return grad
            else:
                in_shape = tuple(int(d) for d in grad.shape[1:])
                return reshape(grad, out_shape + in_shape)
        else:
            jacs = []
            for grad, arg in zip(all_grads, flat_diff_args, strict=False):
                expected_in_shape = tuple(int(d) for d in arg.shape)
                if out_shape == ():
                    grad_shape = tuple(int(d) for d in grad.shape)
                    if total_out == 1:
                        if grad_shape == expected_in_shape:
                            jacs.append(grad)
                        elif len(grad_shape) == len(expected_in_shape) + 1 and grad_shape[0] == 1:
                            jacs.append(reshape(grad, expected_in_shape))
                        else:
                            jacs.append(reshape(grad, expected_in_shape))
                    else:
                        jacs.append(grad)
                else:
                    in_shape = tuple(int(d) for d in grad.shape[1:])
                    jacs.append(reshape(grad, out_shape + in_shape))
            return tuple(jacs)
    else:
        raise NotImplementedError(
            "jacrev with multiple output tensors is not yet supported. "
            "Wrap your function to return a single tensor."
        )


__all__ = ["jacrev"]
