# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian via reverse-mode autodiff (jacrev)."""

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
    """Compute Jacobian of *fn* via reverse-mode (one VJP per output element)."""

    def jacrev_fn(*args: Any) -> Any:
        from ..core.common.pytree import tree_flatten, tree_unflatten
        from ..core.tensor.api import Tensor
        from ..ops.view.shape import stack
        from .vjp import vjp

        target_fn = fn
        if getattr(target_fn, "_nabla_is_grad_wrapper", False) and not getattr(
            target_fn, "_nabla_grad_create_graph", False
        ):
            from ..core.autograd.api import grad as grad_transform

            target_fn = grad_transform(
                getattr(target_fn, "_nabla_grad_fun"),
                argnums=getattr(target_fn, "_nabla_grad_argnums", 0),
                create_graph=True,
                realize=False,
            )

        diff_args, partial_func = create_jacobian_helpers(target_fn, argnums, args)
        needs_higher_order = any(
            isinstance(leaf, Tensor) and (leaf.is_traced or leaf.tangent is not None)
            for leaf in tree_flatten(diff_args)[0]
        )
        if has_aux:
            output, pullback, aux = vjp(
                partial_func,
                *diff_args,
                has_aux=True,
                create_graph=needs_higher_order,
            )
        else:
            output, pullback = vjp(
                partial_func, *diff_args, create_graph=needs_higher_order
            )
            aux = None

        flat_out, out_td = tree_flatten(output, is_leaf=lambda x: isinstance(x, Tensor))
        sizes, cotangent_basis = std_basis(flat_out)
        cotangent_basis = lift_basis_to_batch_prefix(cotangent_basis, flat_out)
        basis_tree = tree_unflatten(out_td, cotangent_basis)

        total_out = sum(sizes)
        flat_basis, _ = tree_flatten(basis_tree, is_leaf=lambda x: isinstance(x, Tensor))

        pullback_rows: list[Any] = []
        for i in range(total_out):
            row_basis = tree_unflatten(out_td, [b[i] for b in flat_basis])
            pullback_rows.append(pullback(row_basis))

        first_row_flat, row_td = tree_flatten(
            pullback_rows[0], is_leaf=lambda x: isinstance(x, Tensor)
        )
        stacked_rows: list[list[Tensor]] = [[] for _ in range(len(first_row_flat))]
        for row in pullback_rows:
            row_flat, _ = tree_flatten(row, is_leaf=lambda x: isinstance(x, Tensor))
            for j, t in enumerate(row_flat):
                stacked_rows[j].append(t)

        batched_grads = tree_unflatten(
            row_td,
            [stack(row_tensors, axis=0) for row_tensors in stacked_rows],
        )
        flat_diff_args, _ = tree_flatten(
            diff_args, is_leaf=lambda x: isinstance(x, Tensor)
        )

        jacobian = _reshape_jacrev(
            batched_grads, flat_out, flat_diff_args, sizes, diff_args
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
    """Reshape vmap(pullback) results into Jacobian shape ``(*out, *in)``."""
    from ..ops.view.shape import reshape

    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1
    all_grads = (
        ([batched_grads[0]] if isinstance(batched_grads, tuple) else [batched_grads])
        if single_arg
        else list(batched_grads)
    )
    total_out = sum(sizes)

    if len(flat_out) == 1:
        out_shape = tuple(int(d) for d in flat_out[0].shape)
        if len(all_grads) == 1:
            grad = all_grads[0]
            in_shape = tuple(int(d) for d in grad.shape[1:])
            if out_shape == ():
                # scalar output → grad shape is (1, *in_shape) → squeeze to in_shape
                return reshape(grad, in_shape) if total_out == 1 else grad
            else:
                return reshape(grad, out_shape + in_shape)
        else:
            jacs = []
            for grad in all_grads:
                in_shape = tuple(int(d) for d in grad.shape[1:])
                if out_shape == ():
                    jacs.append(reshape(grad, in_shape) if total_out == 1 else grad)
                else:
                    jacs.append(reshape(grad, out_shape + in_shape))
            return tuple(jacs)
    else:
        # Multiple output leaves → split and reshape for each
        # TODO: Handle multi-output case (tuple/list/dict outputs)
        # For now, raise for complex cases
        raise NotImplementedError(
            "jacrev with multiple output tensors is not yet supported. "
            "Wrap your function to return a single tensor."
        )


__all__ = ["jacrev"]
