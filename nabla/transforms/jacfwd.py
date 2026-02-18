# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian via forward-mode autodiff (jacfwd).

Uses ``vmap(jvp)`` — the same pattern as JAX and the original nabla
implementation.  The outer ``vmap`` adds a batch dimension over the standard
basis directions so that all JVPs are computed in a single batched call.
This naturally composes for higher-order derivatives: each nesting level
adds another batch dimension, and all intermediate ops handle batch dims
generically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.tensor.api import Tensor

from .utils import (
    create_jacobian_helpers,
    lift_basis_to_batch_prefix,
    split_aux,
    std_basis,
)


def jacfwd(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Compute Jacobian of *fn* via forward-mode (``vmap`` over JVP directions)."""

    def jacfwd_fn(*args: Any) -> Any:
        from ..core.common.pytree import tree_flatten, tree_unflatten
        from ..core.tensor.api import Tensor
        from ..ops.view.shape import stack
        from .jvp import jvp

        diff_args, partial_func = create_jacobian_helpers(fn, argnums, args)

        # ── Build standard basis for INPUT arguments ──
        flat_inputs, in_treedef = tree_flatten(
            diff_args, is_leaf=lambda x: isinstance(x, Tensor)
        )
        sizes, tangent_basis = std_basis(flat_inputs)
        tangent_basis = lift_basis_to_batch_prefix(tangent_basis, flat_inputs)

        primals_tree = tree_unflatten(in_treedef, list(flat_inputs))
        if not isinstance(primals_tree, tuple):
            primals_tree = (primals_tree,)

        total_in = sum(sizes)
        tangent_rows: list[Tensor] = []
        for i in range(total_in):
            tangents_flat = [basis_leaf[i] for basis_leaf in tangent_basis]
            tangents_tree = tree_unflatten(in_treedef, tangents_flat)
            if not isinstance(tangents_tree, tuple):
                tangents_tree = (tangents_tree,)

            if has_aux:
                _, tangent_out, _aux = jvp(
                    partial_func, primals_tree, tangents_tree, has_aux=True
                )
            else:
                _, tangent_out = jvp(partial_func, primals_tree, tangents_tree)

            tangent_flat, _ = tree_flatten(
                tangent_out, is_leaf=lambda x: isinstance(x, Tensor)
            )
            if len(tangent_flat) != 1:
                raise NotImplementedError(
                    "jacfwd with multiple output tensors is not yet supported. "
                    "Wrap your function to return a single tensor."
                )
            tangent_rows.append(tangent_flat[0])

        output_tangents = tangent_rows[0] if len(tangent_rows) == 1 else stack(tangent_rows, axis=0)

        # ── Get test output once for shape information ──
        raw = partial_func(*diff_args)
        test_output, aux = split_aux(raw, has_aux, name="jacfwd")

        # ── Reshape into Jacobian ──
        jacobian = _reshape_jacfwd(
            output_tangents, test_output, flat_inputs, sizes, diff_args
        )

        if has_aux:
            return jacobian, aux
        return jacobian

    return jacfwd_fn


def _reshape_jacfwd(
    output_tangents: Any,
    test_output: Any,
    flat_inputs: list[Tensor],
    sizes: list[int],
    diff_args: tuple[Any, ...],
) -> Any:
    """Reshape vmap(jvp) results into Jacobian shape ``(*out, *in)``.

    After vmap, ``output_tangents`` has shape ``(total_in, *out_shape)``.
    We split by input sizes, reshape each block to ``(in_i_shape, *out_shape)``,
    then permute to ``(*out_shape, *in_i_shape)``."""
    from ..core.common.pytree import tree_flatten
    from ..core.tensor.api import Tensor
    from ..ops.view.axes import permute
    from ..ops.view.shape import reshape

    flat_out, _ = tree_flatten(test_output, is_leaf=lambda x: isinstance(x, Tensor))

    total_in = sum(sizes)
    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1

    if isinstance(output_tangents, Tensor):
        out_shape = tuple(int(d) for d in flat_out[0].shape) if flat_out else ()

        if single_arg:
            in_shape = tuple(int(d) for d in flat_inputs[0].shape)
            if in_shape == ():
                return (
                    reshape(output_tangents, out_shape)
                    if total_in == 1
                    else output_tangents
                )
            else:
                # Reshape (total_in, *out_shape) → (*in_shape, *out_shape)
                intermediate = reshape(output_tangents, in_shape + out_shape)
                if out_shape:
                    in_ndim = len(in_shape)
                    out_ndim = len(out_shape)
                    perm = tuple(range(in_ndim, in_ndim + out_ndim)) + tuple(
                        range(in_ndim)
                    )
                    return permute(intermediate, perm)
                else:
                    return reshape(output_tangents, in_shape)
        else:
            # Multiple input arguments → split along axis 0
            chunks = _split_by_sizes(output_tangents, sizes, axis=0)

            jacs = []
            for inp, chunk in zip(flat_inputs, chunks, strict=False):
                in_shape = tuple(int(d) for d in inp.shape)
                if in_shape == ():
                    J = reshape(chunk, out_shape + in_shape) if out_shape else chunk
                else:
                    intermediate = reshape(chunk, in_shape + out_shape)
                    if out_shape:
                        in_ndim = len(in_shape)
                        out_ndim = len(out_shape)
                        perm = tuple(range(in_ndim, in_ndim + out_ndim)) + tuple(
                            range(in_ndim)
                        )
                        J = permute(intermediate, perm)
                    else:
                        J = reshape(chunk, in_shape)
                jacs.append(J)

            return tuple(jacs)
    else:
        raise NotImplementedError(
            "jacfwd with multiple output tensors is not yet supported. "
            "Wrap your function to return a single tensor."
        )


def _split_by_sizes(
    x: Tensor, sizes: list[int], axis: int = 0
) -> list[Tensor]:
    """Split *x* along *axis* into chunks of given *sizes*."""
    from ..ops.view.shape import slice_tensor

    result = []
    offset = 0
    rank = len(x.shape)
    for sz in sizes:
        start = [0] * rank
        size = [int(d) for d in x.shape]
        start[axis] = offset
        size[axis] = sz
        result.append(slice_tensor(x, start=tuple(start), size=tuple(size)))
        offset += sz
    return result


__all__ = ["jacfwd"]
