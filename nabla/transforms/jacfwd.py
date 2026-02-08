# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian via forward-mode autodiff (jacfwd)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .utils import create_jacobian_helpers, split_aux, std_basis


def jacfwd(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Compute Jacobian of *fn* via forward-mode (one JVP per input element)."""

    def jacfwd_fn(*args: Any) -> Any:
        from .jvp import jvp
        from .vmap import vmap
        from ..core.common.pytree import tree_flatten, tree_unflatten
        from ..core.tensor.api import Tensor
        from ..ops.view.shape import reshape

        diff_args, partial_func = create_jacobian_helpers(fn, argnums, args)

        # Evaluate once to get output shape (must happen before vmap).
        raw = partial_func(*diff_args)
        test_output, aux = split_aux(raw, has_aux, name="jacfwd")

        flat_inputs, in_treedef = tree_flatten(
            diff_args, is_leaf=lambda x: isinstance(x, Tensor)
        )
        sizes, tangent_basis = std_basis(flat_inputs)
        total_in = sum(sizes)

        # Primals captured via closure (not through vmap) to avoid batch_dims mismatch.

        def jvp_one_dir(*tangents_flat):
            primals_tree = tree_unflatten(in_treedef, list(flat_inputs))
            tangents_tree = tree_unflatten(in_treedef, list(tangents_flat))
            if not isinstance(primals_tree, tuple):
                primals_tree = (primals_tree,)
            if not isinstance(tangents_tree, tuple):
                tangents_tree = (tangents_tree,)

            if has_aux:
                _, tangent_out, aux_out = jvp(
                    partial_func, primals_tree, tangents_tree, has_aux=True
                )
                return tangent_out  # aux handled separately
            else:
                _, tangent_out = jvp(partial_func, primals_tree, tangents_tree)
                return tangent_out

        output_tangents = vmap(jvp_one_dir, in_axes=tuple(0 for _ in tangent_basis))(
            *tangent_basis
        )

        # ── Reshape into Jacobian ──
        jacobian = _reshape_jacfwd(
            output_tangents, test_output, flat_inputs, sizes, diff_args
        )

        if has_aux:
            return jacobian, aux
        return jacobian

    return jacfwd_fn


def _reshape_jacfwd(output_tangents, test_output, flat_inputs, sizes, diff_args):
    """Reshape vmap(jvp) results into Jacobian shape ``(*out, *in)``."""
    from ..core.tensor.api import Tensor
    from ..ops.view.shape import reshape
    from ..ops.multi_output import split
    from ..core.common.pytree import tree_flatten

    flat_out, _ = tree_flatten(
        test_output, is_leaf=lambda x: isinstance(x, Tensor)
    )

    total_in = sum(sizes)
    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1

    if isinstance(output_tangents, Tensor):
        out_shape = tuple(int(d) for d in flat_out[0].shape) if flat_out else ()

        if single_arg:
            in_shape = tuple(int(d) for d in flat_inputs[0].shape)
            if in_shape == ():
                return reshape(output_tangents, out_shape) if total_in == 1 else output_tangents
            else:
                intermediate = reshape(output_tangents, in_shape + out_shape)
                if out_shape:
                    in_ndim = len(in_shape)
                    out_ndim = len(out_shape)
                    perm = tuple(range(in_ndim, in_ndim + out_ndim)) + tuple(range(in_ndim))
                    return _permute_tensor(intermediate, perm)
                else:
                    return reshape(output_tangents, in_shape)
        else:
            if total_in > 1:
                chunks = split(output_tangents, num_splits=total_in, axis=0)
            else:
                chunks = (output_tangents,)

            jacs = []
            offset = 0
            for inp, n_elems in zip(flat_inputs, sizes):
                in_shape = tuple(int(d) for d in inp.shape)
                inp_chunks = chunks[offset:offset + n_elems]
                if len(inp_chunks) == 1:
                    J = reshape(inp_chunks[0], out_shape + in_shape) if in_shape == () else inp_chunks[0]
                else:
                    from ..ops.view.shape import concatenate
                    stacked = concatenate(list(inp_chunks), axis=0)
                    intermediate = reshape(stacked, in_shape + out_shape)
                    if out_shape:
                        in_ndim = len(in_shape)
                        out_ndim = len(out_shape)
                        perm = tuple(range(in_ndim, in_ndim + out_ndim)) + tuple(range(in_ndim))
                        J = _permute_tensor(intermediate, perm)
                    else:
                        J = reshape(stacked, in_shape)
                jacs.append(J)
                offset += n_elems

            return tuple(jacs)
    else:
        raise NotImplementedError(
            "jacfwd with multiple output tensors is not yet supported. "
            "Wrap your function to return a single tensor."
        )


def _permute_tensor(t, perm: tuple[int, ...]):
    from ..ops.view.axes import swap_axes
    ndim = len(perm)
    current = list(range(ndim))

    for target_pos in range(ndim):
        target_dim = perm[target_pos]
        current_pos = current.index(target_dim)
        if current_pos != target_pos:
            t = swap_axes(t, target_pos, current_pos)
            current[target_pos], current[current_pos] = current[current_pos], current[target_pos]

    return t


__all__ = ["jacfwd"]
