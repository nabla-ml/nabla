# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian computation via forward-mode autodiff (jacfwd).

``jacfwd`` computes the Jacobian by running one JVP per input element.
For *n* inputs, it performs *n* forward-mode passes via ``vmap(jvp)``.

Usage matches JAX::

    jac = jacfwd(f)(x)           # shape: (*f_shape, *x_shape)
    jac = jacfwd(f, argnums=0)(x, y)
    jac = jacfwd(f, argnums=(0, 1))(x, y)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jacfwd(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Compute the Jacobian of *fn* using forward-mode autodiff.

    For a function ``f : R^n -> R^m``, the Jacobian is an ``(m, n)`` matrix.
    ``jacfwd`` computes it row-by-row: one JVP per input element,
    vectorised with ``vmap``.

    Args:
        fn: Function to differentiate.
        argnums: Which positional arguments to differentiate w.r.t.
            ``None`` means all, an ``int`` selects one, a tuple selects several.
        has_aux: If True, *fn* returns ``(output, aux)``.

    Returns:
        A function with the same signature as *fn* that returns the Jacobian
        (or ``(jacobian, aux)`` when *has_aux*).
    """

    def jacfwd_fn(*args: Any) -> Any:
        from .jvp import jvp
        from .vmap import vmap
        from .jacrev import _create_jacobian_helpers, _std_basis
        from ..core.common.pytree import tree_flatten, tree_unflatten
        from ..core.tensor.api import Tensor
        from ..ops.view.shape import reshape

        # ── 1. Resolve argnums and create partial function ──────────────
        diff_args, partial_func = _create_jacobian_helpers(fn, argnums, args)

        # ── 2. Evaluate function once to get output shape/structure ─────
        # Must happen BEFORE vmap to avoid stale graph state.
        aux = None
        if has_aux:
            raw = partial_func(*diff_args)
            if isinstance(raw, tuple) and len(raw) == 2:
                test_output, aux = raw
            else:
                test_output = raw
        else:
            test_output = partial_func(*diff_args)

        # ── 3. Build standard tangent basis for inputs ──────────────────
        # Flatten all diff args to get input tensor leaves.
        flat_inputs, in_treedef = tree_flatten(
            diff_args, is_leaf=lambda x: isinstance(x, Tensor)
        )
        sizes, tangent_basis = _std_basis(flat_inputs)
        # tangent_basis: list of Tensors with shape (total_in_elems, *input_leaf.shape)

        total_in = sum(sizes)

        # ── 4. Build vmapped JVP function ───────────────────────────────
        # Primals are captured via closure (NOT passed through vmap) to
        # avoid batch_dims mismatches on multi-dimensional tensors.
        # Only the tangent basis is vectorized through vmap.

        def jvp_one_dir(*tangents_flat):
            """Run jvp with one tangent direction.

            Primals come from closure (flat_inputs), tangents come from vmap.
            """
            # Reconstruct pytree structures
            primals_tree = tree_unflatten(in_treedef, list(flat_inputs))
            tangents_tree = tree_unflatten(in_treedef, list(tangents_flat))

            # Ensure they're tuples for jvp
            if not isinstance(primals_tree, tuple):
                primals_tree = (primals_tree,)
            if not isinstance(tangents_tree, tuple):
                tangents_tree = (tangents_tree,)

            if has_aux:
                _, tangent_out, aux_out = jvp(
                    partial_func, primals_tree, tangents_tree, has_aux=True
                )
                return tangent_out  # We'll handle aux separately
            else:
                _, tangent_out = jvp(partial_func, primals_tree, tangents_tree)
                return tangent_out

        # Only tangent basis goes through vmap (axis 0)
        tangent_axes = tuple(0 for _ in tangent_basis)

        # ── 5. Run vmap(jvp) ────────────────────────────────────────────
        output_tangents = vmap(jvp_one_dir, in_axes=tangent_axes)(
            *tangent_basis
        )
        # output_tangents has shape (total_in, *out_shape) for single output

        # ── 6. Reshape into Jacobian ────────────────────────────────────
        jacobian = _reshape_jacfwd(
            output_tangents, test_output, flat_inputs, sizes, diff_args
        )

        if has_aux:
            return jacobian, aux
        return jacobian

    return jacfwd_fn


def _reshape_jacfwd(
    output_tangents, test_output, flat_inputs, sizes, diff_args,
):
    """Reshape vmap(jvp) results into proper Jacobian shape.

    For f: R^n -> R^m, Jacobian J has shape (*out_shape, *in_shape).
    output_tangents has shape (total_in_elems, *out_shape).
    We need to split by input component and transpose to (*out_shape, *in_shape).
    """
    from ..core.tensor.api import Tensor
    from ..ops.view.shape import reshape
    from ..ops.view.axes import swap_axes
    from ..ops.multi_output import split
    from ..core.common.pytree import tree_flatten

    flat_out, _ = tree_flatten(
        test_output, is_leaf=lambda x: isinstance(x, Tensor)
    )

    total_in = sum(sizes)
    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1

    if isinstance(output_tangents, Tensor):
        # Single output tensor
        out_shape = tuple(int(d) for d in flat_out[0].shape) if flat_out else ()

        if single_arg:
            # Single input, single output
            in_shape = tuple(int(d) for d in flat_inputs[0].shape)
            if in_shape == ():
                # Scalar input → tangents shape is (1, *out_shape) → squeeze
                return reshape(output_tangents, out_shape) if total_in == 1 else output_tangents
            else:
                # Reshape: (total_in, *out_shape) → (*in_shape, *out_shape)
                intermediate = reshape(output_tangents, in_shape + out_shape)
                # Transpose to (*out_shape, *in_shape) to match JAX convention
                if out_shape:
                    in_ndim = len(in_shape)
                    out_ndim = len(out_shape)
                    # Permutation: move out dims first, then in dims
                    perm = tuple(range(in_ndim, in_ndim + out_ndim)) + tuple(range(in_ndim))
                    return _permute_tensor(intermediate, perm)
                else:
                    # Scalar output → no transpose needed, just flatten
                    return reshape(output_tangents, in_shape)
        else:
            # Multiple inputs, single output
            # Split the batch dimension by input sizes
            if total_in > 1:
                chunks = split(output_tangents, num_splits=total_in, axis=0)
                # chunks is a tuple of (1, *out_shape) tensors
            else:
                chunks = (output_tangents,)

            jacs = []
            offset = 0
            for inp, n_elems in zip(flat_inputs, sizes):
                in_shape = tuple(int(d) for d in inp.shape)
                # Gather the n_elems chunks for this input
                inp_chunks = chunks[offset:offset + n_elems]
                if len(inp_chunks) == 1:
                    # Scalar input
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
    """Permute tensor dimensions using successive swap_axes."""
    from ..ops.view.axes import swap_axes

    # Implement general permutation using transpositions
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
