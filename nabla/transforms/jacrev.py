# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Jacobian computation via reverse-mode autodiff (jacrev).

``jacrev`` computes the Jacobian by running one VJP per output element.
For *m* outputs, it performs *m* reverse-mode passes via ``vmap(vjp_fn)``.

Usage matches JAX::

    jac = jacrev(f)(x)           # shape: (*f_shape, *x_shape)
    jac = jacrev(f, argnums=0)(x, y)
    jac = jacrev(f, argnums=(0, 1))(x, y)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def jacrev(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None = None,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """Compute the Jacobian of *fn* using reverse-mode autodiff.

    For a function ``f : R^n -> R^m``, the Jacobian is an ``(m, n)`` matrix.
    ``jacrev`` computes it column-by-column: one VJP per output element,
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

    def jacrev_fn(*args: Any) -> Any:
        from .vjp import vjp
        from .vmap import vmap
        from ..core.common.pytree import tree_flatten, tree_unflatten, tree_structure
        from ..core.tensor.api import Tensor
        from ..ops.creation import zeros_like
        from ..ops.view.shape import reshape

        # ── 1. Resolve argnums and create partial function ──────────────
        diff_args, partial_func = _create_jacobian_helpers(fn, argnums, args)

        # ── 2. VJP to get output + pullback ─────────────────────────────
        if has_aux:
            output, pullback, aux = vjp(partial_func, *diff_args, has_aux=True)
        else:
            output, pullback = vjp(partial_func, *diff_args)

        # ── 3. Build standard cotangent basis for output ────────────────
        # For each scalar entry in the output we need a one-hot cotangent
        # with the same pytree structure, batched along axis-0.
        flat_out, out_treedef = tree_flatten(output, is_leaf=lambda x: isinstance(x, Tensor))
        sizes, cotangent_basis = _std_basis(flat_out)
        # cotangent_basis: list of Tensors with shape (total_out_elems, *leaf.shape)

        # Reassemble into the output pytree so pullback can consume it.
        # The pullback expects a single cotangent matching the output structure.
        # We pass a batched version and vmap over axis 0.
        basis_tree = tree_unflatten(out_treedef, cotangent_basis)

        # ── 4. vmap(pullback) over the cotangent basis ──────────────────
        # in_axes: axis 0 for every cotangent leaf in the basis tree.
        # The pullback returns a tuple of gradient pytrees, one per diff arg.
        batched_grads = vmap(pullback, in_axes=0)(basis_tree)
        # batched_grads: tuple of pytrees, each tensor has shape (total_out, *arg_shape)

        # ── 5. Split batched gradients per output leaf and reshape ──────
        flat_diff_args, _ = tree_flatten(diff_args, is_leaf=lambda x: isinstance(x, Tensor))

        jacobian = _reshape_jacrev(
            batched_grads, flat_out, flat_diff_args, sizes, diff_args
        )

        if has_aux:
            return jacobian, aux
        return jacobian

    return jacrev_fn


def _reshape_jacrev(
    batched_grads, flat_out, flat_diff_args, sizes, diff_args,
):
    """Reshape vmap(pullback) results into proper Jacobian shape.

    For a function f: R^n -> R^m, the Jacobian J has shape (*out_shape, *in_shape).
    batched_grads has shape (total_out_elems, *in_shape) for each arg.
    We split along axis 0 by output component sizes, then reshape.
    """
    from ..core.tensor.api import Tensor
    from ..ops.view.shape import reshape
    from ..ops.multi_output import split
    from ..core.common.pytree import tree_flatten, tree_unflatten, tree_structure

    single_arg = not isinstance(diff_args, tuple) or len(diff_args) == 1

    # When there's only one diff arg, batched_grads is (grad,) — a 1-tuple
    # When there are multiple, batched_grads is a tuple of grads per arg
    if single_arg:
        # batched_grads is a 1-tuple; the tensor has shape (total_out, *in_shape)
        all_grads = [batched_grads[0]] if isinstance(batched_grads, tuple) else [batched_grads]
    else:
        all_grads = list(batched_grads)

    # For each output component, reshape the gradient slice
    # Final Jacobian structure: one entry per output leaf, each being
    # a pytree matching diff_args with shape (*out_leaf_shape, *in_leaf_shape)
    total_out = sum(sizes)

    if len(flat_out) == 1:
        # Single output: no need to split
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


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities for jacrev and jacfwd
# ═══════════════════════════════════════════════════════════════════════════


def _create_jacobian_helpers(
    fn: Callable,
    argnums: int | tuple[int, ...] | list[int] | None,
    args: tuple,
) -> tuple[tuple, Callable]:
    """Resolve argnums and create a partial function over non-diff args.

    Returns:
        (diff_args, partial_func) where partial_func(*diff_args) == fn(*full_args).
    """
    if argnums is None:
        selected = tuple(range(len(args)))
    elif isinstance(argnums, int):
        selected = (argnums,)
    else:
        selected = tuple(argnums)

    # Normalize negative indices
    normalized = tuple(a if a >= 0 else len(args) + a for a in selected)
    for a in normalized:
        if not (0 <= a < len(args)):
            raise ValueError(
                f"argnum {a} out of bounds for function with {len(args)} arguments"
            )

    diff_args = tuple(args[i] for i in normalized)

    def partial_func(*diff_args_inner):
        full_args = list(args)
        for idx, arg in zip(normalized, diff_args_inner, strict=False):
            full_args[idx] = arg
        return fn(*full_args)

    return diff_args, partial_func


def _std_basis(flat_tensors: list) -> tuple[list[int], list]:
    """Create standard basis (one-hot) vectors for Jacobian computation.

    For each tensor in the flat list, creates a batched tensor of shape
    ``(total_elements, *tensor.shape)`` where each slice along axis 0
    is a one-hot vector activating exactly one element.

    The total batch size equals the sum of all elements across all tensors,
    so ``vmap(pullback)`` or ``vmap(jvp)`` computes one row/column of
    the Jacobian per batch element.

    Args:
        flat_tensors: List of Tensor leaves (output leaves for jacrev,
                      input leaves for jacfwd).

    Returns:
        (sizes, basis_tensors):
            sizes: list of int — number of elements per tensor.
            basis_tensors: list of Tensors — batched one-hot bases.
    """
    import numpy as np
    from ..core.tensor.api import Tensor

    # Compute total elements across all tensors
    sizes = []
    for t in flat_tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        sizes.append(max(n, 1))  # Scalars count as 1 element

    total = sum(sizes)

    basis_tensors = []
    offset = 0

    for t, n_elems in zip(flat_tensors, sizes):
        t_shape = tuple(int(d) for d in t.shape)

        if t_shape == ():
            # Scalar: basis is a vector of length total with 1 at positions [offset:offset+1]
            basis_np = np.zeros((total,), dtype=np.float32)
            basis_np[offset] = 1.0
            # Shape: (total,) — vmap will vectorize over axis 0
            basis_tensor = Tensor.from_dlpack(basis_np)
        else:
            # Tensor: construct (total, *t_shape) one-hot basis
            # Each of the n_elems slices in [offset:offset+n_elems]
            # has exactly one 1.0 at a different position.
            batched_shape = (total,) + t_shape
            basis_np = np.zeros(batched_shape, dtype=np.float32)

            # Fill the identity block
            flat_view = basis_np.reshape(total, n_elems)
            for j in range(n_elems):
                flat_view[offset + j, j] = 1.0

            basis_tensor = Tensor.from_dlpack(basis_np)

        basis_tensors.append(basis_tensor)
        offset += n_elems

    return sizes, basis_tensors


__all__ = ["jacrev"]
