# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #
"""Shared utilities for nabla transforms — deduplicates repeated patterns."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.tensor.api import Tensor

# Non-differentiable dtypes (lazily initialised).
_NON_DIFF_DTYPES: frozenset | None = None


def _get_non_diff_dtypes() -> frozenset:
    global _NON_DIFF_DTYPES
    if _NON_DIFF_DTYPES is None:
        from max.dtype import DType

        _NON_DIFF_DTYPES = frozenset(
            {
                DType.bool,
                DType.int8,
                DType.int16,
                DType.int32,
                DType.int64,
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
            }
        )
    return _NON_DIFF_DTYPES


def split_aux(raw: Any, has_aux: bool, name: str) -> tuple[Any, Any]:
    """Split ``(output, aux)`` when *has_aux*, else return ``(raw, None)``."""
    if not has_aux:
        return raw, None
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ValueError(
            f"{name} with has_aux=True expects fn to return (output, aux), "
            f"got {type(raw)}"
        )
    return raw[0], raw[1]


def resolve_argnums(
    argnums: int | tuple[int, ...] | list[int] | None,
    n_args: int,
) -> tuple[int, ...]:
    """Normalise *argnums* to a canonical ``tuple[int, ...]``."""
    if argnums is None:
        sel = tuple(range(n_args))
    elif isinstance(argnums, int):
        sel = (argnums,)
    else:
        sel = tuple(argnums)
    norm = tuple(a if a >= 0 else n_args + a for a in sel)
    for a in norm:
        if not (0 <= a < n_args):
            raise ValueError(f"argnum {a} out of bounds for {n_args} arguments")
    return norm


def select_argnums(
    grads_struct: tuple[Any, ...], argnums: int | tuple[int, ...]
) -> Any:
    """Index *grads_struct* by *argnums* (int → element, tuple → tuple)."""
    if isinstance(argnums, int):
        return grads_struct[argnums] if len(grads_struct) > argnums else grads_struct
    if isinstance(argnums, (tuple, list)):
        return tuple(grads_struct[i] for i in argnums)
    return grads_struct


def collect_grads(
    grads_map: dict[Tensor, Tensor],
    input_leaves: list[Any],
    *,
    skip_non_diff: bool = True,
) -> list[Tensor | None]:
    """Collect per-leaf gradients from a backward *grads_map*."""
    from ..core.tensor.api import Tensor
    from ..ops.creation import zeros_like

    non_diff = _get_non_diff_dtypes() if skip_non_diff else frozenset()
    result: list[Tensor | None] = []
    for leaf in input_leaves:
        if not isinstance(leaf, Tensor) or skip_non_diff and leaf.dtype in non_diff:
            result.append(None)
        elif leaf in grads_map:
            result.append(grads_map[leaf])
        else:
            result.append(zeros_like(leaf))
    return result


def realize_tensors(tensors: list[Any]) -> None:
    """Batch-realize any lazy Tensors in *tensors*."""
    from ..core.tensor.api import Tensor, realize_all

    unrealized = [t for t in tensors if isinstance(t, Tensor) and not t.is_realized]
    if unrealized:
        realize_all(*unrealized)


def create_jacobian_helpers(
    fn: Callable[..., Any],
    argnums: int | tuple[int, ...] | list[int] | None,
    args: tuple[Any, ...],
) -> tuple[tuple[Any, ...], Callable[..., Any]]:
    """Resolve *argnums* and build a partial that fixes non-diff args."""
    norm = resolve_argnums(argnums, len(args))
    diff_args = tuple(args[i] for i in norm)

    def partial_func(*diff_args_inner):
        full = list(args)
        for idx, arg in zip(norm, diff_args_inner, strict=False):
            full[idx] = arg
        return fn(*full)

    return diff_args, partial_func


def std_basis(flat_tensors: list[Tensor]) -> tuple[list[int], list[Tensor]]:
    """One-hot basis vectors for Jacobian computation (shared by jacfwd/jacrev)."""
    import numpy as np

    from ..core.tensor.api import Tensor

    sizes: list[int] = []
    for t in flat_tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        sizes.append(max(n, 1))
    total = sum(sizes)
    basis: list[Tensor] = []
    offset = 0
    for t, n in zip(flat_tensors, sizes, strict=False):
        sh = tuple(int(d) for d in t.shape)
        if sh == ():
            bp = np.zeros((total,), dtype=np.float32)
            bp[offset] = 1.0
        else:
            bp = np.zeros((total,) + sh, dtype=np.float32)
            bp.reshape(total, n)[range(n), range(n)] = 0.0  # init
            for j in range(n):
                bp.reshape(total, n)[offset + j, j] = 1.0
        basis.append(Tensor.from_dlpack(bp))
        offset += n
    return sizes, basis


def lift_basis_to_batch_prefix(
    basis: list[Tensor], references: list[Tensor]
) -> list[Tensor]:
    """Lift Jacobian basis tensors to match reference batch prefixes.

    This is required for nested transform contexts like ``vmap(jacrev(...))`` and
    ``vmap(jacfwd(...))`` where basis vectors must preserve already-active outer
    batch dimensions.
    """
    from ..ops.view.batch import broadcast_batch_dims

    if len(basis) != len(references):
        raise ValueError(
            f"basis/reference length mismatch: {len(basis)} != {len(references)}"
        )

    lifted: list[Tensor] = []
    for b, ref in zip(basis, references, strict=False):
        if ref.batch_dims <= 0:
            lifted.append(b)
            continue

        phys = ref.physical_global_shape or ref.local_shape
        if phys is None:
            lifted.append(b)
            continue

        ref_batch_shape = tuple(int(d) for d in phys[: ref.batch_dims])
        if b.batch_dims < ref.batch_dims:
            lifted.append(broadcast_batch_dims(b, ref_batch_shape))
        else:
            lifted.append(b)

    return lifted
