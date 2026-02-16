# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import is_tensor, realize_all, tree_leaves
from .tensor.api import Tensor


def _collect_unrealized(obj: Any) -> list[Tensor]:
    tensors: list[Tensor] = []
    for leaf in tree_leaves(obj):
        if is_tensor(leaf) and not leaf.real:
            tensors.append(leaf)
    return tensors


def batch_realize(*objs: Any) -> None:
    """Batch-realize all Nabla tensors found in arbitrary pytree-like objects."""
    to_realize: list[Tensor] = []
    for obj in objs:
        to_realize.extend(_collect_unrealized(obj))
    if to_realize:
        realize_all(*to_realize)


def _to_numpy_like(obj: Any) -> Any:
    if is_tensor(obj):
        return obj.to_numpy()

    torch_mod = None
    try:
        import torch as torch_mod  # type: ignore
    except Exception:
        torch_mod = None

    if torch_mod is not None and isinstance(obj, torch_mod.Tensor):
        return obj.detach().cpu().numpy()

    if isinstance(obj, Mapping):
        return {k: _to_numpy_like(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(_to_numpy_like(v) for v in obj)

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_numpy_like(v) for v in obj]

    return np.asarray(obj)


def assert_allclose(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    realize: bool = True,
) -> None:
    """Assert numerical closeness across Nabla/JAX/PyTorch/NumPy objects.

    By default realizes Nabla tensors first; pass `realize=False` if callers
    already did an explicit `batch_realize(...)` for efficiency.
    """
    if realize:
        batch_realize(actual, expected)

    np.testing.assert_allclose(
        _to_numpy_like(actual),
        _to_numpy_like(expected),
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
