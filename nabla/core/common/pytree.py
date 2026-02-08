# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Pytree utilities for structural manipulation.

Optimized for high-throughput traversals using JAX-style logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from ..tensor.api import Tensor

T = TypeVar("T")


_K_LEAF = 0
_K_NONE = 1
_K_LIST = 2
_K_TUPLE = 3
_K_DICT = 4


class _LeafMarker:
    """Sentinel marking a leaf position."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "*"


_LEAF = _LeafMarker()


class PyTreeDef:
    """Immutable definition of a pytree's structure."""

    __slots__ = ("_kind", "_meta", "_children", "num_leaves")

    def __init__(self, kind: int, meta: Any, children: tuple[PyTreeDef, ...], num_leaves: int) -> None:
        self._kind = kind
        self._meta = meta
        self._children = children
        self.num_leaves = num_leaves

    def __repr__(self) -> str:
        if self._kind == _K_LEAF:
            return "*"
        if self._kind == _K_NONE:
            return "None"
        if self._kind == _K_LIST:
            return f"list[{len(self._children)}]"
        if self._kind == _K_TUPLE:
            return f"tuple[{len(self._children)}]"
        if self._kind == _K_DICT:
            return f"dict{list(self._meta)}"
        return "PyTreeDef(?)"

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, PyTreeDef):
            return False
        return (
            self._kind == other._kind
            and self._meta == other._meta
            and self._children == other._children
        )

    def __hash__(self) -> int:
        return hash((self._kind, self._meta, self._children))

    def unflatten(self, leaves: list[Any]) -> Any:
        """Reconstruct pytree from leaves."""
        return tree_unflatten(self, leaves)


def tree_flatten(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], PyTreeDef]:
    """Flatten a pytree into leaves and structure."""
    leaves = []

    def _flatten(node: Any) -> PyTreeDef:
        if is_leaf is not None and is_leaf(node):
            leaves.append(node)
            return PyTreeDef(_K_LEAF, None, (), 1)

        if node is None:
            return PyTreeDef(_K_NONE, None, (), 0)

        if isinstance(node, list):
            children = tuple(_flatten(v) for v in node)
            num = sum(c.num_leaves for c in children)
            return PyTreeDef(_K_LIST, None, children, num)

        if isinstance(node, tuple):
            children = tuple(_flatten(v) for v in node)
            num = sum(c.num_leaves for c in children)
            return PyTreeDef(_K_TUPLE, type(node), children, num)

        if isinstance(node, dict):
            keys = sorted(node.keys())
            children = tuple(_flatten(node[k]) for k in keys)
            num = sum(c.num_leaves for c in children)
            return PyTreeDef(_K_DICT, tuple(keys), children, num)

        leaves.append(node)
        return PyTreeDef(_K_LEAF, None, (), 1)

    treedef = _flatten(tree)
    return leaves, treedef


def tree_unflatten(treedef: PyTreeDef, leaves: list[Any]) -> Any:
    """Reconstruct a pytree from structure info and leaves."""
    if len(leaves) != treedef.num_leaves:
        raise ValueError(f"Expected {treedef.num_leaves} leaves, got {len(leaves)}")

    leaves_iter = iter(leaves)

    def _build(def_: PyTreeDef) -> Any:
        k = def_._kind

        if k == _K_LEAF:
            return next(leaves_iter)

        if k == _K_NONE:
            return None

        if k == _K_LIST:
            return [_build(c) for c in def_._children]

        if k == _K_TUPLE:

            cls = def_._meta
            items = (_build(c) for c in def_._children)

            return cls(items) if cls is tuple else cls(*items)

        if k == _K_DICT:
            return {
                key: _build(child)
                for key, child in zip(def_._meta, def_._children, strict=False)
            }

        raise ValueError(f"Unknown node kind: {k}")

    return _build(treedef)


def tree_flatten_full(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[list, PyTreeDef]:
    """Flatten a pytree into leaves and structure, treating None as a leaf."""
    full_is_leaf = lambda x: (x is None) or (is_leaf(x) if is_leaf else False)
    return tree_flatten(tree, is_leaf=full_is_leaf)


def tree_unflatten_full(treedef: PyTreeDef, leaves: list[Any]) -> Any:
    """Reconstruct a pytree from structure info and leaves, supporting None leaves."""
    return tree_unflatten(treedef, leaves)


def tree_leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """Get all leaves from a pytree (optimized version - doesn't build treedef)."""
    leaves = []

    def _collect(node: Any) -> None:
        if is_leaf is not None and is_leaf(node):
            leaves.append(node)
            return

        if node is None:
            return

        if isinstance(node, list):
            for v in node:
                _collect(v)
            return

        if isinstance(node, tuple):
            for v in node:
                _collect(v)
            return

        if isinstance(node, dict):
            for k in sorted(node.keys()):
                _collect(node[k])
            return

        leaves.append(node)

    _collect(tree)
    return leaves


def tree_structure(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTreeDef:
    """Get structure info from a pytree."""
    _, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_map(
    fn: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Apply a function to every leaf of a pytree."""

    def _map(primary: Any, *others: Any) -> Any:
        if is_leaf is not None and is_leaf(primary):
            return fn(primary, *others)

        if primary is None:
            if not all(x is None for x in others):
                raise ValueError("Tree structure mismatch: Expected None")
            return None

        if isinstance(primary, list):
            if not all(isinstance(x, list) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: List length or type")
            return [_map(c, *[x[i] for x in others]) for i, c in enumerate(primary)]

        if isinstance(primary, tuple):
            if not all(isinstance(x, tuple) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: Tuple length or type")

            result_tuple = tuple(
                _map(c, *[x[i] for x in others]) for i, c in enumerate(primary)
            )
            if type(primary) is tuple:
                return result_tuple
            return type(primary)(*result_tuple)

        if isinstance(primary, dict):
            if not all(isinstance(x, dict) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: Dict size or type")

            return {k: _map(v, *[x[k] for x in others]) for k, v in primary.items()}

        return fn(primary, *others)

    return _map(tree, *rest)


def traced(tree: Any) -> Any:
    """Mark all tensors in a pytree as traced."""
    from ..tensor.api import Tensor

    return tree_map(
        lambda x: (
            setattr(x._impl, "is_traced", True) or x if isinstance(x, Tensor) else x
        ),
        tree,
    )


def untraced(tree: Any) -> Any:
    """Mark all tensors in a pytree as untraced."""
    from ..tensor.api import Tensor

    return tree_map(
        lambda x: (
            setattr(x._impl, "is_traced", False) or x if isinstance(x, Tensor) else x
        ),
        tree,
    )


def with_batch_dims(tree: Any, delta: int) -> Any:
    """Adjust batch_dims on all tensors in a pytree."""
    from ..tensor.api import Tensor

    def _adj(x: Any) -> Any:
        if isinstance(x, Tensor):
            x.batch_dims += delta
        return x

    return tree_map(_adj, tree)


def tensor_leaves(tree: Any) -> list[Tensor]:
    """Get only Tensor leaves from a pytree."""
    from ..tensor.api import Tensor

    return [x for x in tree_leaves(tree) if isinstance(x, Tensor)]


def is_tensor(obj: Any) -> bool:
    from ..tensor.api import Tensor

    return isinstance(obj, Tensor)


def is_tensor_value(obj: Any) -> bool:
    from max import graph

    return isinstance(obj, (graph.TensorValue, graph.BufferValue))
