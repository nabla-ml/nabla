# ===----------------------------------------------------------------------=== #
# Nabla 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Pytree utilities for the eager module.

Optimized for high-throughput traversals using JAX-style structural handling:
- Single-pass tree_map (avoids intermediate flatten/unflatten overhead).
- Lightweight tuple-based structure definitions (avoids object instantiation).
- Strict structural matching and NamedTuple support.

A pytree is a tree-like container of "leaves" and "nodes":
- Leaves: Any object not in the node registry (e.g., Tensors, scalars)
- Nodes: Container types (list, tuple, dict)
"""

from __future__ import annotations

import collections
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# =============================================================================
# Structure markers (Internals)
# =============================================================================

# Integer constants are faster than type(obj) checks in tight loops
_K_LEAF = 0
_K_NONE = 1
_K_LIST = 2
_K_TUPLE = 3
_K_DICT = 4

class _LeafMarker:
    """Sentinel marking a leaf position."""
    __slots__ = ()
    def __repr__(self) -> str: return "*"

_LEAF = _LeafMarker()


# =============================================================================
# PyTreeDef
# =============================================================================

class PyTreeDef:
    """Immutable definition of a pytree's structure.
    
    Optimized with __slots__ to minimize memory footprint during
    complex graph traversals.
    """
    __slots__ = ("_kind", "_meta", "_children", "num_leaves")

    def __init__(self, kind: int, meta: Any, children: tuple, num_leaves: int):
        self._kind = kind
        self._meta = meta          # e.g., dict keys, or None
        self._children = children  # tuple of sub-PyTreeDefs
        self.num_leaves = num_leaves

    def __repr__(self) -> str:
        if self._kind == _K_LEAF:  return "*"
        if self._kind == _K_NONE:  return "None"
        if self._kind == _K_LIST:  return f"list[{len(self._children)}]"
        if self._kind == _K_TUPLE: return f"tuple[{len(self._children)}]"
        if self._kind == _K_DICT:  return f"dict{list(self._meta)}"
        return "PyTreeDef(?)"

    def __eq__(self, other: Any) -> bool:
        if self is other: return True
        if not isinstance(other, PyTreeDef): return False
        return (self._kind == other._kind and 
                self._meta == other._meta and 
                self._children == other._children)

    def __hash__(self) -> int:
        return hash((self._kind, self._meta, self._children))

    def unflatten(self, leaves: list) -> Any:
        """Reconstruct a pytree from this structure and a list of leaves."""
        return tree_unflatten(self, leaves)


# =============================================================================
# Core functions
# =============================================================================

def tree_flatten(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[list, PyTreeDef]:
    """Flatten a pytree into a list of leaves and structure info.
    
    Returns:
        (leaves, treedef)
    """
    leaves = []

    def _flatten(node: Any) -> PyTreeDef:
        # 1. Custom Leaf Predicate
        if is_leaf is not None and is_leaf(node):
            leaves.append(node)
            return PyTreeDef(_K_LEAF, None, (), 1)

        # 2. Native None
        if node is None:
            return PyTreeDef(_K_NONE, None, (), 0)

        # 3. List
        if isinstance(node, list):
            children = tuple(_flatten(v) for v in node)
            num = sum(c.num_leaves for c in children)
            return PyTreeDef(_K_LIST, None, children, num)

        # 4. Tuple (handles NamedTuple via generic tuple check)
        if isinstance(node, tuple):
            children = tuple(_flatten(v) for v in node)
            num = sum(c.num_leaves for c in children)
            # We store the specific class type in meta to reconstruct NamedTuples correctly
            return PyTreeDef(_K_TUPLE, type(node), children, num)

        # 5. Dict
        if isinstance(node, dict):
            keys = sorted(node.keys()) # Sorted for deterministic order
            children = tuple(_flatten(node[k]) for k in keys)
            num = sum(c.num_leaves for c in children)
            return PyTreeDef(_K_DICT, tuple(keys), children, num)

        # 6. Default Leaf
        leaves.append(node)
        return PyTreeDef(_K_LEAF, None, (), 1)

    treedef = _flatten(tree)
    return leaves, treedef


def tree_unflatten(treedef: PyTreeDef, leaves: list) -> Any:
    """Reconstruct a pytree from structure info and leaves."""
    if len(leaves) != treedef.num_leaves:
        raise ValueError(
            f"Expected {treedef.num_leaves} leaves, got {len(leaves)}"
        )
    
    # Use iterator to consume leaves sequentially
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
            # def_._meta holds the tuple class (e.g. tuple, or a NamedTuple type)
            cls = def_._meta 
            items = (_build(c) for c in def_._children)
            # NamedTuples/tuples can be constructed via cls(iterator)
            return cls(items) if cls is tuple else cls(*items)
            
        if k == _K_DICT:
            return {
                key: _build(child) 
                for key, child in zip(def_._meta, def_._children)
            }
            
        raise ValueError(f"Unknown node kind: {k}")

    return _build(treedef)


def tree_leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list:
    """Get all leaves from a pytree without structure info."""
    leaves, _ = tree_flatten(tree, is_leaf)
    return leaves


def tree_structure(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTreeDef:
    """Get structure info from a pytree without leaves."""
    _, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_map(
    fn: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Apply a function to every leaf of a pytree.
    
    Optimized Single-Pass Implementation:
    Traverses all trees simultaneously, applying fn to leaves.
    Raises ValueError if structures do not match exactly.
    """
    
    def _map(primary: Any, *others: Any) -> Any:
        # 1. Leaf Detection
        if is_leaf is not None and is_leaf(primary):
            return fn(primary, *others)

        # 2. None
        if primary is None:
            if not all(x is None for x in others):
                raise ValueError("Tree structure mismatch: Expected None")
            return None

        # 3. List
        if isinstance(primary, list):
            if not all(isinstance(x, list) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: List length or type")
            return [
                _map(c, *[x[i] for x in others]) 
                for i, c in enumerate(primary)
            ]

        # 4. Tuple (and NamedTuple)
        if isinstance(primary, tuple):
            if not all(isinstance(x, tuple) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: Tuple length or type")
            
            # Reconstruct using the primary's type (preserves NamedTuple)
            result_gen = (
                _map(c, *[x[i] for x in others]) 
                for i, c in enumerate(primary)
            )
            return type(primary)(result_gen)

        # 5. Dict
        if isinstance(primary, dict):
            if not all(isinstance(x, dict) and len(x) == len(primary) for x in others):
                raise ValueError("Tree structure mismatch: Dict size or type")
            
            # Check keys match exactly
            # We iterate primary keys; strictness ensures others must match
            return {
                k: _map(v, *[x[k] for x in others])
                for k, v in primary.items()
            }

        # 6. Default Leaf
        return fn(primary, *others)

    return _map(tree, *rest)

# =============================================================================
# Transform helpers (Tensor-aware utilities)
# =============================================================================

def traced(tree: Any) -> Any:
    """Mark all tensors in a pytree as traced (In-place)."""
    from .tensor import Tensor
    return tree_map(lambda x: setattr(x._impl, 'traced', True) or x if isinstance(x, Tensor) else x, tree)

def untraced(tree: Any) -> Any:
    """Mark all tensors in a pytree as untraced (In-place)."""
    from .tensor import Tensor
    return tree_map(lambda x: setattr(x._impl, 'traced', False) or x if isinstance(x, Tensor) else x, tree)

def with_batch_dims(tree: Any, delta: int) -> Any:
    """Adjust batch_dims on all tensors in a pytree (In-place)."""
    from .tensor import Tensor
    def _adj(x: Any) -> Any:
        if isinstance(x, Tensor): x._impl.batch_dims += delta
        return x
    return tree_map(_adj, tree)

def tensor_leaves(tree: Any) -> list:
    """Get only Tensor leaves from a pytree."""
    from .tensor import Tensor
    return [x for x in tree_leaves(tree) if isinstance(x, Tensor)]

def is_tensor(obj: Any) -> bool:
    from .tensor import Tensor
    return isinstance(obj, Tensor)

def is_tensor_value(obj: Any) -> bool:
    from max import graph
    return isinstance(obj, (graph.TensorValue, graph.BufferValue))