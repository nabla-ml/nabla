# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Pytree utilities for the eager module.

A pytree is a tree-like container of "leaves" and "nodes":
- Leaves: Any object not in the node registry (e.g., Tensors, scalars)
- Nodes: Container types (list, tuple, dict)

This module follows JAX's pytree design:
- PyTreeDef captures structure without values
- is_leaf predicate controls what counts as a leaf
- Prefix matching for axis specifications (vmap)
- Transforms: flatten → process leaves → unflatten

Example:
    from eager import pytree
    
    params = {'w': tensor1, 'b': tensor2}
    leaves, treedef = pytree.tree_flatten(params)
    # leaves = [tensor1, tensor2]
    # treedef = PyTreeDef({'w': *, 'b': *})
    
    new_leaves = [leaf * 2 for leaf in leaves]
    doubled = pytree.tree_unflatten(treedef, new_leaves)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# =============================================================================
# Structure markers
# =============================================================================

class _Leaf:
    """Sentinel class marking a leaf position in the structure."""
    __slots__ = ()
    
    def __repr__(self) -> str:
        return "*"


_LEAF = _Leaf()


@dataclass(frozen=True)
class _DictNode:
    """Structure info for a dict node."""
    keys: tuple  # Sorted keys for deterministic ordering
    
    def __repr__(self) -> str:
        return f"dict[{', '.join(repr(k) for k in self.keys)}]"


@dataclass(frozen=True)
class _ListNode:
    """Structure info for a list node."""
    length: int
    
    def __repr__(self) -> str:
        return f"list[{self.length}]"


@dataclass(frozen=True)
class _TupleNode:
    """Structure info for a tuple node."""
    length: int
    
    def __repr__(self) -> str:
        return f"tuple[{self.length}]"


# =============================================================================
# PyTreeDef
# =============================================================================

@dataclass(frozen=True)
class PyTreeDef:
    """Immutable definition of a pytree's structure.
    
    Captures the container structure of a pytree without storing the actual
    leaf values. This allows structure to be reused across different sets
    of leaves.
    
    Attributes:
        structure: Internal tree representation with _Leaf markers
        num_leaves: Total number of leaves in the tree
    """
    structure: Any
    num_leaves: int
    
    def __repr__(self) -> str:
        return f"PyTreeDef({self.structure})"
    
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
    
    Args:
        tree: A pytree (nested containers of leaves)
        is_leaf: Optional predicate to determine if a node should be treated
            as a leaf. If None, uses default leaf detection (non-containers).
    
    Returns:
        A tuple of (leaves, treedef) where:
        - leaves: List of leaf values in left-to-right, depth-first order
        - treedef: PyTreeDef capturing the structure
    
    Example:
        >>> params = {'w': tensor1, 'b': tensor2}
        >>> leaves, treedef = tree_flatten(params)
        >>> leaves
        [tensor1, tensor2]
    """
    leaves: list = []
    
    def _flatten(node: Any) -> Any:
        # Check custom is_leaf predicate first
        if is_leaf is not None and is_leaf(node):
            leaves.append(node)
            return _LEAF
        
        # None is treated as an empty container (like JAX)
        if node is None:
            return None
        
        # Handle container types
        if isinstance(node, dict):
            keys = tuple(sorted(node.keys()))
            children = tuple(_flatten(node[k]) for k in keys)
            return (_DictNode(keys), children)
        
        if isinstance(node, list):
            children = tuple(_flatten(item) for item in node)
            return (_ListNode(len(node)), children)
        
        if isinstance(node, tuple):
            children = tuple(_flatten(item) for item in node)
            return (_TupleNode(len(node)), children)
        
        # Everything else is a leaf
        leaves.append(node)
        return _LEAF
    
    structure = _flatten(tree)
    return leaves, PyTreeDef(structure, len(leaves))


def tree_unflatten(treedef: PyTreeDef, leaves: list) -> Any:
    """Reconstruct a pytree from structure info and leaves.
    
    Args:
        treedef: PyTreeDef from tree_flatten
        leaves: List of leaf values (must match treedef.num_leaves)
    
    Returns:
        Reconstructed pytree with original structure
    
    Raises:
        ValueError: If number of leaves doesn't match treedef
    
    Example:
        >>> params = tree_unflatten(treedef, [new_w, new_b])
        >>> params
        {'w': new_w, 'b': new_b}
    """
    if len(leaves) != treedef.num_leaves:
        raise ValueError(
            f"Expected {treedef.num_leaves} leaves, got {len(leaves)}"
        )
    
    leaves_iter = iter(leaves)
    
    def _unflatten(structure: Any) -> Any:
        if structure is _LEAF:
            return next(leaves_iter)
        
        # None is preserved as None
        if structure is None:
            return None
        
        node_type, children = structure
        
        if isinstance(node_type, _DictNode):
            return {
                k: _unflatten(child)
                for k, child in zip(node_type.keys, children)
            }
        
        if isinstance(node_type, _ListNode):
            return [_unflatten(child) for child in children]
        
        if isinstance(node_type, _TupleNode):
            return tuple(_unflatten(child) for child in children)
        
        raise ValueError(f"Unknown node type: {node_type}")
    
    return _unflatten(treedef.structure)


def tree_leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list:
    """Get all leaves from a pytree without structure info.
    
    Args:
        tree: A pytree
        is_leaf: Optional predicate for leaf detection
    
    Returns:
        List of leaves in traversal order
    """
    leaves, _ = tree_flatten(tree, is_leaf)
    return leaves


def tensor_leaves(tree: Any) -> list:
    """Get only Tensor leaves from a pytree, ignoring scalars and other types.
    
    Args:
        tree: A pytree that may contain Tensors and other values
    
    Returns:
        List of Tensor objects found in the tree
    """
    from .tensor import Tensor
    return [leaf for leaf in tree_leaves(tree) if isinstance(leaf, Tensor)]


def tree_structure(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTreeDef:
    """Get structure info from a pytree without leaves.
    
    Args:
        tree: A pytree
        is_leaf: Optional predicate for leaf detection
    
    Returns:
        PyTreeDef capturing the structure
    """
    _, treedef = tree_flatten(tree, is_leaf)
    return treedef


def tree_map(
    fn: Callable[[T], Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Apply a function to every leaf of a pytree.
    
    Args:
        fn: Function to apply to each leaf. If rest is provided, fn takes
            len(rest)+1 arguments (one from each tree).
        tree: A pytree
        *rest: Additional pytrees with matching structure
        is_leaf: Optional predicate for leaf detection
    
    Returns:
        A new pytree with the same structure as tree, with fn applied to leaves
    
    Example:
        >>> doubled = tree_map(lambda x: x * 2, params)
        >>> summed = tree_map(lambda x, y: x + y, params1, params2)
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    
    if rest:
        # Flatten all additional trees and verify structure match
        all_leaves = [leaves]
        for other_tree in rest:
            other_leaves, other_treedef = tree_flatten(other_tree, is_leaf)
            # Check both num_leaves and structure match
            if other_treedef.structure != treedef.structure:
                raise ValueError(
                    f"Tree structures don't match: {treedef} vs {other_treedef}"
                )
            all_leaves.append(other_leaves)
        
        # Apply function to corresponding leaves
        new_leaves = [fn(*args) for args in zip(*all_leaves)]
    else:
        new_leaves = [fn(leaf) for leaf in leaves]
    
    return tree_unflatten(treedef, new_leaves)


# =============================================================================
# Prefix matching (for vmap in_axes, out_axes)
# =============================================================================

def broadcast_prefix(prefix_tree: Any, full_tree: Any) -> Any:
    """Broadcast a prefix tree to match a full tree's structure.
    
    A prefix tree is a tree where leaves can appear earlier than in the full
    tree. Each leaf in the prefix tree is broadcast to all corresponding
    leaves in the full tree.
    
    This is used by vmap: `in_axes=0` broadcasts to match any input structure.
    
    Args:
        prefix_tree: A prefix of full_tree's structure (or a single leaf)
        full_tree: The full pytree structure to match
    
    Returns:
        A tree with full_tree's structure, with prefix_tree's values broadcast
    
    Example:
        >>> broadcast_prefix(0, {'a': x, 'b': y})
        {'a': 0, 'b': 0}
        >>> broadcast_prefix({'a': 0, 'b': 1}, {'a': x, 'b': y})
        {'a': 0, 'b': 1}
    """
    def _broadcast(prefix: Any, full: Any) -> Any:
        # If prefix is a leaf at this point, broadcast it to all of full's leaves
        if not _is_container(prefix):
            # Prefix is a leaf, apply it to all leaves in full
            return tree_map(lambda _: prefix, full)
        
        # Both are containers, recurse with matching structure
        if isinstance(prefix, dict) and isinstance(full, dict):
            # All keys in prefix must exist in full
            result = {}
            for k in full.keys():
                if k in prefix:
                    result[k] = _broadcast(prefix[k], full[k])
                else:
                    raise ValueError(
                        f"Prefix dict missing key '{k}' that exists in full tree"
                    )
            return result
        
        if isinstance(prefix, (list, tuple)) and isinstance(full, (list, tuple)):
            if len(prefix) != len(full):
                raise ValueError(
                    f"Prefix length {len(prefix)} doesn't match full tree "
                    f"length {len(full)}"
                )
            result = [_broadcast(p, f) for p, f in zip(prefix, full)]
            return type(full)(result)
        
        # Mismatched container types
        raise ValueError(
            f"Cannot broadcast prefix {type(prefix)} to full tree {type(full)}"
        )
    
    return _broadcast(prefix_tree, full_tree)


def _is_container(obj: Any) -> bool:
    """Check if an object is a pytree container."""
    return isinstance(obj, (dict, list, tuple))


# =============================================================================
# Transform helpers (Tensor-aware utilities)
# =============================================================================

def traced(tree: Any) -> Any:
    """Mark all tensors in a pytree as traced for autograd.
    
    Note: This mutates tensors IN-PLACE. The returned tree has the same
    tensor objects, now with traced=True.
    
    Args:
        tree: A pytree containing Tensors
    
    Returns:
        The same tree structure (tensors are mutated in-place)
    """
    from .tensor import Tensor
    
    def _set_traced(leaf: Any) -> Any:
        if isinstance(leaf, Tensor):
            leaf._impl.traced = True
        return leaf
    
    return tree_map(_set_traced, tree)


def untraced(tree: Any) -> Any:
    """Mark all tensors in a pytree as untraced.
    
    Note: This mutates tensors IN-PLACE. The returned tree has the same
    tensor objects, now with traced=False.
    
    Args:
        tree: A pytree containing Tensors
    
    Returns:
        The same tree structure (tensors are mutated in-place)
    """
    from .tensor import Tensor
    
    def _set_untraced(leaf: Any) -> Any:
        if isinstance(leaf, Tensor):
            leaf._impl.traced = False
        return leaf
    
    return tree_map(_set_untraced, tree)


def with_batch_dims(tree: Any, delta: int) -> Any:
    """Adjust batch_dims on all tensors in a pytree.
    
    Note: This mutates tensors IN-PLACE.
    
    Args:
        tree: A pytree containing Tensors
        delta: Amount to add to batch_dims (positive or negative)
    
    Returns:
        The same tree structure (tensors are mutated in-place)
    """
    from .tensor import Tensor
    
    def _adjust_batch_dims(leaf: Any) -> Any:
        if isinstance(leaf, Tensor):
            leaf._impl.batch_dims += delta
        return leaf
    
    return tree_map(_adjust_batch_dims, tree)
