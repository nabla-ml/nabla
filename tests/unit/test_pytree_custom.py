# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import nabla as nb


class Pair:
    def __init__(self, left, right):
        self.left = left
        self.right = right


def _pair_flatten(node: Pair):
    return [node.left, node.right], None


def _pair_unflatten(_aux, children):
    return Pair(children[0], children[1])


nb.register_pytree_node(Pair, _pair_flatten, _pair_unflatten)


def test_custom_pytree_roundtrip_and_map():
    p = Pair({"x": 1}, [2, 3])
    leaves, treedef = nb.tree_flatten(p)
    rebuilt = nb.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, Pair)
    assert rebuilt.left == {"x": 1}
    assert rebuilt.right == [2, 3]

    mapped = nb.tree_map(lambda x: x + 1 if isinstance(x, int) else x, p)
    assert isinstance(mapped, Pair)
    assert mapped.left == {"x": 2}
    assert mapped.right == [3, 4]
