# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import Dict, Optional
from utils import Variant
from nabla.api.array import Array


@value
struct MoTree(Copyable, Movable):
    var data: List[Dict[String, Variant[Array, List[Array], Self, List[Self]]]]

    fn __init__(out self) raises:
        self.data = List[
            Dict[String, Variant[Array, List[Array], Self, List[Self]]]
        ]()

    fn insert(
        mut self,
        key: String,
        value: Variant[Array, List[Array], Self, List[Self]],
    ) raises:
        if len(self.data) == 0:
            self.data.append(
                Dict[String, Variant[Array, List[Array], Self, List[Self]]]()
            )

        if value.isa[Array]():
            var val = value[Array]
            self.data[0][key] = val
        elif value.isa[Self]():
            var val = value[Self]
            self.data[0][key] = val
        elif value.isa[List[Array]]():
            var val = value[List[Array]]
            self.data[0][key] = val
        elif value.isa[List[Self]]():
            var val = value[List[Self]]
            self.data[0][key] = val
        else:
            raise "Invalid type"

    fn __setitem__(
        mut self,
        key: String,
        value: Variant[Array, List[Array], Self, List[Self]],
    ) raises:
        self.insert(key, value)

    fn __getitem__(
        self, key: String = ""
    ) raises -> Variant[Array, List[Array], MoTree, List[MoTree]]:
        if len(self.data) != 1 or key not in self.data[0]:
            raise "Key not found" + key
        return self.data[0][key]

    fn retreive_leaves(
        self,
        mut leaves: List[Array],
        curr: Variant[Array, List[Array], Self, List[Self]],
    ) raises:
        if curr.isa[Array]():
            leaves.append(curr[Array])
        elif curr.isa[List[Array]]():
            for val in curr[List[Array]]:
                leaves.append(val[])
        elif curr.isa[Self]():
            for key in curr[Self].data[0].keys():
                self.retreive_leaves(leaves, curr[Self].data[0][key[]])
        elif curr.isa[List[Self]]():
            var subtrees = curr[List[Self]]
            for subtree in subtrees:
                for key in subtree[].data[0].keys():
                    self.retreive_leaves(leaves, subtree[].data[0][key[]])

    fn flatten(self) raises -> List[Array]:
        var leaves = List[Array]()
        for key in self.data[0].keys():
            self.retreive_leaves(leaves, self.data[0][key[]])
        return leaves


fn motree(
    key: String, value: Variant[Array, List[Array], MoTree, List[MoTree]]
) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key, value)
    return new_tree


fn motree(
    key0: String,
    value0: Variant[Array, List[Array], MoTree, List[MoTree]],
    key1: String,
    value1: Variant[Array, List[Array], MoTree, List[MoTree]],
) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key0, value0)
    new_tree.insert(key1, value1)
    return new_tree


fn motree(
    key0: String,
    value0: Variant[Array, List[Array], MoTree, List[MoTree]],
    key1: String,
    value1: Variant[Array, List[Array], MoTree, List[MoTree]],
    key2: String,
    value2: Variant[Array, List[Array], MoTree, List[MoTree]],
) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key0, value0)
    new_tree.insert(key1, value1)
    new_tree.insert(key2, value2)
    return new_tree


fn motree(
    key0: String,
    value0: Variant[Array, List[Array], MoTree, List[MoTree]],
    key1: String,
    value1: Variant[Array, List[Array], MoTree, List[MoTree]],
    key2: String,
    value2: Variant[Array, List[Array], MoTree, List[MoTree]],
    key3: String,
    value3: Variant[Array, List[Array], MoTree, List[MoTree]],
) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key0, value0)
    new_tree.insert(key1, value1)
    new_tree.insert(key2, value2)
    new_tree.insert(key3, value3)
    return new_tree


fn motree(*key_value_pairs: Tuple[String, Array]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[][0], key_value_pair[][1])
    return new_tree


fn motree(*key_value_pairs: Tuple[String, List[Array]]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[][0], key_value_pair[][1])
    return new_tree


fn motree(*key_value_pairs: Tuple[String, MoTree]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[][0], key_value_pair[][1])
    return new_tree


fn motree(*key_value_pairs: Tuple[String, List[MoTree]]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[][0], key_value_pair[][1])
    return new_tree


fn motree(*trees: MoTree) raises -> MoTree:
    var new_tree = MoTree()
    var subtrees = List[MoTree]()
    for motree in trees:
        subtrees.append(motree[])
    new_tree.insert("_", subtrees)
    return new_tree
