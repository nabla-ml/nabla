# ===----------------------------------------------------------------------=== #
# Nabla 2026
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

"""Core Trace primitive for capturing and manipulating computation subgraphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .pytree import tree_flatten, tree_leaves
from .tensor import Tensor

if TYPE_CHECKING:
    from .tensor_impl import TensorImpl


class Trace:
    """Represents a captured computation subgraph.
    
    A Trace allows viewing and manipulating the computation graph between
    a set of input tensors (boundary) and output tensors.
    
    Attributes:
        inputs: The original input pytree structure.
        outputs: The original output pytree structure.
        nodes: Topologically sorted list of unique TensorImpls in the subgraph.
    """
    
    def __init__(self, inputs: Any, outputs: Any):
        self.inputs = inputs
        self.outputs = outputs
        self._computed = False
        self.nodes: list[TensorImpl] = []
        
        # Flatten inputs to establish the boundary
        self._input_nodes = {
            id(t._impl) for t in tree_leaves(inputs) if isinstance(t, Tensor)
        }
        
    def compute(self) -> None:
        """Compute the topological ordering of the subgraph."""
        if self._computed:
            return
            
        visited: set[int] = set()
        nodes: list[TensorImpl] = []
        
        # We search backwards from outputs
        output_leaves = [
            t._impl for t in tree_leaves(self.outputs) if isinstance(t, Tensor)
        ]
        
        def dfs(node: TensorImpl) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            
            # If we hit an input boundary, we stop recursing but include the node
            # This effectively makes it a "leaf" for this specific trace
            if node_id in self._input_nodes:
                visited.add(node_id)
                return
            
            # Recurse into parents
            for parent in node.parents:
                dfs(parent)
            
            visited.add(node_id)
            nodes.append(node)
            
        for leaf in output_leaves:
            dfs(leaf)
            
        self.nodes = nodes
        self._computed = True
        
    def __str__(self) -> str:
        """Pretty-print the trace using the GraphPrinter."""
        from ..utils.debug import GraphPrinter
        if not self._computed:
            self.compute()
        return GraphPrinter(self).to_string()
