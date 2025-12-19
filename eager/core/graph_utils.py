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

"""Graph traversal utilities for traced computation graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .tensor_impl import TensorImpl


def get_operations_topological(outputs: list[TensorImpl]) -> list[tuple[object, list[TensorImpl]]]:
    """Get operations in reverse topological order (outputs first, inputs last).
    
    This properly handles multi-output operations by deduplicating via OutputRefs.
    Each operation appears exactly once, even if it has multiple outputs.
    
    Args:
        outputs: List of output TensorImpls to traverse from.
        
    Returns:
        List of (operation_id, output_impls) tuples in reverse topological order.
        The operation_id is `id(output_refs)` which uniquely identifies an op call.
        output_impls is the list of ALL outputs (alive) from that operation.
    """
    visited_ops: set[int] = set()  # Track visited operations via id(output_refs)
    visited_impls: set[int] = set()  # Track visited TensorImpls
    result: list[tuple[object, list[TensorImpl]]] = []
    
    def visit(impl: TensorImpl) -> None:
        """DFS traversal of the computation graph."""
        impl_id = id(impl)
        if impl_id in visited_impls:
            return
        visited_impls.add(impl_id)
        
        # First, recurse on parents (inputs)
        for parent in impl.parents:
            visit(parent)
        
        # Then, record this operation (if not already recorded)
        if impl.output_refs is not None:
            op_id = id(impl.output_refs)
            if op_id not in visited_ops:
                visited_ops.add(op_id)
                
                # Collect all alive siblings
                alive_outputs = [
                    out for out in impl.output_refs.get_alive_outputs()
                    if out is not None
                ]
                
                # Record the operation with all its outputs
                result.append((op_id, alive_outputs))
    
    # Start traversal from all outputs
    for output in outputs:
        visit(output)
    
    return result


def get_all_impls_topological(outputs: list[TensorImpl]) -> list[TensorImpl]:
    """Get all TensorImpls in forward topological order (inputs first, outputs last).
    
    Unlike get_operations_topological, this returns individual TensorImpls,
    not grouped by operation. Useful for general graph traversal.
    
    Args:
        outputs: List of output TensorImpls to traverse from.
        
    Returns:
        List of TensorImpls in topological order (dependencies before dependents).
    """
    visited: set[int] = set()
    result: list[TensorImpl] = []
    
    def visit(impl: TensorImpl) -> None:
        impl_id = id(impl)
        if impl_id in visited:
            return
        visited.add(impl_id)
        
        # Visit parents first (inputs before outputs)
        for parent in impl.parents:
            visit(parent)
        
        result.append(impl)
    
    for output in outputs:
        visit(output)
    
    return result


def print_trace_graph(outputs: list[TensorImpl], show_siblings: bool = True) -> None:
    """Print a human-readable representation of the traced computation graph.
    
    Args:
        outputs: List of output TensorImpls to start from.
        show_siblings: If True, show all outputs of multi-output operations.
    """
    ops = get_operations_topological(outputs)
    
    print(f"Computation Graph ({len(ops)} operations):")
    print("-" * 60)
    
    for idx, (op_id, op_outputs) in enumerate(reversed(ops)):  # Reverse to show forward order
        # Get operation info from first output
        first_out = op_outputs[0]
        op_name = first_out.op_name or "unknown"
        num_inputs = len(first_out.parents)
        
        print(f"{idx:3d}. {op_name}")
        print(f"      Inputs: {num_inputs}")
        print(f"      Outputs: {len(op_outputs)}")
        
        if show_siblings and len(op_outputs) > 1:
            for i, out in enumerate(op_outputs):
                shape = out.logical_shape
                print(f"        [{i}] shape={shape}, index={out.output_index}")
    
    print("-" * 60)


def apply_to_operations(
    outputs: list[TensorImpl],
    fn: Callable[[object, list[TensorImpl]], None]
) -> None:
    """Apply a function to each operation in reverse topological order.
    
    Useful for VJP backward pass where you process operations from outputs to inputs.
    
    Args:
        outputs: Starting points for traversal.
        fn: Function to apply to each operation.
            Receives (operation_id, output_impls) for each op.
    """
    ops = get_operations_topological(outputs)
    for op_id, op_outputs in ops:
        fn(op_id, op_outputs)
