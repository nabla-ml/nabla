# ===----------------------------------------------------------------------=== #
# Nabla 2026
# SPDX-License-Identifier: Apache-2.0
# ===----------------------------------------------------------------------=== #

"""Graph traversal utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..tensor.impl import TensorImpl



def get_operations_topological(outputs: list[TensorImpl]) -> list[tuple[object, list[TensorImpl]]]:
    """Get operations in reverse topological order (outputs first)."""
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
    """Get all TensorImpls in forward topological order."""
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
    """Print the traced computation graph."""
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
    """Apply a function to each operation in reverse topological order."""
    ops = get_operations_topological(outputs)
    for op_id, op_outputs in ops:
        fn(op_id, op_outputs)
