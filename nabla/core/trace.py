
"""Core tracing functionality for the Nabla framework.

This module implements a simple and efficient tracing system using a "traced" field
to mark nodes that belong to a specific execution trace. This allows easy
distinction between nodes from different function calls and selective graph traversal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .array import Array


class Trace:
    """A simple trace container that holds the computation graph."""
    
    def __init__(self, inputs: list[Array], outputs: list[Array]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.trace: list[Array] = []
        self._computed = False
    
    def get_traced_nodes(self) -> list[Array]:
        """Get all nodes that belong to this trace in topological order."""
        if not self._computed:
            self._compute_trace()
        return self.trace
    
    def _compute_trace(self) -> None:
        """Compute the topological ordering of traced nodes."""
        visited = set()
        self.trace = []
        
        for output in self.outputs:
            self._dfs_visit(output, visited)
        
        self._computed = True
    
    def _dfs_visit(self, node: Array, visited: set[Array]) -> None:
        """DFS traversal to build topological ordering."""
        if node in visited or not node.traced:
            return
        
        # Visit children first (post-order)
        for arg in node.args:
            self._dfs_visit(arg, visited)
        
        # Add current node after visiting children
        visited.add(node)
        self.trace.append(node)
    
    def __str__(self) -> str:
        """Return a JAX-like string representation of the trace."""
        if not self._computed:
            self._compute_trace()
        
        # ANSI color codes
        PURPLE = '\033[95m'
        RESET = '\033[0m'
        
        def format_dtype(dtype) -> str:
            """Format dtype for display."""
            # Convert DType to string representation
            dtype_str = str(dtype).lower()
            if 'float32' in dtype_str:
                return 'f32'
            elif 'float64' in dtype_str:
                return 'f64'
            elif 'int32' in dtype_str:
                return 'i32'
            elif 'int64' in dtype_str:
                return 'i64'
            else:
                return dtype_str
        
        def format_shape_and_dtype(array) -> str:
            """Format shape and dtype in JAX style."""
            dtype_str = format_dtype(array.dtype)
            if array.shape:
                shape_str = ','.join(map(str, array.shape))
                return f"{PURPLE}{dtype_str}[{shape_str}]{RESET}"
            else:
                return f"{PURPLE}{dtype_str}[]{RESET}"
        
        # Build variable name mapping
        var_names = {}
        var_counter = 0
        
        # Assign names to inputs first with type annotations
        input_vars = []
        for inp in self.inputs:
            var_name = chr(ord('a') + var_counter)
            var_names[id(inp)] = var_name
            type_annotation = format_shape_and_dtype(inp)
            input_vars.append(f"{var_name}:{type_annotation}")
            var_counter += 1
        
        # Build the equation lines
        equations = []
        for node in self.trace:
            node_id = id(node)
            
            # Skip if this is an input (already named)
            if node_id in var_names:
                continue
            
            # Assign a variable name to this node
            var_name = chr(ord('a') + var_counter)
            var_names[node_id] = var_name
            var_counter += 1
            
            # Build the operation description
            if node.args:
                # Get argument variable names
                arg_vars = []
                for arg in node.args:
                    arg_id = id(arg)
                    if arg_id in var_names:
                        arg_vars.append(var_names[arg_id])
                    else:
                        # This shouldn't happen in a well-formed trace
                        arg_vars.append("?")
                
                # Format the equation with type annotation
                op_name = node.name or "unknown"
                type_annotation = format_shape_and_dtype(node)
                
                if len(arg_vars) == 1:
                    equation = f"    {var_name}:{type_annotation} = {op_name} {arg_vars[0]}"
                else:
                    equation = f"    {var_name}:{type_annotation} = {op_name} {' '.join(arg_vars)}"
                
                equations.append(equation)
        
        # Get output variable names
        output_vars = []
        for out in self.outputs:
            out_id = id(out)
            if out_id in var_names:
                output_vars.append(var_names[out_id])
            else:
                output_vars.append("?")
        
        # Format the final representation
        input_sig = f"({', '.join(input_vars)})"
        output_sig = f"({', '.join(output_vars)})" if len(output_vars) > 1 else output_vars[0]
        
        result = f"{{ lambda {input_sig} ;\n"
        result += "  let\n"
        for eq in equations:
            result += f"{eq}\n"
        result += f"  in {output_sig} }}"
        
        return result
    
    def print_trace(self) -> None:
        """Print the trace in a nice format."""
        print(self)



def trace_function(func: Callable[[list[Array]], list[Array]], inputs: list[Array]) -> Trace:
    """
    Trace a function execution and return a Trace object.
    
    This function:
    1. Marks all input arrays as traced
    2. Executes the function 
    3. Collects all traced nodes in the output
    4. Returns a Trace object
    
    Args:
        func: Function that takes list of Arrays and returns list of Arrays
        inputs: Input Arrays to trace
        
    Returns:
        Trace object containing the computation graph
    """
    # Step 1: Mark input arrays as traced
    for inp in inputs:
        inp.traced = True
    
    # Step 2: Execute the function
    outputs = func(inputs)
    
    # Step 3: Create trace object
    trace = Trace(inputs, outputs)
    
    return trace


def reset_traced_flags(arrays: list[Array]) -> None:
    """Reset traced flags for a list of arrays and their dependencies."""
    visited = set()
    
    def reset_recursive(node: Array) -> None:
        if node in visited:
            return
        visited.add(node)
        node.traced = False
        for arg in node.args:
            reset_recursive(arg)
    
    for array in arrays:
        reset_recursive(array)