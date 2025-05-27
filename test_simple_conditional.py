"""Simple test for conditional tracing using the traced field approach."""

import numpy as np
from nabla.core.array import Array
from nabla.core.trace import trace_function, reset_traced_flags
from nabla.ops.creation import array
from nabla.ops.binary import add, mul
from nabla.ops.unary import sin, cos


def simple_conditional_function(inputs: list[Array]) -> list[Array]:
    """Simple function with if/else that takes different execution paths."""
    x = inputs[0]
    
    # Realize the array and get the first element to make a decision
    x.realize()
    first_element = x.impl.to_numpy().flat[0]
    
    if first_element > 0:
        # Path A: Positive values
        print("Taking positive path: sin(x) + x")
        result = sin(x)
        result = add(result, x)
    else:
        # Path B: Non-positive values  
        print("Taking negative path: x * x + x")
        result = mul(x, x)
        result = add(result, x)
    
    return [result]


def test_conditional_tracing():
    """Test that conditional execution produces different traces."""
    print("=== Testing Conditional Tracing ===\n")
    
    # Test Case 1: Positive input (should take Path A)
    print("--- Test Case 1: Positive input ---")
    x_pos = array([2.0, 3.0, 1.0])
    print(f"Input: {x_pos.impl.to_numpy()}")
    
    trace_pos = trace_function(simple_conditional_function, [x_pos])
    traced_nodes_pos = trace_pos.get_traced_nodes()
    
    print(f"Output: {trace_pos.outputs[0].impl.to_numpy()}")
    print(f"Number of traced nodes: {len(traced_nodes_pos)}")
    print("Operations in trace:")
    for i, node in enumerate(traced_nodes_pos):
        print(f"  {i}: {node.name or 'unnamed'} {node.shape}")
    
    # Reset traced flags
    reset_traced_flags(trace_pos.inputs + trace_pos.outputs)
    
    print("\n" + "="*50 + "\n")
    
    # Test Case 2: Negative input (should take Path B)  
    print("--- Test Case 2: Negative input ---")
    x_neg = array([-1.0, -2.0, -0.5])
    print(f"Input: {x_neg.impl.to_numpy()}")
    
    trace_neg = trace_function(simple_conditional_function, [x_neg])
    traced_nodes_neg = trace_neg.get_traced_nodes()
    
    print(f"Output: {trace_neg.outputs[0].impl.to_numpy()}")
    print(f"Number of traced nodes: {len(traced_nodes_neg)}")
    print("Operations in trace:")
    for i, node in enumerate(traced_nodes_neg):
        print(f"  {i}: {node.name or 'unnamed'} {node.shape}")
    
    # Compare the traces
    print("\n" + "="*50 + "\n")
    print("--- Trace Comparison ---")
    
    pos_ops = [node.name for node in traced_nodes_pos if node.name]
    neg_ops = [node.name for node in traced_nodes_neg if node.name]
    
    print(f"Positive path operations: {pos_ops}")
    print(f"Negative path operations: {neg_ops}")
    print(f"Traces are different: {pos_ops != neg_ops}")
    print(f"Positive path has {len(traced_nodes_pos)} nodes")
    print(f"Negative path has {len(traced_nodes_neg)} nodes")
    
    # Clean up
    reset_traced_flags(trace_neg.inputs + trace_neg.outputs)
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_conditional_tracing()
