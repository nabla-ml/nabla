"""Test the new JAX-like trace printing functionality."""

import numpy as np
from nabla.core.array import Array
from nabla.core.trace import trace_function, reset_traced_flags
from nabla.ops.creation import array
from nabla.ops.binary import add, mul
from nabla.ops.unary import sin, cos


def simple_function(inputs: list[Array]) -> list[Array]:
    """Simple function for testing trace representation."""
    x = inputs[0]
    
    # x^2 + sin(x)
    x_squared = mul(x, x)
    sin_x = sin(x)
    result = add(x_squared, sin_x)
    
    return [result]


def multi_input_function(inputs: list[Array]) -> list[Array]:
    """Function with multiple inputs and outputs."""
    x, y = inputs[0], inputs[1]
    
    # Multiple operations
    xy = mul(x, y)
    x_sin = sin(x)
    y_cos = cos(y)
    
    # Two outputs
    out1 = add(xy, x_sin)
    out2 = mul(y_cos, x)
    
    return [out1, out2]


def conditional_function(inputs: list[Array]) -> list[Array]:
    """Conditional function to show different traces."""
    x = inputs[0]
    
    # Realize to check value
    x.realize()
    first_val = x.impl.to_numpy().flat[0]
    
    if first_val > 0:
        result = add(sin(x), x)  # sin(x) + x
    else:
        result = add(mul(x, x), x)  # x^2 + x
    
    return [result]


def test_trace_printing():
    """Test the new trace printing functionality."""
    print("=== Testing JAX-like Trace Printing ===\n")
    
    # Test 1: Simple single-input function
    print("--- Test 1: Simple Function ---")
    x = array([1.0, 2.0])
    trace1 = trace_function(simple_function, [x])
    
    print("Function: f(x) = x^2 + sin(x)")
    print("Trace representation:")
    print(trace1)
    print()
    
    reset_traced_flags(trace1.inputs + trace1.outputs)
    
    # Test 2: Multi-input, multi-output function
    print("--- Test 2: Multi-Input/Output Function ---")
    x = array([1.0, 2.0])
    y = array([0.5, 1.5])
    trace2 = trace_function(multi_input_function, [x, y])
    
    print("Function: f(x,y) = (x*y + sin(x), cos(y)*x)")
    print("Trace representation:")
    print(trace2)
    print()
    
    reset_traced_flags(trace2.inputs + trace2.outputs)
    
    # Test 3: Conditional execution (positive path)
    print("--- Test 3: Conditional Function (Positive Path) ---")
    x_pos = array([2.0, 3.0])
    trace3_pos = trace_function(conditional_function, [x_pos])
    
    print("Function: f(x) = sin(x) + x (when x[0] > 0)")
    print("Trace representation:")
    print(trace3_pos)
    print()
    
    reset_traced_flags(trace3_pos.inputs + trace3_pos.outputs)
    
    # Test 4: Conditional execution (negative path)
    print("--- Test 4: Conditional Function (Negative Path) ---")
    x_neg = array([-1.0, -2.0])
    trace4_neg = trace_function(conditional_function, [x_neg])
    
    print("Function: f(x) = x^2 + x (when x[0] <= 0)")
    print("Trace representation:")
    print(trace4_neg)
    print()
    
    reset_traced_flags(trace4_neg.inputs + trace4_neg.outputs)
    
    # Test 5: Compare trace lengths
    print("--- Test 5: Trace Comparison ---")
    print(f"Simple function trace length: {len(trace1.get_traced_nodes())}")
    print(f"Multi-input function trace length: {len(trace2.get_traced_nodes())}")
    print(f"Conditional positive trace length: {len(trace3_pos.get_traced_nodes())}")
    print(f"Conditional negative trace length: {len(trace4_neg.get_traced_nodes())}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_trace_printing()
