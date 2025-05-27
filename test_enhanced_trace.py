"""Test enhanced trace printing with different shapes and dtypes."""

import numpy as np
from nabla.core.array import Array
from nabla.core.trace import trace_function, reset_traced_flags
from nabla.ops.creation import array, zeros, ones
from nabla.ops.binary import add, mul
from nabla.ops.unary import sin
from max.dtype import DType


def mixed_shapes_function(inputs: list[Array]) -> list[Array]:
    """Function with different shaped arrays."""
    scalar, vector, matrix = inputs[0], inputs[1], inputs[2]
    
    # Operations that preserve or change shapes
    result1 = add(vector, scalar)  # Broadcasting
    result2 = mul(matrix, scalar)  # Broadcasting
    result3 = sin(vector)
    
    return [result1, result2, result3]


def test_enhanced_trace_printing():
    """Test the enhanced trace printing with various shapes and dtypes."""
    print("=== Testing Enhanced JAX-like Trace Printing ===\n")
    
    # Test 1: Different shapes and broadcasting
    print("--- Test 1: Mixed Shapes and Broadcasting ---")
    scalar = array([5.0])  # Shape (1,)
    vector = array([1.0, 2.0, 3.0])  # Shape (3,)
    matrix = array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2)
    
    trace1 = trace_function(mixed_shapes_function, [scalar, vector, matrix])
    
    print("Function with scalar, vector, and matrix inputs:")
    print(trace1)
    print()
    
    reset_traced_flags(trace1.inputs + trace1.outputs)
    
    # Test 2: Single scalar operation
    print("--- Test 2: Scalar Operations ---")
    def scalar_function(inputs: list[Array]) -> list[Array]:
        x = inputs[0]
        y = mul(x, x)  # x^2
        z = add(y, x)  # x^2 + x
        return [z]
    
    x_scalar = array([3.14])
    trace2 = trace_function(scalar_function, [x_scalar])
    
    print("Scalar function: f(x) = x^2 + x")
    print(trace2)
    print()
    
    reset_traced_flags(trace2.inputs + trace2.outputs)
    
    # Test 3: Larger arrays
    print("--- Test 3: Larger Arrays ---")
    def matrix_function(inputs: list[Array]) -> list[Array]:
        A = inputs[0]
        B = mul(A, A)  # Element-wise square
        C = add(B, A)  # B + A
        return [C]
    
    large_matrix = array(np.random.randn(3, 4).astype(np.float32))
    trace3 = trace_function(matrix_function, [large_matrix])
    
    print("Matrix function: f(A) = A^2 + A (element-wise)")
    print(trace3)
    print()
    
    reset_traced_flags(trace3.inputs + trace3.outputs)
    
    # Test 4: Show color formatting works
    print("--- Test 4: Visual Verification ---")
    print("Notice the purple color on type annotations (if terminal supports ANSI colors)")
    simple_trace = trace_function(lambda inputs: [add(inputs[0], inputs[0])], [array([1.0, 2.0])])
    print(simple_trace)
    
    reset_traced_flags(simple_trace.inputs + simple_trace.outputs)
    
    print("\n=== Enhanced Trace Testing Complete ===")


if __name__ == "__main__":
    test_enhanced_trace_printing()
