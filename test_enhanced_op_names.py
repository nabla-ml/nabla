"""Test to showcase all the enhanced JAX-like operation names."""

import numpy as np
from nabla.core.array import Array
from nabla.core.trace import trace_function, reset_traced_flags
from nabla.ops.creation import array, zeros, ones
from nabla.ops.binary import add, mul
from nabla.ops.unary import sin, cos, negate, cast
from nabla.ops.view import transpose, reshape, broadcast_to
from nabla.ops.reduce import reduce_sum
from nabla.ops.linalg import matmul
from max.dtype import DType


def comprehensive_operations_function(inputs: list[Array]) -> list[Array]:
    """Function that uses many different operations to showcase enhanced names."""
    x, y = inputs[0], inputs[1]
    
    # Unary operations
    neg_x = negate(x)
    sin_x = sin(x)
    cos_y = cos(y)
    
    # Binary operations
    xy = mul(x, y)
    sum_result = add(sin_x, cos_y)
    
    # View operations
    x_transposed = transpose(x)
    x_reshaped = reshape(x, (4,))
    y_broadcast = broadcast_to(y, (2, 2))
    
    # Reduce operations
    x_sum = reduce_sum(x)
    
    # Linear algebra
    matrix_mult = matmul(x, y_broadcast)
    
    # Type casting
    x_int = cast(x, DType.int32)
    
    # Complex combination
    final_result = add(matrix_mult, x_sum)
    
    return [final_result, x_int]


def view_operations_function(inputs: list[Array]) -> list[Array]:
    """Function focused on view operations."""
    x = inputs[0]  # Should be (2, 2)
    
    # Transpose with different axes
    x_t = transpose(x, 0, 1)
    
    # Reshape to vector
    x_flat = reshape(x, (4,))
    
    # Broadcast to larger shape
    x_big = broadcast_to(x_flat, (3, 4))
    
    return [x_t, x_big]


def reduction_function(inputs: list[Array]) -> list[Array]:
    """Function focused on reduction operations."""
    x = inputs[0]  # Should be (2, 3)
    
    # Reduce sum along different axes
    sum_all = reduce_sum(x)  # Reduce all
    sum_axis0 = reduce_sum(x, axes=0)  # Reduce axis 0
    sum_axis1 = reduce_sum(x, axes=1)  # Reduce axis 1
    
    return [sum_all, sum_axis0, sum_axis1]


def test_enhanced_operation_names():
    """Test all the enhanced JAX-like operation names."""
    print("=== Testing Enhanced JAX-like Operation Names ===\n")
    
    # Test 1: Comprehensive operations
    print("--- Test 1: Comprehensive Operations ---")
    x = array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    y = array([[0.5, 1.0], [1.5, 2.0]])  # (2, 2)
    
    trace1 = trace_function(comprehensive_operations_function, [x, y])
    print("Complex function with multiple operation types:")
    print(trace1)
    print()
    
    reset_traced_flags(trace1.inputs + trace1.outputs)
    
    # Test 2: View operations
    print("--- Test 2: View Operations ---")
    x = array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    
    trace2 = trace_function(view_operations_function, [x])
    print("Function showcasing view operations:")
    print(trace2)
    print()
    
    reset_traced_flags(trace2.inputs + trace2.outputs)
    
    # Test 3: Reduction operations
    print("--- Test 3: Reduction Operations ---")
    x = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    
    trace3 = trace_function(reduction_function, [x])
    print("Function showcasing reduction operations:")
    print(trace3)
    print()
    
    reset_traced_flags(trace3.inputs + trace3.outputs)
    
    # Test 4: Simple operations with clear names
    print("--- Test 4: Simple Binary and Unary Operations ---")
    
    def simple_ops(inputs: list[Array]) -> list[Array]:
        x = inputs[0]
        y = negate(x)
        z = sin(y)
        w = add(x, z)
        return [w]
    
    x = array([1.0, 2.0, 3.0])
    trace4 = trace_function(simple_ops, [x])
    print("Simple function: f(x) = x + sin(-x)")
    print(trace4)
    print()
    
    reset_traced_flags(trace4.inputs + trace4.outputs)
    
    print("=== Enhanced Operation Names Test Complete ===")
    print("\nKey improvements:")
    print("- 'neg' instead of 'negate'")
    print("- 'convert_element_type[new_dtype=...]' instead of 'cast'") 
    print("- 'transpose[permutation=(...)]' instead of 'transpose'")
    print("- 'reshape[new_sizes=...]' instead of 'reshape'")
    print("- 'broadcast_in_dim[shape=...]' instead of 'broadcast_to'")
    print("- 'reduce_sum[axes=...]' instead of 'reduce_sum'")
    print("- 'dot_general' instead of 'matmul'")
    print("- 'rng_normal[shape=...]' instead of 'randn_...'")


if __name__ == "__main__":
    test_enhanced_operation_names()
