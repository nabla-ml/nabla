#!/usr/bin/env python3

"""Comprehensive tests for VJP with various function signatures."""

import numpy as np

import nabla as nb


def test_vjp_single_arg():
    """Test VJP with single positional argument."""
    print("=== Test 1: Single Argument ===")

    def func(x):
        return x * x + nb.array([2.0]) * x + nb.array([1.0])

    x = nb.array([2.0, 3.0])
    outputs, vjp_fn = nb.vjp(func, x)

    cotangent = nb.array([1.0, 1.0])
    gradient = vjp_fn(cotangent)  # Should return single Array, not tuple

    print(f"Input: {x}")
    print(f"Output: {outputs}")
    print(f"Gradient: {gradient}")
    print(f"Gradient type: {type(gradient)}")

    # Expected: df/dx = 2x + 2, so at x=[2,3] -> [6, 8]
    expected = np.array([6.0, 8.0])
    # JAX always returns tuple, so access first element
    actual_gradient = gradient[0]
    assert np.allclose(actual_gradient.to_numpy(), expected), (
        f"Expected {expected}, got {actual_gradient.to_numpy()}"
    )
    print("✓ Single argument test passed\n")


def test_vjp_multiple_args():
    """Test VJP with multiple positional arguments."""
    print("=== Test 2: Multiple Arguments ===")

    def func(x, y, z):
        return x * y + y * z + x * z

    x = nb.array([1.0, 2.0])
    y = nb.array([3.0, 4.0])
    z = nb.array([5.0, 6.0])

    outputs, vjp_fn = nb.vjp(func, x, y, z)

    cotangent = nb.array([1.0, 1.0])
    gradients = vjp_fn(cotangent)  # Should return tuple of 3 gradients

    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Input z: {z}")
    print(f"Output: {outputs}")
    print(f"Gradients type: {type(gradients)}")
    print(f"Number of gradients: {len(gradients)}")
    print(f"Gradient w.r.t. x: {gradients[0]}")
    print(f"Gradient w.r.t. y: {gradients[1]}")
    print(f"Gradient w.r.t. z: {gradients[2]}")

    # Expected: df/dx = y + z, df/dy = x + z, df/dz = y + x
    # At x=[1,2], y=[3,4], z=[5,6]:
    # df/dx = [3+5, 4+6] = [8, 10]
    # df/dy = [1+5, 2+6] = [6, 8]
    # df/dz = [3+1, 4+2] = [4, 6]
    expected_x = np.array([8.0, 10.0])
    expected_y = np.array([6.0, 8.0])
    expected_z = np.array([4.0, 6.0])

    assert np.allclose(gradients[0].to_numpy(), expected_x), (
        f"Expected {expected_x}, got {gradients[0].to_numpy()}"
    )
    assert np.allclose(gradients[1].to_numpy(), expected_y), (
        f"Expected {expected_y}, got {gradients[1].to_numpy()}"
    )
    assert np.allclose(gradients[2].to_numpy(), expected_z), (
        f"Expected {expected_z}, got {gradients[2].to_numpy()}"
    )
    print("✓ Multiple arguments test passed\n")


def test_vjp_kwargs_only():
    """Test VJP with keyword-only functions using functools.partial (JAX-compatible approach)."""
    print("=== Test 3: Keyword Arguments via functools.partial ===")

    def func(*, x, y):
        return x * y + x

    x = nb.array([2.0, 3.0])
    y = nb.array([4.0, 5.0])

    # JAX-compatible approach: use functools.partial for keyword arguments
    import functools
    func_with_args = functools.partial(func, x=x, y=y)
    outputs, vjp_fn = nb.vjp(func_with_args)

    cotangent = nb.array([1.0, 1.0])
    gradients = vjp_fn(cotangent)  # Returns gradients for the partial function (empty tuple)

    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Output: {outputs}")
    print(f"Gradients type: {type(gradients)}")
    print("Note: Gradients are empty since all inputs are captured in partial")

    # Expected: f(x, y) = x * y + x, with x=[2, 3], y=[4, 5]
    # f = [2*4 + 2, 3*5 + 3] = [10, 18]
    expected = np.array([10.0, 18.0])
    
    assert np.allclose(outputs.to_numpy(), expected), (
        f"Expected {expected}, got {outputs.to_numpy()}"
    )
    print("✓ Keyword arguments (via partial) test passed\n")


def test_vjp_mixed_args_kwargs():
    """Test VJP with multiple positional arguments (kwargs not supported in JAX mode)."""
    print("=== Test 4: Multiple Args (No Kwargs) ===")

    def func(a, b, scale, offset):
        return scale * (a * b + offset)

    a = nb.array([1.0, 2.0])
    b = nb.array([3.0, 4.0])
    scale = nb.array([2.0])
    offset = nb.array([1.0])

    outputs, vjp_fn = nb.vjp(func, a, b, scale, offset)

    cotangent = nb.array([1.0, 1.0])
    gradients = vjp_fn(cotangent)  # Returns tuple of gradients

    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Input scale: {scale}")
    print(f"Input offset: {offset}")
    print(f"Output: {outputs}")
    grad_a, grad_b, grad_scale, grad_offset = gradients
    print(f"Gradient w.r.t. a: {grad_a}")
    print(f"Gradient w.r.t. b: {grad_b}")
    print(f"Gradient w.r.t. scale: {grad_scale}")
    print(f"Gradient w.r.t. offset: {grad_offset}")

    # Expected: f = scale * (a * b + offset)
    # df/da = scale * b = 2 * [3, 4] = [6, 8]
    # df/db = scale * a = 2 * [1, 2] = [2, 4]
    # df/dscale = (a * b + offset) = [1*3+1, 2*4+1] = [4, 9] -> sum = 13
    # df/doffset = scale = 2 -> sum = 4
    expected_a = np.array([6.0, 8.0])
    expected_b = np.array([2.0, 4.0])
    expected_scale = np.array([13.0])  # sum of [4, 9]
    expected_offset = np.array([4.0])  # sum of [2, 2]

    assert np.allclose(grad_a.to_numpy(), expected_a), (
        f"Expected {expected_a}, got {grad_a.to_numpy()}"
    )
    assert np.allclose(grad_b.to_numpy(), expected_b), (
        f"Expected {expected_b}, got {grad_b.to_numpy()}"
    )
    assert np.allclose(grad_scale.to_numpy(), expected_scale), (
        f"Expected {expected_scale}, got {grad_scale.to_numpy()}"
    )
    assert np.allclose(grad_offset.to_numpy(), expected_offset), (
        f"Expected {expected_offset}, got {grad_offset.to_numpy()}"
    )
    print("✓ Multiple args test passed\n")


def test_vjp_nested_structures():
    """Test VJP with nested data structures (lists, dicts)."""
    print("=== Test 5: Nested Structures ===")

    def func(data):
        x = data["x"]
        y_list = data["y"]
        return x * y_list[0] + x * y_list[1]

    x = nb.array([2.0, 3.0])
    y1 = nb.array([4.0, 5.0])
    y2 = nb.array([6.0, 7.0])

    data = {"x": x, "y": [y1, y2]}

    outputs, vjp_fn = nb.vjp(func, data)

    cotangent = nb.array([1.0, 1.0])
    gradient = vjp_fn(cotangent)  # Returns tuple, need to unpack

    print(f"Input data: {data}")
    print(f"Output: {outputs}")
    print(f"Gradient type: {type(gradient)}")
    # Unpack the single-element tuple
    actual_gradient = gradient[0]
    print(f"Gradient keys: {list(actual_gradient.keys())}")
    print(f"Gradient w.r.t. x: {actual_gradient['x']}")
    print(f"Gradient w.r.t. y[0]: {actual_gradient['y'][0]}")
    print(f"Gradient w.r.t. y[1]: {actual_gradient['y'][1]}")

    # Expected: f = x * y[0] + x * y[1] = x * (y[0] + y[1])
    # df/dx = y[0] + y[1] = [4+6, 5+7] = [10, 12]
    # df/dy[0] = x = [2, 3]
    # df/dy[1] = x = [2, 3]
    expected_x = np.array([10.0, 12.0])
    expected_y0 = np.array([2.0, 3.0])
    expected_y1 = np.array([2.0, 3.0])

    assert np.allclose(actual_gradient["x"].to_numpy(), expected_x), (
        f"Expected {expected_x}, got {actual_gradient['x'].to_numpy()}"
    )
    assert np.allclose(actual_gradient["y"][0].to_numpy(), expected_y0), (
        f"Expected {expected_y0}, got {actual_gradient['y'][0].to_numpy()}"
    )
    assert np.allclose(actual_gradient["y"][1].to_numpy(), expected_y1), (
        f"Expected {expected_y1}, got {actual_gradient['y'][1].to_numpy()}"
    )
    print("✓ Nested structures test passed\n")


def test_vjp_list_input():
    """Test VJP with list input (like the original failing example)."""
    print("=== Test 6: List Input (Legacy Style) ===")

    def func(inputs):
        return inputs[0] ** 3  # Cubic function

    x = nb.array([2.0])

    outputs, vjp_fn = nb.vjp(func, [x])  # Pass list as single argument

    cotangent = [nb.array([1.0])]
    gradient = vjp_fn(cotangent)  # Returns tuple, need to unpack

    print(f"Input: {[x]}")
    print(f"Output: {outputs}")
    print(f"Gradient type: {type(gradient)}")
    print(f"Gradient: {gradient}")
    # Unpack the single-element tuple
    actual_gradient = gradient[0]
    print(f"Gradient[0]: {actual_gradient}")

    # Expected: f = x^3, df/dx = 3x^2 = 3 * 4 = 12
    expected = np.array([12.0])
    assert np.allclose(actual_gradient[0].to_numpy(), expected), (
        f"Expected {expected}, got {actual_gradient[0].to_numpy()}"
    )
    print("✓ List input test passed\n")


def test_vjp_complex_computation():
    """Test VJP with more complex mathematical operations."""
    print("=== Test 7: Complex Computation ===")

    def complex_func(x, y):
        # f(x, y) = sin(x) * cos(y) + exp(x) * log(y)
        import nabla as nb

        return nb.sin(x) * nb.cos(y) + nb.exp(x) * nb.log(y)

    x = nb.array([0.5])  # sin'(0.5) = cos(0.5), exp'(0.5) = exp(0.5)
    y = nb.array([1.0])  # cos'(1.0) = -sin(1.0), log'(1.0) = 1.0

    outputs, vjp_fn = nb.vjp(complex_func, x, y)

    cotangent = nb.array([1.0])
    gradients = vjp_fn(cotangent)

    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Output: {outputs}")
    print(f"Gradient w.r.t. x: {gradients[0]}")
    print(f"Gradient w.r.t. y: {gradients[1]}")

    # Expected:
    # df/dx = cos(x) * cos(y) + exp(x) * log(y)
    # df/dy = -sin(x) * sin(y) + exp(x) / y
    # At x=0.5, y=1.0:
    # df/dx = cos(0.5) * cos(1.0) + exp(0.5) * log(1.0) = cos(0.5) * cos(1.0) + exp(0.5) * 0
    # df/dy = -sin(0.5) * sin(1.0) + exp(0.5) / 1.0 = -sin(0.5) * sin(1.0) + exp(0.5)

    expected_x = np.cos(0.5) * np.cos(1.0)
    expected_y = -np.sin(0.5) * np.sin(1.0) + np.exp(0.5)

    print(f"Expected gradient x: {expected_x}")
    print(f"Expected gradient y: {expected_y}")

    assert np.allclose(gradients[0].to_numpy(), expected_x, rtol=1e-5), (
        f"Expected {expected_x}, got {gradients[0].to_numpy()}"
    )
    assert np.allclose(gradients[1].to_numpy(), expected_y, rtol=1e-5), (
        f"Expected {expected_y}, got {gradients[1].to_numpy()}"
    )
    print("✓ Complex computation test passed\n")


def test_vjp_multiple_outputs():
    """Test VJP with multiple outputs."""
    print("=== Test 8: Multiple Outputs ===")

    def multi_output_func(x, y):
        return x + y, x * y

    x = nb.array([2.0, 3.0])
    y = nb.array([4.0, 5.0])

    outputs, vjp_fn = nb.vjp(multi_output_func, x, y)

    # Need cotangents for both outputs
    cotangent1 = nb.array([1.0, 1.0])
    cotangent2 = nb.array([1.0, 1.0])

    gradients = vjp_fn((cotangent1, cotangent2))

    print(f"Input x: {x}")
    print(f"Input y: {y}")
    print(f"Outputs: {outputs}")
    print(f"Gradient w.r.t. x: {gradients[0]}")
    print(f"Gradient w.r.t. y: {gradients[1]}")

    # Expected:
    # f1 = x + y, f2 = x * y
    # df1/dx = 1, df1/dy = 1
    # df2/dx = y, df2/dy = x
    # Total: df/dx = 1 + y = [1+4, 1+5] = [5, 6]
    # Total: df/dy = 1 + x = [1+2, 1+3] = [3, 4]
    expected_x = np.array([5.0, 6.0])
    expected_y = np.array([3.0, 4.0])

    assert np.allclose(gradients[0].to_numpy(), expected_x), (
        f"Expected {expected_x}, got {gradients[0].to_numpy()}"
    )
    assert np.allclose(gradients[1].to_numpy(), expected_y), (
        f"Expected {expected_y}, got {gradients[1].to_numpy()}"
    )
    print("✓ Multiple outputs test passed\n")


def test_vjp_edge_cases():
    """Test VJP edge cases."""
    print("=== Test 9: Edge Cases ===")

    # Test with constants
    def func_with_constants(x):
        return x * nb.array([5.0]) + nb.array(
            [3.0]
        )  # Constants shouldn't affect gradients

    x = nb.array([1.0, 2.0])
    outputs, vjp_fn = nb.vjp(func_with_constants, x)

    cotangent = nb.array([1.0, 1.0])
    gradient = vjp_fn(cotangent)

    print(f"Input: {x}")
    print(f"Output: {outputs}")
    print(f"Gradient: {gradient}")

    # Expected: df/dx = 5.0
    expected = np.array([5.0, 5.0])
    # Unpack the single-element tuple
    actual_gradient = gradient[0]
    assert np.allclose(actual_gradient.to_numpy(), expected), (
        f"Expected {expected}, got {actual_gradient.to_numpy()}"
    )
    print("✓ Constants test passed")

    # Test with zero gradients
    def zero_grad_func(x):
        return nb.array([42.0])  # Constant output, zero gradient

    x = nb.array([1.0, 2.0])
    outputs, vjp_fn = nb.vjp(zero_grad_func, x)

    cotangent = nb.array([1.0])
    gradient = vjp_fn(cotangent)

    print(f"Zero gradient input: {x}")
    print(f"Zero gradient output: {outputs}")
    print(f"Zero gradient: {gradient}")

    # Expected: df/dx = 0
    expected_zero = np.array([0.0, 0.0])
    # Unpack the single-element tuple
    actual_zero_gradient = gradient[0]
    assert np.allclose(actual_zero_gradient.to_numpy(), expected_zero), (
        f"Expected {expected_zero}, got {actual_zero_gradient.to_numpy()}"
    )
    print("✓ Zero gradients test passed\n")


if __name__ == "__main__":
    print("🧪 Comprehensive VJP Testing")
    print("=" * 50)

    try:
        test_vjp_single_arg()
        test_vjp_multiple_args()
        test_vjp_kwargs_only()
        test_vjp_mixed_args_kwargs()
        test_vjp_nested_structures()
        test_vjp_list_input()
        test_vjp_complex_computation()
        test_vjp_multiple_outputs()
        test_vjp_edge_cases()

        print("🎉 All comprehensive VJP tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
