#!/usr/bin/env python3
"""Test JVP (forward-mode autodiff) functionality."""

import nabla as nb
from nabla.core.trafos import jvp


def test_basic_jvp():
    """Test basic JVP functionality."""
    print("=== Testing Basic JVP ===")

    def square_fn(inputs):
        x = inputs[0]
        return [x * x]  # f(x) = x²

    # Create input and tangent
    x = nb.array([3.0])
    tangent = nb.array([1.0])  # dx = 1

    # Compute JVP: f'(x) * dx where f'(x) = 2x
    outputs, output_tangents = jvp(square_fn, [x], [tangent])

    print(f"f(3) = {outputs[0]} (expected: 9)")
    print(f"f'(3) * 1 = {output_tangents[0]} (expected: 6)")

    assert abs(outputs[0].get_numpy()[0] - 9.0) < 1e-6
    assert abs(output_tangents[0].get_numpy()[0] - 6.0) < 1e-6
    print("✓ Basic JVP test passed\n")


def test_cubic_jvp():
    """Test JVP with cubic function."""
    print("=== Testing Cubic JVP ===")

    def cubic_fn(inputs):
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    # For f(x) = x³, f'(x) = 3x²
    x = nb.array([2.0])
    tangent = nb.array([1.0])

    outputs, output_tangents = jvp(cubic_fn, [x], [tangent])

    print(f"f(2) = {outputs[0]} (expected: 8)")
    print(f"f'(2) * 1 = {output_tangents[0]} (expected: 12)")

    assert abs(outputs[0].get_numpy()[0] - 8.0) < 1e-6
    assert abs(output_tangents[0].get_numpy()[0] - 12.0) < 1e-6
    print("✓ Cubic JVP test passed\n")


def test_multivariable_jvp():
    """Test JVP with multiple variables."""
    print("=== Testing Multivariable JVP ===")

    def fn(inputs):
        x, y = inputs
        return [x * y + x * x]  # f(x,y) = xy + x²

    # For f(x,y) = xy + x², ∇f = (y + 2x, x)
    # At (x=2, y=3): ∇f = (3 + 4, 2) = (7, 2)
    x = nb.array([2.0])
    y = nb.array([3.0])
    dx = nb.array([1.0])  # tangent for x
    dy = nb.array([0.5])  # tangent for y

    outputs, output_tangents = jvp(fn, [x, y], [dx, dy])

    # Expected: f(2,3) = 2*3 + 2*2 = 10
    # Expected tangent: ∇f · (dx, dy) = (7, 2) · (1, 0.5) = 7 + 1 = 8
    print(f"f(2,3) = {outputs[0]} (expected: 10)")
    print(f"∇f · (1, 0.5) = {output_tangents[0]} (expected: 8)")

    assert abs(outputs[0].get_numpy()[0] - 10.0) < 1e-6
    assert abs(output_tangents[0].get_numpy()[0] - 8.0) < 1e-6
    print("✓ Multivariable JVP test passed\n")


def test_jvp_vjp_consistency():
    """Test that JVP and VJP are consistent for simple functions."""
    print("=== Testing JVP-VJP Consistency ===")

    def simple_fn(inputs):
        x = inputs[0]
        return [x * x + x]  # f(x) = x² + x

    x = nb.array([4.0])

    # JVP: compute f'(4) using forward mode
    tangent = nb.array([1.0])
    _, jvp_result = jvp(simple_fn, [x], [tangent])

    # VJP: compute f'(4) using reverse mode
    from nabla.core.trafos import vjp

    _, vjp_fn = vjp(simple_fn, [x])
    cotangent = nb.array([1.0])
    vjp_result = vjp_fn([cotangent])

    print(f"JVP result: {jvp_result[0]}")
    print(f"VJP result: {vjp_result[0]}")

    # For f(x) = x² + x, f'(x) = 2x + 1
    # f'(4) = 2*4 + 1 = 9
    expected = 9.0

    assert abs(jvp_result[0].get_numpy()[0] - expected) < 1e-6
    assert abs(vjp_result[0].get_numpy()[0] - expected) < 1e-6
    assert abs(jvp_result[0].get_numpy()[0] - vjp_result[0].get_numpy()[0]) < 1e-6

    print(f"Expected: {expected}")
    print("✓ JVP-VJP consistency test passed\n")


def test_higher_order_jvp():
    """Test higher-order derivatives using nested JVP calls."""
    print("=== Testing Higher-Order JVP ===")

    def cubic_fn(inputs):
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    # Create input
    x = nb.array([2.0])
    tangent = nb.array([1.0])

    # First-order: compute f(x) and f'(x) * tangent
    values, first_order = jvp(cubic_fn, [x], [tangent])

    print(f"f(2) = {values[0]} (expected: 8)")
    print(f"f'(2) * 1 = {first_order[0]} (expected: 12)")

    # For second-order derivatives, create a jacobian function
    def jacobian_fn(inputs):
        x = inputs[0]
        _, tangents = jvp(cubic_fn, [x], [nb.ones(x.shape)])
        return [tangents[0]]

    # Second-order: compute the derivative of the jacobian function
    _, second_order = jvp(jacobian_fn, [x], [tangent])

    print(f"f''(2) * 1 = {second_order[0]} (expected: 12)")

    # Verify results
    # For f(x) = x³:
    # f(2) = 8
    # f'(x) = 3x², so f'(2) = 12
    # f''(x) = 6x, so f''(2) = 12
    assert abs(values[0].get_numpy()[0] - 8.0) < 1e-6
    assert abs(first_order[0].get_numpy()[0] - 12.0) < 1e-6
    assert abs(second_order[0].get_numpy()[0] - 12.0) < 1e-6

    print("✓ Higher-order JVP test passed\n")


if __name__ == "__main__":
    print("Testing JVP (Forward-Mode Autodiff)")
    print("=" * 50)

    test_basic_jvp()
    test_cubic_jvp()
    test_multivariable_jvp()
    test_jvp_vjp_consistency()
    test_higher_order_jvp()

    print("=" * 50)
    print("🎉 All JVP tests passed!")
    print("\nJVP (forward-mode autodiff) is working correctly!")
