#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Nabla 2025
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

"""Test nested transformations and composability of VJP and JVP."""

import nabla as nb


def test_jvp_of_vjp():
    """Test JVP of VJP: computing second derivatives using forward-over-reverse mode."""
    print("=== Testing JVP(VJP(f)) - Hessian via Forward-over-Reverse ===")

    def square_fn(inputs):
        x = inputs[0]
        return [x * x]  # f(x) = x²

    def grad_fn(inputs):
        """Gradient function: returns df/dx"""
        x = inputs[0]
        # Use VJP to compute gradient
        _, vjp_fn = nb.vjp(square_fn, [x])
        cotangent = nb.array([1.0])
        grads = vjp_fn([cotangent])
        return grads  # returns [2x]

    # Now compute JVP of the gradient function to get second derivative
    x = nb.array([3.0])
    tangent = nb.array([1.0])

    # JVP(VJP(f)) should give us the Hessian
    grad_values, hessian_values = nb.jvp(grad_fn, [x], [tangent])

    grad_at_3 = grad_values[0].to_numpy()[0]
    hessian_at_3 = hessian_values[0].to_numpy()[0]

    print(f"f'(3) = {grad_at_3} (expected: 6)")
    print(f"f''(3) = {hessian_at_3} (expected: 2)")

    # For f(x) = x², f'(x) = 2x, f''(x) = 2
    assert abs(grad_at_3 - 6.0) < 1e-6, f"First derivative incorrect: {grad_at_3}"
    assert (
        abs(hessian_at_3 - 2.0) < 1e-6
    ), f"Second derivative incorrect: {hessian_at_3}"

    print("✓ JVP(VJP(f)) test passed!")


def test_vjp_of_jvp():
    """Test VJP of JVP: computing mixed derivatives using reverse-over-forward mode."""
    print("\n=== Testing VJP(JVP(f)) - Mixed Second Derivatives ===")

    def cubic_fn(inputs):
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    def jvp_fn_factory(tangent_value):
        """Factory to create JVP function with fixed tangent"""

        def jvp_fn(inputs):
            x = inputs[0]
            tangent = nb.array([tangent_value])
            # Use JVP to compute directional derivative
            values, tangents = nb.jvp(cubic_fn, [x], [tangent])
            return tangents  # returns [3x² * tangent_value]

        return jvp_fn

    # Create JVP function with tangent = 1.0
    jvp_fn = jvp_fn_factory(1.0)

    # Now compute VJP of the JVP function
    x = nb.array([2.0])
    jvp_values, vjp_fn = nb.vjp(jvp_fn, [x])
    cotangent = nb.array([1.0])
    vjp_gradients = vjp_fn([cotangent])

    jvp_at_2 = jvp_values[0].to_numpy()[0]
    mixed_deriv_at_2 = vjp_gradients[0].to_numpy()[0]

    print(f"JVP(f)(2) = {jvp_at_2} (expected: 12)")
    print(f"d/dx[JVP(f)](2) = {mixed_deriv_at_2} (expected: 12)")

    # For f(x) = x³, JVP gives 3x²*tangent = 3x² (tangent=1)
    # d/dx[3x²] = 6x, so at x=2: 6*2 = 12
    assert abs(jvp_at_2 - 12.0) < 1e-6, f"JVP value incorrect: {jvp_at_2}"
    assert (
        abs(mixed_deriv_at_2 - 12.0) < 1e-6
    ), f"Mixed derivative incorrect: {mixed_deriv_at_2}"

    print("✓ VJP(JVP(f)) test passed!")


def test_double_nested_transformations():
    """Test deeply nested transformations: JVP(VJP(JVP(f)))"""
    print("\n=== Testing JVP(VJP(JVP(f))) - Triple Nested ===")

    def simple_fn(inputs):
        x = inputs[0]
        return [x * x * x * x]  # f(x) = x⁴

    def jvp_fn(inputs):
        """First level: JVP of f"""
        x = inputs[0]
        tangent = nb.array([1.0])
        values, tangents = nb.jvp(simple_fn, [x], [tangent])
        return tangents  # returns [4x³]

    def vjp_jvp_fn(inputs):
        """Second level: VJP of JVP of f"""
        x = inputs[0]
        values, vjp_fn = nb.vjp(jvp_fn, [x])
        cotangent = nb.array([1.0])
        grads = vjp_fn([cotangent])
        return grads  # returns [12x²]

    # Third level: JVP of VJP of JVP of f
    x = nb.array([2.0])
    tangent = nb.array([1.0])

    final_values, final_tangents = nb.jvp(vjp_jvp_fn, [x], [tangent])

    result_value = final_values[0].to_numpy()[0]
    result_tangent = final_tangents[0].to_numpy()[0]

    print(f"VJP(JVP(f))(2) = {result_value} (expected: 48)")
    print(f"JVP(VJP(JVP(f)))(2) = {result_tangent} (expected: 48)")

    # For f(x) = x⁴:
    # JVP(f) = 4x³
    # VJP(JVP(f)) = d/dx[4x³] = 12x²
    # JVP(VJP(JVP(f))) = d/dx[12x²] = 24x
    # At x=2: 24*2 = 48
    assert (
        abs(result_value - 48.0) < 1e-6
    ), f"VJP(JVP(f)) value incorrect: {result_value}"
    assert (
        abs(result_tangent - 48.0) < 1e-6
    ), f"JVP(VJP(JVP(f))) value incorrect: {result_tangent}"

    print("✓ Triple nested transformation test passed!")


def test_multivariable_nested():
    """Test nested transformations with multiple variables."""
    print("\n=== Testing Nested Transformations with Multiple Variables ===")

    def multivariable_fn(inputs):
        x, y = inputs
        return [x * x * y + y * y * y]  # f(x,y) = x²y + y³

    def partial_x_fn(inputs):
        """Compute ∂f/∂x using VJP"""
        x, y = inputs
        values, vjp_fn = nb.vjp(multivariable_fn, [x, y])
        cotangent = nb.array([1.0])
        grads = vjp_fn([cotangent])
        return [grads[0]]  # return only ∂f/∂x

    # Compute ∂²f/∂x∂y using JVP of the partial_x function
    x = nb.array([2.0])
    y = nb.array([3.0])

    # Tangent in y direction to get ∂/∂y[∂f/∂x]
    tangent_x = nb.array([0.0])  # no change in x
    tangent_y = nb.array([1.0])  # unit change in y

    partial_x_values, mixed_partial = nb.jvp(
        partial_x_fn, [x, y], [tangent_x, tangent_y]
    )

    partial_x_at_2_3 = partial_x_values[0].to_numpy()[0]
    mixed_partial_at_2_3 = mixed_partial[0].to_numpy()[0]

    print(f"∂f/∂x(2,3) = {partial_x_at_2_3} (expected: 12)")
    print(f"∂²f/∂x∂y(2,3) = {mixed_partial_at_2_3} (expected: 4)")

    # For f(x,y) = x²y + y³:
    # ∂f/∂x = 2xy, at (2,3): 2*2*3 = 12
    # ∂²f/∂x∂y = ∂/∂y[2xy] = 2x, at (2,3): 2*2 = 4
    assert (
        abs(partial_x_at_2_3 - 12.0) < 1e-6
    ), f"Partial derivative incorrect: {partial_x_at_2_3}"
    assert (
        abs(mixed_partial_at_2_3 - 4.0) < 1e-6
    ), f"Mixed partial incorrect: {mixed_partial_at_2_3}"

    print("✓ Multivariable nested transformation test passed!")


if __name__ == "__main__":
    print("Testing Nested Transformations and Composability")
    print("=" * 60)

    test_jvp_of_vjp()
    test_vjp_of_jvp()
    test_double_nested_transformations()
    test_multivariable_nested()

    print("\n" + "=" * 60)
    print("🎉 All nested transformation tests passed!")
    print("\nNabla's transformation system is fully composable:")
    print("- JVP(VJP(f)) works correctly ✓")
    print("- VJP(JVP(f)) works correctly ✓")
    print("- Triple nested transformations work ✓")
    print("- Multivariable mixed partials work ✓")
    print("\nThis demonstrates JAX-level composability!")
