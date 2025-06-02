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

import numpy as np
import pytest

import nabla as nb


def test_vjp_cubic_function():
    """Test VJP computation for a cubic function f(x) = x³."""
    device = nb.device("cpu")

    def cubic_fn(inputs: list[nb.Array]) -> list[nb.Array]:
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    # Test case: x = 2.0
    x = nb.array([2.0]).to(device)

    # Compute VJP
    values, vjp_fn = nb.vjp(cubic_fn, [x])
    cotangent = [nb.ones(values[0].shape).to(device)]
    gradients = vjp_fn(cotangent)

    # Assertions
    # f(2) = 8
    assert np.isclose(values[0].to_numpy(), 8.0, rtol=1e-6), (
        f"Expected f(2)=8, got {values[0].to_numpy()}"
    )

    # f'(x) = 3x², so f'(2) = 12
    expected_gradient = 3 * 2.0**2  # 12.0
    assert np.isclose(gradients[0][0].to_numpy(), expected_gradient, rtol=1e-6), (
        f"Expected gradient 12.0, got {gradients[0][0].to_numpy()}"
    )


def test_vjp_second_order_derivatives():
    """Test second-order derivatives using nested VJP calls."""
    device = nb.device("cpu")

    def cubic_fn(inputs: list[nb.Array]) -> list[nb.Array]:
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    x = nb.array([2.0]).to(device)

    # Define jacobian function for second-order derivatives
    def jacobian_fn(inputs):
        x_inner = inputs[0]
        _, vjp_fn = nb.vjp(cubic_fn, [x_inner])
        cotangent = [nb.ones((1,)).to(device)]
        gradients = vjp_fn(cotangent)
        return [gradients[0][0]]

    # Compute second-order derivative
    _, hessian_fn = nb.vjp(jacobian_fn, [x])
    cotangent2 = [nb.ones((1,)).to(device)]
    second_order_grad = hessian_fn(cotangent2)

    # f''(x) = 6x, so f''(2) = 12
    expected_second_order = 6 * 2.0  # 12.0
    assert np.isclose(
        second_order_grad[0][0].to_numpy(), expected_second_order, rtol=1e-6
    ), (
        f"Expected second-order derivative 12.0, got {second_order_grad[0][0].to_numpy()}"
    )


def test_vjp_multiple_inputs():
    """Test VJP with multiple input arrays."""
    device = nb.device("cpu")

    def multi_input_fn(inputs: list[nb.Array]) -> list[nb.Array]:
        x, y = inputs[0], inputs[1]
        return [x * y + x * x]  # f(x,y) = xy + x²

    x = nb.array([3.0]).to(device)
    y = nb.array([4.0]).to(device)

    values, vjp_fn = nb.vjp(multi_input_fn, [x, y])
    cotangent = [nb.ones(values[0].shape).to(device)]
    gradients = vjp_fn(cotangent)

    # f(3,4) = 3*4 + 3² = 12 + 9 = 21
    assert np.isclose(values[0].to_numpy(), 21.0, rtol=1e-6), (
        f"Expected f(3,4)=21, got {values[0].to_numpy()}"
    )

    # ∂f/∂x = y + 2x = 4 + 6 = 10
    # ∂f/∂y = x = 3
    assert np.isclose(gradients[0][0].to_numpy(), 10.0, rtol=1e-6), (
        f"Expected ∂f/∂x=10, got {gradients[0][0].to_numpy()}"
    )
    assert np.isclose(gradients[0][1].to_numpy(), 3.0, rtol=1e-6), (
        f"Expected ∂f/∂y=3, got {gradients[0][1].to_numpy()}"
    )


@pytest.mark.parametrize("x_val", [0.0, 1.0, -2.0, 5.0])
def test_vjp_parametrized_inputs(x_val):
    """Test VJP with different input values."""
    device = nb.device("cpu")

    def square_fn(inputs: list[nb.Array]) -> list[nb.Array]:
        x = inputs[0]
        return [x * x]  # f(x) = x²

    x = nb.array([x_val]).to(device)
    values, vjp_fn = nb.vjp(square_fn, [x])
    cotangent = [nb.ones(values[0].shape).to(device)]
    gradients = vjp_fn(cotangent)

    # f(x) = x²
    expected_value = x_val**2
    assert np.isclose(values[0].to_numpy(), expected_value, rtol=1e-6), (
        f"Expected f({x_val})={expected_value}, got {values[0].to_numpy()}"
    )

    # f'(x) = 2x
    expected_gradient = 2 * x_val
    assert np.isclose(gradients[0][0].to_numpy(), expected_gradient, rtol=1e-6), (
        f"Expected gradient {expected_gradient}, got {gradients[0][0].to_numpy()}"
    )


if __name__ == "__main__":
    # Run all tests when script is executed directly
    test_vjp_cubic_function()
    test_vjp_second_order_derivatives()
    test_vjp_multiple_inputs()
    print("All VJP tests passed!")

    # Run parametrized test for a few values
    for val in [0.0, 1.0, -2.0]:
        test_vjp_parametrized_inputs(val)
    print("Parametrized tests passed!")
