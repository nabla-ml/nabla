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

"""Test JVP (forward-mode autodiff) functionality."""

import numpy as np
import pytest

import nabla as nb


def test_jvp_cubic_function():
    """Test JVP computation for a cubic function f(x) = x³."""
    device = nb.device("cpu")

    def cubic_fn(inputs):
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    # Test case: x = 2.0, tangent = 1.0
    x = nb.tensor([2.0]).to(device)
    tangent = nb.tensor([1.0]).to(device)

    values, tangent_out = nb.jvp(cubic_fn, [x], [tangent])

    # Assertions
    # f(2) = 8
    assert np.isclose(values[0].to_numpy(), 8.0, rtol=1e-6), (
        f"Expected f(2)=8, got {values[0].to_numpy()}"
    )

    # f'(x) = 3x², so f'(2) = 12
    # JVP computes f'(x) * tangent = 12 * 1 = 12
    expected_tangent_out = 3 * 2.0**2 * 1.0  # 12.0
    assert np.isclose(tangent_out[0].to_numpy(), expected_tangent_out, rtol=1e-6), (
        f"Expected tangent output 12.0, got {tangent_out[0].to_numpy()}"
    )


def test_jvp_higher_order_derivatives():
    """Test second-order derivatives using nested JVP calls."""
    device = nb.device("cpu")

    def cubic_fn(inputs):
        x = inputs[0]
        return [x * x * x]  # f(x) = x³

    x = nb.tensor([2.0]).to(device)
    tangent = nb.tensor([1.0]).to(device)

    # First-order JVP
    values, first_order = nb.jvp(cubic_fn, [x], [tangent])

    # Define jacobian function for second-order derivatives
    def jacobian_fn(inputs):
        x_inner = inputs[0]
        ones_tangent = nb.ones((1,)).to(device)
        _, tangents = nb.jvp(cubic_fn, [x_inner], [ones_tangent])
        return [tangents[0]]

    # Second-order JVP
    _, second_order = nb.jvp(jacobian_fn, [x], [tangent])

    # f''(x) = 6x, so f''(2) = 12
    expected_second_order = 6 * 2.0 * 1.0  # 12.0
    assert np.isclose(second_order[0].to_numpy(), expected_second_order, rtol=1e-6), (
        f"Expected second-order derivative 12.0, got {second_order[0].to_numpy()}"
    )


def test_jvp_multiple_inputs():
    """Test JVP with multiple input tensors."""
    device = nb.device("cpu")

    def multi_input_fn(inputs):
        x, y = inputs[0], inputs[1]
        return [x * y + x * x]  # f(x,y) = xy + x²

    x = nb.tensor([3.0]).to(device)
    y = nb.tensor([4.0]).to(device)
    tangent_x = nb.tensor([1.0]).to(device)
    tangent_y = nb.tensor([1.0]).to(device)

    values, tangent_out = nb.jvp(multi_input_fn, [x, y], [tangent_x, tangent_y])

    # f(3,4) = 3*4 + 3² = 12 + 9 = 21
    assert np.isclose(values[0].to_numpy(), 21.0, rtol=1e-6), (
        f"Expected f(3,4)=21, got {values[0].to_numpy()}"
    )

    # JVP: (∂f/∂x * tangent_x) + (∂f/∂y * tangent_y)
    # ∂f/∂x = y + 2x = 4 + 6 = 10
    # ∂f/∂y = x = 3
    # JVP = 10 * 1 + 3 * 1 = 13
    expected_tangent_out = 10.0 * 1.0 + 3.0 * 1.0  # 13.0
    assert np.isclose(tangent_out[0].to_numpy(), expected_tangent_out, rtol=1e-6), (
        f"Expected tangent output 13.0, got {tangent_out[0].to_numpy()}"
    )


@pytest.mark.parametrize(
    "x_val,tangent_val", [(0.0, 1.0), (1.0, 2.0), (-2.0, 0.5), (5.0, -1.0)]
)
def test_jvp_parametrized_inputs(x_val, tangent_val):
    """Test JVP with different input and tangent values."""
    device = nb.device("cpu")

    def square_fn(inputs):
        x = inputs[0]
        return [x * x]  # f(x) = x²

    x = nb.tensor([x_val]).to(device)
    tangent = nb.tensor([tangent_val]).to(device)

    values, tangent_out = nb.jvp(square_fn, [x], [tangent])

    # f(x) = x²
    expected_value = x_val**2
    assert np.isclose(values[0].to_numpy(), expected_value, rtol=1e-6), (
        f"Expected f({x_val})={expected_value}, got {values[0].to_numpy()}"
    )

    # JVP: f'(x) * tangent = 2x * tangent
    expected_tangent_out = 2 * x_val * tangent_val
    assert np.isclose(tangent_out[0].to_numpy(), expected_tangent_out, rtol=1e-6), (
        f"Expected tangent output {expected_tangent_out}, got {tangent_out[0].to_numpy()}"
    )


def test_jvp_with_squeeze_unsqueeze():
    """Test JVP with squeeze and unsqueeze operations."""
    device = nb.device("cpu")

    def squeeze_unsqueeze_fn(inputs):
        x = nb.unsqueeze(nb.unsqueeze(inputs[0], [0]), [0])
        x = nb.squeeze(nb.squeeze(x, [0]), [0])
        return [x * x * x]  # f(x) = x³

    x = nb.tensor([2.0]).to(device)
    tangent = nb.tensor([1.0]).to(device)

    values, tangent_out = nb.jvp(squeeze_unsqueeze_fn, [x], [tangent])

    # Should behave the same as cubic function
    assert np.isclose(values[0].to_numpy(), 8.0, rtol=1e-6), (
        f"Expected f(2)=8, got {values[0].to_numpy()}"
    )
    assert np.isclose(tangent_out[0].to_numpy(), 12.0, rtol=1e-6), (
        f"Expected tangent output 12.0, got {tangent_out[0].to_numpy()}"
    )


if __name__ == "__main__":
    # Run all tests when script is executed directly
    test_jvp_cubic_function()
    test_jvp_higher_order_derivatives()
    test_jvp_multiple_inputs()
    test_jvp_with_squeeze_unsqueeze()
    print("All JVP tests passed!")

    # Run parametrized test for a few values
    for x_val, tangent_val in [(0.0, 1.0), (1.0, 2.0), (-2.0, 0.5)]:
        test_jvp_parametrized_inputs(x_val, tangent_val)
    print("Parametrized tests passed!")
