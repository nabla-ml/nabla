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

"""Test JVP (forward-mode autodiff) functionality."""

import nabla
from nabla.core.trafos import jvp


def test_higher_order_jvp():
    """Test higher-order derivatives using nested JVP calls."""
    # print("=== Testing Higher-Order JVP ===")

    device = nabla.device("gpu:0")  # Change to "cpu" for CPU testing

    def cubic_fn(inputs):
        x = nabla.unsqueeze(nabla.unsqueeze(inputs[0], [0]), [0])
        x = nabla.squeeze(nabla.squeeze(x, [0]), [0])
        return [x * x * x]  # f(x) = x³

    x = nabla.array([2.0]).to(device)
    tangent = nabla.array([1.0]).to(device)  # Tangent vector for JVP

    values, first_order = jvp(cubic_fn, [x], [tangent])

    # print(values[0].shape)

    def jacobian_fn(inputs):
        x = inputs[0]
        ones_tangent = nabla.ones((1,)).to(nabla.device("gpu:0"))
        _, tangents = jvp(cubic_fn, [x], [ones_tangent])
        return [tangents[0]]

    _, second_order = jvp(jacobian_fn, [x], [tangent])

    print("Values:", values)
    print("First-order derivative:", first_order)
    print("Second-order derivative:", second_order)


if __name__ == "__main__":
    print("Testing JVP (Forward-Mode Autodiff)")
    test_higher_order_jvp()
