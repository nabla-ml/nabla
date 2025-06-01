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

import nabla as nb


def test_basic_vjp_transform_with_xpr_prints():
    """Test the new VJPTransform class."""
    print("=== Testing VJPTransform ===")

    device = nb.device("cpu")  # Change to "cpu" for CPU testing

    def square_fn(inputs: list[nb.Array]) -> list[nb.Array]:
        x = inputs[0]
        return [x * x * x]

    # Create input
    x = nb.array([2.0]).to(device)
    print("\nOrignal Function:", nb.xpr(square_fn, [x]))

    values, jacobian = nb.vjp(square_fn, [x])
    cotangent = [nb.ones(values[0].shape).to(device)]
    print("\nJacobian:", nb.xpr(jacobian, cotangent))
    d1 = jacobian(cotangent)
    print(f"Value: {values[0]}")
    print(f"First-order derivative: {d1[0]}")

    # For second-order derivatives, we need a proper jacobian function
    def jacobian_fn(inputs):
        x = inputs[0]
        _, vjp_fn = nb.vjp(square_fn, [x])
        cotangent = [nb.ones((1,)).to(device)]
        return [vjp_fn(cotangent)[0]]

    _, hessian = nb.vjp(jacobian_fn, [x])
    cotangent2 = [nb.ones((1,)).to(device)]
    print("\nHessian:", nb.xpr(hessian, cotangent2))
    d2 = hessian(cotangent2)
    print(f"Second-order derivative: {d2[0]}")


if __name__ == "__main__":
    print("\nTesting with xpr prints")
    test_basic_vjp_transform_with_xpr_prints()
