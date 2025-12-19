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

"""Test Dynamic JIT compilation"""

import nabla as nb


def test_jit_with_if_else_backward():
    """Test JIT compilation with conditional statements"""

    def func(inputs: list[nb.Tensor]) -> list[nb.Tensor]:
        input = inputs[0].requires_grad_(True)
        x = input + 1
        if x.item() == 5.0:
            x = x * x
        else:
            x = x * x * x

        x.backward(nb.ones_like(x))
        return [input.grad]

    jitted_func = nb.djit(func)

    for _ in range(10):
        x0 = nb.tensor([4.0])
        outputs0 = jitted_func([x0])
        print("Output:", outputs0[0])

        x1 = nb.tensor([5.0])
        outputs1 = jitted_func([x1])
        print("Output:", outputs1[0])



def test_jit_with_if_else_grad_imperative():
    """Test JIT compilation with conditional statements"""

    def func(inputs: list[nb.Tensor]) -> list[nb.Tensor]:
        input = inputs[0].requires_grad_(True)
        x = input + 1
        if x.item() == 5.0:
            x = x * x
        else:
            x = x * x * x

        x_grad = nb.transforms.utils.grad(x, input)
        return [x_grad]

    jitted_func = nb.djit(func)

    for _ in range(10):
        x0 = nb.tensor([4.0])
        outputs0 = jitted_func([x0])
        print("Output:", outputs0[0])

        x1 = nb.tensor([5.0])
        outputs1 = jitted_func([x1])
        print("Output:", outputs1[0])


def test_jit_with_if_else_grad_functional():
    """Test JIT compilation with conditional statements"""

    @nb.grad
    def func(inputs: list[nb.Tensor]) -> list[nb.Tensor]:
        input = inputs[0]
        x = input + 1
        if x.item() == 5.0:
            x = x * x
        else:
            x = x * x * x

        return [x]

    jitted_func = nb.djit(func)

    for _ in range(10):
        x0 = nb.tensor([4.0])
        outputs0 = jitted_func([x0])
        print("Output:", outputs0[0])

        x1 = nb.tensor([5.0])
        outputs1 = jitted_func([x1])
        print("Output:", outputs1[0])


if __name__ == "__main__":
    # Expected outputs:
    # For input 4.0: gradient should be 10
    # For input 5.0: gradient should be 108
    print("Testing Dynamic JIT Compilation")
    print("\n--- Test 1: JIT with If-Else ---")
    test_jit_with_if_else_backward()
    print("\n--- Test 2: JIT with If-Else (Imperative) ---")
    test_jit_with_if_else_grad_imperative()
    print("\n--- Test 3: JIT with If-Else (Functional) ---")
    test_jit_with_if_else_grad_functional()
