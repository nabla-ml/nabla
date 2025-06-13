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

"""Test JIT compilation"""

import nabla as nb


def test_jit_with_if_else():
    """Test JIT compilation with conditional statements"""

    device = nb.device("cpu")

    def func(inputs: list[nb.Array]) -> list[nb.Array]:
        x = inputs[0]
        x = nb.sin(x)

        x = nb.negate(x) if x.to_numpy().item() > 0.5 else x + nb.array([1000.0])

        x = x * 2
        return [x]

    jitted_func = nb.djit(func)

    for _ in range(10):
        x0 = nb.array([2.0]).to(device)
        outputs0 = jitted_func([x0])
        print("Output:", outputs0[0])

        x1 = nb.array([3.0]).to(device)
        outputs1 = jitted_func([x1])
        print("Output:", outputs1[0])


if __name__ == "__main__":
    print("Testing JIT Compilation")
    test_jit_with_if_else()
