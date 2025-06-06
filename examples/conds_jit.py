# ===----------------------------------------------------------------------=== #
# Endia 2025
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

import endia as nd


def test_jit_with_if_else():
    """Test JIT compilation with conditional statements"""

    device = nd.device("cpu")

    def func(inputs: list[nd.Array]) -> list[nd.Array]:
        x = inputs[0]
        x = nd.sin(x)

        x = nd.negate(x) if x.to_numpy().item() > 0.5 else x + nd.array([1000.0])

        x = x * 2
        return [x]

    jitted_func = func  # nd.jit(func)

    for _ in range(10):
        x0 = nd.array([2.0]).to(device)
        outputs0 = jitted_func([x0])
        print("Output:", outputs0[0])

        x1 = nd.array([3.0]).to(device)
        outputs1 = jitted_func([x1])
        print("Output:", outputs1[0])


if __name__ == "__main__":
    print("Testing JIT Compilation")
    test_jit_with_if_else()
