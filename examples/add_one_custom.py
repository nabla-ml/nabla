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

from pathlib import Path

import max  # noqa: A004

import nabla as nb


class AddOneCustomOp(nb.UnaryOperation):
    """Custom unary operation for demonstration."""

    def __init__(self):
        super().__init__("add_one_custom")

    def maxpr(self, args: list[max.graph.Value], output: nb.Array) -> None:
        custom_result = max.graph.ops.custom(
            name="add_one_custom",
            values=args,
            out_types=[args[0].type],
            device=args[0].device,
        )
        output.tensor_value = custom_result[0]

    def custom_kernel_path(self):
        return Path(__file__).parent / "custom_kernels"

    def eagerxpr(self, args: list[nb.Array], output: nb.Array) -> None:
        np_result = args[0].to_numpy() + 1
        output.impl = max.driver.Tensor.from_numpy(np_result)

    def vjp_rule(
        self, primals: list[nb.Array], cotangent: nb.Array, output: nb.Array
    ) -> list[nb.Array]:
        raise NotImplementedError("VJP not implemented for AddOneCustomOp")

    def jvp_rule(
        self, primals: list[nb.Array], tangents: list[nb.Array], output: nb.Array
    ) -> nb.Array:
        raise NotImplementedError("JVP not implemented for AddOneCustomOp")


def add_one_custom(arg: nb.Array) -> list[nb.Array]:
    """Custom unary operation that adds one to the input."""
    return [AddOneCustomOp().forward(arg)]


if __name__ == "__main__":
    a = nb.ndarange((2, 3))
    print(a)

    jitted_add_one_custom = nb.jit(add_one_custom)

    res = jitted_add_one_custom(a)
    print(res)
