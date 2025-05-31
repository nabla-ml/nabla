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

if __name__ == "__main__":

    def dot(args: list[nb.Array]) -> list[nb.Array]:
        return [
            nb.reduce_sum(
                args[0] * args[1],
                axes=[0],
            )
        ]

    def mv_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(dot, [0, None])(args)

    def mm_prod(args: list[nb.Array]) -> list[nb.Array]:
        return nb.vmap(mv_prod, [None, 1], [1])(args)

    def batched_matmul(args: list[nb.Array]) -> list[nb.Array]:
        return [nb.vmap(mm_prod, [0, None])([args[0], args[1]])[0]]

    batch_a = nb.arange((2, 3, 4), nb.DType.float32)
    mat_b = nb.arange((4, 5), nb.DType.float32)
    print(nb.xpr(batched_matmul, [batch_a, mat_b]))
    res = batched_matmul([batch_a, mat_b])
    print(res[0])
