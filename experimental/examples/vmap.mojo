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

import math
from time import perf_counter
import nabla


fn test_vmap() raises:
    fn dot(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = [
            nabla.sum(
                args[0] * args[1],
                axis=[
                    0,
                ],
            ),
        ]
        return res

    fn mv_prod(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(dot, [nabla.none, 1])(args)
        return res

    fn mm_prod(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(mv_prod, [0, nabla.none])(args)
        return res

    fn batched_matmul(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(mm_prod, [0, nabla.none])([args[0], args[1]])[0]
        return [
            res,
        ]

    var batch_a = nabla.ndarange((2, 3, 4), DType.float32)
    var mat_b = nabla.ndarange((4, 5), DType.float32)

    print(nabla.xpr(batched_matmul)([batch_a, mat_b]))
    var res = batched_matmul([batch_a, mat_b])
    print(res[0])


def test_vmap2():
    fn vv(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [
            nabla.sum(args[0] * args[1]),
        ]

    fn mv(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.vmap(vv, [0, nabla.none])(args)

    fn mm(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.vmap(
            mv,
            [nabla.none, 1],
            [
                1,
            ],
        )(args)

    var a = nabla.ndarange((2, 3), DType.float32)
    var b = nabla.ndarange((3, 4), DType.float32)

    var res = mm([a, b])[0]
    print(res)


def test_vmap3():
    fn br_foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [
            nabla.broadcast_to(args[0], (1, 3, 9)),
        ]

    var res = nabla.vmap(br_foo)(
        [
            nabla.ndarange((2, 9), DType.float32),
        ]
    )[0]
    print(res)


fn test_vmap4() raises:
    fn dot(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [
            nabla.sum(
                args[0] * args[1],
                axis=[
                    0,
                ],
            ),
        ]

    var mv_prod = nabla.vmap(dot, [nabla.none, 1])
    var mm_prod = nabla.vmap(mv_prod, [0, nabla.none])
    var batched_matmul = nabla.vmap(mm_prod, [0, nabla.none])

    var batch_a = nabla.ndarange((2, 3, 4), DType.float32)
    var mat_b = nabla.ndarange((4, 5), DType.float32)

    var res = batched_matmul([batch_a, mat_b])[0]
    print(res)
