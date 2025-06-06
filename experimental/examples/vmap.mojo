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

import math
from time import perf_counter
import endia


fn test_vmap() raises:
    fn dot(args: List[endia.Array]) raises -> List[endia.Array]:
        var res = [
            endia.sum(
                args[0] * args[1],
                axis=[
                    0,
                ],
            ),
        ]
        return res

    fn mv_prod(args: List[endia.Array]) raises -> List[endia.Array]:
        var res = endia.vmap(dot, [endia.none, 1])(args)
        return res

    fn mm_prod(args: List[endia.Array]) raises -> List[endia.Array]:
        var res = endia.vmap(mv_prod, [0, endia.none])(args)
        return res

    fn batched_matmul(args: List[endia.Array]) raises -> List[endia.Array]:
        var res = endia.vmap(mm_prod, [0, endia.none])([args[0], args[1]])[0]
        return [
            res,
        ]

    var batch_a = endia.arange((2, 3, 4), DType.float32)
    var mat_b = endia.arange((4, 5), DType.float32)

    print(endia.xpr(batched_matmul)([batch_a, mat_b]))
    var res = batched_matmul([batch_a, mat_b])
    print(res[0])


def test_vmap2():
    fn vv(args: List[endia.Array]) raises -> List[endia.Array]:
        return [
            endia.sum(args[0] * args[1]),
        ]

    fn mv(args: List[endia.Array]) raises -> List[endia.Array]:
        return endia.vmap(vv, [0, endia.none])(args)

    fn mm(args: List[endia.Array]) raises -> List[endia.Array]:
        return endia.vmap(
            mv,
            [endia.none, 1],
            [
                1,
            ],
        )(args)

    var a = endia.arange((2, 3), DType.float32)
    var b = endia.arange((3, 4), DType.float32)

    var res = mm([a, b])[0]
    print(res)


def test_vmap3():
    fn br_foo(args: List[endia.Array]) raises -> List[endia.Array]:
        return [
            endia.broadcast_to(args[0], (1, 3, 9)),
        ]

    var res = endia.vmap(br_foo)(
        [
            endia.arange((2, 9), DType.float32),
        ]
    )[0]
    print(res)


fn test_vmap4() raises:
    fn dot(args: List[endia.Array]) raises -> List[endia.Array]:
        return [
            endia.sum(
                args[0] * args[1],
                axis=[
                    0,
                ],
            ),
        ]

    var mv_prod = endia.vmap(dot, [endia.none, 1])
    var mm_prod = endia.vmap(mv_prod, [0, endia.none])
    var batched_matmul = endia.vmap(mm_prod, [0, endia.none])

    var batch_a = endia.arange((2, 3, 4), DType.float32)
    var mat_b = endia.arange((4, 5), DType.float32)

    var res = batched_matmul([batch_a, mat_b])[0]
    print(res)
