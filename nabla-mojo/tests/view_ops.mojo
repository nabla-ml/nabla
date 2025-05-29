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

import nabla


fn test_broadcast0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.broadcast_to(x, (5, 2, 3, 4))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(5, 2, 3, 4):
        print("✅ Test (broadcast0) passed.")
    else:
        print("❌ Test (broadcast0) failed.")
        print("Expected shape: (5, 2, 3, 4), but got: ", res.shape().__str__())


fn test_broadcast1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (5, 3, 4))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 5, 3, 4):
        print("✅ Test (broadcast1) passed.")
    else:
        print("❌ Test (broadcast1) failed.")
        print("Expected shape: (2, 5, 3, 4), but got: ", res.shape().__str__())


fn test_broadcast2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (5, 4))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 5, 4):
        print("✅ Test (broadcast2) passed.")
    else:
        print("❌ Test (broadcast2) failed.")
        print("Expected shape: (2, 3, 5, 4), but got: ", res.shape().__str__())


fn test_broadcast3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (5,))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4, 5):
        print("✅ Test (broadcast3) passed.")
    else:
        print("❌ Test (broadcast3) failed.")
        print("Expected shape: (2, 3, 4, 5), but got: ", res.shape().__str__())


fn test_broadcast4() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.broadcast_to(x, (4, 5, 2, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(4, 5, 2, 3):
        print("✅ Test (broadcast3) passed.")
    else:
        print("❌ Test (broadcast3) failed.")
        print("Expected shape: (4, 5, 2, 3), but got: ", res.shape().__str__())


fn test_broadcast5() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (4, 5, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 4, 5, 3):
        print("✅ Test (broadcast3) passed.")
    else:
        print("❌ Test (broadcast3) failed.")
        print("Expected shape: (2, 4, 5, 3), but got: ", res.shape().__str__())


fn test_broadcast6() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (4, 5))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4, 5):
        print("✅ Test (broadcast3) passed.")
    else:
        print("❌ Test (broadcast3) failed.")
        print("Expected shape: (2, 3, 4, 5), but got: ", res.shape().__str__())


fn test_reshape0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.reshape(x, (2, 3, 4, 5))
        return [
            x,
        ]

    var x = nabla.arange((6, 20))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4, 5):
        print("✅ Test (reshape0) passed.")
    else:
        print("❌ Test (reshape0) failed.")
        print("Expected shape: (2, 3, 4, 5), but got: ", res.shape().__str__())


fn test_reshape1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.reshape(x, (4, 5))
        return [
            x,
        ]

    var x = nabla.arange((6, 20))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(6, 4, 5):
        print("✅ Test (reshape1) passed.")
    else:
        print("❌ Test (reshape1) failed.")
        print("Expected shape: (6, 4, 5), but got: ", res.shape().__str__())


fn test_reshape2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.reshape(x, (1, 1))
        return [
            x,
        ]

    var x = nabla.arange((6, 20))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(6, 20, 1, 1):
        print("✅ Test (reshape2) passed.")
    else:
        print("❌ Test (reshape2) failed.")
        print("Expected shape: (6, 20, 1, 1), but got: ", res.shape().__str__())


fn test_reshape3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.reshape(x, (120,))
        return [
            x,
        ]

    var x = nabla.arange((6, 20))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(
        120,
    ):
        print("✅ Test (reshape3) passed.")
    else:
        print("❌ Test (reshape3) failed.")
        print("Expected shape: (120,), but got: ", res.shape().__str__())


fn test_squeeze0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.squeeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((1, 2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (squeeze) passed.")
    else:
        print("❌ Test (squeeze) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_squeeze1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.squeeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 1, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (squeeze) passed.")
    else:
        print("❌ Test (squeeze) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_squeeze2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.squeeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 1, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (squeeze) passed.")
    else:
        print("❌ Test (squeeze) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_squeeze3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.squeeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4, 1))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (squeeze) passed.")
    else:
        print("❌ Test (squeeze) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_unsqueeze0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.unsqueeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(1, 2, 3, 4):
        print("✅ Test (unsqueeze) passed.")
    else:
        print("❌ Test (unsqueeze) failed.")
        print("Expected shape: (1, 2, 3, 4), but got: ", res.shape().__str__())


fn test_unsqueeze1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.unsqueeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 1, 3, 4):
        print("✅ Test (unsqueeze) passed.")
    else:
        print("❌ Test (unsqueeze) failed.")
        print("Expected shape: (2, 1, 3, 4), but got: ", res.shape().__str__())


fn test_unsqueeze2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.unsqueeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 1, 4):
        print("✅ Test (unsqueeze) passed.")
    else:
        print("❌ Test (unsqueeze) failed.")
        print("Expected shape: (2, 3, 1, 4), but got: ", res.shape().__str__())


fn test_unsqueeze3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.unsqueeze(x, List(0))
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4, 1):
        print("✅ Test (unsqueeze) passed.")
    else:
        print("❌ Test (unsqueeze) failed.")
        print("Expected shape: (2, 3, 4, 1), but got: ", res.shape().__str__())


fn test_slice0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = x[1:2]
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(1, 3, 4):
        print("✅ Test (slice) passed.")
    else:
        print("❌ Test (slice) failed.")
        print("Expected shape: (1, 3, 4), but got: ", res.shape().__str__())


fn test_slice1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = x[1:3]
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 2, 4):
        print("✅ Test (slice) passed.")
    else:
        print("❌ Test (slice) failed.")
        print("Expected shape: (2, 2, 4), but got: ", res.shape().__str__())


fn test_slice2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = x[0:2]
        return [
            x,
        ]

    var x = nabla.arange((2, 3, 4))
    var res = test_func(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 2):
        print("✅ Test (slice) passed.")
    else:
        print("❌ Test (slice) failed.")
        print("Expected shape: (2, 3, 2), but got: ", res.shape().__str__())


fn test_broadcast_vjp0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.broadcast_to(x, (4, 2, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var tangent = nabla.ones((4, 2, 3))
    var res = nabla.vjp(
        test_func,
        [
            x,
        ],
    )[1](
        [
            tangent,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3):
        print("✅ Test (broadcast_vjp0) passed.")
    else:
        print("❌ Test (broadcast_vjp0) failed.")
        print("Expected shape: (2, 3), but got: ", res.shape().__str__())


fn test_broadcast_vjp1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.broadcast_to(x, (4, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var tangent = nabla.ones((2, 4, 3))
    tangent = nabla.incr_batch_dim_ctr(tangent)
    var res = nabla.vjp(
        test_func,
        [
            x,
        ],
    )[1](
        [
            tangent,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 3):
        print("✅ Test (broadcast_vjp1) passed.")
    else:
        print("❌ Test (broadcast_vjp1) failed.")
        print("Expected shape: (2, 3), but got: ", res.shape().__str__())


fn test_broadcast_vjp2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.broadcast_to(x, (4, 2, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var tangent = nabla.ones((2, 4, 2, 3))
    tangent = nabla.incr_batch_dim_ctr(tangent)
    var res = nabla.vjp(
        test_func,
        [
            x,
        ],
    )[1](
        [
            tangent,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(2, 2, 3):
        print("✅ Test (broadcast_vjp2) passed.")
    else:
        print("❌ Test (broadcast_vjp2) failed.")
        print("Expected shape: (2, 2, 3), but got: ", res.shape().__str__())


fn test_broadcast_jacrev0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.broadcast_to(x, (4, 2, 3))
        return [
            x,
        ]

    var x = nabla.arange((2, 3))
    var res = nabla.jacrev(test_func)(
        [
            x,
        ]
    )[0]
    _ = res.load(0)

    if res.shape() == List(4, 2, 3, 2, 3):
        print("✅ Test (broadcast_jacrev0) passed.")
    else:
        print("❌ Test (broadcast_jacrev0) failed.")
        print(
            "Expected shape: (4, 2, 3, 2, 3), but got: ", res.shape().__str__()
        )
