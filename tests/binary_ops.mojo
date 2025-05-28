# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import nabla


fn test_binary_op0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 3, 4))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op0) passed.")
    else:
        print("❌ Test (binary op0) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        x = nabla.incr_batch_dim_ctr(x)
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 3, 4))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op1) passed.")
    else:
        print("❌ Test (binary op1) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        y = nabla.incr_batch_dim_ctr(y)
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 3, 4))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op2) passed.")
    else:
        print("❌ Test (binary op2) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        y = nabla.incr_batch_dim_ctr(y)
        y = nabla.incr_batch_dim_ctr(y)
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 3, 4))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op3) passed.")
    else:
        print("❌ Test (binary op3) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op4() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((3, 4))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op4) passed.")
    else:
        print("❌ Test (binary op4) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op5() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        y = nabla.incr_batch_dim_ctr(y)
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((4,))
    var y = nabla.arange((2, 3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op5) passed.")
    else:
        print("❌ Test (binary op5) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op6() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        x = nabla.incr_batch_dim_ctr(x)
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 4))
    var y = nabla.arange((3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op6) passed.")
    else:
        print("❌ Test (binary op6) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_binary_op7() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        var y = args[1]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        y = nabla.incr_batch_dim_ctr(y)
        var res = x + y
        return List(res)

    var x = nabla.arange((2, 3))
    var y = nabla.arange((3, 4))
    var res = test_func([x, y])[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (binary op7) passed.")
    else:
        print("❌ Test (binary op7) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())
