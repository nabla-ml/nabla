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


fn test_sum0() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.sum(x, List(0))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(3, 4):
        print("✅ Test (sum0) passed.")
    else:
        print("❌ Test (sum0) failed.")
        print("Expected shape: (3, 4), but got: ", res.shape().__str__())


fn test_sum1() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 4):
        print("✅ Test (sum1) passed.")
    else:
        print("❌ Test (sum1) failed.")
        print("Expected shape: (2, 4), but got: ", res.shape().__str__())


fn test_sum2() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 3):
        print("✅ Test (sum2) passed.")
    else:
        print("❌ Test (sum2) failed.")
        print("Expected shape: (2, 3), but got: ", res.shape().__str__())


fn test_sum3() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 4):
        print("✅ Test (sum3) passed.")
    else:
        print("❌ Test (sum3) failed.")
        print("Expected shape: (2, 3, 4), but got: ", res.shape().__str__())


fn test_sum4() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.sum(x, List(0, 1))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(4):
        print("✅ Test (sum4) passed.")
    else:
        print("❌ Test (sum4) failed.")
        print("Expected shape: (4), but got: ", res.shape().__str__())


fn test_sum5() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0, 1))
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2):
        print("✅ Test (sum5) passed.")
    else:
        print("❌ Test (sum5) failed.")
        print("Expected shape: (2), but got: ", res.shape().__str__())


fn test_sum6() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0), act_on_batch_dims=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(3, 4):
        print("✅ Test (sum6) passed.")
    else:
        print("❌ Test (sum6) failed.")
        print("Expected shape: (3, 4), but got: ", res.shape().__str__())


fn test_sum7() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0, 1), act_on_batch_dims=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(4):
        print("✅ Test (sum7) passed.")
    else:
        print("❌ Test (sum7) failed.")
        print("Expected shape: (4), but got: ", res.shape().__str__())


fn test_sum8() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.sum(x, List(0), keep_dim=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(1, 3, 4):
        print("✅ Test (sum8) passed.")
    else:
        print("❌ Test (sum8) failed.")
        print("Expected shape: (1, 3, 4), but got: ", res.shape().__str__())


fn test_sum9() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0), keep_dim=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 1, 4):
        print("✅ Test (sum9) passed.")
    else:
        print("❌ Test (sum9) failed.")
        print("Expected shape: (2, 1, 4), but got: ", res.shape().__str__())


fn test_sum10() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0), keep_dim=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 3, 1):
        print("✅ Test (sum10) passed.")
    else:
        print("❌ Test (sum10) failed.")
        print("Expected shape: (2, 3, 1), but got: ", res.shape().__str__())


fn test_sum11() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.sum(x, List(0, 1), keep_dim=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(1, 1, 4):
        print("✅ Test (sum11) passed.")
    else:
        print("❌ Test (sum11) failed.")
        print("Expected shape: (1, 1, 4), but got: ", res.shape().__str__())


fn test_sum12() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0, 1), keep_dim=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(2, 1, 1):
        print("✅ Test (sum12) passed.")
    else:
        print("❌ Test (sum12) failed.")
        print("Expected shape: (2, 1, 1), but got: ", res.shape().__str__())


fn test_sum13() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0), keep_dim=True, act_on_batch_dims=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(1, 3, 4):
        print("✅ Test (sum13) passed.")
    else:
        print("❌ Test (sum13) failed.")
        print("Expected shape: (1, 3, 4), but got: ", res.shape().__str__())


fn test_sum14() raises:
    fn test_func(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var x = args[0]
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.incr_batch_dim_ctr(x)
        x = nabla.sum(x, List(0, 1), keep_dim=True, act_on_batch_dims=True)
        return List(x)

    var x = nabla.arange((2, 3, 4))
    var res = test_func(List(x))[0]
    _ = res.load(0)

    if res.shape() == List(1, 1, 4):
        print("✅ Test (sum14) passed.")
    else:
        print("❌ Test (sum14) failed.")
        print("Expected shape: (1, 1, 4), but got: ", res.shape().__str__())
