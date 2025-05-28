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

#
from memory import ArcPointer
from collections import Optional
from nabla.core.device_array import DeviceArray
from nabla.api.utils import ExecutionContext
from nabla.core.utils import ShapeType
import nabla.core.device_array as devarr


@value
struct Array(Copyable, Movable, Writable, Stringable, Representable):
    var device_array: ArcPointer[DeviceArray]

    fn __init__(out self, read device_array: DeviceArray) raises:
        self.device_array = ArcPointer(device_array)
        self.device_array[].not_to_be_materialized_(False)

    fn __del__(owned self) -> None:
        if self.device_array.count() == 1:
            self.device_array[].not_to_be_materialized_(True)

    fn tangent(self) raises -> Array:
        return Array(self.device_array[].tangent())

    fn cotangent(self) raises -> Array:
        return Array(self.device_array[].cotangent())

    fn grad(self) raises -> Array:
        return Array(self.device_array[].grad())

    fn zero_tangent(mut self) raises -> None:
        self.device_array[].zero_tangent()

    fn zero_cotangent(mut self) raises -> None:
        self.device_array[].zero_cotangent()

    fn zero_grad(mut self) raises -> None:
        self.device_array[].zero_grad()

    fn __repr__(self) -> String:
        return self.device_array[].__repr__()

    fn __str__(self) -> String:
        return String(self.device_array[])

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))

    fn no_tangent(mut self) raises -> None:
        self.device_array[].no_tangent()

    fn checkpoint(mut self, value: Bool = True) raises -> None:
        self.device_array[].checkpoint(value)

    fn requires_pullback(self) raises -> Bool:
        return self.device_array[].requires_pullback()

    fn requires_pullback_(mut self, value: Bool = True) raises -> None:
        self.device_array[].requires_pullback_(value)

    fn requires_grad(self) raises -> Bool:
        return self.device_array[].requires_grad()

    fn requires_grad_(mut self, value: Bool = True) raises -> None:
        self.device_array[].requires_grad_(value)

    fn num_elements(self) raises -> Int:
        var num_elements = 1
        for dim in self.shape():
            num_elements *= dim[]
        return num_elements

    fn shape(self) raises -> List[Int]:
        return self.device_array[].shape()

    fn shape_(mut self, shape: List[Int]) raises -> None:
        self.device_array[].shape_(shape)

    fn dtype(self) raises -> DType:
        return self.device_array[].dtype()

    fn batch_dim_ctr(self) raises -> Int:
        return self.device_array[].batch_dim_ctr()

    fn batch_dim_ctr_(mut self, value: Int) raises -> None:
        self.device_array[].batch_dim_ctr_(value)

    fn backward(mut self, remat: Bool = False) raises -> None:
        self.device_array[].backward(remat)

    fn item[
        type: DType = DType.float64
    ](
        self, execution_context: Optional[ExecutionContext] = None
    ) raises -> Scalar[type]:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        return self.device_array[].item[type](_execution_context)

    fn load[
        type: DType = DType.float32, width: Int = 1
    ](
        self, idx: Int, execution_context: Optional[ExecutionContext] = None
    ) raises -> SIMD[type, width]:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        return self.device_array[].load[type, width](idx, _execution_context)

    fn store[
        type: DType, width: Int
    ](
        mut self,
        idx: Int,
        value: SIMD[type, width],
        execution_context: Optional[ExecutionContext] = None,
    ) raises -> None:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        self.device_array[].store[type, width](idx, value, _execution_context)

    fn __getitem__(self, *slices: Slice) raises -> Array:
        var slice_list = List[Slice]()
        for slice in slices:
            slice_list.append(slice[])
        return Array(self.device_array[].__getitem__(slice_list))

    fn __add__(self, other: Array) raises -> Array:
        return Array(self.device_array[] + other.device_array[])

    fn __add__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] + other)

    fn __add__(self, other: Int) raises -> Array:
        return Array(self.device_array[] + other)

    fn __radd__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other + self.device_array[])

    fn __radd__(self, other: Int) raises -> Array:
        return Array(other + self.device_array[])

    fn __iadd__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] + other.device_array[])

    fn __iadd__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] + other)

    fn __iadd__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] + other)

    fn __mul__(self, other: Array) raises -> Array:
        return Array(self.device_array[] * other.device_array[])

    fn __mul__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] * other)

    fn __mul__(self, other: Int) raises -> Array:
        return Array(self.device_array[] * other)

    fn __rmul__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other * self.device_array[])

    fn __rmul__(self, other: Int) raises -> Array:
        return Array(other * self.device_array[])

    fn __imul__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] * other.device_array[])

    fn __imul__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] * other)

    fn __imul__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] * other)

    fn __sub__(self, other: Array) raises -> Array:
        return Array(self.device_array[] - other.device_array[])

    fn __sub__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] - other)

    fn __sub__(self, other: Int) raises -> Array:
        return Array(self.device_array[] - other)

    fn __rsub__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other - self.device_array[])

    fn __rsub__(self, other: Int) raises -> Array:
        return Array(other - self.device_array[])

    fn __isub__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] - other.device_array[])

    fn __isub__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] - other)

    fn __isub__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] - other)

    fn __truediv__(self, other: Array) raises -> Array:
        return Array(self.device_array[] / other.device_array[])

    fn __truediv__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] / other)

    fn __truediv__(self, other: Int) raises -> Array:
        return Array(self.device_array[] / other)

    fn __rtruediv__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other / self.device_array[])

    fn __rtruediv__(self, other: Int) raises -> Array:
        return Array(other / self.device_array[])

    fn __itruediv__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] / other.device_array[])

    fn __itruediv__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] / other)

    fn __itruediv__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] / other)

    fn __neg__(self) raises -> Array:
        return Array(-self.device_array[])

    fn __matmul__(self, other: Array) raises -> Array:
        return Array(self.device_array[] @ other.device_array[])

    fn T(self, x: Int = -2, y: Int = -1) raises -> Array:
        return Array(self.device_array[].T(x, y))

    fn reshape(self, shape: List[Int]) raises -> Array:
        return Array(self.device_array[].reshape(shape))

    fn __pow__(self, exp: DeviceArray) raises -> Array:
        return Array(self.device_array[] ** exp)

    fn __pow__(self, exp: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] ** exp)

    fn __pow__(self, exp: Int) raises -> Array:
        return Array(self.device_array[] ** exp)

    fn __rpow__(self, exp: SIMD[_, 1]) raises -> Array:
        return Array(exp ** self.device_array[])

    fn __rpow__(self, exp: Int) raises -> Array:
        return Array(exp ** self.device_array[])


fn ones(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devarr.ones(shape, dtype, requires_grad, execution_context))


fn ones_like(
    array: Array,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devarr.ones_like(
            array.device_array[], dtype, requires_pullback, execution_context
        )
    )


fn full(
    shape: ShapeType,
    fill_value: SIMD[_, 1],
    dtype: DType = fill_value.dtype,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devarr.full(
            shape, fill_value, dtype, requires_pullback, execution_context
        )
    )


fn arange(
    start: Float32,
    end: Float32,
    step: Float32,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devarr.arange(
            start, end, step, dtype, requires_pullback, execution_context
        )
    )


fn arange(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devarr.arange(shape, dtype, requires_grad, execution_context))


fn zeros(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devarr.zeros(shape, dtype, requires_grad, execution_context))


fn zeros_like(
    array: Array,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devarr.zeros_like(
            array.device_array[], dtype, requires_pullback, execution_context
        )
    )


fn randn(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    mean: Float64 = Float64(0.0),
    variance: Float64 = Float64(1.0),
) raises -> Array:
    return Array(
        devarr.randn(
            shape,
            dtype,
            requires_grad,
            execution_context,
            seed,
            mean,
            variance,
        )
    )


fn rand(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    min: Float64 = Float64(0.0),
    max: Float64 = Float64(1.0),
) raises -> Array:
    return Array(
        devarr.rand(
            shape, dtype, requires_grad, execution_context, seed, min, max
        )
    )


fn he_normal(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
) raises -> Array:
    return Array(
        devarr.he_normal(shape, dtype, requires_grad, execution_context, seed)
    )
