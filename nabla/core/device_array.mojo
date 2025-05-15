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

from memory import memset_zero, ArcPointer, UnsafePointer, memcpy
from collections import Dict, Optional
from nabla.compiler.graph import Symbol
from nabla.compiler.tensor import TensorSpec, Tensor
import random
import math
from utils import Variant

from nabla.api.utils import ExecutionContext
from .utils import ShapeType, getshape, compact_dtype_repr
from nabla.engine.trafos.vjp_trafo import backward
from nabla.engine.executor import Executor


from nabla.ops.binary_ops import (
    add,
    mul,
    sub,
    div,
    matmul,
    gt,
    pow,
)
from nabla.ops.reduce_ops import sum
from nabla.ops.unary_ops import (
    sin,
    negate,
    cos,
    relu,
    log,
    gt_zero,
    incr_batch_dim_ctr,
    decr_batch_dim_ctr,
    cast,
)
from nabla.ops.view_ops import (
    permute,
    transpose,
    reshape,
    flatten,
    broadcast_to,
    stack,
    concat,
    split,
    squeeze,
    unsqueeze,
    array_slice,
)


fn default_setup(
    mut res: DeviceArray,
    args: List[DeviceArray] = List[DeviceArray](),
    name: String = "",
) raises -> None:
    pass


fn default_jvp(
    primals: List[DeviceArray],
    tangents: List[DeviceArray],
    output: DeviceArray,
) raises -> DeviceArray:
    if len(output.impl[].tangents) == 0:
        raise "Error in default JVP: array does not have a tangent"
    return DeviceArray(output.impl[].tangents[0])


fn default_eagerxpr(
    mut res: DeviceArray,
    args: List[DeviceArray] = List[DeviceArray](),
) raises -> None:
    pass


@value
struct ArrayImpl(Copyable, Movable):
    var id: Int
    var name: String
    var spec: TensorSpec
    var tangents: List[ArcPointer[Self]]
    var cotangent: List[ArcPointer[Self]]
    var requires_pullback: Bool
    var execution_context: Optional[ExecutionContext]
    var is_checkpoint: Bool
    var shape: List[Int]
    var dtype: DType
    var _batch_dim_ctr: Int
    var runtime_info: List[List[Int]]
    var _args: List[ArcPointer[Self]]
    var _data: UnsafePointer[Scalar[DType.uint8]]
    var _visited: Bool
    var _max_symbol: Optional[Symbol]
    var _diffable: Bool
    var _not_to_be_materialized: Bool
    var _maxpr: Optional[fn (List[Symbol], DeviceArray) raises -> Symbol]
    var _vjp: Optional[
        fn (
            List[DeviceArray], DeviceArray, DeviceArray
        ) raises -> List[DeviceArray]
    ]
    var _jvp: Optional[
        fn (
            List[DeviceArray], List[DeviceArray], DeviceArray
        ) raises -> DeviceArray
    ]
    var _eagerxpr: Optional[
        fn (mut DeviceArray, List[DeviceArray]) raises -> None
    ]
    var _compute_jvp: Bool
    var _tmp_is_input: Bool
    var _tmp_is_output: Bool
    var _dual: List[ArcPointer[Self]]
    var tmp_name: String

    fn __init__(
        out self,
        shape: List[Int],
        dtype: DType,
        requires_pullback: Bool,
        execution_context: Optional[ExecutionContext],
        owned ptr: UnsafePointer[Scalar[DType.uint8]],
        _maxpr: Optional[
            fn (List[Symbol], DeviceArray) raises -> Symbol
        ] = None,
        name: String = "",
    ) raises:
        self.id = -1
        self.spec = TensorSpec(dtype, shape)
        self.runtime_info = List[List[Int]]()
        if ptr != UnsafePointer[Scalar[DType.uint8]]():
            self._data = ptr
        else:
            self._data = UnsafePointer[Scalar[DType.uint8]].alloc(
                self.spec.bytecount()
            )
        self.tangents = List[ArcPointer[Self]]()
        self.cotangent = List[ArcPointer[Self]]()
        self._args = List[ArcPointer[Self]]()
        self._visited = False
        self._max_symbol = None
        self._diffable = requires_pullback
        self.requires_pullback = (
            False if self.spec.num_elements() == 0 else requires_pullback
        )
        self._not_to_be_materialized = True
        self.is_checkpoint = False
        self._maxpr = _maxpr
        self._vjp = None
        self._jvp = None
        self._eagerxpr = None
        if execution_context:
            self.execution_context = execution_context.value()
        else:
            self.execution_context = None
        self.shape = shape
        self.dtype = dtype
        self._batch_dim_ctr = 0
        self.name = name
        self._compute_jvp = False
        self._tmp_is_input = False
        self._tmp_is_output = False
        self._dual = List[ArcPointer[Self]]()
        self.tmp_name = ""

    fn __copyinit__(out self, read other: Self):
        self.id = other.id
        self.name = other.name
        self.spec = other.spec
        self.tangents = other.tangents
        self.cotangent = other.cotangent
        self.requires_pullback = other.requires_pullback
        self.execution_context = other.execution_context
        self.is_checkpoint = other.is_checkpoint
        self.shape = other.shape
        self.dtype = other.dtype
        self._batch_dim_ctr = other._batch_dim_ctr
        self.runtime_info = other.runtime_info
        self._args = other._args
        self._data = UnsafePointer[Scalar[DType.uint8]].alloc(
            self.spec.bytecount()
        )
        memcpy(self._data, other._data, self.spec.bytecount())
        self._visited = other._visited
        self._max_symbol = None
        self._diffable = other._diffable
        self._not_to_be_materialized = other._not_to_be_materialized
        self._maxpr = other._maxpr
        self._vjp = other._vjp
        self._jvp = other._jvp
        self._eagerxpr = other._eagerxpr
        self._compute_jvp = other._compute_jvp
        self._tmp_is_input = other._tmp_is_input
        self._tmp_is_output = other._tmp_is_output
        self._dual = List[ArcPointer[Self]]()
        self.tmp_name = other.tmp_name

    fn __moveinit__(out self, owned other: Self):
        self.id = other.id
        self.name = other.name
        self.spec = other.spec
        self.tangents = other.tangents
        self.cotangent = other.cotangent
        self.requires_pullback = other.requires_pullback
        self.execution_context = other.execution_context
        self.is_checkpoint = other.is_checkpoint
        self.shape = other.shape
        self.dtype = other.dtype
        self._batch_dim_ctr = other._batch_dim_ctr
        self.runtime_info = other.runtime_info
        self._args = other._args
        self._data = UnsafePointer[Scalar[DType.uint8]].alloc(
            self.spec.bytecount()
        )
        memcpy(self._data, other._data, self.spec.bytecount())
        self._visited = other._visited
        self._max_symbol = None
        self._diffable = other._diffable
        self._not_to_be_materialized = other._not_to_be_materialized
        self._maxpr = other._maxpr
        self._vjp = other._vjp
        self._jvp = other._jvp
        self._eagerxpr = other._eagerxpr
        self._compute_jvp = other._compute_jvp
        self._tmp_is_input = other._tmp_is_input
        self._tmp_is_output = other._tmp_is_output
        self._dual = List[ArcPointer[Self]]()
        self.tmp_name = other.tmp_name

    fn __del__(owned self):
        self._data.free()


@value
struct DeviceArray(Copyable, Movable, Writable, Stringable):
    var impl: ArcPointer[ArrayImpl]

    fn __init__(
        out self,
        shape: ShapeType,
        dtype: DType,
        requires_pullback: Bool = False,
        execution_context: Optional[ExecutionContext] = None,
        ptr: UnsafePointer[Scalar[DType.uint8]] = UnsafePointer[
            Scalar[DType.uint8]
        ](),
        _maxpr: Optional[
            fn (List[Symbol], DeviceArray) raises -> Symbol
        ] = None,
        name: String = "",
    ) raises:
        self.impl = ArcPointer(
            ArrayImpl(
                getshape(shape),
                dtype,
                requires_pullback,
                execution_context,
                ptr,
                _maxpr,
                name,
            )
        )
        if execution_context:
            self.impl[].execution_context = execution_context.value()

    fn __init__(out self, impl: ArcPointer[ArrayImpl]):
        self.impl = impl

    fn __copyinit__(out self, read other: Self):
        self.impl = other.impl

    fn __moveinit__(out self, owned other: Self):
        self.impl = other.impl^

    fn num_elements(self) raises -> Int:
        return self.impl[].spec.num_elements()

    fn visited(self) -> Bool:
        return self.impl[]._visited

    fn visited_(mut self, visited: Bool) -> None:
        self.impl[]._visited = visited

    fn is_tmp_input(self) -> Bool:
        return self.impl[]._tmp_is_input

    fn is_tmp_input_(mut self, is_tmp_input: Bool) -> None:
        self.impl[]._tmp_is_input = is_tmp_input

    fn id(self) -> Int:
        return self.impl[].id

    fn id_(mut self, id: Int) -> None:
        self.impl[].id = id

    fn not_to_be_materialized(self) -> Bool:
        return self.impl[]._not_to_be_materialized

    fn not_to_be_materialized_(mut self, not_to_be_materialized: Bool) -> None:
        self.impl[]._not_to_be_materialized = not_to_be_materialized

    fn is_tmp_output(self) -> Bool:
        return self.impl[]._tmp_is_output

    fn is_tmp_output_(mut self, is_tmp_output: Bool) -> None:
        self.impl[]._tmp_is_output = is_tmp_output

    fn compute_jvp(self) -> Bool:
        return self.impl[]._compute_jvp

    fn compute_jvp_(mut self, compute_jvp: Bool) -> None:
        self.impl[]._compute_jvp = compute_jvp

    fn tmp_name_(mut self, name: String) -> None:
        self.impl[].tmp_name = name

    fn tmp_name(self) -> String:
        return self.impl[].tmp_name

    fn has_tangent(self) -> Bool:
        return len(self.impl[].tangents) == 1

    fn has_cotangent(self) -> Bool:
        return len(self.impl[].cotangent) == 1

    fn to_max[dtype: DType](self) raises -> Tensor[dtype]:
        var s = self
        s.realize()
        var max_array = Tensor[dtype](self.impl[].spec)
        var max_array_ptr = max_array.unsafe_uint8_ptr()
        memcpy(max_array_ptr, self.impl[]._data, self.impl[].spec.bytecount())
        return max_array

    fn tangent(self) raises -> DeviceArray:
        if len(self.impl[].tangents) == 0:
            raise "No gradient found for array with id: " + String(self.id())
        return DeviceArray(self.impl[].tangents[0])

    fn cotangent(self) raises -> DeviceArray:
        if len(self.impl[].cotangent) == 0:
            return zeros_like(self)
        return DeviceArray(self.impl[].cotangent[0])

    fn grad(self) raises -> DeviceArray:
        return self.cotangent()

    fn tangent_(mut self, grad: DeviceArray) raises -> None:
        self.impl[].tangents = List(grad.impl)

    fn zero_tangent(mut self) raises -> None:
        # self.realize()
        self.impl[].tangents.clear()

    fn cotangent_(mut self, cotangent: DeviceArray) raises -> None:
        self.impl[].cotangent = List(cotangent.impl)

    fn has_dual(self) raises -> Bool:
        return len(self.impl[]._dual) == 1

    fn dual(self) raises -> DeviceArray:
        if not self.has_dual():
            raise "Error in retreiving dual: DeviceArray has no dual."
        return DeviceArray(self.impl[]._dual[0])

    fn dual_(mut self, other: Self) raises -> None:
        self.impl[]._dual = List(other.impl)

    fn name(self) raises -> String:
        return self.impl[].name

    fn name_(mut self, val: String) raises -> None:
        self.impl[].name = val

    fn zero_cotangent(mut self) raises -> None:
        # self.realize()
        self.impl[].cotangent.clear()

    fn zero_grad(mut self) raises -> None:
        self.zero_cotangent()

    fn args(self) raises -> List[DeviceArray]:
        var args_list = List[DeviceArray]()
        for arg in self.impl[]._args:
            args_list.append(DeviceArray(arg[]))
        return args_list

    fn clear_args(mut self) raises -> None:
        self.impl[]._args.clear()

    fn args_(mut self, _args: List[DeviceArray]) raises -> None:
        var args_impl = List[ArcPointer[ArrayImpl]]()
        for arg in _args:
            args_impl.append(arg[].impl)
        self.impl[]._args = args_impl

    fn __str__(self) -> String:
        try:
            var out_str: String = ""
            out_str += ""
            var dtype = self.impl[].spec.dtype()

            if dtype == DType.float16:
                out_str += String(self.to_max[DType.float16]())
            elif dtype == DType.float32:
                out_str += String(self.to_max[DType.float32]())
            elif dtype == DType.float64:
                out_str += String(self.to_max[DType.float64]())
            elif dtype == DType.int8:
                out_str += String(self.to_max[DType.int8]())
            elif dtype == DType.int16:
                out_str += String(self.to_max[DType.int16]())
            elif dtype == DType.int32:
                out_str += String(self.to_max[DType.int32]())
            elif dtype == DType.int64:
                out_str += String(self.to_max[DType.int64]())
            elif dtype == DType.uint8:
                out_str += String(self.to_max[DType.uint8]())
            elif dtype == DType.uint16:
                out_str += String(self.to_max[DType.uint16]())
            elif dtype == DType.uint32:
                out_str += String(self.to_max[DType.uint32]())
            elif dtype == DType.uint64:
                out_str += String(self.to_max[DType.uint64]())
            else:
                out_str += "Unsupported dtype" + String(dtype)

            out_str = out_str[7:]
            out_str = out_str[: out_str.find(", dtype=")]
            out_str += "\033[95m"
            out_str += ":"
            out_str += compact_dtype_repr(self.impl[].dtype) + String("[")
            for i in range(len(self.shape())):
                out_str += String(self.shape()[i])
                if i != len(self.shape()) - 1:
                    out_str += ","
            out_str += "]\033[0m"

            return out_str
        except Exception:
            return (
                "DeviceArray with id: "
                + String(self.id())
                + " can not be converted to string"
                + String(Exception)
            )

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn realize(
        mut self, execution_context: Optional[ExecutionContext] = None
    ) raises -> None:
        if self.num_elements() == 0:
            var _execution_context: Optional[ExecutionContext] = None
            if self.impl[].execution_context:
                _execution_context = self.impl[].execution_context
            elif execution_context:
                _execution_context = execution_context

            var outs = List(self)
            executor = Executor(outs, _execution_context)
            executor.realize()

    fn no_tangent(mut self) raises -> None:
        self.impl[]._diffable = False
        self.impl[].requires_pullback = False

    fn checkpoint(mut self, value: Bool = True) raises -> None:
        self.impl[].is_checkpoint = value

    fn requires_pullback_(mut self, value: Bool = True) raises -> None:
        self.impl[].requires_pullback = value
        self.impl[]._diffable = value

    fn requires_pullback(self) -> Bool:
        return self.impl[].requires_pullback

    fn requires_grad_(mut self, value: Bool = True) raises -> None:
        self.requires_pullback_(value)

    fn requires_grad(self) -> Bool:
        return self.requires_pullback()

    fn shape(self) -> List[Int]:
        return self.impl[].shape

    fn shape_(mut self, shape: List[Int]) raises -> None:
        self.impl[].shape = shape

    fn dtype(self) -> DType:
        return self.impl[].dtype

    fn dtype_(mut self, dtype: DType) raises -> None:
        self.impl[].dtype = dtype

    fn batch_dim_ctr(self) -> Int:
        return self.impl[]._batch_dim_ctr

    fn batch_dim_ctr_(mut self, batch_dim_ctr: Int) raises -> None:
        self.impl[]._batch_dim_ctr = batch_dim_ctr

    fn backward(mut self, remat: Bool = False) raises -> None:
        var output = self
        backward(output, remat)

    fn item[
        type: DType = DType.float32
    ](
        self, execution_context: Optional[ExecutionContext] = None
    ) raises -> Scalar[type]:
        var s = self
        s.realize(execution_context)
        if self.num_elements() != 1:
            raise "Item only supported for arrays with one element, got shape: " + self.shape().__str__()
        return self.load[type](0, execution_context)

    fn load[
        type: DType = DType.float32, width: Int = 1
    ](
        self, idx: Int, execution_context: Optional[ExecutionContext] = None
    ) raises -> SIMD[type, width]:
        var s = self
        s.realize(execution_context)
        if idx >= s.num_elements():
            raise "Index out of range in load"

        var dtype = self.impl[].spec.dtype()
        if dtype == type:
            return (
                self.impl[]
                ._data.bitcast[SIMD[type, 1]]()
                .load[width=width](idx)
            )
        else:
            if dtype == DType.float16:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.float16, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.float32:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.float32, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.float64:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.float64, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.int8:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.int8, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.int16:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.int16, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.int32:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.int32, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.int64:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.int64, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.uint8:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.uint8, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.uint16:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.uint16, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.uint32:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.uint32, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            elif dtype == DType.uint64:
                return (
                    self.impl[]
                    ._data.bitcast[SIMD[DType.uint64, 1]]()
                    .load[width=width](idx)
                    .cast[type]()
                )
            else:
                raise "Unsupported dtype: " + String(type)

    fn store[
        type: DType, width: Int
    ](
        mut self,
        idx: Int,
        value: SIMD[type, width],
        execution_context: Optional[ExecutionContext] = None,
    ) raises -> None:
        self.realize(execution_context)
        if idx >= self.num_elements():
            raise "Index out of range in store"

        var dtype = self.impl[].spec.dtype()
        if dtype == type:
            self.impl[]._data.bitcast[SIMD[type, 1]]().store(idx, value)
        else:
            if dtype == DType.float16:
                self.impl[]._data.bitcast[SIMD[DType.float16, 1]]().store(
                    idx, value.cast[DType.float16]()
                )
            elif dtype == DType.float32:
                self.impl[]._data.bitcast[SIMD[DType.float32, 1]]().store(
                    idx, value.cast[DType.float32]()
                )
            elif dtype == DType.float64:
                self.impl[]._data.bitcast[SIMD[DType.float64, 1]]().store(
                    idx, value.cast[DType.float64]()
                )
            elif dtype == DType.int8:
                self.impl[]._data.bitcast[SIMD[DType.int8, 1]]().store(
                    idx, value.cast[DType.int8]()
                )
            elif dtype == DType.int16:
                self.impl[]._data.bitcast[SIMD[DType.int16, 1]]().store(
                    idx, value.cast[DType.int16]()
                )
            elif dtype == DType.int32:
                self.impl[]._data.bitcast[SIMD[DType.int32, 1]]().store(
                    idx, value.cast[DType.int32]()
                )
            elif dtype == DType.int64:
                self.impl[]._data.bitcast[SIMD[DType.int64, 1]]().store(
                    idx, value.cast[DType.int64]()
                )
            elif dtype == DType.uint8:
                self.impl[]._data.bitcast[SIMD[DType.uint8, 1]]().store(
                    idx, value.cast[DType.uint8]()
                )
            elif dtype == DType.uint16:
                self.impl[]._data.bitcast[SIMD[DType.uint16, 1]]().store(
                    idx, value.cast[DType.uint16]()
                )
            elif dtype == DType.uint32:
                self.impl[]._data.bitcast[SIMD[DType.uint32, 1]]().store(
                    idx, value.cast[DType.uint32]()
                )
            elif dtype == DType.uint64:
                self.impl[]._data.bitcast[SIMD[DType.uint64, 1]]().store(
                    idx, value.cast[DType.uint64]()
                )
            else:
                raise "Unsupported dtype: " + String(type)

    fn __getitem__(self, slices: List[Slice]) raises -> DeviceArray:
        return array_slice(self, slices)

    fn __add__(self, other: DeviceArray) raises -> DeviceArray:
        return add(self, other)

    fn __add__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return add(self, _other)

    fn __add__(self, other: Int) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return add(self, _other)

    fn __radd__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        return self + other

    fn __radd__(self, other: Int) raises -> DeviceArray:
        return self + other

    fn __iadd__(mut self, other: DeviceArray) raises -> None:
        self = self + other

    fn __iadd__(mut self, other: SIMD[_, 1]) raises -> None:
        self = self + other

    fn __iadd__(mut self, other: Int) raises -> None:
        self = self + other

    fn __mul__(self, other: DeviceArray) raises -> DeviceArray:
        return mul(self, other)

    fn __mul__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return mul(self, _other)

    fn __mul__(self, other: Int) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return mul(self, _other)

    fn __rmul__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        return self * other

    fn __rmul__(self, other: Int) raises -> DeviceArray:
        return self * other

    fn __imul__(mut self, other: DeviceArray) raises -> None:
        self = self * other

    fn __imul__(mut self, other: SIMD[_, 1]) raises -> None:
        self = self * other

    fn __imul__(mut self, other: Int) raises -> None:
        self = self * other

    fn __sub__(self, other: DeviceArray) raises -> DeviceArray:
        return sub(self, other)

    fn __sub__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return sub(self, _other)

    fn __sub__(self, other: Int) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return sub(self, _other)

    fn __rsub__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return _other - self

    fn __rsub__(self, other: Int) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return _other - self

    fn __isub__(mut self, other: DeviceArray) raises -> None:
        self = self - other

    fn __isub__(mut self, other: SIMD[_, 1]) raises -> None:
        self = self - other

    fn __isub__(mut self, other: Int) raises -> None:
        self = self - other

    fn __truediv__(self, other: DeviceArray) raises -> DeviceArray:
        return div(self, other)

    fn __truediv__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        if other == 0:
            raise "Division by zero"
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return div(self, _other)

    fn __truediv__(self, other: Int) raises -> DeviceArray:
        if other == 0:
            raise "Division by zero"
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return div(self, _other)

    fn __rtruediv__(self, other: SIMD[_, 1]) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, other)
        return _other / self

    fn __rtruediv__(self, other: Int) raises -> DeviceArray:
        var _other = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _other.store(0, Float32(other))
        return _other / self

    fn __itruediv__(mut self, other: DeviceArray) raises -> None:
        self = self / other

    fn __itruediv__(mut self, other: SIMD[_, 1]) raises -> None:
        self = self / other

    fn __itruediv__(mut self, other: Int) raises -> None:
        self = self / other

    fn __neg__(self) raises -> DeviceArray:
        return negate(self)

    fn __matmul__(self, other: DeviceArray) raises -> DeviceArray:
        return matmul(self, other)

    fn T(self, x: Int = -2, y: Int = -1) raises -> DeviceArray:
        return transpose(self, x, y)

    fn reshape(self, shape: List[Int]) raises -> DeviceArray:
        return reshape(self, shape)

    fn __pow__(self, exp: DeviceArray) raises -> DeviceArray:
        return ops.binary_ops.pow(self, exp)

    fn __pow__(self, exp: SIMD[_, 1]) raises -> DeviceArray:
        var _exp = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _exp.store(0, exp)
        return ops.binary_ops.pow(self, _exp)

    fn __pow__(self, exp: Int) raises -> DeviceArray:
        var _exp = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _exp.store(0, Float32(exp))
        return ops.binary_ops.pow(self, _exp)

    fn __rpow__(self, exp: SIMD[_, 1]) raises -> DeviceArray:
        var _exp = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _exp.store(0, exp)
        return _exp**self

    fn __rpow__(self, exp: Int) raises -> DeviceArray:
        var _exp = DeviceArray(
            (1,), self.impl[].spec.dtype(), False, self.impl[].execution_context
        )
        _exp.store(0, Float32(exp))
        return _exp**self


fn ones(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    for i in range(res.num_elements()):
        res.store(i, Float32(1.0))
    return res


fn ones_like(
    x: DeviceArray,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = ones(x.shape(), dtype, requires_pullback, execution_context)
    res.impl[]._batch_dim_ctr = x.batch_dim_ctr()
    return res


fn full(
    shape: ShapeType,
    fill_value: SIMD[_, 1],
    dtype: DType = fill_value.dtype,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    for i in range(res.num_elements()):
        res.store(i, fill_value)
    return res


fn arange(
    start: Float32,
    end: Float32,
    step: Float32,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = DeviceArray(
        (Int((end - start) / step),),
        dtype,
        requires_pullback,
        execution_context,
    )
    for i in range(res.num_elements()):
        res.store(i, start + i * step)
    return res


fn arange(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    for i in range(res.num_elements()):
        res.store(i, Float32(i))
    return res


fn zeros(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    for i in range(res.num_elements()):
        res.store(i, Float32(0.0))
    return res


fn zeros_like(
    x: DeviceArray,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var res = zeros(x.shape(), dtype, requires_pullback, execution_context)
    res.impl[]._batch_dim_ctr = x.batch_dim_ctr()
    return res


fn randn(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    mean: Float64 = Float64(0.0),
    variance: Float64 = Float64(1.0),
) raises -> DeviceArray:
    random.seed() if seed == None else random.seed(seed.value())
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    var size = res.num_elements()
    if dtype == DType.float16:
        random.randn(
            res.impl[]._data.bitcast[SIMD[DType.float16, 1]](),
            size,
            mean,
            variance,
        )
    elif dtype == DType.float32:
        random.randn(
            res.impl[]._data.bitcast[SIMD[DType.float32, 1]](),
            size,
            mean,
            variance,
        )
    elif dtype == DType.float64:
        random.randn(
            res.impl[]._data.bitcast[SIMD[DType.float64, 1]](),
            size,
            mean,
            variance,
        )
    else:
        raise "Unsupported dtype"
    return res


fn rand(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    min: Float64 = Float64(0.0),
    max: Float64 = Float64(1.0),
) raises -> DeviceArray:
    random.seed() if seed == None else random.seed(seed.value())
    var res = DeviceArray(shape, dtype, requires_pullback, execution_context)
    var size = res.num_elements()
    if dtype == DType.float16:
        random.rand(
            res.impl[]._data.bitcast[SIMD[DType.float16, 1]](),
            size,
            min=min,
            max=max,
        )
    elif dtype == DType.float32:
        random.rand(
            res.impl[]._data.bitcast[SIMD[DType.float32, 1]](),
            size,
            min=min,
            max=max,
        )
    elif dtype == DType.float64:
        random.rand(
            res.impl[]._data.bitcast[SIMD[DType.float64, 1]](),
            size,
            min=min,
            max=max,
        )
    else:
        raise "Unsupported dtype"
    return res


fn he_normal(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
) raises -> DeviceArray:
    random.seed() if seed == None else random.seed(seed.value())
    var result = randn(shape, dtype, requires_pullback, execution_context)
    var _shape = result.impl[].spec.shape
    var fan_in = _shape[-1]
    var scaling_factor = Float64(math.sqrt(2.0 / fan_in))

    for i in range(result.num_elements()):
        result.store(i, result.load[DType.float32](i) * Float32(scaling_factor))

    return result


fn kronecker_delta(
    dim: Int,
    num_dims: Int,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> DeviceArray:
    var shape = List[Int]()
    for _ in range(num_dims):
        shape.append(dim)

    var strides = List[Int]()
    for _ in range(num_dims):
        strides.append(1)
    for i in range(num_dims - 2, -1, -1):
        strides[i] = strides[i + 1] * dim

    var result = zeros(shape, dtype, requires_pullback, execution_context)

    for i in range(dim):
        var offset = 0
        for j in range(num_dims):
            offset += i * strides[j]
        result.store(offset, Float32(1.0))

    return result
