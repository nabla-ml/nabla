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

from memory import ArcPointer, memcpy
from collections import Dict
from nabla.engine.utils import TrafoMeta, GraphRepr, Callable
from nabla.compiler.engine import Model
from nabla.core.device_array import DeviceArray
from nabla.engine.executor import Executor
from python import Python, PythonObject


alias none: Int = -55555


@value
struct ExecutionContext(Copyable, Movable):
    var dict: ArcPointer[Dict[Int, ArcPointer[Model]]]

    fn __init__(out self):
        self.dict = ArcPointer(Dict[Int, ArcPointer[Model]]())

    fn __getitem__(self, key: Int) raises -> ArcPointer[Model]:
        return self.dict[][key]

    fn __setitem__(mut self, key: Int, value: ArcPointer[Model]) -> None:
        if key in self.dict[]:
            print("Warning: key-value pair alrey present in model cache.")
            return
        self.dict[][key] = value

    fn __contains__(self, key: Int) -> Bool:
        return key in self.dict[]

    fn clear(mut self) -> None:
        self.dict[].clear()


fn xpr(callable: Callable) raises -> GraphRepr:
    return GraphRepr(callable)


fn xpr(func: fn (List[Array]) raises -> List[Array]) raises -> GraphRepr:
    return GraphRepr(jit(func))


def realize(
    mut args: List[Array], ctx: Optional[ExecutionContext] = None
) -> None:
    var outs = List[DeviceArray]()
    for array in args:
        outs.append(array[].device_array[])
    executor = Executor(outs, ctx)
    executor.realize()


def realize(mut args: Array, ctx: Optional[ExecutionContext] = None) -> None:
    var outs = List(args)
    realize(outs, ctx)


fn to_numpy(input: nabla.Array) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var num_elements = 1
    for dim in input.shape():
        num_elements *= dim[]
    array = np.zeros(Python.list(input.shape()), dtype=np.float32)
    var dst = array.__array_interface__["data"][0].unsafe_get_as_pointer[
        DType.float32
    ]()
    var src = input.device_array[].impl[]._data.bitcast[Scalar[DType.float32]]()
    var length = 1
    for dim in input.shape():
        length *= dim[]
    memcpy(dst, src, length)
    return array
